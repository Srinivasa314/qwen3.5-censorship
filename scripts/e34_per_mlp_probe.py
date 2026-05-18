"""Per-MLP verdict-decodability probe.

For each reader-band layer L in 20..31, cache the MLP output residual at the
last prompt token. Train a per-MLP linear probe (4-class) to predict the
verdict register from that single MLP's output, and report the
cross-validated accuracy — high accuracy across the band indicates the
verdict is redundantly decodable from every late MLP.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.hooks import installed
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch
from qwc.probes import cv_acc_softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(20, 32)))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e34_per_mlp_probe.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    keep = [it for it in items if it["id"] not in excl]
    # 4-class labels: tia / ccp_other / harmful / harmless+neutral combined
    def cls_label(it):
        if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen":
            return 0
        if it["class"] == "prc_sensitive":
            return 1
        if it["class"] == "harmful":
            return 2
        return 3  # harmless or neutral_political
    y = np.array([cls_label(it) for it in keep])
    prompts = [it["chat"] for it in keep]

    lm = model_mod.load_posttrain()
    captures: dict[int, list[torch.Tensor]] = {L: [] for L in args.layers}

    def make_hook(L):
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            captures[L].append(h[:, -1, :].detach().float().cpu())
            return output
        return hook

    BATCH = 8
    for bi in range(0, len(prompts), BATCH):
        chunk = prompts[bi : bi + BATCH]
        rendered = render_chat_batch(lm.tokenizer, chunk)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        specs = [(lm.layers[L].mlp, "forward", make_hook(L))
                 for L in args.layers if hasattr(lm.layers[L], "mlp")]
        with installed(specs), torch.inference_mode():
            lm.model(**enc, use_cache=False, return_dict=True)

    results: dict[int, float] = {}
    for L, parts in captures.items():
        if not parts:
            continue
        X = torch.cat(parts, 0).numpy()
        acc = cv_acc_softmax(X, y, n_classes=4, k=5)
        results[L] = float(acc)

    write_json(args.out, {"cv_acc_per_layer": results})
    print("\nPer-MLP CV accuracy:")
    for L, acc in sorted(results.items()):
        print(f"  L{L}: {acc:.3f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
