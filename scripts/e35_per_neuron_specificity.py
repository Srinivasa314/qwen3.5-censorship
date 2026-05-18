"""Per-neuron class specificity at reader-band MLPs.

For each MLP layer L in 20..31, compute per-neuron mean activation per class.
Score by max cross-class mean-difference in SDs. Report the count of neurons
above thresholds (5, 7, 10 SDs) per layer and a small registry of the most
specific neurons.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.hooks import installed
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(20, 32)))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e35_per_neuron_specificity.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    keep = [it for it in items if it["id"] not in excl]
    cls_map = {}
    for it in keep:
        if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen":
            cls_map[it["id"]] = "tia"
        elif it["class"] == "prc_sensitive":
            cls_map[it["id"]] = "ccp_other"
        else:
            cls_map[it["id"]] = it["class"]

    lm = model_mod.load_posttrain()
    captures: dict[int, list[torch.Tensor]] = {L: [] for L in args.layers}

    def make_hook(L):
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            captures[L].append(h[:, -1, :].detach().float().cpu())
            return output
        return hook

    BATCH = 8
    for bi in range(0, len(keep), BATCH):
        chunk = keep[bi : bi + BATCH]
        rendered = render_chat_batch(lm.tokenizer, [it["chat"] for it in chunk])
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        specs = [(lm.layers[L].mlp, "forward", make_hook(L))
                 for L in args.layers if hasattr(lm.layers[L], "mlp")]
        with installed(specs), torch.inference_mode():
            lm.model(**enc, use_cache=False, return_dict=True)

    out_log = {"per_layer_stats": {}, "top_neurons": {}}
    for L in args.layers:
        if not captures[L]:
            continue
        X = torch.cat(captures[L], 0).numpy()  # [N, D]
        # Aggregate per class
        cls_to_idx = defaultdict(list)
        for i, it in enumerate(keep):
            cls_to_idx[cls_map[it["id"]]].append(i)
        # Per-neuron mean per class, plus a per-neuron pooled std
        means = {cls: X[idxs].mean(0) for cls, idxs in cls_to_idx.items()}
        pooled_std = np.sqrt(np.var(X, axis=0) + 1e-6)
        classes = list(means.keys())
        # For each neuron, max pairwise mean-diff / pooled std
        D = X.shape[1]
        max_disc = np.zeros(D)
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                disc = np.abs(means[classes[i]] - means[classes[j]]) / pooled_std
                max_disc = np.maximum(max_disc, disc)
        out_log["per_layer_stats"][L] = {
            "neurons_above_5_sd":  int((max_disc > 5).sum()),
            "neurons_above_7_sd":  int((max_disc > 7).sum()),
            "neurons_above_10_sd": int((max_disc > 10).sum()),
        }
        top = np.argsort(-max_disc)[:8].tolist()
        out_log["top_neurons"][L] = [
            {"neuron": int(n), "disc_sd": float(max_disc[n]),
             "per_class_mean": {cls: float(means[cls][n]) for cls in classes}}
            for n in top
        ]

    write_json(args.out, out_log)
    print("Per-layer counts above 5 SD:")
    for L, info in out_log["per_layer_stats"].items():
        print(f"  L{L}: {info}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
