"""Per-position logit lens.

Cache residuals at the last K positions at every tap; apply the logit lens;
report the CJK-top1 fraction per position-offset per class. This localises
which prompt positions the Chinese-template commitment concentrates at
(e.g. final user token vs chat-template boundary vs structural newline).
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
from qwc.activations import cache_per_position_residuals
from qwc.io import write_json
from qwc.logit_lens import cjk_token_mask, lm_head_vocab_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--tap", type=int, default=24)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e44_per_position_logit_lens.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    keep = [it for it in items if it["id"] not in excl]
    by_cls = {
        "tia":       [it for it in keep if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen"],
        "ccp_other": [it for it in keep if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen"],
        "harmful":   [it for it in keep if it["class"] == "harmful"],
        "harmless":  [it for it in keep if it["class"] == "harmless"],
        "neutral":   [it for it in keep if it["class"] == "neutral_political"],
    }

    lm = model_mod.load_posttrain()
    cjk_mask = cjk_token_mask(lm.tokenizer, lm_head_vocab_size(lm))

    out: dict = {"tap": args.tap, "max_offset": args.positions, "per_class": {}}
    inner = lm.model.model if hasattr(lm.model, "model") else lm.model
    norm = inner.norm; head = lm.model.lm_head
    for cls, subset in by_cls.items():
        prompts = [it["chat"] for it in subset]
        resid, mask = cache_per_position_residuals(
            lm, prompts, max_offset=args.positions, batch_size=4)
        # resid: [N, n_taps, K, H]
        t_resid = torch.from_numpy(resid[:, args.tap, :, :].astype(np.float32))
        N, K, H = t_resid.shape
        t_resid = t_resid.to(lm.device).to(torch.bfloat16).reshape(-1, H)
        with torch.inference_mode():
            logits = head(norm(t_resid))
            top1 = logits.argmax(-1).cpu().numpy().reshape(N, K)
        cjk_frac = cjk_mask[top1].astype(np.float32) * mask.astype(np.float32)  # mask out pad
        denom = mask.sum(axis=0).clip(min=1)
        per_offset = (cjk_frac.sum(axis=0) / denom).tolist()
        # offsets are -K..-1
        out["per_class"][cls] = {
            "per_offset_cjk_top1": dict(zip(range(-K, 0), per_offset)),
            "n_prompts": len(subset),
        }

    write_json(args.out, out)
    print(f"\nCJK-top1 fraction per offset (tap {args.tap}):")
    offsets = list(range(-args.positions, 0))
    print("  offset:  " + "  ".join(f"{o:>4}" for o in offsets))
    for cls, info in out["per_class"].items():
        row = info["per_offset_cjk_top1"]
        print(f"  {cls:>10}: " + "  ".join(f"{100*row.get(o,0):>3.0f}%" for o in offsets))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
