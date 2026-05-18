"""Logit lens at every tap on the last prompt token.

At each tap, apply `final_norm + lm_head` to the cached residual at the last
prompt token; record the top-1 token and the CJK (Chinese)-token fraction
per class. This traces where (and in which language) the verdict commits
through the depth of the stack.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.io import load_npz, write_json
from qwc.logit_lens import cjk_token_mask, lm_head_vocab_size, top1_per_tap, chinese_top1_fraction_per_tap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e19_logit_lens.json"))
    args = ap.parse_args()

    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    items = data.all_items()
    cls_map = {it["id"]: it["class"] for it in items}
    topic_map = {it["id"]: it["topic"] for it in items}

    lm = model_mod.load_posttrain()
    cjk_mask = cjk_token_mask(lm.tokenizer, lm_head_vocab_size(lm))
    print("Computing top-1 per tap on the last prompt token ...", flush=True)
    top1 = top1_per_tap(acts["hidden"], lm)

    cls_idx: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(ids):
        cls = cls_map.get(pid)
        if cls is None:
            continue
        if cls == "prc_sensitive" and topic_map[pid] == "tiananmen":
            cls_idx["tia"].append(i)
        elif cls == "prc_sensitive":
            cls_idx["ccp_other"].append(i)
        elif cls == "neutral_political":
            cls_idx["neutral"].append(i)
        else:
            cls_idx[cls].append(i)

    out = {"cjk_top1_fraction_per_tap": {}, "top1_token_per_class_per_tap": {}}
    for cls, idx in cls_idx.items():
        out["cjk_top1_fraction_per_tap"][cls] = chinese_top1_fraction_per_tap(top1[idx], cjk_mask).tolist()
        # Per-class most-common top-1 token at each tap
        n_taps = top1.shape[1]
        per_tap = {}
        for tap in range(n_taps):
            ids_here = top1[idx, tap].tolist()
            from collections import Counter
            c = Counter(ids_here)
            most_common = c.most_common(3)
            decoded = [(lm.tokenizer.decode([tid]), count) for tid, count in most_common]
            per_tap[tap] = decoded
        out["top1_token_per_class_per_tap"][cls] = per_tap

    write_json(args.out, out)
    print("Class -> CJK-top1 fraction at key taps (20, 24, 28, 32):")
    for cls, fracs in out["cjk_top1_fraction_per_tap"].items():
        keys = [20, 24, 28, 32]
        s = "  ".join(f"tap{k}: {100*fracs[k]:.0f}%" for k in keys if k < len(fracs))
        print(f"  {cls:>10}:  {s}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
