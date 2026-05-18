"""Base-vs-posttrain mid-stack Chinese commitment under the chat template.

Runs the same 200 prompts under the same chat-template wrapper on both
checkpoints; for each tap, applies the logit lens and records the top-1
token plus the CJK-token fraction per class.

Also compares the diff-of-means directions extracted from each checkpoint:
their cosines drop in the mid-stack (the band where posttraining
concentrates its weight changes), and class-membership linear probes hit
high accuracy on both, which suggests the class representations are
already present in pretraining.

Run this after `00_cache_activations.py --checkpoint base` and
`00_cache_activations.py --checkpoint posttrain` have both completed.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import diff_of_means, pairwise_cosines, _unit_per_tap
from qwc.io import load_npz, write_json
from qwc.logit_lens import cjk_token_mask, lm_head_vocab_size, top1_per_tap, chinese_top1_fraction_per_tap
from qwc.probes import cv_acc_logreg


def per_class_cjk(top1_ids: np.ndarray, ids: list[str], items: list[dict],
                  cjk_mask: np.ndarray) -> dict[str, np.ndarray]:
    """For each class, return the CJK-top1 fraction per tap."""
    cls_map = {it["id"]: it["class"] for it in items}
    by_cls: dict[str, list[int]] = {}
    for i, pid in enumerate(ids):
        c = cls_map.get(pid)
        if c is None:
            continue
        by_cls.setdefault(c, []).append(i)
    out = {}
    for cls, idxs in by_cls.items():
        out[cls] = chinese_top1_fraction_per_tap(top1_ids[idxs], cjk_mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e2_base_vs_posttrain.json"))
    args = ap.parse_args()

    base = load_npz(config.RESULTS_DIR / "activations_base.npz")
    post = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")

    base_ids = [str(x) for x in base["ids"]]
    post_ids = [str(x) for x in post["ids"]]
    assert base_ids == post_ids, "ID order must match across checkpoints"

    items = data.all_items()
    by_id_class = {it["id"]: it["class"] for it in items}

    out = {"cjk_top1_per_class_per_tap": {}, "direction_cosines_per_tap": {},
           "probe_acc_per_tap": {}}

    # Logit-lens CJK commitment on both checkpoints.
    print("Loading posttrain for logit-lens ...", flush=True)
    lm_post = model_mod.load_posttrain()
    cjk_mask = cjk_token_mask(lm_post.tokenizer, lm_head_vocab_size(lm_post))

    print("Computing top1 per tap (posttrain) ...", flush=True)
    top1_post = top1_per_tap(post["hidden"], lm_post)
    out["cjk_top1_per_class_per_tap"]["posttrain"] = {
        cls: arr.tolist() for cls, arr in
        per_class_cjk(top1_post, post_ids, items, cjk_mask).items()
    }
    del lm_post

    print("Loading base for logit-lens ...", flush=True)
    lm_base = model_mod.load_base()
    cjk_mask_b = cjk_token_mask(lm_base.tokenizer, lm_head_vocab_size(lm_base))
    print("Computing top1 per tap (base) ...", flush=True)
    top1_base = top1_per_tap(base["hidden"], lm_base)
    out["cjk_top1_per_class_per_tap"]["base"] = {
        cls: arr.tolist() for cls, arr in
        per_class_cjk(top1_base, base_ids, items, cjk_mask_b).items()
    }
    del lm_base

    # Direction cosines per tap.
    groups = data.class_means_groups()
    for name, (pos_grp, neg_grp) in [
        ("d_prc",    ("all_prc",   "neutral")),
        ("d_refuse", ("harmful",   "harmless")),
        ("d_style",  ("tiananmen", "prc_other")),
    ]:
        d_post = _unit_per_tap(diff_of_means(post["hidden"], post_ids, groups[pos_grp], groups[neg_grp]))
        d_base = _unit_per_tap(diff_of_means(base["hidden"], base_ids, groups[pos_grp], groups[neg_grp]))
        cos_per_tap = (d_post * d_base).sum(axis=-1).tolist()
        out["direction_cosines_per_tap"][name] = cos_per_tap

    # Class-membership probe accuracies at canonical taps on both checkpoints.
    for name, (pos_grp, neg_grp, tap) in [
        ("prc_vs_neutral",   ("all_prc", "neutral", 14)),
        ("harmful_vs_harmless", ("harmful", "harmless", 19)),
    ]:
        for ckpt_label, blob, blob_ids in [("posttrain", post, post_ids), ("base", base, base_ids)]:
            id_to_idx = {pid: i for i, pid in enumerate(blob_ids)}
            pos_idx = [id_to_idx[i] for i in groups[pos_grp]]
            neg_idx = [id_to_idx[i] for i in groups[neg_grp]]
            X = np.concatenate([blob["hidden"][pos_idx, tap], blob["hidden"][neg_idx, tap]]).astype(np.float32)
            y = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
            acc = cv_acc_logreg(X, y, k=5)
            out["probe_acc_per_tap"].setdefault(name, {})[ckpt_label] = {"tap": tap, "cv_acc": acc}

    write_json(args.out, out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
