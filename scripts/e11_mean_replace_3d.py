"""Mean-replace ONLY the (d_prc, d_refuse, d_style) coordinates at L19 output.

Leaves the ~4093-D orthogonal complement untouched. Comparing this
off-propaganda rate against the full-residual mean-replace tells you how
much of the load-bearing reader-band signal lives inside vs outside the
3D writer subspace.
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
from qwc.activations import cache_per_position_residuals
from qwc.directions import Directions, qr_orthonormalize
from qwc.generate import generate
from qwc.hooks import make_subspace_patch_hook
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome


def _three_class_breakdown(judge_results):
    """Counts and fractions of the three outcome classes over a list of
    JudgeResult objects, computed via classify_outcome."""
    counts = {"off_propaganda": 0, "on_propaganda": 0, "incoherent": 0}
    for jr in judge_results:
        counts[classify_outcome(jr)] += 1
    total = max(1, len(judge_results))
    fractions = {k: v / total for k, v in counts.items()}
    return {"counts": counts, "fractions": fractions, "n": len(judge_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e11_mean_replace_3d.json"))
    ap.add_argument("--positions", type=int, default=7)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    target_items = [it for it in items if it["class"] == "prc_sensitive"
                    and it["topic"] != "tiananmen" and it["id"] not in excl]
    source_items = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]
    target_prompts = [it["chat"] for it in target_items]
    source_prompts = [it["chat"] for it in source_items]

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    tap = 20  # output of L19
    B = qr_orthonormalize([dirs.d_prc[tap], dirs.d_refuse[tap], dirs.d_style[tap]])

    lm = model_mod.load_posttrain()
    src_resid, _ = cache_per_position_residuals(lm, source_prompts, max_offset=args.positions, batch_size=4)
    # mean over source prompts at tap == output of L19
    src_mean = src_resid[:, tap, :, :].astype(np.float32).mean(0)  # [K, H]
    src_coords = (src_mean @ B).astype(np.float32)  # [K, 3]

    Bt = torch.from_numpy(B).to(lm.device).to(torch.float32)
    coords_t = torch.from_numpy(src_coords).to(lm.device).unsqueeze(0)  # [1, K, 3]

    # Restrict to last K positions only.
    K = args.positions

    def hook(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < K:
            return output
        proj = h[:, -K:, :] @ Bt.to(h.dtype)  # [B, K, 3]
        delta = (coords_t.to(h.dtype) - proj) @ Bt.to(h.dtype).T  # [B, K, H]
        h_new = h.clone()
        h_new[:, -K:, :] = h_new[:, -K:, :] + delta
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    hook_specs = ((lm.layers[19], "forward", hook),)
    texts = generate(lm, target_prompts, hook_specs=hook_specs,
                     max_new_tokens=128, batch_size=8, verbose=True)

    judge_items = [{"id": it["id"], "question": it["chat"], "response": t}
                   for it, t in zip(target_items, texts)]
    judged = judge_all(judge_items)
    offs = sum(1 for it in target_items if judged[it["id"]].register != "prc_propaganda")
    legacy_rate = offs / max(1, len(target_items))
    three_class = _three_class_breakdown([judged[it["id"]] for it in target_items])
    rollouts = [{"id": it["id"], "topic": it["topic"], "text": t,
                 "register": judged[it["id"]].register,
                 "coherence": judged[it["id"]].coherence}
                for it, t in zip(target_items, texts)]
    write_json(args.out, {
        "three_class": three_class,
        "legacy_binary_rate": legacy_rate,
        "n": len(target_items),
        "positions_replaced": K,
        "rollouts": rollouts,
    })
    tc = three_class["counts"]
    print(f"\n3-class: off={tc['off_propaganda']} on={tc['on_propaganda']} "
          f"incoherent={tc['incoherent']} (n={three_class['n']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
