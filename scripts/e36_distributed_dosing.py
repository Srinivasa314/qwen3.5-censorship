"""Distributed-dosing test.

Hold the total steering dose constant and split it across N writer-band
layers instead of concentrating at the canonical writer. Measures whether
spreading the same total α across layers preserves or loses the effect
(it loses it if each per-layer dose falls below the writer's sigmoid
threshold).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_steer_hook, numpy_to_device_unit
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
    ap.add_argument("--direction", choices=["d_prc", "d_refuse"], default="d_refuse")
    ap.add_argument("--alpha-total", type=float, default=-25.0)
    ap.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or str(config.RESULTS_DIR / f"e36_distributed_dosing_{args.direction}.json")
    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")

    if args.direction == "d_refuse":
        items = [it for it in data.all_items() if it["class"] == "harmful" and it["id"] not in data.excluded_ids()]
        canonical_L = 18
        default_register = "safety_refusal"
        # Spread across L11..L18 for N=8
        candidate_layers = [11, 12, 13, 14, 15, 16, 17, 18]
    else:
        items = [it for it in data.all_items() if it["class"] == "prc_sensitive" and it["id"] not in data.excluded_ids()]
        canonical_L = 13
        default_register = "prc_propaganda"  # or prc_deflection for tia
        candidate_layers = [6, 8, 10, 11, 12, 13, 14, 15]

    prompts = [it["chat"] for it in items]
    lm = model_mod.load_posttrain()
    tap = config.DIRECTION_LAYOUT[args.direction]["tap"]
    d_unit = numpy_to_device_unit(getattr(dirs, args.direction)[tap], lm.device)

    judge_items = []
    log = {}
    for N in args.ns:
        layers_used = candidate_layers[-N:]  # closer to canonical
        per_layer_alpha = args.alpha_total / N
        specs = tuple(
            (lm.layers[L], "forward", make_steer_hook(d_unit, per_layer_alpha))
            for L in layers_used
        )
        texts = generate(lm, prompts, hook_specs=specs,
                         max_new_tokens=96, batch_size=16, verbose=False)
        for it, t in zip(items, texts):
            judge_items.append({"id": f"N{N}__{it['id']}", "question": it["chat"], "response": t})
        log[N] = {"layers": layers_used, "per_layer_alpha": per_layer_alpha,
                  "n": len(items)}

    judged = judge_all(judge_items)
    rollouts = []
    for N in args.ns:
        results = [judged[f"N{N}__{it['id']}"] for it in items]
        offs = sum(1 for jr in results if jr.register != default_register)
        log[N]["legacy_binary_rate"] = offs / max(1, len(items))
        log[N]["three_class"] = _three_class_breakdown(results)
        for it in items:
            jr = judged[f"N{N}__{it['id']}"]
            rollouts.append({"id": f"N{N}__{it['id']}", "n_split": N,
                             "register": jr.register, "coherence": jr.coherence})

    write_json(out_path, {"direction": args.direction, "alpha_total": args.alpha_total,
                          "results": log, "rollouts": rollouts})
    print("\nN    layers          α/L         (off/on/incoherent)")
    for N, info in log.items():
        tc = info["three_class"]["counts"]
        print(f"  N={N:>2}: {info['layers']!s:<24} {info['per_layer_alpha']:+.2f}    "
              f"({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
