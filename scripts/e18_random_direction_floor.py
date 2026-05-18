"""Random-direction brittleness floor.

Generate K random unit-norm vectors in residual-stream space. For each, run
the same steering protocol at the writer layer over the same α grid. Average
off-default rate across the random directions gives the per-class
brittleness floor (the rate at which a random direction breaks the trained
template, distinct from a real direction's causal effect).
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
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_steer_hook
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome, is_off_propaganda


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
    ap.add_argument("--n-directions", type=int, default=8)
    ap.add_argument("--alphas", nargs="+", type=float, default=[-15.0, -25.0, -30.0])
    ap.add_argument("--steer-layer", type=int, default=18)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e18_random_direction_floor.json"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    by_cls = {
        "harmful":   [it for it in items if it["class"] == "harmful" and it["id"] not in excl],
        "ccp_other": [it for it in items if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen" and it["id"] not in excl],
        "tia":       [it for it in items if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen" and it["id"] not in excl],
    }
    # Each arm scores "left the trained default register" against its OWN
    # default. The harmful and tia defaults are single registers, so a plain
    # register-mismatch is the correct off-default test there. The ccp_other
    # default is the propaganda behaviour, whose escape must additionally
    # exclude the whitewash/denial register and require a coherent rollout.
    target_register = {
        "harmful": "safety_refusal",
        "tia": "prc_deflection",
    }

    def left_default(cls: str, jr) -> bool:
        if cls == "ccp_other":
            return is_off_propaganda(jr)
        return jr.register != target_register[cls]

    lm = model_mod.load_posttrain()
    H = lm.hidden_size

    rng = np.random.default_rng(args.seed)
    random_dirs = []
    for k in range(args.n_directions):
        v = rng.standard_normal(H).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-12)
        random_dirs.append(v)

    judge_items = []
    log = {cls: {alpha: [] for alpha in args.alphas} for cls in by_cls}
    for k, v in enumerate(random_dirs):
        d_unit = torch.from_numpy(v).to(lm.device).to(torch.float32)
        for alpha in args.alphas:
            specs = ((lm.layers[args.steer_layer], "forward", make_steer_hook(d_unit, alpha)),)
            for cls, subset in by_cls.items():
                prompts = [it["chat"] for it in subset]
                texts = generate(lm, prompts, hook_specs=specs,
                                 max_new_tokens=args.max_new, batch_size=8, verbose=False)
                for it, t in zip(subset, texts):
                    judge_items.append({"id": f"rnd{k}__a{alpha:+g}__{cls}__{it['id']}",
                                        "question": it["chat"], "response": t})
                log[cls][alpha].append({"dir_index": k, "n": len(subset)})
        print(f"random dir {k+1}/{args.n_directions} generated", flush=True)

    judged = judge_all(judge_items)
    summary: dict = {}
    details = []
    for cls in by_cls:
        summary[cls] = {}
        for alpha in args.alphas:
            rates = []
            for k in range(args.n_directions):
                offs = 0
                for it in by_cls[cls]:
                    key = f"rnd{k}__a{alpha:+g}__{cls}__{it['id']}"
                    jr = judged[key]
                    if left_default(cls, jr):
                        offs += 1
                    details.append({"id": key, "class": cls, "alpha": alpha,
                                    "dir_index": k, "register": jr.register,
                                    "coherence": jr.coherence})
                rates.append(offs / max(1, len(by_cls[cls])))
            # 3-class breakdown over the same items pooled across all random
            # directions for this (class, alpha) cell.
            pooled = [judged[f"rnd{k}__a{alpha:+g}__{cls}__{it['id']}"]
                      for k in range(args.n_directions) for it in by_cls[cls]]
            summary[cls][alpha] = {
                "mean_off_rate": float(np.mean(rates)),
                "std_off_rate":  float(np.std(rates)),
                "per_direction_rates": rates,
                "three_class": _three_class_breakdown(pooled),
            }

    write_json(args.out, {"summary": summary, "n_directions": args.n_directions,
                          "details": details})
    print("\nRandom-direction floor:")
    for cls, by_alpha in summary.items():
        print(f"  {cls}:")
        for alpha, s in by_alpha.items():
            tc = s["three_class"]["counts"]
            print(f"    α={alpha:+g}: mean={100*s['mean_off_rate']:.1f}% ± "
                  f"{100*s['std_off_rate']:.1f}%   3-class off/on/incoh="
                  f"{tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
