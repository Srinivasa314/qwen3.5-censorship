"""d_style thinking-mode dose-response.

Sweep the d_style steering dose on the Tiananmen prompt set in thinking
mode, scoring the post-</think> answer with the 3-class judge on the
within-PRC register axis (deflection vs propaganda). d_style is the
register toggle rather than a decensoring direction, so the signal of
interest is the deflection->propaganda crossover as a function of |alpha|,
not an off-propaganda rate.

Steering uses make_steer_hook with no position window (all positions,
including decode steps), the same hook configuration as e28.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_steer_hook, numpy_to_device_unit
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome

ALPHAS = [0.0, -4.0, -8.0, -12.0, -16.0, -20.0, -25.0, -30.0, -40.0]
EXPECTED_DEFAULT = "prc_deflection"   # tia register if the verdict is NOT broken


def _three_class(judge_results):
    counts = {"off_propaganda": 0, "on_propaganda": 0, "incoherent": 0}
    for jr in judge_results:
        counts[classify_outcome(jr)] += 1
    n = max(1, len(judge_results))
    return {"counts": counts,
            "fractions": {k: v / n for k, v in counts.items()},
            "n": len(judge_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e47_dstyle_thinkdose.json"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    tia = [it for it in items
           if it["id"] not in excl and it["class"] == "prc_sensitive"
           and it["topic"] == "tiananmen"]
    print(f"tia prompts: {len(tia)}", flush=True)

    lm = model_mod.load_posttrain()
    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    tap = config.DIRECTION_LAYOUT["d_style"]["tap"]
    L = config.DIRECTION_LAYOUT["d_style"]["steer_layer"]
    d_unit = numpy_to_device_unit(getattr(dirs, "d_style")[tap], lm.device)
    prompts = [it["chat"] for it in tia]

    summary, rollouts = [], []
    for alpha in ALPHAS:
        specs = () if alpha == 0.0 else (
            (lm.layers[L], "forward", make_steer_hook(d_unit, alpha)),)
        texts = generate(lm, prompts, hook_specs=specs, enable_thinking=True,
                         max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size, verbose=False)
        parsed = []
        for t in texts:
            if "</think>" in t:
                parsed.append((t.split("</think>", 1)[1].strip(), True))
            else:
                parsed.append((t.strip(), False))
        ji = [{"id": f"a{alpha}__{it['id']}", "question": it["chat"], "response": a}
              for it, (a, _) in zip(tia, parsed)]
        judged = judge_all(ji)
        res = [judged[f"a{alpha}__{it['id']}"] for it in tia]
        tc = _three_class(res)
        n_reached = sum(1 for _, ok in parsed if ok)
        legacy = sum(1 for jr in res if jr.register != EXPECTED_DEFAULT) / max(1, len(res))
        summary.append({"alpha": alpha, "three_class": tc,
                         "n_reached_answer": n_reached,
                         "legacy_binary_rate": legacy, "n": len(tia)})
        for it, t, (a, ok) in zip(tia, texts, parsed):
            jr = judged[f"a{alpha}__{it['id']}"]
            rollouts.append({"alpha": alpha, "id": it["id"], "text": t,
                             "answer": a, "reached_answer": ok,
                             "register": jr.register, "coherence": jr.coherence})
        off = tc["fractions"]["off_propaganda"]
        print(f"  alpha={alpha:>6}: off={off:.2f} "
              f"(off/on/incoh {tc['counts']['off_propaganda']}/"
              f"{tc['counts']['on_propaganda']}/{tc['counts']['incoherent']}) "
              f"reached={n_reached}/{len(tia)}", flush=True)

    # first |alpha| (least negative) at which off-propaganda crosses 50%, if any
    half = next((s["alpha"] for s in summary
                 if s["three_class"]["fractions"]["off_propaganda"] >= 0.5), None)
    out = {"direction": "d_style", "target": "tia", "tap": tap, "steer_layer": L,
           "enable_thinking": True, "max_new_tokens": args.max_new_tokens,
           "alphas": ALPHAS, "half_dose_alpha": half,
           "summary": summary, "rollouts": rollouts}
    write_json(args.out, out)
    print(f"\nThinking-mode half-dose alpha: {half}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
