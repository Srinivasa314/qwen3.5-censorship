"""Cumulative writer-output ablation.

For each writer-band layer L in 15..18, project d_style out of the layer's
MLP output at all positions. Sweep singles, pairs, and the full L15-L18
combined condition to see how much of the verdict survives when the
writer-band MLPs no longer write d_style (later layers may keep writing it
after the canonical writer band).
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_ablate_hook, numpy_to_device_unit
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
    ap.add_argument("--layers", nargs="+", type=int, default=[15, 16, 17, 18])
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e37_cumulative_writer_ablation.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    target_items = [it for it in items if it["class"] == "prc_sensitive"
                    and it["topic"] != "tiananmen" and it["id"] not in excl]

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    lm = model_mod.load_posttrain()
    tap = config.DIRECTION_LAYOUT["d_style"]["tap"]
    d_unit = numpy_to_device_unit(dirs.d_style[tap], lm.device)

    conditions = [tuple()]  # baseline
    for L in args.layers:
        conditions.append((L,))
    conditions.append(tuple(args.layers))  # all combined

    judge_items = []
    log = {}
    prompts = [it["chat"] for it in target_items]
    for cond in conditions:
        if cond:
            specs = tuple((lm.layers[L].mlp, "forward", make_ablate_hook(d_unit)) for L in cond)
            label = "L" + "_".join(str(L) for L in cond)
        else:
            specs = ()
            label = "baseline"
        texts = generate(lm, prompts, hook_specs=specs,
                         max_new_tokens=96, batch_size=16, verbose=False)
        for it, t in zip(target_items, texts):
            judge_items.append({"id": f"{label}__{it['id']}", "question": it["chat"], "response": t})
        log[label] = {"layers": list(cond), "n": len(target_items)}

    judged = judge_all(judge_items)
    rollouts = []
    for label, info in log.items():
        results = [judged[f"{label}__{it['id']}"] for it in target_items]
        offs = sum(1 for jr in results if jr.register != "prc_propaganda")
        info["legacy_binary_rate"] = offs / max(1, len(target_items))
        info["three_class"] = _three_class_breakdown(results)
        for it in target_items:
            jr = judged[f"{label}__{it['id']}"]
            rollouts.append({"id": f"{label}__{it['id']}", "condition": label,
                             "register": jr.register, "coherence": jr.coherence})

    out = {"conditions": log, "rollouts": rollouts}
    write_json(args.out, out)
    print("\nCondition          (off/on/incoherent)")
    for label, info in log.items():
        tc = info["three_class"]["counts"]
        print(f"  {label:<18} ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
