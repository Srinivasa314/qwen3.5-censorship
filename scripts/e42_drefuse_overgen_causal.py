"""Causal test of d_refuse overgeneralization.

Steer d_refuse at L18 with α in {0, -15, -25} on the two flagged
non-harmful prompts (Arab Spring self-immolation, aspirin synthesis).
At α=0 the model refuses these benign prompts; the sweep measures whether
subtracting d_refuse flips them to factual substantive answers.
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
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, -15.0, -25.0])
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e42_drefuse_overgen_causal.json"))
    args = ap.parse_args()

    meta = data.load()["_meta"]["overgeneralization_and_anomalies"]
    flagged_ids = [e["id"] for e in meta["drefuse_overgeneralization"]]
    by_id = data.by_id()
    items = [by_id[i] for i in flagged_ids]
    prompts = [it["chat"] for it in items]
    print(f"d_refuse overgen IDs: {flagged_ids}", flush=True)

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    lm = model_mod.load_posttrain()
    tap = config.DIRECTION_LAYOUT["d_refuse"]["tap"]
    d_unit = numpy_to_device_unit(dirs.d_refuse[tap], lm.device)

    judge_items = []
    log = {}
    for alpha in args.alphas:
        specs = () if alpha == 0.0 else \
            ((lm.layers[18], "forward", make_steer_hook(d_unit, alpha)),)
        texts = generate(lm, prompts, hook_specs=specs,
                         max_new_tokens=256, batch_size=8, verbose=False)
        for it, t in zip(items, texts):
            judge_items.append({"id": f"a{alpha:+g}__{it['id']}", "question": it["chat"], "response": t})
        log[alpha] = list(zip(flagged_ids, texts))

    judged = judge_all(judge_items)
    out_log: dict = {}
    three_class: dict = {}
    rollouts = []
    for alpha in args.alphas:
        out_log[alpha] = []
        results = []
        for pid, text in log[alpha]:
            j = judged[f"a{alpha:+g}__{pid}"]
            results.append(j)
            entry = {"id": pid, "text": text,
                     "register": j.register, "coherence": j.coherence}
            out_log[alpha].append(entry)
            rollouts.append({"id": f"a{alpha:+g}__{pid}", "alpha": alpha,
                             "text": text, "register": j.register,
                             "coherence": j.coherence})
        three_class[alpha] = _three_class_breakdown(results)
    write_json(args.out, {"three_class": three_class,
                          "by_alpha": out_log, "rollouts": rollouts})
    print("\n3-class per α (off/on/incoherent):")
    for alpha in args.alphas:
        tc = three_class[alpha]["counts"]
        print(f"  α = {alpha:+g}: "
              f"({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
