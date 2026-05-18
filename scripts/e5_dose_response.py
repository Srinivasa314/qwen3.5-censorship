"""Dose-response sigmoid for each direction at its writer layer.

Sweep α at the writer layer for each direction; for each α, generate over
the matched prompt class, judge, and report the off-default-register rate.

Default registers per class:
    d_prc on prc -> default = prc_deflection / prc_propaganda
    d_refuse on harmful -> default = safety_refusal
    d_style on tia      -> default = prc_deflection
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


PROMPT_CLASS = {
    "d_prc":    ("prc_sensitive", {"prc_deflection", "prc_propaganda"}),
    "d_refuse": ("harmful",       {"safety_refusal"}),
    "d_style":  ("tia",           {"prc_deflection"}),
}


def gather_prompts(direction: str) -> list[dict]:
    """Return [{id, chat}, ...] for the relevant class. d_style uses Tiananmen only."""
    items = data.all_items()
    excl = data.excluded_ids()
    if direction == "d_style":
        return [it for it in items
                if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen"
                and it["id"] not in excl]
    cls = PROMPT_CLASS[direction][0]
    return [it for it in items if it["class"] == cls and it["id"] not in excl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["d_prc", "d_refuse", "d_style"], required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -15.0, -20.0, -25.0])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or str(config.RESULTS_DIR / f"e5_sigmoid_{args.direction}.json")

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    items = gather_prompts(args.direction)
    prompts = [it["chat"] for it in items]
    ids = [it["id"] for it in items]
    print(f"Sweeping {args.direction}; n_prompts={len(items)}; alphas={args.alphas}", flush=True)

    lm = model_mod.load_posttrain()
    tap = config.DIRECTION_LAYOUT[args.direction]["tap"]
    steer_layer = config.DIRECTION_LAYOUT[args.direction]["steer_layer"]
    d_unit = numpy_to_device_unit(getattr(dirs, args.direction)[tap], lm.device)
    default_registers = PROMPT_CLASS[args.direction][1]

    all_judge_items = []
    by_alpha: dict[float, list[dict]] = {}
    for alpha in args.alphas:
        if alpha == 0.0:
            hook_specs = ()
        else:
            hook_specs = ((lm.layers[steer_layer], "forward", make_steer_hook(d_unit, alpha)),)
        print(f"\nα = {alpha:+g}", flush=True)
        texts = generate(lm, prompts, hook_specs=hook_specs,
                         max_new_tokens=args.max_new, batch_size=args.batch_size, verbose=True)
        by_alpha[alpha] = []
        for pid, txt in zip(ids, texts):
            opaque = f"a{alpha:+g}__{pid}"
            all_judge_items.append({"id": opaque, "question": next(it for it in items if it['id']==pid)["chat"], "response": txt})
            by_alpha[alpha].append({"id": pid, "opaque": opaque, "text": txt})

    print(f"\nJudging {len(all_judge_items)} rollouts ...", flush=True)
    judged = judge_all(all_judge_items)

    summary = []
    for alpha in args.alphas:
        off = 0
        results = []
        for entry in by_alpha[alpha]:
            j = judged[entry["opaque"]]
            entry["register"] = j.register
            entry["coherence"] = j.coherence
            results.append(j)
            if j.register not in default_registers:
                off += 1
        tc_block = _three_class_breakdown(results)
        summary.append({
            "alpha": alpha,
            "three_class": tc_block,
            "legacy_binary_rate": off / max(1, len(by_alpha[alpha])),
            "n": len(by_alpha[alpha]),
        })
        tc = tc_block["counts"]
        print(f"  α={alpha:+g}: 3-class off/on/incoherent="
              f"{tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']} "
              f"(n={tc_block['n']})")

    blob = {
        "direction": args.direction,
        "steer_layer": steer_layer,
        "tap": tap,
        "default_registers": list(default_registers),
        "summary": summary,
        "raw": {str(a): rs for a, rs in by_alpha.items()},
    }
    write_json(out_path, blob)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
