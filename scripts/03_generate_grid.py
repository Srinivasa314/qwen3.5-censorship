"""Generate the steering grid: 352 cells × 3 rollouts.

For each cell, install a forward hook on the cell's steer_layer that adds
alpha * direction_unit to the residual at every position, then decode
max_new_tokens tokens three times: greedy, sample seed=1234, sample seed=1235.

Output: results/steering_grid_generated.json — same schema as the published
file but without judge / coherence / in_predicted_window (those are filled
by the next script).
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, model as model_mod
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_steer_hook, numpy_to_device_unit
from qwc.io import read_json, write_json


def _normalise_cell_groups(cells: list[dict]):
    """Group cells by (direction, alpha, steer_layer) so each unique condition
    runs once over its prompts. Keeps the cell→prompt mapping for later merge.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        key = (c["direction"], float(c["alpha"]), int(c["steer_layer"]))
        groups[key].append(c)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs",  default=str(config.GRID_INPUTS_PATH))
    ap.add_argument("--directions", default=str(config.RESULTS_DIR / "directions_posttrain.npz"))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "steering_grid_generated.json"))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rollouts", nargs="+",
                    default=["greedy", "s0", "s1"],
                    help="any subset of greedy, s0, s1")
    ap.add_argument("--cell-limit", type=int, default=None,
                    help="for smoke-testing, cap the number of cells")
    args = ap.parse_args()

    inputs = read_json(args.inputs)
    meta = inputs["_meta"]
    cells = inputs["cells"]
    if args.cell_limit:
        cells = cells[: args.cell_limit]
        print(f"NB: limiting to first {len(cells)} cells (smoke test)", flush=True)

    print(f"Loading model from {config.MODEL_POSTTRAIN} ...", flush=True)
    lm = model_mod.load_posttrain()
    print(f"  loaded. layers={lm.num_layers}", flush=True)

    print(f"Loading directions from {args.directions} ...", flush=True)
    dirs = Directions.load(args.directions)

    prompts_meta = {p["id"]: p for p in meta["prompts"]}
    rollout_specs = {
        "greedy": dict(do_sample=False, seed=None),
        "s0":     dict(do_sample=True,  temperature=0.7, top_p=0.9, seed=1234),
        "s1":     dict(do_sample=True,  temperature=0.7, top_p=0.9, seed=1235),
    }

    groups = _normalise_cell_groups(cells)
    print(f"\n{len(cells)} cells -> {len(groups)} unique (direction, alpha, layer) conditions", flush=True)

    results_by_cell: dict[tuple, dict] = {}
    for c in cells:
        key = (c["prompt_id"], c["direction"], float(c["alpha"]))
        results_by_cell[key] = {**c, "rollouts": {}}

    t_overall = time.time()
    cond_index = 0
    for (direction, alpha, steer_layer), cells_in_cond in groups.items():
        cond_index += 1
        meta_dir = meta["directions"][direction]
        tap = meta_dir["steer_tap"]
        d_unit = numpy_to_device_unit(
            getattr(dirs, direction)[tap], lm.device, dtype=torch.float32
        )

        prompt_texts = [prompts_meta[c["prompt_id"]]["text"] for c in cells_in_cond]
        layer_module = lm.layers[steer_layer]

        if alpha == 0.0:
            hook_specs = ()
        else:
            hook_specs = ((layer_module, "forward", make_steer_hook(d_unit, alpha)),)

        print(f"\n[{cond_index}/{len(groups)}] direction={direction} alpha={alpha:+g} L={steer_layer}  "
              f"n_prompts={len(prompt_texts)}", flush=True)
        for roll_name in args.rollouts:
            spec = rollout_specs[roll_name]
            t0 = time.time()
            texts = generate(
                lm, prompt_texts,
                hook_specs=hook_specs,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                **spec,
            )
            for c, txt in zip(cells_in_cond, texts):
                key = (c["prompt_id"], c["direction"], float(c["alpha"]))
                results_by_cell[key]["rollouts"][roll_name] = {"text": txt}
            print(f"  rollout {roll_name}: {time.time()-t0:.1f}s", flush=True)

    out_blob = {
        "_meta": {
            **meta,
            "generated": time.strftime("%Y-%m-%d"),
            "max_new_tokens": args.max_new_tokens,
            "rollouts": [
                "greedy (do_sample=False)" if r == "greedy"
                else "s0 (T=0.7 seed=1234)" if r == "s0"
                else "s1 (seed=1235)" if r == "s1"
                else r
                for r in args.rollouts
            ],
        },
        "cells": list(results_by_cell.values()),
    }
    write_json(args.out, out_blob)
    print(f"\nDone in {time.time()-t_overall:.1f}s.  Saved {args.out}")


if __name__ == "__main__":
    main()
