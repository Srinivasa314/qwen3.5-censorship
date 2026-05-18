"""Blind LLM-judge pass over a generated grid.

For each rollout, an LLM judge sees only the (question, response) pair and
classifies it into one of 8 registers plus a coherence flag. Cell IDs are
opaque/shuffled; the judge gets no direction, alpha, or source metadata.

Output: <input>.judged.json with rollouts.{roll}.{judge, coherence,
in_predicted_window} filled in.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config
from qwc.io import read_json, write_json
from qwc.judge import judge_all
from qwc.taxonomy import in_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.RESULTS_DIR / "steering_grid_generated.json"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    output = args.output or args.input.replace(".json", ".judged.json")

    blob = read_json(args.input)
    prompts_meta = {p["id"]: p for p in blob["_meta"]["prompts"]}

    items = []
    refs = []  # parallel list of (cell_index, rollout_name)
    for ci, cell in enumerate(blob["cells"]):
        for roll_name, roll in cell["rollouts"].items():
            opaque = f"cell{ci:04d}_{roll_name}"
            items.append({
                "id": opaque,
                "question": prompts_meta[cell["prompt_id"]]["text"],
                "response": roll["text"],
            })
            refs.append((ci, roll_name, opaque))
    print(f"Judging {len(items)} rollouts from {len(blob['cells'])} cells ...", flush=True)

    results = judge_all(items)

    # Merge back into the blob in original order.
    cells = blob["cells"]
    matches = 0
    for ci, roll_name, opaque in refs:
        cell = cells[ci]
        res = results[opaque]
        roll = cell["rollouts"][roll_name]
        roll["judge"] = res.register
        roll["coherence"] = res.coherence
        roll["in_predicted_window"] = in_window(res.register, cell["predicted_labels"])
        if roll["in_predicted_window"]:
            matches += 1

    print(f"In-window: {matches}/{len(items)} ({100*matches/len(items):.1f}%)")

    write_json(output, blob)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
