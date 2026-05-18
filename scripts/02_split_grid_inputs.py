"""Split steering_grid_public.json into an inputs-only file.

steering_grid_public.json carries both inputs (prompt × direction × alpha
× predicted labels) and outputs (rollouts with judge labels). This script
extracts the inputs in isolation so a fresh run of script 03 can generate
the outputs in the same schema.

Output: data/steering_grid_inputs.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config
from qwc.io import read_json, write_json


def main():
    pub = read_json(config.GRID_PUBLIC_PATH)

    # _meta passthrough but drop the "generated" stamp (we'll re-add at gen time).
    meta = dict(pub["_meta"])
    meta.pop("generated", None)

    cells_in: list[dict] = []
    for c in pub["cells"]:
        cells_in.append({
            "prompt_id":        c["prompt_id"],
            "class":            c["class"],
            "direction":        c["direction"],
            "steer_layer":      c["steer_layer"],
            "alpha":            c["alpha"],
            "predicted_labels": c["predicted_labels"],
        })

    out = {"_meta": meta, "cells": cells_in}
    write_json(config.GRID_INPUTS_PATH, out)
    print(f"Wrote {config.GRID_INPUTS_PATH}  ({len(cells_in)} cells)")


if __name__ == "__main__":
    main()
