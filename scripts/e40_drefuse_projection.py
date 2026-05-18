"""d_refuse projection across all classes.

For every prompt, project its tap-19 last-token residual onto unit d_refuse.
Report per-class statistics and flag the outliers (Arab Spring, aspirin
synthesis, OTC overdose) that sit at the tail of their nominal classes.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data
from qwc.directions import Directions, project_onto, per_class_stats
from qwc.io import load_npz, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e40_drefuse_projection.json"))
    args = ap.parse_args()

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    tap = config.DIRECTION_LAYOUT["d_refuse"]["tap"]
    proj = project_onto(acts["hidden"], dirs.d_refuse[tap], tap)

    items = data.all_items()
    excl = data.excluded_ids()
    groups = {"prc_sensitive": [], "neutral_political": [], "harmful": [], "harmless": []}
    for it in items:
        if it["id"] in excl:
            continue
        groups[it["class"]].append(it["id"])
    stats = per_class_stats(proj, ids, groups)

    id_to_proj = {pid: float(proj[i]) for i, pid in enumerate(ids)}
    # High-tail outliers per class (anti-monotone for harmful)
    out = {"tap": tap, "per_class": stats}
    out["neutral_top6"] = sorted(
        [(pid, id_to_proj[pid]) for pid in groups["neutral_political"]],
        key=lambda kv: -kv[1])[:6]
    out["harmless_top6"] = sorted(
        [(pid, id_to_proj[pid]) for pid in groups["harmless"]],
        key=lambda kv: -kv[1])[:6]
    out["harmful_bot6"] = sorted(
        [(pid, id_to_proj[pid]) for pid in groups["harmful"]],
        key=lambda kv: kv[1])[:6]
    out["flagged_ids"] = [(pid, id_to_proj[pid]) for pid in data.excluded_ids() if pid in id_to_proj]

    write_json(args.out, out)
    print(f"\nd_refuse projection at tap {tap}:")
    for cls, st in stats.items():
        print(f"  {cls:>20}: mean {st['mean']:+.2f} ± {st['std']:.2f}  n={st['n']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
