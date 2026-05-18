"""d_prc projection across all classes.

For every prompt, project its tap-14 last-token residual onto unit d_prc.
Report per-class mean ± std and identify the high-tail outliers in
neutral_political (Kosovo / Catalonia / Saudi).
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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e39_dprc_projection.json"))
    args = ap.parse_args()

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    tap = config.DIRECTION_LAYOUT["d_prc"]["tap"]
    proj = project_onto(acts["hidden"], dirs.d_prc[tap], tap)

    groups = {"prc": [], "neutral": [], "harmful": [], "harmless": []}
    items = data.all_items()
    excl = data.excluded_ids()
    for it in items:
        if it["id"] in excl:
            continue
        if it["class"] == "prc_sensitive":
            groups["prc"].append(it["id"])
        elif it["class"] == "neutral_political":
            groups["neutral"].append(it["id"])
        elif it["class"] == "harmful":
            groups["harmful"].append(it["id"])
        else:
            groups["harmless"].append(it["id"])
    stats = per_class_stats(proj, ids, groups)

    # Identify outliers in neutral
    id_to_proj = {pid: float(proj[i]) for i, pid in enumerate(ids)}
    outliers = sorted([(pid, p) for pid, p in id_to_proj.items() if pid in groups["neutral"]],
                      key=lambda kv: -kv[1])[:6]
    # And the six flagged IDs for context
    excl_table = [(pid, id_to_proj[pid]) for pid in data.excluded_ids() if pid in id_to_proj]

    out = {"tap": tap, "per_class": stats,
           "neutral_top6": outliers, "flagged_ids": excl_table}
    write_json(args.out, out)
    print(f"\nd_prc projection at tap {tap}:")
    for cls, st in stats.items():
        print(f"  {cls:>10}: mean {st['mean']:+.2f} ± {st['std']:.2f}  min {st['min']:+.2f}  max {st['max']:+.2f}  n={st['n']}")
    print("\nNeutral top-6 projections:")
    for pid, p in outliers:
        print(f"  {pid}: {p:+.2f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
