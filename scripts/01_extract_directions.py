"""Extract d_prc, d_refuse, d_style from cached activations.

Output: results/directions_{label}.npz (per-tap unit directions).

Also prints sanity checks: per-class projection ranges, pairwise cosines,
and per-topic d_prc cosines (the seven PRC topics should share one axis).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data
from qwc.directions import (
    extract_three_axes,
    diff_of_means,
    project_onto,
    per_class_stats,
    pairwise_cosines,
    _unit_per_tap,
)
from qwc.io import load_npz, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="posttrain")
    args = ap.parse_args()

    act_path = config.RESULTS_DIR / f"activations_{args.label}.npz"
    blob = load_npz(act_path)
    residuals = blob["hidden"]  # [N, n_taps, H]
    ids = [str(x) for x in blob["ids"]]
    print(f"Loaded {act_path}. shape={residuals.shape}", flush=True)

    dirs = extract_three_axes(residuals, ids)
    out_path = config.RESULTS_DIR / f"directions_{args.label}.npz"
    dirs.save(out_path)
    print(f"Saved {out_path}")

    groups = data.class_means_groups()
    report = {"per_class_projection": {}, "pairwise_cosines": {}, "per_topic_dprc_cosines": {}}

    for name in ("d_prc", "d_refuse", "d_style"):
        tap = config.DIRECTION_LAYOUT[name]["tap"]
        d = getattr(dirs, name)[tap]
        proj = project_onto(residuals, d, tap)
        report["per_class_projection"][name] = {"tap": tap}
        report["per_class_projection"][name]["stats"] = per_class_stats(proj, ids, groups)

    canonical = {n: getattr(dirs, n)[config.DIRECTION_LAYOUT[n]["tap"]] for n in ("d_prc", "d_refuse", "d_style")}
    pc = pairwise_cosines(canonical)
    report["pairwise_cosines"] = {f"{a}|{b}": v for (a, b), v in pc.items() if a < b}

    by_id = data.by_id()
    by_topic = defaultdict(list)
    for it in data.by_class()["prc_sensitive"]:
        if it["id"] in data.excluded_ids():
            continue
        by_topic[it["topic"]].append(it["id"])
    neutral_ids = groups["neutral"]
    per_topic_dirs = {}
    for topic, topic_ids in by_topic.items():
        if len(topic_ids) < 3:
            continue
        raw = diff_of_means(residuals, ids, topic_ids, neutral_ids)
        unit = _unit_per_tap(raw)
        per_topic_dirs[topic] = unit[config.DIRECTION_LAYOUT["d_prc"]["tap"]]
    if per_topic_dirs:
        pc_topic = pairwise_cosines(per_topic_dirs)
        report["per_topic_dprc_cosines"] = {f"{a}|{b}": v for (a, b), v in pc_topic.items() if a < b}

    rep_path = config.RESULTS_DIR / f"directions_{args.label}_report.json"
    write_json(rep_path, report)
    print(f"Saved {rep_path}")

    print("\nPairwise cosines @ canonical taps:")
    for k, v in report["pairwise_cosines"].items():
        print(f"  {k}: {v:+.3f}")

    print("\nPer-class projection ranges (mean ± std):")
    for name, sec in report["per_class_projection"].items():
        print(f"  {name} @ tap {sec['tap']}:")
        for cls, st in sec["stats"].items():
            print(f"    {cls:>10}: {st['mean']:+.2f} ± {st['std']:.2f}  n={st['n']}")


if __name__ == "__main__":
    main()
