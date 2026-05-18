"""Compare a freshly-generated, judged grid against the public reference.

Reports:
  - In-window rate (overall, by class, by direction, by rollout).
  - Verdict-match rate against the reference labels (per cell, per rollout).
  - Cells where our judge disagrees with the reference, with diffs printed.
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config
from qwc.io import read_json


def _cell_key(c):
    return (c["prompt_id"], c["direction"], float(c["alpha"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours",      default=str(config.RESULTS_DIR / "steering_grid_generated.judged.json"))
    ap.add_argument("--reference", default=str(config.GRID_PUBLIC_PATH))
    ap.add_argument("--show-diffs", type=int, default=20,
                    help="print up to N cells where our judge label differs from the reference")
    args = ap.parse_args()

    ours = read_json(args.ours)
    ref  = read_json(args.reference)

    ours_by_key = {_cell_key(c): c for c in ours["cells"]}
    ref_by_key  = {_cell_key(c): c for c in ref["cells"]}

    common = set(ours_by_key) & set(ref_by_key)
    print(f"Cells: ours={len(ours_by_key)}  reference={len(ref_by_key)}  common={len(common)}")

    rollouts = sorted({r for c in ours["cells"] for r in c["rollouts"]} &
                      {r for c in ref["cells"]  for r in c["rollouts"]})

    # In-window rate (independent of reference).
    in_win_total = Counter(); in_win_match = Counter()
    judge_agree = Counter(); judge_total = Counter()
    by_dir_in_win  = defaultdict(lambda: [0, 0])  # match, total
    by_cls_in_win  = defaultdict(lambda: [0, 0])
    diffs = []

    for key in sorted(common):
        ours_cell = ours_by_key[key]
        ref_cell  = ref_by_key[key]
        for roll in rollouts:
            o = ours_cell["rollouts"].get(roll)
            r = ref_cell["rollouts"].get(roll)
            if not o or not r:
                continue
            in_win_total[roll] += 1
            if o.get("in_predicted_window"):
                in_win_match[roll] += 1
            by_dir_in_win[ours_cell["direction"]][1] += 1
            if o.get("in_predicted_window"):
                by_dir_in_win[ours_cell["direction"]][0] += 1
            by_cls_in_win[ours_cell["class"]][1] += 1
            if o.get("in_predicted_window"):
                by_cls_in_win[ours_cell["class"]][0] += 1

            judge_total[roll] += 1
            if o.get("judge") == r.get("judge"):
                judge_agree[roll] += 1
            else:
                diffs.append({
                    "cell": key,
                    "rollout": roll,
                    "ours_judge": o.get("judge"),
                    "ref_judge":  r.get("judge"),
                    "ours_text":  (o.get("text") or "")[:200],
                    "ref_text":   (r.get("text") or "")[:200],
                })

    print("\n=== In-window rate (ours, per rollout) ===")
    for roll in rollouts:
        t = in_win_total[roll]; m = in_win_match[roll]
        print(f"  {roll}: {m}/{t}  ({100*m/max(1,t):.1f}%)")

    print("\n=== In-window rate by direction (ours) ===")
    for d, (m, t) in sorted(by_dir_in_win.items()):
        print(f"  {d}: {m}/{t}  ({100*m/max(1,t):.1f}%)")

    print("\n=== In-window rate by class (ours) ===")
    for c, (m, t) in sorted(by_cls_in_win.items()):
        print(f"  {c}: {m}/{t}  ({100*m/max(1,t):.1f}%)")

    print("\n=== Judge-label agreement with reference (per rollout) ===")
    for roll in rollouts:
        t = judge_total[roll]; m = judge_agree[roll]
        print(f"  {roll}: {m}/{t}  ({100*m/max(1,t):.1f}%)")

    if args.show_diffs and diffs:
        print(f"\n=== Showing first {min(args.show_diffs, len(diffs))} of {len(diffs)} judge-label divergences ===")
        for d in diffs[: args.show_diffs]:
            print(f"\n  {d['cell']} [{d['rollout']}]: ours={d['ours_judge']}  ref={d['ref_judge']}")
            print(f"    ours: {d['ours_text']!r}")
            print(f"    ref : {d['ref_text']!r}")


if __name__ == "__main__":
    main()
