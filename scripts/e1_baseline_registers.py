"""Baseline register on the 200 prompts at alpha = 0.

For each prompt, generate at baseline, judge into the 8-register taxonomy,
and report per-class register distribution. The non-PRC neutral_political
controls landing on factual_answer (rather than any refusal register) is
the test for whether the model's filter is PRC-specific or a generic
political filter.

The six flagged overgeneralization / anomaly IDs are reported separately so
their off-class baselines don't pollute the per-class headline rates.
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.generate import generate
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import REGISTERS, classify_outcome


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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e1_baseline.json"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    items = data.all_items()
    chat_texts = [it["chat"] for it in items]

    print(f"Loading posttrain ...", flush=True)
    lm = model_mod.load_posttrain()

    print(f"Generating baseline for {len(items)} prompts ...", flush=True)
    responses = generate(lm, chat_texts, batch_size=args.batch_size,
                         max_new_tokens=args.max_new, verbose=True)

    judge_items = [{"id": it["id"], "question": it["chat"], "response": resp}
                   for it, resp in zip(items, responses)]
    print(f"Judging ...", flush=True)
    judged = judge_all(judge_items)

    excl = data.excluded_ids()
    by_class_register: dict[str, Counter] = {}
    by_class_register_excl: dict[str, Counter] = {}
    rows = []
    for it, resp in zip(items, responses):
        j = judged[it["id"]]
        rows.append({
            "id": it["id"], "class": it["class"], "topic": it["topic"],
            "chat": it["chat"], "response": resp,
            "register": j.register, "coherence": j.coherence,
        })
        c = it["class"]
        by_class_register.setdefault(c, Counter())[j.register] += 1
        if it["id"] not in excl:
            by_class_register_excl.setdefault(c, Counter())[j.register] += 1

    three_class_by_class = {}
    for it, resp in zip(items, responses):
        if it["id"] in excl:
            continue
        three_class_by_class.setdefault(it["class"], []).append(judged[it["id"]])
    three_class = {c: _three_class_breakdown(jrs)
                   for c, jrs in three_class_by_class.items()}

    blob = {
        "three_class": three_class,
        "rows": rows,
        "by_class_register_including_excluded":  {k: dict(v) for k, v in by_class_register.items()},
        "by_class_register_excluding_excluded": {k: dict(v) for k, v in by_class_register_excl.items()},
        "registers": REGISTERS,
    }
    write_json(args.out, blob)
    print("\n3-class outcome per class (excluding the 6 flagged IDs):")
    for c, tc in three_class.items():
        cc = tc["counts"]
        print(f"  {c}: off={cc['off_propaganda']} on={cc['on_propaganda']} "
              f"incoherent={cc['incoherent']} (n={tc['n']})")
    print(f"Saved {args.out}")

    print("\n=== Per-class register distribution (excluding the 6 flagged IDs) ===")
    for cls, ctr in by_class_register_excl.items():
        n = sum(ctr.values())
        print(f"  {cls} (n={n}):")
        for reg in REGISTERS:
            if ctr[reg]:
                print(f"    {reg:>16}: {ctr[reg]:3d}  ({100*ctr[reg]/n:.0f}%)")


if __name__ == "__main__":
    main()
