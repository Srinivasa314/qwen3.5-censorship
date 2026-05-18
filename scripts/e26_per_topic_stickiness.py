"""Per-topic stickiness sweep.

For each PRC subtopic (Hong Kong, Xi, Tibet, Xinjiang, Taiwan, Falun Gong)
gather its prompts; run d_prc steering at L13 with α=-10 and K samples per
prompt; LLM-judge each generation.

The off-propaganda rate per topic measures stickiness — how readily that
topic's propaganda template comes off under a fixed d_prc suppression dose.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
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
from qwc.taxonomy import PROPAGANDA_REGISTERS, classify_outcome, is_off_propaganda


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
    ap.add_argument("--alpha", type=float, default=-10.0)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e26_per_topic_stickiness.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    prc_other = [it for it in items if it["class"] == "prc_sensitive"
                 and it["topic"] != "tiananmen" and it["id"] not in excl]
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for it in prc_other:
        by_topic[it["topic"]].append(it)
    print("Topic counts:", {k: len(v) for k, v in by_topic.items()}, flush=True)

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    lm = model_mod.load_posttrain()
    tap = config.DIRECTION_LAYOUT["d_prc"]["tap"]
    L = config.DIRECTION_LAYOUT["d_prc"]["steer_layer"]
    d_unit = numpy_to_device_unit(dirs.d_prc[tap], lm.device)

    judge_items = []
    log: dict[str, dict] = {}
    for topic, subset in by_topic.items():
        prompts = [it["chat"] for it in subset] * args.n_samples
        ids_rep = [it["id"] for it in subset] * args.n_samples
        # Baseline + steered
        for label, alpha, do_sample, seed in [("baseline", 0.0, True, 1234),
                                              ("steered",  args.alpha, True, 1234)]:
            specs = () if alpha == 0.0 else \
                ((lm.layers[L], "forward", make_steer_hook(d_unit, alpha)),)
            texts = generate(lm, prompts, hook_specs=specs,
                             max_new_tokens=128, do_sample=do_sample,
                             temperature=0.7, top_p=0.9, seed=seed,
                             batch_size=8, verbose=False)
            for j, (pid, t) in enumerate(zip(ids_rep, texts)):
                judge_items.append({"id": f"{topic}__{label}__{j}__{pid}", "question": prompts[j], "response": t})

    judged = judge_all(judge_items)
    details = []
    for topic, subset in by_topic.items():
        baseline_prop = 0; baseline_total = 0; steered_off = 0; steered_total = 0
        steered_results = []
        for label in ("baseline", "steered"):
            for j, pid in enumerate([it["id"] for it in subset] * args.n_samples):
                key = f"{topic}__{label}__{j}__{pid}"
                if key in judged:
                    jr = judged[key]
                    details.append({"id": key, "topic": topic, "label": label,
                                    "register": jr.register, "coherence": jr.coherence})
                    if label == "baseline":
                        baseline_total += 1
                        if jr.register in PROPAGANDA_REGISTERS:
                            baseline_prop += 1
                    else:
                        steered_total += 1
                        steered_results.append(jr)
                        if is_off_propaganda(jr):
                            steered_off += 1
        log[topic] = {
            "baseline_propaganda_rate": baseline_prop / max(1, baseline_total),
            "off_propaganda_at_alpha":  steered_off / max(1, steered_total),
            "n_baseline": baseline_total, "n_steered": steered_total,
            # 3-class breakdown over the steered (post-suppression) rollouts.
            "three_class_steered": _three_class_breakdown(steered_results),
        }

    write_json(args.out, {"alpha": args.alpha, "by_topic": log, "details": details})
    print("\nTopic       baseline_prop   off_prop@α   (off/on/incoherent)")
    for topic, info in sorted(log.items(), key=lambda kv: -kv[1]["off_propaganda_at_alpha"]):
        tc = info["three_class_steered"]["counts"]
        print(f"  {topic:<12} {100*info['baseline_propaganda_rate']:>6.1f}%      "
              f"{100*info['off_propaganda_at_alpha']:>6.1f}%   "
              f"({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
