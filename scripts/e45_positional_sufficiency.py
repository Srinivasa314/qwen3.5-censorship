"""Positional sufficiency of the verdict-writer signal.

Steer the verdict direction at only the last K positions of the prompt
(prefill; generation proceeds untouched), sweep K, and measure the full
three-class outcome distribution. An all-position condition is the ceiling.

This characterises how much of the writable verdict can be set from the
end of the prompt alone versus needing the whole context. Success counts
only coherent, genuinely-decensored output (classify_outcome ->
off_propaganda); denial-template and incoherent collapses are reported
as their own classes, not as success.
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_steer_hook, numpy_to_device_unit
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome


def _dist(judge_results):
    c = Counter(classify_outcome(jr) for jr in judge_results)
    n = max(1, sum(c.values()))
    return {"counts": dict(c),
            "fractions": {k: c.get(k, 0) / n for k in
                          ("off_propaganda", "on_propaganda", "incoherent")},
            "n": sum(c.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 16])
    ap.add_argument("--alpha", type=float, default=-12.0)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e45_positional_sufficiency.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    target = [it for it in items if it["class"] == "prc_sensitive"
              and it["topic"] != "tiananmen" and it["id"] not in excl]
    prompts = [it["chat"] for it in target]

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    tap = config.DIRECTION_LAYOUT["d_prc"]["tap"]
    steer_layer = tap - 1
    lm = model_mod.load_posttrain()
    d_unit = numpy_to_device_unit(dirs.d_prc[tap], lm.device)

    # condition label -> position_window (None = all positions = ceiling)
    conditions = [(f"K{K}", (-K, None)) for K in args.ks] + [("all", None)]

    judge_items = []
    text_by_id: dict[str, str] = {}
    cond_by_id: dict[str, str] = {}
    for label, window in conditions:
        hook = make_steer_hook(d_unit, args.alpha, position_window=window)
        specs = ((lm.layers[steer_layer], "forward", hook),)
        print(f"\n{label}: generating ...", flush=True)
        texts = generate(lm, prompts, hook_specs=specs,
                         max_new_tokens=args.max_new, batch_size=8, verbose=False)
        for it, t in zip(target, texts):
            jid = f"{label}__{it['id']}"
            judge_items.append({"id": jid, "question": it["chat"], "response": t})
            text_by_id[jid] = t
            cond_by_id[jid] = label

    judged = judge_all(judge_items)

    by_cond: dict[str, list] = {label: [] for label, _ in conditions}
    for jid, label in cond_by_id.items():
        by_cond[label].append(judged[jid])
    summary = {label: _dist(by_cond[label]) for label, _ in conditions}

    rollouts = [{"id": jid, "condition": cond_by_id[jid], "text": text_by_id[jid],
                 "register": judged[jid].register, "coherence": judged[jid].coherence}
                for jid in text_by_id]

    write_json(args.out, {
        "direction": "d_prc", "tap": tap, "steer_layer": steer_layer,
        "alpha": args.alpha, "n_prompts": len(target),
        "summary": summary, "rollouts": rollouts,
    })
    print("\ncond   off%   on%   incoh%")
    for label, _ in conditions:
        f = summary[label]["fractions"]
        print(f"  {label:>4}: {100*f['off_propaganda']:5.1f} "
              f"{100*f['on_propaganda']:5.1f} {100*f['incoherent']:5.1f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
