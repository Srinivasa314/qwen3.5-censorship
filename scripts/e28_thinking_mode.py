"""Thinking-mode commitment + steering.

Tests whether the same writer circuit drives thinking mode:
(a) at the first <think>\\n token position, measure how well the
    (d_prc, d_refuse, d_style) coordinates separate the four classes;
(b) apply the same steering hooks at the same writer layers and measure
    whether each thinking-mode verdict breaks the way it does in
    no-think mode.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.activations import cache_last_token_residuals
from qwc.directions import Directions
from qwc.generate import generate
from qwc.hooks import make_steer_hook, numpy_to_device_unit
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome


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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e28_thinking_mode.json"))
    ap.add_argument("--batch-size", type=int, default=8)
    # Thinking mode emits a long <think>...</think> reasoning trace before
    # the answer; a small budget truncates before the answer is produced,
    # so the judge sees only reasoning. Give a generous budget and score
    # the post-</think> answer (see below).
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    keep = [it for it in items if it["id"] not in excl]
    by_cls = {
        "tia":       [it for it in keep if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen"],
        "ccp_other": [it for it in keep if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen"],
        "harmful":   [it for it in keep if it["class"] == "harmful"],
        "harmless":  [it for it in keep if it["class"] == "harmless"],
    }

    lm = model_mod.load_posttrain()
    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")

    # (a) Project last-token residual in thinking mode onto each direction at canonical taps.
    print("Caching last-token residuals in thinking mode ...", flush=True)
    chat_texts = [it["chat"] for cls in by_cls for it in by_cls[cls]]
    class_labels = [cls for cls in by_cls for _ in by_cls[cls]]
    resid = cache_last_token_residuals(lm, chat_texts, enable_thinking=True,
                                       batch_size=args.batch_size)

    out = {"projections": {}}
    for name in ("d_prc", "d_refuse", "d_style"):
        tap = config.DIRECTION_LAYOUT[name]["tap"]
        d = getattr(dirs, name)[tap]
        d = d / max(np.linalg.norm(d), 1e-12)
        proj = resid[:, tap, :].astype(np.float32) @ d
        per_cls: dict[str, dict] = {}
        for cls in by_cls:
            mask = np.array([c == cls for c in class_labels])
            if mask.sum():
                arr = proj[mask]
                per_cls[cls] = {"mean": float(arr.mean()), "std": float(arr.std()),
                                 "min": float(arr.min()), "max": float(arr.max()), "n": int(mask.sum())}
        out["projections"][name] = {"tap": tap, "per_class": per_cls}

    # (b) Steer at canonical layer in thinking mode and judge breakthrough.
    out["steering_breakthrough"] = {}
    rollouts = []
    for direction, target_cls, alpha, expected_off in [
        ("d_prc",    "tia",        -15.0, "prc_deflection"),
        ("d_refuse", "harmful",    -25.0, "safety_refusal"),
        ("d_style",  "tia",        -10.0, "prc_deflection"),
    ]:
        tap = config.DIRECTION_LAYOUT[direction]["tap"]
        L = config.DIRECTION_LAYOUT[direction]["steer_layer"]
        d_unit = numpy_to_device_unit(getattr(dirs, direction)[tap], lm.device)
        items_here = by_cls[target_cls]
        prompts = [it["chat"] for it in items_here]
        specs = ((lm.layers[L], "forward", make_steer_hook(d_unit, alpha)),)
        texts = generate(lm, prompts, hook_specs=specs, enable_thinking=True,
                         max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size, verbose=False)
        # Score the answer that follows the </think> terminator, not the
        # reasoning trace. If </think> never appears the model used the
        # whole budget on reasoning without answering -> a genuine
        # no-answer outcome at this budget; judge the raw text (will be
        # classified incoherent/partial) and flag it.
        parsed = []
        for t in texts:
            if "</think>" in t:
                parsed.append((t.split("</think>", 1)[1].strip(), True))
            else:
                parsed.append((t.strip(), False))
        judge_items = [{"id": f"{direction}__{it['id']}", "question": it["chat"],
                        "response": ans}
                       for it, (ans, _) in zip(items_here, parsed)]
        judged = judge_all(judge_items)
        results = [judged[f"{direction}__{it['id']}"] for it in items_here]
        offs = sum(1 for jr in results if jr.register != expected_off)
        n_reached = sum(1 for _, ok in parsed if ok)
        out["steering_breakthrough"][direction] = {
            "n": len(items_here),
            "alpha": alpha,
            "max_new_tokens": args.max_new_tokens,
            "n_reached_answer": n_reached,
            "three_class": _three_class_breakdown(results),
            "legacy_binary_rate": offs / max(1, len(items_here)),
            "expected_default": expected_off,
        }
        for it, t, (ans, ok) in zip(items_here, texts, parsed):
            jr = judged[f"{direction}__{it['id']}"]
            rollouts.append({"id": f"{direction}__{it['id']}", "condition": direction,
                             "target_class": target_cls,
                             "text": t, "answer": ans, "reached_answer": ok,
                             "register": jr.register, "coherence": jr.coherence})

    out["rollouts"] = rollouts
    write_json(args.out, out)
    print("\nThinking-mode projections (per-class means at canonical taps):")
    for name, sec in out["projections"].items():
        print(f"  {name} @ tap {sec['tap']}:")
        for cls, st in sec["per_class"].items():
            print(f"    {cls:>10}: {st['mean']:+.2f}")
    print("\nSteering breakthrough (off/on/incoherent) [answers reached]:")
    for k, v in out["steering_breakthrough"].items():
        tc = v["three_class"]["counts"]
        print(f"  {k}: ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})"
              f"  reached_answer={v['n_reached_answer']}/{v['n']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
