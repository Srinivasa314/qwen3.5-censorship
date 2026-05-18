"""Residual geometry vs reader-band template channel.

For each PRC prompt, cache the residual at tap 24 at α=0 (baseline) and
α=-10 (d_prc-suppressed). Project both onto a propaganda↔factual probe
direction; measure the per-topic shift.

If stickiness lived upstream in the residual, sticky topics would move less.
If stickiness lives downstream (reader-band template channel), every topic's
residual moves the same distance but downstream rendering differs.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions
from qwc.hooks import installed, make_steer_hook, numpy_to_device_unit
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch


def capture_resid_at_tap(lm, prompts: list[str], tap: int, hook_specs=()) -> np.ndarray:
    rendered = render_chat_batch(lm.tokenizer, prompts)
    enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
    with installed(hook_specs), torch.no_grad():
        r = lm.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    return r.hidden_states[tap][:, -1, :].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=-10.0)
    ap.add_argument("--read-tap", type=int, default=24)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e27_residual_geometry.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    prc_other = [it for it in items if it["class"] == "prc_sensitive"
                 and it["topic"] != "tiananmen" and it["id"] not in excl]
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for it in prc_other:
        by_topic[it["topic"]].append(it)

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    lm = model_mod.load_posttrain()
    tap = args.read_tap
    L = config.DIRECTION_LAYOUT["d_prc"]["steer_layer"]
    d_unit = numpy_to_device_unit(dirs.d_prc[config.DIRECTION_LAYOUT["d_prc"]["tap"]], lm.device)

    # Probe direction: just use d_prc[tap] as a proxy for propaganda↔factual axis.
    probe = dirs.d_prc[tap]

    by_topic_stats: dict[str, dict] = {}
    for topic, subset in by_topic.items():
        prompts = [it["chat"] for it in subset]
        base = capture_resid_at_tap(lm, prompts, tap)
        specs = ((lm.layers[L], "forward", make_steer_hook(d_unit, args.alpha)),)
        steered = capture_resid_at_tap(lm, prompts, tap, hook_specs=specs)
        proj_base = base @ probe
        proj_steered = steered @ probe
        by_topic_stats[topic] = {
            "n": len(subset),
            "mean_projection_baseline": float(proj_base.mean()),
            "mean_projection_steered":  float(proj_steered.mean()),
            "delta": float((proj_steered - proj_base).mean()),
        }

    write_json(args.out, {"alpha": args.alpha, "read_tap": tap, "by_topic": by_topic_stats})
    print(f"\nResidual projection at tap {tap} (probe = d_prc):")
    print(f"  topic        baseline    steered     delta")
    for topic, s in by_topic_stats.items():
        print(f"  {topic:<12} {s['mean_projection_baseline']:+8.2f}  {s['mean_projection_steered']:+8.2f}  {s['delta']:+6.2f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
