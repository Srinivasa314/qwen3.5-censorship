"""Writer-layer identification by two converging sweeps.

(a) Subspace-patch tap sweep: copy the 3D-subspace coordinates from a
    source prompt's residual at each tap onto a target prompt; measure
    verdict flip rate.
(b) α-effectiveness sweep across layers: with a fixed α and direction,
    steer at each layer in turn; measure off-default rate. The layer that
    peaks is the canonical writer for that direction.
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
from qwc.directions import Directions, qr_orthonormalize
from qwc.generate import generate
from qwc.hooks import make_steer_hook, make_subspace_patch_hook, numpy_to_device_unit
from qwc.io import load_npz, write_json
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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e6_writer_layer_id.json"))
    ap.add_argument("--direction", choices=["d_prc", "d_refuse", "d_style"], default="d_prc")
    ap.add_argument("--alpha-scan-layers", nargs="+", type=int,
                    default=[5, 9, 13, 17, 21, 25, 29])
    ap.add_argument("--alpha", type=float, default=-12.0)
    ap.add_argument("--tap-scan", nargs="+", type=int,
                    default=[8, 12, 14, 16, 18, 20, 22])
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    groups = data.class_means_groups()

    # Pick prompt class to steer over based on the direction.
    if args.direction == "d_refuse":
        cls_items = [it for it in data.all_items()
                     if it["class"] == "harmful" and it["id"] not in data.excluded_ids()]
        off_register_check = lambda r: r != "safety_refusal"
    elif args.direction == "d_style":
        cls_items = [it for it in data.all_items()
                     if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen"
                     and it["id"] not in data.excluded_ids()]
        off_register_check = lambda r: r != "prc_deflection"
    else:
        cls_items = [it for it in data.all_items()
                     if it["class"] == "prc_sensitive" and it["id"] not in data.excluded_ids()]
        off_register_check = lambda r: r not in {"prc_deflection", "prc_propaganda"}
    prompts = [it["chat"] for it in cls_items]
    ids_used = [it["id"] for it in cls_items]
    print(f"Direction {args.direction}: n_prompts = {len(prompts)}", flush=True)

    lm = model_mod.load_posttrain()
    judge_items = []
    log = {"alpha_layer_scan": [], "tap_subspace_scan": []}

    # (a) Alpha-effectiveness across layers
    d_unit_canon = numpy_to_device_unit(
        getattr(dirs, args.direction)[config.DIRECTION_LAYOUT[args.direction]["tap"]],
        lm.device,
    )
    for L in args.alpha_scan_layers:
        hook_specs = ((lm.layers[L], "forward", make_steer_hook(d_unit_canon, args.alpha)),)
        texts = generate(lm, prompts, hook_specs=hook_specs,
                         max_new_tokens=args.max_new, batch_size=8, verbose=False)
        for pid, t in zip(ids_used, texts):
            judge_items.append({"id": f"alpha_L{L}__{pid}", "question": next(it["chat"] for it in cls_items if it["id"]==pid), "response": t})
        log["alpha_layer_scan"].append({"layer": L, "n": len(prompts)})
        print(f"  alpha α={args.alpha} @ L={L}: queued {len(prompts)} judge items", flush=True)

    # (b) Subspace-patch tap sweep (full 3D)
    src_items = [it for it in data.all_items() if it["class"] == "harmless" and it["id"] not in data.excluded_ids()]
    src_ids = [it["id"] for it in src_items]
    id_to_idx = {p: i for i, p in enumerate(ids)}
    for tap in args.tap_scan:
        B = qr_orthonormalize([dirs.d_prc[tap], dirs.d_refuse[tap], dirs.d_style[tap]])
        Bt = torch.from_numpy(B).to(lm.device).to(torch.float32)
        # source class-mean coords at this tap
        src_mean = acts["hidden"][[id_to_idx[i] for i in src_ids], tap].astype(np.float32).mean(0)
        src_coords = (B.T @ src_mean).reshape(1, 1, 3)
        coords_t = torch.from_numpy(src_coords).to(lm.device)
        layer_to_hook = tap - 1  # hook L_{tap-1} so its output is the tap-th residual
        if layer_to_hook < 0:
            continue
        hook_specs = ((lm.layers[layer_to_hook], "forward", make_subspace_patch_hook(Bt, coords_t)),)
        texts = generate(lm, prompts, hook_specs=hook_specs,
                         max_new_tokens=args.max_new, batch_size=8, verbose=False)
        for pid, t in zip(ids_used, texts):
            judge_items.append({"id": f"tap_T{tap}__{pid}", "question": next(it["chat"] for it in cls_items if it["id"]==pid), "response": t})
        log["tap_subspace_scan"].append({"tap": tap, "layer_hooked": layer_to_hook, "n": len(prompts)})
        print(f"  subspace tap={tap}: queued", flush=True)

    print(f"\nJudging {len(judge_items)} rollouts ...", flush=True)
    judged = judge_all(judge_items)

    # Per-condition outcome breakdowns
    rollouts = []
    for entry in log["alpha_layer_scan"]:
        L = entry["layer"]
        results = [judged[f"alpha_L{L}__{pid}"] for pid in ids_used]
        offs = sum(1 for jr in results if off_register_check(jr.register))
        entry["legacy_binary_rate"] = offs / max(1, len(ids_used))
        entry["three_class"] = _three_class_breakdown(results)
        for pid in ids_used:
            jr = judged[f"alpha_L{L}__{pid}"]
            rollouts.append({"id": f"alpha_L{L}__{pid}", "condition": f"alpha_L{L}",
                             "register": jr.register, "coherence": jr.coherence})
    for entry in log["tap_subspace_scan"]:
        tap = entry["tap"]
        results = [judged[f"tap_T{tap}__{pid}"] for pid in ids_used]
        offs = sum(1 for jr in results if off_register_check(jr.register))
        entry["legacy_binary_rate"] = offs / max(1, len(ids_used))
        entry["three_class"] = _three_class_breakdown(results)
        for pid in ids_used:
            jr = judged[f"tap_T{tap}__{pid}"]
            rollouts.append({"id": f"tap_T{tap}__{pid}", "condition": f"tap_T{tap}",
                             "register": jr.register, "coherence": jr.coherence})

    log["rollouts"] = rollouts
    write_json(args.out, log)
    print("\nα-effectiveness across layers (off/on/incoherent):")
    for entry in log["alpha_layer_scan"]:
        tc = entry["three_class"]["counts"]
        print(f"  L={entry['layer']:>2}: ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print("\nSubspace-patch tap sweep (off/on/incoherent):")
    for entry in log["tap_subspace_scan"]:
        tc = entry["three_class"]["counts"]
        print(f"  tap={entry['tap']:>2} (hook L{entry['layer_hooked']:>2}): "
              f"({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
