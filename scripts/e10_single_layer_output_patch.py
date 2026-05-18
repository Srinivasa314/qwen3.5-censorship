"""Single-layer output patch, L20-L28.

For each L in 20..28, replace L's output residual at the last K prompt-end
positions with the per-position layer-specific mean computed over non-PRC
source prompts; resume generation and measure off-propaganda. A flat
profile across layers indicates no single reader-band layer is the
commit layer.

NB: a single mean vector broadcast to *every* position destroys all
content (the model emits garbage, ~100% off-prop), so this uses the same
per-position class-mean replacement over the last K positions as the L19
mean-replace script, applied at each L20-L28 in turn. K is the number of
tail positions; default matches the last-7 convention.
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
from qwc.activations import cache_per_position_residuals
from qwc.generate import generate
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome, is_off_propaganda


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
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(20, 29)))
    ap.add_argument("--positions", type=int, default=7,
                    help="last-K tail positions to overwrite (matches E8's window)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e10_single_layer_output_patch.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    target_items = [it for it in items if it["class"] == "prc_sensitive"
                    and it["topic"] != "tiananmen" and it["id"] not in excl]
    source_items = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]
    target_prompts = [it["chat"] for it in target_items]
    source_prompts = [it["chat"] for it in source_items]

    lm = model_mod.load_posttrain()
    print("Caching source per-position residuals ...", flush=True)
    src_resid, _ = cache_per_position_residuals(lm, source_prompts,
                                                max_offset=args.positions, batch_size=4)
    # mean over source prompts, shape [n_taps, K, H]
    src_mean = src_resid.astype(np.float32).mean(0)
    K = args.positions

    judge_items = []
    summary: dict = {}
    for L in args.layers:
        # Use tap L+1 (output of L) for the class-mean residual at this layer.
        layer_mean = torch.from_numpy(src_mean[L + 1]).to(lm.device).to(torch.float32)  # [K, H]
        layer_mean = layer_mean.unsqueeze(0)  # [1, K, H]

        def hook_factory(K_local, mean_local):
            def hook(module, args_, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] < K_local:
                    return output
                h_new = h.clone()
                h_new[:, -K_local:, :] = mean_local.to(h.dtype).expand(h.shape[0], K_local, h.shape[-1])
                return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
            return hook

        hook_specs = ((lm.layers[L], "forward", hook_factory(K, layer_mean)),)
        print(f"\nL={L}: generating ...", flush=True)
        texts = generate(lm, target_prompts, hook_specs=hook_specs,
                         max_new_tokens=128, batch_size=8, verbose=False)
        for it, t in zip(target_items, texts):
            judge_items.append({"id": f"L{L}__{it['id']}", "question": it["chat"], "response": t})
        summary.setdefault(L, {})["n"] = len(target_items)

    judged = judge_all(judge_items)
    details = []
    for L in args.layers:
        offs = sum(1 for it in target_items if is_off_propaganda(judged[f"L{L}__{it['id']}"]))
        summary[L]["off_propaganda_rate"] = offs / max(1, len(target_items))
        summary[L]["three_class"] = _three_class_breakdown(
            [judged[f"L{L}__{it['id']}"] for it in target_items])
        for it in target_items:
            jr = judged[f"L{L}__{it['id']}"]
            details.append({"id": f"L{L}__{it['id']}", "layer": L,
                            "register": jr.register, "coherence": jr.coherence})

    write_json(args.out, {"summary": summary, "positions": K, "details": details})
    print("\nL  off-prop   (off/on/incoherent)")
    for L in args.layers:
        tc = summary[L]["three_class"]["counts"]
        print(f"  L{L}: {100*summary[L]['off_propaganda_rate']:.1f}%   "
              f"({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
