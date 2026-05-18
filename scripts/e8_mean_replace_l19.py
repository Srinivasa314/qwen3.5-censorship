"""Mean-replace the L19-output residual at the last seven positions.

For each PRC prompt, overwrite the L19-output residual at positions -7..-1
with the per-position mean over non-PRC prompts; resume generation and
measure the off-propaganda rate. This probes how much of the verdict
survives a full-residual (not subspace-restricted) overwrite at L19.
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
from qwc.hooks import make_mean_replace_hook
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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e8_mean_replace_l19.json"))
    ap.add_argument("--positions", type=int, default=7)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    target_items = [it for it in items
                    if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen"
                    and it["id"] not in excl]
    source_items = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]
    target_prompts = [it["chat"] for it in target_items]
    source_prompts = [it["chat"] for it in source_items]
    print(f"Target (ccp_other): {len(target_items)}; source (harmless): {len(source_items)}", flush=True)

    lm = model_mod.load_posttrain()

    # Cache last-K residuals for the source set; mean over prompts gives a
    # [K, H] tensor to broadcast.
    print("Caching source per-position residuals ...", flush=True)
    src_resid, src_mask = cache_per_position_residuals(lm, source_prompts,
                                                       max_offset=args.positions, batch_size=4)
    # Average over prompts at tap 20 (output of L19 == input of L20).
    src_mean_l19 = src_resid[:, 20, :, :].astype(np.float32).mean(0)  # [K, H]
    mean_tensor = torch.from_numpy(src_mean_l19).to(lm.device).to(torch.float32)
    # Hook expects [B, T, H] or [H]; we broadcast over batch.
    mean_tensor = mean_tensor.unsqueeze(0)  # [1, K, H]

    def hook_factory():
        K = mean_tensor.shape[1]
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] < K:
                return output
            h_new = h.clone()
            h_new[:, -K:, :] = mean_tensor.to(h.dtype).expand(h.shape[0], K, h.shape[-1])
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
        return hook

    hook_specs = ((lm.layers[19], "forward", hook_factory()),)
    print(f"Generating on {len(target_prompts)} target prompts with mean-replace ...", flush=True)
    texts = generate(lm, target_prompts, hook_specs=hook_specs,
                     max_new_tokens=128, batch_size=8, verbose=True)

    judge_items = [{"id": it["id"], "question": it["chat"], "response": t}
                   for it, t in zip(target_items, texts)]
    judged = judge_all(judge_items)

    off_prop = sum(1 for it in target_items if is_off_propaganda(judged[it["id"]]))
    rate = off_prop / max(1, len(target_items))
    three_class = _three_class_breakdown([judged[it["id"]] for it in target_items])
    write_json(args.out, {
        "off_propaganda_rate": rate,
        "three_class": three_class,
        "n": len(target_items),
        "positions_replaced": args.positions,
        "details": [
            {"id": it["id"], "topic": it["topic"], "text": t,
             "register": judged[it["id"]].register,
             "coherence": judged[it["id"]].coherence}
            for it, t in zip(target_items, texts)
        ],
    })
    print(f"\nOff-propaganda rate: {100*rate:.1f}%  ({off_prop}/{len(target_items)})")
    tc = three_class["counts"]
    print(f"3-class: off={tc['off_propaganda']} on={tc['on_propaganda']} "
          f"incoherent={tc['incoherent']} (n={three_class['n']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
