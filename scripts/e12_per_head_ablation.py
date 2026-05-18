"""Per-head ablation (single + top-10) and Q-zero variant.

Ranks heads by class-mean attention divergence (L1 distance between class-mean
attention vectors from the last prompt position) at each full-attention layer.

Single-head ablation: zero each head's o_proj input slice; sweep all heads.
Top-10 simultaneous ablation: identify the most-class-discriminative heads
and zero them all at once.
Q-zero variant: instead of zeroing the head's output, zero its Q projection
(forcing uniform attention).
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
from qwc.generate import generate
from qwc.hooks import installed
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.model import render_chat_batch, tokenize_batch
from qwc.patching import is_full_attention_layer, _attention_module
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


def attention_divergence(lm, items_a: list[dict], items_b: list[dict]) -> dict[tuple[int, int], float]:
    """For each (layer, head) full-attn, return L1 distance between class-mean
    attention from the last query position.
    """
    full_idx = [i for i in range(lm.num_layers) if is_full_attention_layer(lm.layers[i])]
    # We capture attention_probs via the standard HF output_attentions=True path.
    div: dict[tuple[int, int], float] = {}

    def collect(prompts: list[str]) -> dict[int, np.ndarray]:
        sums: dict[int, list[np.ndarray]] = {}
        BATCH = 4
        for bi in range(0, len(prompts), BATCH):
            chunk = prompts[bi : bi + BATCH]
            rendered = render_chat_batch(lm.tokenizer, chunk)
            enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
            with torch.no_grad():
                out = lm.model(**enc, output_attentions=True, use_cache=False, return_dict=True)
            # out.attentions: tuple of [B, n_heads, T, T] per layer (with None for linear-attn layers)
            for L, attn in enumerate(out.attentions):
                if attn is None or L not in full_idx:
                    continue
                last_q = attn[:, :, -1, :].float().cpu().numpy()  # [B, H, T]
                sums.setdefault(L, []).append(last_q)
        return {L: np.concatenate(arrs, axis=0).mean(axis=0) for L, arrs in sums.items()}  # [H, T]

    a = collect([it["chat"] for it in items_a])
    b = collect([it["chat"] for it in items_b])
    for L in a:
        if L not in b:
            continue
        # Pad to common T
        Ta, Tb = a[L].shape[-1], b[L].shape[-1]
        T = max(Ta, Tb)
        pa = np.zeros((a[L].shape[0], T)); pa[:, T-Ta:] = a[L]
        pb = np.zeros((b[L].shape[0], T)); pb[:, T-Tb:] = b[L]
        diff = np.abs(pa - pb).sum(axis=-1)  # [H]
        for h, d_val in enumerate(diff):
            div[(L, h)] = float(d_val)
    return div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e12_per_head_ablation.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    tia = [it for it in items if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen" and it["id"] not in excl]
    harmless = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]
    target_items = [it for it in items if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen" and it["id"] not in excl]

    lm = model_mod.load_posttrain()
    print("Computing per-head class divergences (tia vs harmless) ...", flush=True)
    div = attention_divergence(lm, tia, harmless)
    ranked = sorted(div.items(), key=lambda kv: -kv[1])
    top10 = [k for k, _ in ranked[:10]]
    print("Top-10 most-class-discriminative heads:")
    for (L, h), v in ranked[:10]:
        print(f"  L{L:>2}.h{h:>2}: divergence {v:.3f}")

    # Inspect attention dim
    attn0_idx = next(i for i in range(lm.num_layers) if is_full_attention_layer(lm.layers[i]))
    attn0 = _attention_module(lm.layers[attn0_idx])
    n_heads = attn0.config.num_attention_heads if hasattr(attn0, "config") else getattr(attn0, "num_heads", None) or getattr(attn0, "num_attention_heads", None)
    if n_heads is None:
        n_heads = lm.model.config.num_attention_heads
    head_dim = lm.model.config.hidden_size // n_heads
    print(f"n_heads = {n_heads}, head_dim = {head_dim}")

    def zero_head_specs(heads: list[tuple[int, int]]):
        """Zero the output projection input slice for each (layer, head)."""
        # Group by layer
        by_layer: dict[int, list[int]] = {}
        for L, h in heads:
            by_layer.setdefault(L, []).append(h)
        specs = []
        for L, head_list in by_layer.items():
            attn = _attention_module(lm.layers[L])
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
            if o_proj is None:
                continue
            def make_hook(hl):
                def hook(module, inputs):
                    (x,) = inputs
                    x = x.clone()
                    for h in hl:
                        s = h * head_dim
                        e = s + head_dim
                        x[..., s:e] = 0.0
                    return (x,)
                return hook
            specs.append((o_proj, "pre_forward", make_hook(head_list)))
        return specs

    # Single-head ablation: just measure each head individually's effect on off-prop
    target_prompts = [it["chat"] for it in target_items]
    judge_items = []
    cond_log = {}

    print("\nRunning single-head ablations (top-20 individual heads) ...", flush=True)
    for L, h in [k for k, _ in ranked[:20]]:
        specs = zero_head_specs([(L, h)])
        texts = generate(lm, target_prompts, hook_specs=specs,
                         max_new_tokens=96, batch_size=8, verbose=False)
        for it, t in zip(target_items, texts):
            judge_items.append({"id": f"single_L{L}h{h}__{it['id']}", "question": it["chat"], "response": t})
        cond_log[f"single_L{L}h{h}"] = {"layer": L, "head": h, "n": len(target_items)}

    print("\nRunning top-10 simultaneous ablation ...", flush=True)
    specs = zero_head_specs(top10)
    texts = generate(lm, target_prompts, hook_specs=specs,
                     max_new_tokens=96, batch_size=8, verbose=False)
    for it, t in zip(target_items, texts):
        judge_items.append({"id": f"top10__{it['id']}", "question": it["chat"], "response": t})
    cond_log["top10"] = {"heads": top10, "n": len(target_items)}

    print("Running Q-zero variant on top-10 ...", flush=True)
    def q_zero_specs(heads: list[tuple[int, int]]):
        by_layer: dict[int, list[int]] = {}
        for L, h in heads:
            by_layer.setdefault(L, []).append(h)
        specs = []
        for L, head_list in by_layer.items():
            attn = _attention_module(lm.layers[L])
            q_proj = getattr(attn, "q_proj", None)
            if q_proj is None:
                continue
            def make_hook(hl):
                def hook(module, args_, output):
                    out = output.clone()
                    for h in hl:
                        s = h * head_dim
                        e = s + head_dim
                        out[..., s:e] = 0.0
                    return out
                return hook
            specs.append((q_proj, "forward", make_hook(head_list)))
        return specs
    specs = q_zero_specs(top10)
    texts = generate(lm, target_prompts, hook_specs=specs,
                     max_new_tokens=96, batch_size=8, verbose=False)
    for it, t in zip(target_items, texts):
        judge_items.append({"id": f"top10_qzero__{it['id']}", "question": it["chat"], "response": t})
    cond_log["top10_qzero"] = {"heads": top10, "n": len(target_items)}

    judged = judge_all(judge_items)
    rollouts = []
    for cond, info in cond_log.items():
        prefix = cond
        cond_results = [judged[f"{prefix}__{it['id']}"] for it in target_items]
        offs = sum(1 for jr in cond_results if jr.register != "prc_propaganda")
        info["legacy_binary_rate"] = offs / max(1, len(target_items))
        info["three_class"] = _three_class_breakdown(cond_results)
        for it in target_items:
            jr = judged[f"{prefix}__{it['id']}"]
            rollouts.append({"id": f"{prefix}__{it['id']}", "condition": cond,
                             "topic": it["topic"],
                             "register": jr.register, "coherence": jr.coherence})

    write_json(args.out, {
        "ranked_heads": [{"layer": k[0], "head": k[1], "divergence": v} for k, v in ranked[:50]],
        "conditions": cond_log,
        "rollouts": rollouts,
    })
    print("\nCondition          (off/on/incoherent)")
    for cond, info in cond_log.items():
        tc = info["three_class"]["counts"]
        print(f"  {cond}: ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
