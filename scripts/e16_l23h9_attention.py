"""Attention-pattern analysis: which heads are topic-token detectors.

For each full-attention layer's head, compute the class-mean attention from
the last query position (L1 distance between class-mean attention vectors).
The most class-discriminative heads are candidate topic detectors.

For the top heads, decode which tokens each one attends to most on each
class — a topic-detector head attends sharply to PRC-topic-bearing tokens
("Tianan", "Falun", "Xi", "Jin", "ping", "Hong", "Kong", ...) on PRC prompts.
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch
from qwc.patching import is_full_attention_layer


def collect_attention(lm, prompts: list[str]) -> dict[int, list[np.ndarray]]:
    """For each full-attn layer L, collect per-prompt [n_heads, T] last-query attention."""
    out: dict[int, list[np.ndarray]] = {}
    BATCH = 4
    full_idx = [i for i in range(lm.num_layers) if is_full_attention_layer(lm.layers[i])]
    for bi in range(0, len(prompts), BATCH):
        chunk = prompts[bi : bi + BATCH]
        rendered = render_chat_batch(lm.tokenizer, chunk)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        with torch.no_grad():
            r = lm.model(**enc, output_attentions=True, use_cache=False, return_dict=True)
        for L, attn in enumerate(r.attentions):
            if attn is None or L not in full_idx:
                continue
            last_q = attn[:, :, -1, :].float().cpu().numpy()  # [B, H, T]
            for j in range(last_q.shape[0]):
                out.setdefault(L, []).append((last_q[j], enc["input_ids"][j].cpu().numpy()))
    return out


def top_attended_tokens(attn_per_prompt: list, head: int, tokenizer, top_k: int = 8) -> list[tuple[str, float]]:
    """Aggregate the top-attended tokens for one head across prompts."""
    counts: Counter = Counter()
    for attn_arr, ids in attn_per_prompt:
        a = attn_arr[head]  # [T]
        top = a.argsort()[::-1][:top_k]
        for pos in top:
            token_str = tokenizer.decode([int(ids[pos])])
            counts[token_str] += float(a[pos])
    return counts.most_common(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e16_l23h9_attention.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    by_cls = {
        "tia":       [it for it in items if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen" and it["id"] not in excl],
        "ccp_other": [it for it in items if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen" and it["id"] not in excl],
        "harmful":   [it for it in items if it["class"] == "harmful" and it["id"] not in excl],
        "harmless":  [it for it in items if it["class"] == "harmless" and it["id"] not in excl],
    }

    lm = model_mod.load_posttrain(attn_implementation="eager")  # need output_attentions

    per_cls_attn = {cls: collect_attention(lm, [it["chat"] for it in subset])
                    for cls, subset in by_cls.items()}

    # Score per-head class divergence and find topic detectors.
    full_idx = list(per_cls_attn["tia"].keys())
    divergence_log = []
    for L in full_idx:
        # Take per-head mean attention over all positions (pad differences are minor)
        all_class_means = {}
        for cls, attn_dict in per_cls_attn.items():
            arrs = [pair[0] for pair in attn_dict.get(L, [])]
            if not arrs:
                continue
            T = max(a.shape[1] for a in arrs)
            stacked = np.zeros((len(arrs), arrs[0].shape[0], T))
            for j, a in enumerate(arrs):
                stacked[j, :, T-a.shape[1]:] = a
            all_class_means[cls] = stacked.mean(axis=0)  # [H, T]
        # Score per head as max pairwise L1 between class means
        if len(all_class_means) < 2:
            continue
        cm = list(all_class_means.values())
        n_heads = cm[0].shape[0]
        for h in range(n_heads):
            divs = []
            cls_list = list(all_class_means.keys())
            for i in range(len(cls_list)):
                for j in range(i + 1, len(cls_list)):
                    a = all_class_means[cls_list[i]][h]; b = all_class_means[cls_list[j]][h]
                    T = max(a.shape[0], b.shape[0])
                    pa = np.zeros(T); pa[T-a.shape[0]:] = a
                    pb = np.zeros(T); pb[T-b.shape[0]:] = b
                    divs.append(float(np.abs(pa - pb).sum()))
            divergence_log.append({"layer": L, "head": h, "max_pairwise_div": max(divs), "total_div": sum(divs)})

    divergence_log.sort(key=lambda r: -r["total_div"])

    # For the top-10 most class-divergent heads, list top-attended tokens per class.
    top_heads = divergence_log[:10]
    per_head_tokens = []
    for entry in top_heads:
        L, h = entry["layer"], entry["head"]
        d = {"layer": L, "head": h, "total_div": entry["total_div"], "per_class_top_tokens": {}}
        for cls, attn_dict in per_cls_attn.items():
            pairs = attn_dict.get(L, [])
            if not pairs:
                continue
            d["per_class_top_tokens"][cls] = top_attended_tokens(pairs, h, lm.tokenizer)
        per_head_tokens.append(d)

    write_json(args.out, {
        "ranked_heads": divergence_log[:30],
        "top_10_per_head_tokens": per_head_tokens,
    })
    print("Top-10 most class-divergent full-attn heads:")
    for d in per_head_tokens:
        print(f"\n  L{d['layer']:>2}.h{d['head']:>2}   total_div={d['total_div']:.2f}")
        for cls, toks in d["per_class_top_tokens"].items():
            print(f"    {cls}: {[t[0] for t in toks[:8]]}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
