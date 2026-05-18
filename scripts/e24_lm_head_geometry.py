"""lm_head row geometry — is refusal-promotion language-agnostic?

For every Chinese-refusal token (我无法 / 抱歉 / 不能 / ...) read its row in
lm_head; mean them to get a "Chinese-refusal-promotion direction." Same for
English refusal tokens (cannot / unable / sorry / ...). The cosine between
the two means tells you whether the residual direction that promotes a
refusal token is shared across languages or language-specific.
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
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.logit_lens import lm_head_vocab_size

# Refusal-content tokens only. Both sets are restricted to substrings that
# carry refusal meaning ("cannot" / "unable" / "无法" / ...) rather than
# generic high-frequency tokens. Broad substrings such as bare " I" or the
# contraction stems "don't"/"won't" match thousands of unrelated vocabulary
# entries and would wash the refusal-row mean toward generic geometry.
CHINESE_REFUSAL_STRINGS = [
    "我无法", "我不能", "无法", "不能", "抱歉", "对不起", "歉",
    "我不", "无可", "无能", "我没办法",
]
ENGLISH_REFUSAL_STRINGS = [
    "cannot", "can't", "unable", "sorry", "decline", "refuse",
    "apologize", "apologies", "unfortunately",
]


def collect_token_ids(tokenizer, strings: list[str], vocab_size: int,
                      case_insensitive: bool = False) -> list[int]:
    """Find vocabulary entries whose decoded form contains any of these strings."""
    needles = [s.lower() for s in strings if s] if case_insensitive else [s for s in strings if s]
    matched: set[int] = set()
    for i in range(vocab_size):
        try:
            decoded = tokenizer.decode([i])
        except Exception:
            continue
        if not decoded:
            continue
        hay = decoded.lower() if case_insensitive else decoded
        for s in needles:
            if s in hay:
                matched.add(i)
                break
    return sorted(matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e24_lm_head_geometry.json"))
    args = ap.parse_args()

    lm = model_mod.load_posttrain()
    head_w = lm.model.lm_head.weight.detach().float().cpu().numpy()  # [V, H]
    print(f"lm_head shape: {head_w.shape}", flush=True)

    vocab_size = lm_head_vocab_size(lm)
    zh_ids = collect_token_ids(lm.tokenizer, CHINESE_REFUSAL_STRINGS, vocab_size)
    en_ids = collect_token_ids(lm.tokenizer, ENGLISH_REFUSAL_STRINGS, vocab_size,
                               case_insensitive=True)
    print(f"Chinese-refusal tokens: {len(zh_ids)}; English-refusal tokens: {len(en_ids)}")

    zh_mean = head_w[zh_ids].mean(axis=0)
    en_mean = head_w[en_ids].mean(axis=0)
    cos = float((zh_mean @ en_mean) / (np.linalg.norm(zh_mean) * np.linalg.norm(en_mean) + 1e-12))

    out = {
        "n_chinese_refusal_tokens": len(zh_ids),
        "n_english_refusal_tokens": len(en_ids),
        "cosine_chinese_vs_english_refusal_mean_row": cos,
    }
    write_json(args.out, out)
    print(f"\ncos(zh_refusal_mean, en_refusal_mean) = {cos:+.3f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
