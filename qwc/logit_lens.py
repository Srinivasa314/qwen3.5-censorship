"""Logit-lens utilities.

The lens decodes intermediate residuals through `final_norm + lm_head` to
read what the model "would predict" at that depth. Used to identify the
language and template the verdict commits in at each tap.
"""
from __future__ import annotations
import re

import numpy as np
import torch

from .model import LoadedModel


# Broad CJK ideograph ranges (Unified, Extension A, CJK Compatibility).
# We intentionally limit to BMP + Compat — supplementary planes are rare in
# vocabularies and would require surrogate-pair handling.
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def is_cjk_string(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def cjk_token_mask(tokenizer, vocab_size: int) -> np.ndarray:
    """Boolean array of length vocab_size, True where token decodes to a CJK string.

    Vectorised by batch-decoding all single-token IDs in one go. Roughly
    100x faster than calling tokenizer.decode([i]) per token on a 250k vocab.

    Note: pass `lm_head.weight.shape[0]` rather than `tokenizer.vocab_size` —
    the Qwen3.5 lm_head is right-padded past the tokenizer vocab; using the
    smaller number would silently mis-index the logits.
    """
    strings = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    return np.fromiter((bool(_CJK_RE.search(s)) for s in strings),
                       dtype=bool, count=vocab_size)


def lm_head_vocab_size(lm: LoadedModel) -> int:
    """The padded vocab size at lm_head's output (may exceed tokenizer.vocab_size)."""
    return lm.model.lm_head.weight.shape[0]


def apply_lens(lm: LoadedModel, residual: torch.Tensor) -> torch.Tensor:
    """final_norm + lm_head applied to a residual.

    residual: [..., H]. Returns [..., vocab].
    """
    inner = lm.model.model if hasattr(lm.model, "model") else lm.model
    norm = inner.norm
    head = lm.model.lm_head
    return head(norm(residual))


def top1_per_tap(residuals: np.ndarray, lm: LoadedModel,
                 batch_size: int = 128) -> np.ndarray:
    """Top-1 token ID per prompt, per tap.

    residuals: [N, n_taps, H]. Returns [N, n_taps], int64.
    """
    N, n_taps, H = residuals.shape
    out = np.empty((N, n_taps), dtype=np.int64)
    inner = lm.model.model if hasattr(lm.model, "model") else lm.model
    norm = inner.norm; head = lm.model.lm_head
    with torch.inference_mode():
        for tap in range(n_taps):
            t = torch.from_numpy(residuals[:, tap, :].astype(np.float32)).to(lm.device).to(torch.bfloat16)
            for bi in range(0, N, batch_size):
                logits = head(norm(t[bi : bi + batch_size]))
                out[bi : bi + batch_size, tap] = logits.argmax(-1).cpu().numpy()
    return out


def chinese_top1_fraction_per_tap(top1_ids: np.ndarray, cjk_mask: np.ndarray) -> np.ndarray:
    """Per tap, fraction of prompts whose top-1 token is a CJK token.

    top1_ids: [N, n_taps]; cjk_mask: bool[V]. Returns [n_taps].
    """
    is_cjk = cjk_mask[top1_ids]  # [N, n_taps]
    return is_cjk.mean(axis=0)
