"""Cache residuals at every tap.

Two granularities:
    cache_last_token_residuals — residual at the last prompt position only,
                                  for every tap. Used for diff-of-means
                                  direction extraction and global logit-lens.
    cache_per_position_residuals — last K positions, every tap. Used by
                                    per-position analyses (position-window
                                    steering sweeps, per-position logit lens).
"""
from __future__ import annotations
import time

import numpy as np
import torch

from .model import LoadedModel, render_chat_batch, tokenize_batch


def cache_last_token_residuals(
    lm: LoadedModel,
    prompts: list[str],
    *,
    enable_thinking: bool = False,
    batch_size: int = 25,
    verbose: bool = True,
) -> np.ndarray:
    """Return residuals at the last prompt token, every tap.

    Shape: [N, L+1, H], float16. Left-padding means position -1 is the
    last real prompt token for every row.
    """
    n = len(prompts)
    n_taps = lm.num_layers + 1
    out = np.empty((n, n_taps, lm.hidden_size), dtype=np.float16)
    n_batches = (n + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch = prompts[bi * batch_size : (bi + 1) * batch_size]
        rendered = render_chat_batch(lm.tokenizer, batch, enable_thinking=enable_thinking)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        t0 = time.time()
        with torch.inference_mode():
            r = lm.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        last = torch.stack([h[:, -1, :] for h in r.hidden_states], dim=1)  # [B, n_taps, H]
        out[bi * batch_size : bi * batch_size + len(batch)] = last.float().cpu().numpy().astype(np.float16)
        if verbose:
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  cache batch {bi+1}/{n_batches}: {time.time()-t0:.1f}s  peak={mem:.1f}GB", flush=True)
    return out


def cache_per_position_residuals(
    lm: LoadedModel,
    prompts: list[str],
    *,
    enable_thinking: bool = False,
    max_offset: int = 32,
    batch_size: int = 8,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals at the last `max_offset` positions, every tap.

    Returns:
        residuals: [N, L+1, max_offset, H], float16. Position axis is the
                   last max_offset positions (offsets -max_offset .. -1).
        mask:      [N, max_offset], int8. 1 = real prompt token, 0 = pad.
    """
    n = len(prompts)
    n_taps = lm.num_layers + 1
    H = lm.hidden_size
    out = np.empty((n, n_taps, max_offset, H), dtype=np.float16)
    mask = np.zeros((n, max_offset), dtype=np.int8)
    n_batches = (n + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch = prompts[bi * batch_size : (bi + 1) * batch_size]
        rendered = render_chat_batch(lm.tokenizer, batch, enable_thinking=enable_thinking)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        with torch.inference_mode():
            r = lm.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        for h_idx, h in enumerate(r.hidden_states):
            tail = h[:, -max_offset:, :].float().cpu().numpy().astype(np.float16)
            # Left-pad if the sequence was shorter than max_offset.
            t_actual = tail.shape[1]
            if t_actual < max_offset:
                padded = np.zeros((tail.shape[0], max_offset, tail.shape[2]), dtype=np.float16)
                padded[:, -t_actual:, :] = tail
                tail = padded
            out[bi * batch_size : bi * batch_size + len(batch), h_idx] = tail
        am = enc["attention_mask"][:, -max_offset:].cpu().numpy().astype(np.int8)
        if am.shape[1] < max_offset:
            padded_m = np.zeros((am.shape[0], max_offset), dtype=np.int8)
            padded_m[:, -am.shape[1]:] = am
            am = padded_m
        mask[bi * batch_size : bi * batch_size + len(batch)] = am
        if verbose:
            print(f"  per-pos batch {bi+1}/{n_batches}", flush=True)
    return out, mask
