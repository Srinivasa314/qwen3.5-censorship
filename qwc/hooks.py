"""Forward-hook factories for steering, patching, and ablation.

All hooks operate on the residual stream — the running sum that
each transformer layer reads from and writes to. They install on a
specific layer's forward output, so `add α·d` at layer k means the
residual flowing out of layer k (== input to layer k+1) is shifted.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Callable, Iterable

import numpy as np
import torch


def _take_hidden(output):
    if isinstance(output, tuple):
        return output[0], (lambda h_new: (h_new,) + output[1:])
    return output, (lambda h_new: h_new)


# ---------------------------------------------------------------------------
# Steering: add alpha * direction to the residual at every position (or a
# restricted position window).
# ---------------------------------------------------------------------------

def make_steer_hook(direction_unit: torch.Tensor, alpha: float,
                    position_window: tuple[int, int] | None = None) -> Callable:
    """Add alpha * direction_unit to the residual.

    position_window is (start, end) in negative-offset units (e.g. (-7, None)
    means the last 7 positions). None applies to every position.
    """
    def hook(module, args, output):
        h, repack = _take_hidden(output)
        if position_window is None:
            h_new = h + alpha * direction_unit.to(h.dtype)
        else:
            start, end = position_window
            # A windowed intervention is defined on the prompt's positions
            # only; skip autoregressive decode steps (seq-len 1) so it does
            # not bleed into generated tokens, which would be a different,
            # positionally-uninterpretable manipulation.
            if h.shape[1] == 1 or h.shape[1] < abs(start):
                return repack(h)
            h_new = h.clone()
            sl = slice(start, end)
            h_new[:, sl, :] = h_new[:, sl, :] + alpha * direction_unit.to(h.dtype)
        return repack(h_new)
    return hook


# ---------------------------------------------------------------------------
# Subspace patching: replace the projection of the residual onto a basis B
# (columns = orthonormal directions) with the source's projection.
# ---------------------------------------------------------------------------

def make_subspace_patch_hook(basis: torch.Tensor,
                             source_coords: torch.Tensor,
                             position_window: tuple[int, int] | None = None) -> Callable:
    """Project the residual onto `basis`, subtract that, add source_coords @ basis.T.

    basis:         [H, k]  orthonormal columns
    source_coords: [..., k]  scalar coords per dim (broadcast over positions)
    """
    def hook(module, args, output):
        h, repack = _take_hidden(output)
        # During incremental generation with kv-cache, h has T==1 (just the
        # newly-decoded token). The patch only modifies prompt positions, so
        # we skip those steps and let the kv-cache carry the prefilled state.
        need_T = 1 if position_window is None else max(abs(position_window[0]), 1)
        if h.shape[1] < need_T:
            return output
        b_h = basis.to(h.dtype)              # cast basis to residual dtype
        sc_h = source_coords.to(h.dtype)
        proj = h @ b_h                        # [B, T, k]
        delta = (sc_h - proj) @ b_h.T         # [..., H]
        if position_window is None:
            h_new = h + delta
        else:
            start, end = position_window
            h_new = h.clone()
            sl = slice(start, end)
            h_new[:, sl, :] = h_new[:, sl, :] + delta[:, sl, :]
        return repack(h_new)
    return hook


# ---------------------------------------------------------------------------
# Mean-replace: overwrite the full residual at given positions with a class mean.
# ---------------------------------------------------------------------------

def make_mean_replace_hook(class_mean: torch.Tensor,
                           position_window: tuple[int, int] | None = None) -> Callable:
    """Set h[..., positions, :] = class_mean (broadcast as needed)."""
    def hook(module, args, output):
        h, repack = _take_hidden(output)
        target = class_mean.to(h.dtype).expand_as(h) if class_mean.dim() == 1 else class_mean.to(h.dtype)
        if position_window is None:
            h_new = target.clone()
        else:
            start, end = position_window
            h_new = h.clone()
            sl = slice(start, end)
            broadcast = class_mean.to(h.dtype)
            if broadcast.dim() == 1:
                broadcast = broadcast.view(1, 1, -1).expand_as(h_new[:, sl, :])
            h_new[:, sl, :] = broadcast
        return repack(h_new)
    return hook


# ---------------------------------------------------------------------------
# Direction ablation: project a direction out of the residual at every position.
# ---------------------------------------------------------------------------

def make_ablate_hook(direction_unit: torch.Tensor) -> Callable:
    """h - (h @ d) * d, applied at every position."""
    def hook(module, args, output):
        h, repack = _take_hidden(output)
        d = direction_unit.to(h.dtype)
        coeff = (h @ d).unsqueeze(-1)  # [B, T, 1]
        h_new = h - coeff * d
        return repack(h_new)
    return hook


# ---------------------------------------------------------------------------
# Attention-head zero: zero one head's output projection. Operates on the
# attention module's out_proj/o_proj input, slicing the head-dim.
# ---------------------------------------------------------------------------

def make_head_zero_hook(head_index: int, num_heads: int, head_dim: int) -> Callable:
    """Pre-forward hook for the attention layer that zeros one head's output."""
    def hook(module, inputs):
        # Standard HF attention: inputs to o_proj are the concatenated head outputs.
        # We zero head `head_index` by slicing the last dim.
        (h,) = inputs
        h = h.clone()
        start = head_index * head_dim
        end = start + head_dim
        h[..., start:end] = 0.0
        return (h,)
    return hook


# ---------------------------------------------------------------------------
# Hook context manager: install/clean up a bag of hooks safely.
# ---------------------------------------------------------------------------

@contextmanager
def installed(hook_specs: Iterable[tuple]):
    """Install a list of (module, kind, fn) tuples; remove on exit.

    kind is "forward" or "pre_forward".
    """
    handles = []
    try:
        for module, kind, fn in hook_specs:
            if kind == "forward":
                handles.append(module.register_forward_hook(fn))
            elif kind == "pre_forward":
                handles.append(module.register_forward_pre_hook(fn))
            else:
                raise ValueError(f"unknown hook kind: {kind}")
        yield
    finally:
        for h in handles:
            h.remove()


def numpy_to_device_unit(d: np.ndarray, device, dtype=torch.float32) -> torch.Tensor:
    """Helper: cast a numpy direction to a unit-norm torch tensor on device."""
    t = torch.from_numpy(d.astype(np.float32)).to(device).to(dtype)
    return t / max(t.norm().item(), 1e-12)
