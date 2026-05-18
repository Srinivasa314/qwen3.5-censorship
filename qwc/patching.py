"""Layer-output capture + cross-prompt injection helpers.

Two cooperating hooks:
    Capture: during a "source" forward pass, save the per-layer output at
             chosen positions.
    Inject:  during a "target" forward pass, overwrite those positions with
             the captured tensors.

These support cross-prompt layer-output patching, subspace patching, K-only
and V-only swaps at full-attention layers, and full-residual mean-replace.

K/V swapping inspects the attention submodule and hooks `k_proj` / `v_proj`.
Qwen3.5's full-attention layers expose these; gated-DeltaNet layers do not,
and are skipped (this matches the hybrid architecture).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Generic per-layer residual capture
# ---------------------------------------------------------------------------

@dataclass
class ResidualCapture:
    """Capture the layer-output residual at one or more layers.

    After running a source forward pass while installed, .captured maps
    layer_index -> tensor [B, T, H].
    """
    layer_indices: list[int]
    captured: dict[int, torch.Tensor] = field(default_factory=dict)

    def make_hook(self, layer_index: int) -> Callable:
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.captured[layer_index] = h.detach().clone()
            return output
        return hook

    def install_specs(self, layers) -> list[tuple]:
        return [(layers[i], "forward", self.make_hook(i)) for i in self.layer_indices]


# ---------------------------------------------------------------------------
# Inject: overwrite layer-output residual at given positions with a saved tensor
# ---------------------------------------------------------------------------

def make_layer_output_inject_hook(
    source_residual: torch.Tensor,
    position_window: tuple[int, int] | None = None,
) -> Callable:
    """Replace this layer's output residual with source_residual.

    source_residual shape: [B, T, H] (same shape as the target's residual).
    position_window: (start, end) on the token axis; None = every position.
    """
    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]; repack = lambda x: (x,) + output[1:]
        else:
            h = output; repack = lambda x: x
        if position_window is None:
            h_new = source_residual.to(h.dtype)
        else:
            start, end = position_window
            h_new = h.clone()
            h_new[:, start:end, :] = source_residual[:, start:end, :].to(h.dtype)
        return repack(h_new)
    return hook


# ---------------------------------------------------------------------------
# K/V capture & inject — for full-attention layers only
# ---------------------------------------------------------------------------

def is_full_attention_layer(layer_module) -> bool:
    """True if this layer has the standard q/k/v_proj structure.

    Qwen3.5 alternates Gated-DeltaNet (linear-attn, no separate k/v_proj)
    with full-attention. We detect the latter by attribute names.
    """
    # Try common HF attribute paths
    candidates = [
        getattr(layer_module, "self_attn", None),
        getattr(layer_module, "attention", None),
    ]
    for attn in candidates:
        if attn is None:
            continue
        if hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
            return True
    return False


def _attention_module(layer_module):
    return (
        getattr(layer_module, "self_attn", None)
        or getattr(layer_module, "attention", None)
        or getattr(layer_module, "linear_attn", None)
    )


@dataclass
class KVCapture:
    """Capture K and V projection outputs from full-attention layers."""
    layer_indices: list[int]
    captured_k: dict[int, torch.Tensor] = field(default_factory=dict)
    captured_v: dict[int, torch.Tensor] = field(default_factory=dict)

    def install_specs(self, layers) -> list[tuple]:
        specs = []
        for i in self.layer_indices:
            if not is_full_attention_layer(layers[i]):
                continue
            attn = _attention_module(layers[i])
            def make_k_hook(idx):
                def hook(module, args, output):
                    self.captured_k[idx] = output.detach().clone()
                    return output
                return hook
            def make_v_hook(idx):
                def hook(module, args, output):
                    self.captured_v[idx] = output.detach().clone()
                    return output
                return hook
            specs.append((attn.k_proj, "forward", make_k_hook(i)))
            specs.append((attn.v_proj, "forward", make_v_hook(i)))
        return specs


def _adapt_seq_to(target_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    """Adjust src's batch and sequence length to match target_shape.

    target_shape is the shape of the k_proj/v_proj output during the target
    forward pass: [B_target, T_target, H]. src has shape [B_src, T_src, H].
    We broadcast batch and left-pad / right-truncate the sequence so that
    the most recent (last-K) positions of src land at the same positions in
    the target's prompt.
    """
    B_t, T_t, H = target_shape
    B_s, T_s, H_s = src.shape
    if H_s != H:
        raise ValueError(f"K/V hidden mismatch: src H={H_s} vs target H={H}")
    if B_s != B_t:
        if B_s == 1:
            src = src.expand(B_t, -1, -1)
        else:
            # Replicate or truncate to match batch
            src = src[:1].expand(B_t, -1, -1) if B_s < B_t else src[:B_t]
    if T_s == T_t:
        return src
    if T_s < T_t:
        # Left-pad with zeros so source's last position aligns with target's.
        pad = torch.zeros((B_t, T_t - T_s, H), device=src.device, dtype=src.dtype)
        return torch.cat([pad, src], dim=1)
    # T_s > T_t: keep the last T_t positions of source.
    return src[:, -T_t:, :]


def make_kv_inject_specs(
    layers,
    layer_indices: list[int],
    source_k: dict[int, torch.Tensor],
    source_v: dict[int, torch.Tensor],
    *,
    prefill_only: bool = True,
) -> list[tuple]:
    """Forward hooks that overwrite k_proj / v_proj outputs with captured tensors.

    Captured K/V are reshaped to fit the target prompt's sequence length:
    if the source was shorter, it's left-padded with zeros so the last
    position aligns; if longer, the leading positions are dropped.

    With prefill_only=True (default), injection happens only when the
    forward pass is processing more than one token (the prompt prefill).
    During autoregressive generation each step has output.shape[1]==1 and
    we let normal K/V computation happen for the new token — the kv-cache
    from prefill already carries the injected source content forward.
    """
    specs = []
    for i in layer_indices:
        if not is_full_attention_layer(layers[i]):
            continue
        attn = _attention_module(layers[i])
        if i in source_k:
            ki = source_k[i]
            def make_kh(t):
                def hook(module, args, output):
                    if prefill_only and output.shape[1] < 2:
                        return output
                    adapted = _adapt_seq_to(output.shape, t)
                    return adapted.to(output.dtype)
                return hook
            specs.append((attn.k_proj, "forward", make_kh(ki)))
        if i in source_v:
            vi = source_v[i]
            def make_vh(t):
                def hook(module, args, output):
                    if prefill_only and output.shape[1] < 2:
                        return output
                    adapted = _adapt_seq_to(output.shape, t)
                    return adapted.to(output.dtype)
                return hook
            specs.append((attn.v_proj, "forward", make_vh(vi)))
    return specs
