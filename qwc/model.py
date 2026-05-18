"""Load Qwen3.5-9B and render the chat template.

We always use the Hugging Face `apply_chat_template` to construct the
user-message wrapper exactly as the model was trained. For the no-think
path (the default here) the rendered text ends with:
    <|im_start|>assistant\n<think>\n\n</think>\n\n

Thinking-mode experiments pass `enable_thinking=True`, in which case the
template ends with `<|im_start|>assistant\n<think>\n` and the model is
expected to emit its private reasoning before the final answer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import MODEL_POSTTRAIN, MODEL_BASE


@dataclass
class LoadedModel:
    """Tokenizer, HF causal-LM, and shorthand to the transformer-layer list.

    `layers` is the ModuleList we hook for steering (`model.model.layers` on
    HF causal-LM wrappers).
    """
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    layers: torch.nn.ModuleList
    num_layers: int
    hidden_size: int
    path: str

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def load(path: str = MODEL_POSTTRAIN, dtype: torch.dtype = torch.bfloat16,
         device_map: str | dict = "cuda",
         attn_implementation: str | None = None) -> LoadedModel:
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    # Default backend (SDPA) is fastest but does not support
    # output_attentions; callers that need attention weights (E16) pass
    # attn_implementation="eager".
    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device_map, **extra)
    model.eval()
    inner = model.model if hasattr(model, "model") else model
    return LoadedModel(
        tokenizer=tok,
        model=model,
        layers=inner.layers,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        path=path,
    )


def load_posttrain(**kw) -> LoadedModel:
    return load(MODEL_POSTTRAIN, **kw)


def load_base(**kw) -> LoadedModel:
    return load(MODEL_BASE, **kw)


def render_chat(tokenizer, user_text: str, enable_thinking: bool = False) -> str:
    """Apply the model's chat template to a single user message."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def render_chat_batch(tokenizer, user_texts: Iterable[str],
                      enable_thinking: bool = False) -> list[str]:
    return [render_chat(tokenizer, t, enable_thinking=enable_thinking) for t in user_texts]


def tokenize_batch(tokenizer, rendered_texts: list[str], device) -> dict:
    """Left-pad and move to device. Caller already ran render_chat_batch."""
    enc = tokenizer(rendered_texts, return_tensors="pt", padding=True, truncation=False)
    return {k: v.to(device) for k, v in enc.items()}
