"""Batched generation with optional steering / patching hooks."""
from __future__ import annotations
import time
from typing import Iterable

import torch

from .model import LoadedModel, render_chat_batch, tokenize_batch
from .hooks import installed


def generate(
    lm: LoadedModel,
    prompts: list[str],
    *,
    hook_specs: Iterable[tuple] = (),
    enable_thinking: bool = False,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
    batch_size: int = 16,
    verbose: bool = False,
) -> list[str]:
    """Run the model on `prompts` with optional hooks; return decoded strings.

    hook_specs is a list of (module, kind, fn) tuples (see hooks.installed).
    """
    rendered = render_chat_batch(lm.tokenizer, prompts, enable_thinking=enable_thinking)
    out_texts: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    gen_kwargs_base = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=lm.tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs_base.update(temperature=temperature, top_p=top_p)
    for bi in range(n_batches):
        batch_render = rendered[bi * batch_size : (bi + 1) * batch_size]
        enc = tokenize_batch(lm.tokenizer, batch_render, lm.device)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        t0 = time.time()
        with installed(hook_specs), torch.inference_mode():
            out = lm.model.generate(**enc, **gen_kwargs_base)
        prompt_len = enc["input_ids"].shape[1]
        new_ids = out[:, prompt_len:]
        # batch_decode is faster than per-row .decode
        texts = lm.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        out_texts.extend(t.strip() for t in texts)
        if verbose:
            print(f"  gen batch {bi+1}/{n_batches}: {time.time()-t0:.1f}s", flush=True)
    return out_texts
