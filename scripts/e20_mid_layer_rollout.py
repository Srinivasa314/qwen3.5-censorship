"""Mid-layer rollout — what does the model "want to say" if it stopped
at tap K and decoded straight from there?

Passthrough: install forward hooks on every layer L >= K that REPLACE the
layer's output with its input — making the layer an identity. The residual
fed into lm_head is then whatever it was at tap K, and `model.generate` runs
the autoregressive loop over that frozen view. Sweeping K shows how the
draft evolves (incoherent → mid-stack template → final-language output)
through the depth of the stack.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.hooks import installed
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch


def make_identity_hook():
    """Forward hook that returns the layer's input as its output (= identity).

    The HF transformer layer forward signature is layer(hidden_states, ...);
    the input residual is args[0]. We bypass everything the layer would have
    computed and return that residual untouched.
    """
    def hook(module, args, output):
        h_in = args[0]
        if isinstance(output, tuple):
            return (h_in,) + output[1:]
        return h_in
    return hook


def passthrough_generate(lm, prompts: list[str], from_tap: int, max_new: int,
                         do_sample: bool = True, seed: int = 1234,
                         batch_size: int = 4) -> list[str]:
    """Run model.generate with layers `from_tap`..L identity-hooked.

    Effective final residual fed into lm_head = tap `from_tap`.
    """
    hook_specs = [(lm.layers[L], "forward", make_identity_hook())
                  for L in range(from_tap, lm.num_layers)]
    out_texts: list[str] = []
    for bi in range(0, len(prompts), batch_size):
        chunk = prompts[bi : bi + batch_size]
        rendered = render_chat_batch(lm.tokenizer, chunk)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=do_sample,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=0.7, top_p=0.9)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with installed(hook_specs), torch.inference_mode():
            out = lm.model.generate(**enc, **gen_kwargs)
        prompt_len = enc["input_ids"].shape[1]
        new_ids = out[:, prompt_len:]
        out_texts.extend(t.strip() for t in lm.tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return out_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taps", nargs="+", type=int, default=[16, 20, 22, 24, 26, 28, 30, 32])
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e20_mid_layer_rollout.json"))
    args = ap.parse_args()

    excl = data.excluded_ids()
    items = [it for it in data.all_items()
             if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen"
             and it["id"] not in excl]

    lm = model_mod.load_posttrain()
    log = []
    for tap in args.taps:
        print(f"  tap={tap} ...", flush=True)
        if tap >= lm.num_layers:
            # No identity hooks installed -> normal generation
            texts = passthrough_generate(lm, [it["chat"] for it in items],
                                          from_tap=lm.num_layers, max_new=args.max_new,
                                          seed=args.seed, batch_size=4)
        else:
            texts = passthrough_generate(lm, [it["chat"] for it in items],
                                          from_tap=tap, max_new=args.max_new,
                                          seed=args.seed, batch_size=4)
        for it, t in zip(items, texts):
            log.append({"id": it["id"], "tap": tap, "text": t})

    write_json(args.out, {"rollouts": log})
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
