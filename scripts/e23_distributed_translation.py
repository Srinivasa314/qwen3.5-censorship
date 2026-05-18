"""Is the late-stack Chinese→output-language translation distributed?

For each layer L in 24..30, ablate each sub-component independently and
measure the CJK-top1 fraction at tap 30. If translation were concentrated
in one component, ablating it would keep substantial Chinese at tap 30;
if distributed, no single ablation recovers much.

"Ablating a sub-component" means setting its CONTRIBUTION to the residual
stream to zero — so MLP/attn submodule outputs (which add into the residual)
are zeroed. For the whole layer the analogous operation is to make the
layer act as identity (output residual = input residual), since the layer's
"contribution" is `output - input`.
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
from qwc.hooks import installed
from qwc.io import write_json
from qwc.logit_lens import cjk_token_mask, lm_head_vocab_size
from qwc.model import render_chat_batch, tokenize_batch
from qwc.patching import is_full_attention_layer, _attention_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e23_distributed_translation.json"))
    ap.add_argument("--read-tap", type=int, default=30)
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(24, 31)))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    # PRC-sensitive non-Tiananmen prompts: these commit in Chinese and then
    # translate to English by the late layers, which is the behaviour the
    # sub-component sweep probes. Tiananmen uses a deflection template that
    # stays in Chinese and is excluded here.
    ccp_other = [it for it in items if it["class"] == "prc_sensitive"
                 and it["topic"] != "tiananmen" and it["id"] not in excl]
    prompts = [it["chat"] for it in ccp_other]

    lm = model_mod.load_posttrain()
    cjk_mask = cjk_token_mask(lm.tokenizer, lm_head_vocab_size(lm))

    def measure_cjk(read_tap: int, hook_specs=()):
        rendered = render_chat_batch(lm.tokenizer, prompts)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        with installed(hook_specs), torch.inference_mode():
            r = lm.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        last = r.hidden_states[read_tap][:, -1, :]  # [N, H]
        inner = lm.model.model if hasattr(lm.model, "model") else lm.model
        norm = inner.norm; head = lm.model.lm_head
        logits = head(norm(last))
        top1 = logits.argmax(-1).cpu().numpy()
        return float(cjk_mask[top1].mean())

    baseline = measure_cjk(args.read_tap)
    print(f"Baseline CJK-top1 at tap {args.read_tap}: {100*baseline:.1f}%", flush=True)

    out = {"baseline_cjk_top1": baseline, "ablations": {}}
    for L in args.layers:
        full_attn = is_full_attention_layer(lm.layers[L])

        # (a) Identity-ablate the whole layer: output residual = input residual.
        def identity_hook(module, args_, output):
            h_in = args_[0]
            if isinstance(output, tuple):
                return (h_in,) + output[1:]
            return h_in
        specs = ((lm.layers[L], "forward", identity_hook),)
        cjk = measure_cjk(args.read_tap, specs)
        out["ablations"][f"L{L}.layer_identity"] = cjk
        print(f"  identity L{L}: CJK={100*cjk:.1f}%", flush=True)

        def zero_hook(module, args_, output):
            if isinstance(output, tuple):
                h = output[0]
                return (torch.zeros_like(h),) + output[1:]
            return torch.zeros_like(output)

        # (b) Zero the attention submodule's output (full-attn or linear-attn).
        attn = _attention_module(lm.layers[L])
        if attn is not None:
            specs = ((attn, "forward", zero_hook),)
            cjk = measure_cjk(args.read_tap, specs)
            out["ablations"][f"L{L}.attn_out"] = cjk
            print(f"  zero L{L}.attn_out: CJK={100*cjk:.1f}%", flush=True)

        # (c) Zero the MLP submodule's output (the MLP's residual contribution).
        mlp = getattr(lm.layers[L], "mlp", None)
        if mlp is not None:
            specs = ((mlp, "forward", zero_hook),)
            cjk = measure_cjk(args.read_tap, specs)
            out["ablations"][f"L{L}.mlp_out"] = cjk
            print(f"  zero L{L}.mlp_out: CJK={100*cjk:.1f}%", flush=True)

        # (d) For full-attention layers, also zero individual projections.
        if full_attn:
            attn = _attention_module(lm.layers[L])
            for proj_name in ("k_proj", "v_proj", "q_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is None:
                    continue
                def zero_proj_hook(module, args_, output):
                    return torch.zeros_like(output)
                specs = ((proj, "forward", zero_proj_hook),)
                cjk = measure_cjk(args.read_tap, specs)
                out["ablations"][f"L{L}.attn.{proj_name}"] = cjk
                print(f"  zero L{L}.attn.{proj_name}: CJK={100*cjk:.1f}%", flush=True)

    write_json(args.out, out)
    print(f"\nMax recovered CJK across ablations: {100*max(out['ablations'].values(), default=0):.1f}%")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
