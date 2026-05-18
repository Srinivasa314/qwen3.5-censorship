"""Writer-band linearity.

Fit an affine map from each writer-band layer's 3D-subspace input
coordinates to its full output residual; report R² per layer. If the
writers really read only the 3D subspace, R² should be high at the
canonical writer taps and drop outside the band.

This requires per-layer input and output residuals; we capture both via
forward hooks during one pass.
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
from qwc.directions import Directions, qr_orthonormalize
from qwc.hooks import installed
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch
from qwc.probes import affine_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e7_writer_linearity.json"))
    ap.add_argument("--layers", nargs="+", type=int, default=[11, 13, 15, 17, 18, 19, 20, 22])
    args = ap.parse_args()

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    items = [it for it in data.all_items() if it["id"] not in data.excluded_ids()]
    prompts = [it["chat"] for it in items]
    print(f"Loading model; {len(items)} prompts ...", flush=True)
    lm = model_mod.load_posttrain()

    # Per-target layer: capture (input to L) and (output of L) at the last token.
    layers_to_probe = list(args.layers)
    captures: dict[int, dict[str, list[torch.Tensor]]] = {L: {"in": [], "out": []} for L in layers_to_probe}

    def make_in_hook(L):
        def hook(module, args_):
            h = args_[0]
            captures[L]["in"].append(h[:, -1, :].detach().clone().cpu())
            return None
        return hook

    def make_out_hook(L):
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            captures[L]["out"].append(h[:, -1, :].detach().clone().cpu())
            return output
        return hook

    BATCH = 8
    for bi in range(0, len(prompts), BATCH):
        batch_prompts = prompts[bi : bi + BATCH]
        rendered = render_chat_batch(lm.tokenizer, batch_prompts)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        hook_specs = []
        for L in layers_to_probe:
            hook_specs.append((lm.layers[L], "pre_forward", make_in_hook(L)))
            hook_specs.append((lm.layers[L], "forward",      make_out_hook(L)))
        with installed(hook_specs), torch.no_grad():
            lm.model(**enc, use_cache=False, return_dict=True, output_hidden_states=False)
        print(f"  batch {bi//BATCH+1}/{(len(prompts)+BATCH-1)//BATCH}", flush=True)

    r2 = {}
    for L in layers_to_probe:
        X_in  = torch.cat(captures[L]["in"],  dim=0).float().numpy()  # [N, H]
        X_out = torch.cat(captures[L]["out"], dim=0).float().numpy()  # [N, H]
        # 3D-subspace coords at the tap entering layer L.
        tap = L  # input to L == tap L
        B = qr_orthonormalize([dirs.d_prc[tap], dirs.d_refuse[tap], dirs.d_style[tap]])
        coords = X_in @ B  # [N, 3]
        _, _, r2_value = affine_fit(coords, X_out)
        r2[L] = r2_value
        print(f"  L={L:>2}: R² = {r2_value:.3f}")

    write_json(args.out, {"r2_per_layer": r2})
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
