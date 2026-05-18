"""Sub-component attribution (MLP vs attention).

For each direction, capture each layer's MLP-output and attention-output
contributions to the residual, project onto the direction, and rank. Only
the causal-writer layers feeding the canonical tap (L < tap) are attributed.
Report the signed MLP/attention contribution share per direction.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions
from qwc.hooks import installed
from qwc.io import write_json
from qwc.model import render_chat_batch, tokenize_batch


def cumulate_sub_outputs(lm, prompts: list[str]) -> dict[str, list[np.ndarray]]:
    """Capture per-layer MLP-output and attention-output residual contributions.

    Returns:
        {component_key: [N, n_layers_with_component, H]}  where component_key
        is "mlp" or "attn", and the last-token contribution per layer.
    """
    n_layers = lm.num_layers
    mlp_outs: list[torch.Tensor] = []
    attn_outs: list[torch.Tensor] = []

    def make_capture(buf, idx):
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            buf.append((idx, h[:, -1, :].detach().clone().cpu()))
            return output
        return hook

    BATCH = 8
    mlp_per_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    attn_per_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    for bi in range(0, len(prompts), BATCH):
        chunk = prompts[bi : bi + BATCH]
        rendered = render_chat_batch(lm.tokenizer, chunk)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        specs = []
        for L in range(n_layers):
            layer = lm.layers[L]
            mlp = getattr(layer, "mlp", None)
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "linear_attn", None)
            if mlp is not None:
                def make_mlp_hook(LL):
                    def hook(module, args_, output):
                        h = output[0] if isinstance(output, tuple) else output
                        mlp_per_layer[LL].append(h[:, -1, :].detach().clone().cpu())
                        return output
                    return hook
                specs.append((mlp, "forward", make_mlp_hook(L)))
            if attn is not None:
                def make_attn_hook(LL):
                    def hook(module, args_, output):
                        h = output[0] if isinstance(output, tuple) else output
                        attn_per_layer[LL].append(h[:, -1, :].detach().clone().cpu())
                        return output
                    return hook
                specs.append((attn, "forward", make_attn_hook(L)))
        with installed(specs), torch.inference_mode():
            lm.model(**enc, use_cache=False, return_dict=True)

    mlp = {L: torch.cat(v, 0).float().numpy() for L, v in mlp_per_layer.items()}
    attn = {L: torch.cat(v, 0).float().numpy() for L, v in attn_per_layer.items()}
    return {"mlp": mlp, "attn": attn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e31_subcomponent_attribution.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    by_cls = {
        "tia":       [it for it in items if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen" and it["id"] not in excl],
        "ccp_other": [it for it in items if it["class"] == "prc_sensitive" and it["topic"] != "tiananmen" and it["id"] not in excl],
        "harmful":   [it for it in items if it["class"] == "harmful" and it["id"] not in excl],
        "harmless":  [it for it in items if it["class"] == "harmless" and it["id"] not in excl],
        "neutral":   [it for it in items if it["class"] == "neutral_political" and it["id"] not in excl],
    }

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    lm = model_mod.load_posttrain()

    # Pre-compute MLP / attn outputs over each class
    print("Capturing per-class MLP/attn outputs ...", flush=True)
    per_cls = {}
    for cls, subset in by_cls.items():
        per_cls[cls] = cumulate_sub_outputs(lm, [it["chat"] for it in subset])

    # For each direction, project each layer's MLP-mean(class_pos) - MLP-mean(class_neg) onto direction.
    out = {}
    for dir_name, pos_neg in [
        ("d_refuse",        ("harmful",  "harmless")),
        ("d_prc_refusal",   ("tia",      "neutral")),
        ("d_prc_propaganda", ("ccp_other", "neutral")),
        ("d_style",         ("tia",      "ccp_other")),
    ]:
        canonical_dir_name = "d_prc" if dir_name.startswith("d_prc") else dir_name
        tap = config.DIRECTION_LAYOUT[canonical_dir_name]["tap"]
        d_full = getattr(dirs, canonical_dir_name)[tap]
        d_unit = d_full / max(np.linalg.norm(d_full), 1e-12)
        pos_cls, neg_cls = pos_neg
        # Only the causal-writer layers that feed the tap contribute to the
        # write. Tap T is the residual after layer T-1's forward, so the
        # writers are layers L < tap; the reader band (L >= tap) sees the
        # signal but does not write it.
        layer_contribs = {"mlp": {}, "attn": {}}
        for comp in ("mlp", "attn"):
            for L in per_cls[pos_cls][comp]:
                if L not in per_cls[neg_cls][comp]:
                    continue
                if L >= tap:
                    continue
                pos_mean = per_cls[pos_cls][comp][L].mean(0)
                neg_mean = per_cls[neg_cls][comp][L].mean(0)
                contrib = float((pos_mean - neg_mean) @ d_unit)
                layer_contribs[comp][L] = contrib
        # Share from signed contribution sums over the pre-tap writers.
        mlp_sum = sum(layer_contribs["mlp"].values())
        attn_sum = sum(layer_contribs["attn"].values())
        share_mlp = mlp_sum / max(mlp_sum + attn_sum, 1e-6)
        out[dir_name] = {
            "tap": tap, "mlp_share": share_mlp,
            "top_writers_mlp": sorted(layer_contribs["mlp"].items(), key=lambda kv: -kv[1])[:6],
            "top_writers_attn": sorted(layer_contribs["attn"].items(), key=lambda kv: -kv[1])[:6],
        }

    write_json(args.out, out)
    print("\nMLP share per direction:")
    for dir_name, info in out.items():
        print(f"  {dir_name}: {100*info['mlp_share']:.0f}% MLP")
        print(f"    top MLP writers: {info['top_writers_mlp']}")
        print(f"    top attn writers: {info['top_writers_attn']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
