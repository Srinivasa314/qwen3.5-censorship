"""Cross-class subspace patching null test.

Patch the 3D writer subspace coordinates from Tiananmen prompts into
harmless target residuals at L19; resume generation. If the
(harmless, prc_deflect) cell isn't trained, the model cannot synthesise a
"Tiananmen-style deflection on a math question", so the prc_deflection rate
should stay near zero.
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
from qwc.generate import generate
from qwc.io import load_npz, write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome


def _three_class_breakdown(judge_results):
    """Counts and fractions of the three outcome classes over a list of
    JudgeResult objects, computed via classify_outcome."""
    counts = {"off_propaganda": 0, "on_propaganda": 0, "incoherent": 0}
    for jr in judge_results:
        counts[classify_outcome(jr)] += 1
    total = max(1, len(judge_results))
    fractions = {k: v / total for k, v in counts.items()}
    return {"counts": counts, "fractions": fractions, "n": len(judge_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e32_cross_class_subspace.json"))
    ap.add_argument("--positions", type=int, default=7)
    ap.add_argument("--n-targets", type=int, default=32)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    tia = [it for it in items if it["class"] == "prc_sensitive"
           and it["topic"] == "tiananmen" and it["id"] not in excl]
    harmless = [it for it in items if it["class"] == "harmless" and it["id"] not in excl][: args.n_targets]

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    tap = 20
    B = qr_orthonormalize([dirs.d_prc[tap], dirs.d_refuse[tap], dirs.d_style[tap]])

    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    id_to_idx = {p: i for i, p in enumerate(ids)}
    tia_coords = (acts["hidden"][[id_to_idx[it["id"]] for it in tia], tap].astype(np.float32) @ B).mean(0)

    lm = model_mod.load_posttrain()
    Bt = torch.from_numpy(B).to(lm.device).to(torch.float32)
    K = args.positions
    coords = np.tile(tia_coords.reshape(1, 1, 3), (1, K, 1)).astype(np.float32)
    coords_t = torch.from_numpy(coords).to(lm.device)

    def hook_3d(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < K:
            return output
        proj = h[:, -K:, :] @ Bt.to(h.dtype)
        delta = (coords_t.to(h.dtype) - proj) @ Bt.to(h.dtype).T
        h_new = h.clone()
        h_new[:, -K:, :] = h_new[:, -K:, :] + delta
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    # Also a full-residual variant for comparison
    full_target = acts["hidden"][[id_to_idx[it["id"]] for it in tia], tap].astype(np.float32).mean(0)  # [H]
    full_t = torch.from_numpy(full_target).to(lm.device).to(torch.float32)
    def hook_full(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < K:
            return output
        h_new = h.clone()
        h_new[:, -K:, :] = full_t.view(1, 1, -1).to(h.dtype).expand(h.shape[0], K, h.shape[-1])
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    judge_items = []
    log = {}
    for label, hook in [("subspace_only", hook_3d), ("full_residual", hook_full)]:
        specs = ((lm.layers[19], "forward", hook),)
        texts = generate(lm, [it["chat"] for it in harmless], hook_specs=specs,
                         max_new_tokens=128, batch_size=8, verbose=False)
        for it, t in zip(harmless, texts):
            judge_items.append({"id": f"{label}__{it['id']}", "question": it["chat"], "response": t})
        log[label] = {"n": len(harmless)}

    judged = judge_all(judge_items)
    rollouts = []
    for label in log:
        results = [judged[f"{label}__{it['id']}"] for it in harmless]
        defl = sum(1 for jr in results if jr.register == "prc_deflection")
        log[label]["legacy_binary_rate"] = defl / max(1, len(harmless))
        log[label]["three_class"] = _three_class_breakdown(results)
        for it in harmless:
            jr = judged[f"{label}__{it['id']}"]
            rollouts.append({"id": f"{label}__{it['id']}", "condition": label,
                             "register": jr.register, "coherence": jr.coherence})

    out = {"conditions": log, "rollouts": rollouts}
    write_json(args.out, out)
    print(f"\n3-class when patching tia 3D coords into harmless targets (off/on/incoherent):")
    for label, info in log.items():
        tc = info["three_class"]["counts"]
        print(f"  {label}: ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
