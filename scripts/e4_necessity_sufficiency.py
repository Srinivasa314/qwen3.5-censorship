"""Necessity and sufficiency tests for the three-direction subspace.

Necessity: on the Tiananmen → propaganda transition via subspace patching
at L19, replace only ONE of the three coordinates (d_prc / d_refuse /
d_style) at a time and measure the verdict-flip rate. If each direction
carries a distinction the others miss, only the relevant single-direction
patch should flip the verdict.

Sufficiency: PCA on residuals at tap 19 after projecting out the 3D
subspace, and check whether any remaining component still discriminates
the classes — if not, the 3D subspace captures all linear class signal.
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.directions import Directions, qr_orthonormalize
from qwc.generate import generate
from qwc.hooks import make_subspace_patch_hook
from qwc.io import load_npz, write_json
from qwc.judge import judge_all
from qwc.probes import auc
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


def project(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """vec [H], basis [H, k] -> coords [k]."""
    return basis.T @ vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e4_necessity_sufficiency.json"))
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    tap = 19

    # Build the orthonormal 3D basis at tap 19.
    B = qr_orthonormalize([dirs.d_prc[tap], dirs.d_refuse[tap], dirs.d_style[tap]])  # [H, 3]

    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    groups = data.class_means_groups()

    # Source = mean of prc_other (propaganda baseline). Target prompts = tiananmen.
    id_to_idx = {p: i for i, p in enumerate(ids)}
    src_idx = [id_to_idx[i] for i in groups["prc_other"]]
    src_mean = acts["hidden"][src_idx, tap].astype(np.float32).mean(0)
    src_coords_full = project(src_mean, B)   # [3]

    tia_items = [it for it in data.all_items()
                 if it["class"] == "prc_sensitive" and it["topic"] == "tiananmen"
                 and it["id"] not in data.excluded_ids()]
    tia_prompts = [it["chat"] for it in tia_items]

    # Compute target-prompt coords at tap 19 to leave 2 of 3 dims untouched.
    tgt_idx = [id_to_idx[it["id"]] for it in tia_items]
    tgt_coords = (acts["hidden"][tgt_idx, tap].astype(np.float32) @ B)  # [N, 3]
    tgt_coords_mean = tgt_coords.mean(0)  # [3]

    print(f"Loading model ...", flush=True)
    lm = model_mod.load_posttrain()
    basis_t = torch.from_numpy(B).to(lm.device).to(torch.float32)

    # We hook the L18 output (== input to L19, == tap 19 location).
    steer_layer = 18

    def run_condition(label: str, source_3: np.ndarray):
        """source_3 is the [3]-coord vector to inject (everything outside the patched
        dims is replaced too — see make_subspace_patch_hook semantics)."""
        coords_t = torch.from_numpy(source_3.astype(np.float32)).to(lm.device)
        # Broadcast source_coords over (batch, position) — hook handles broadcasting.
        coords_b = coords_t.view(1, 1, -1)
        hook_specs = (
            (lm.layers[steer_layer], "forward",
             make_subspace_patch_hook(basis_t, coords_b)),
        )
        texts = generate(lm, tia_prompts, hook_specs=hook_specs,
                         max_new_tokens=args.max_new, do_sample=False, batch_size=8)
        return texts

    # Build "patch one of three" coord vectors.
    conditions = {
        "d_style_only":  np.array([tgt_coords_mean[0], tgt_coords_mean[1], src_coords_full[2]], np.float32),
        "d_prc_only":    np.array([src_coords_full[0], tgt_coords_mean[1], tgt_coords_mean[2]], np.float32),
        "d_refuse_only": np.array([tgt_coords_mean[0], src_coords_full[1], tgt_coords_mean[2]], np.float32),
        "all_three":     src_coords_full.copy(),
    }

    all_judge_items = []
    cond_responses = {}
    for label, src_3 in conditions.items():
        print(f"\nCondition: {label}", flush=True)
        texts = run_condition(label, src_3)
        cond_responses[label] = list(zip([it["id"] for it in tia_items], texts))
        for it, txt in zip(tia_items, texts):
            all_judge_items.append({"id": f"{label}__{it['id']}", "question": it["chat"], "response": txt})

    print(f"\nJudging {len(all_judge_items)} rollouts ...", flush=True)
    judged = judge_all(all_judge_items)

    # Legacy binary tally: "flipped" = judge says factual_answer /
    # partial_answer / prc_propaganda / denial (anything that left the tia
    # deflection register). Retained alongside the 3-class breakdown.
    flip_labels = {"factual_answer", "partial_answer", "prc_propaganda", "denial"}
    legacy_flip_rates: dict[str, float] = {}
    three_class: dict[str, dict] = {}
    by_cond: dict[str, list] = {}
    rollouts: list[dict] = []
    for label, pairs in cond_responses.items():
        flips = 0
        results = []
        for pid, txt in pairs:
            j = judged[f"{label}__{pid}"]
            results.append(j)
            by_cond.setdefault(label, []).append(
                {"id": pid, "register": j.register, "coherence": j.coherence})
            rollouts.append({"id": f"{label}__{pid}", "condition": label,
                             "text": txt, "register": j.register,
                             "coherence": j.coherence})
            if j.register in flip_labels:
                flips += 1
        legacy_flip_rates[label] = flips / max(1, len(pairs))
        three_class[label] = _three_class_breakdown(results)

    # Sufficiency: PCA on residuals at tap 19 after projecting out the 3D subspace.
    all_ids_used = groups["all_prc"] + groups["neutral"] + groups["harmful"] + groups["harmless"]
    classes = (
        ["prc"] * len(groups["all_prc"])
        + ["neutral"] * len(groups["neutral"])
        + ["harmful"] * len(groups["harmful"])
        + ["harmless"] * len(groups["harmless"])
    )
    X = np.stack([acts["hidden"][id_to_idx[i], tap].astype(np.float32) for i in all_ids_used])
    # Project out the 3D subspace.
    X_proj = X - (X @ B) @ B.T
    # PCA via SVD
    Xc = X_proj - X_proj.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U * S  # [N, k]
    # Top-1 AUC for each principal component against each class-vs-rest.
    sufficiency: dict[str, list[float]] = {}
    for cls_name in ("prc", "neutral", "harmful", "harmless"):
        y = np.array([1 if c == cls_name else 0 for c in classes])
        aucs = [abs(auc(pcs[:, k], y) - 0.5) + 0.5 for k in range(min(20, pcs.shape[1]))]
        sufficiency[cls_name] = aucs

    out = {
        "three_class": three_class,
        "legacy_binary_rate": legacy_flip_rates,
        "by_condition": by_cond,
        "rollouts": rollouts,
        "sufficiency_pca_aucs_top20": sufficiency,
    }
    write_json(args.out, out)
    print(f"\n3-class outcomes (Tiananmen → propaganda transition):")
    for k, v in three_class.items():
        tc = v["counts"]
        print(f"  {k}: off={tc['off_propaganda']} on={tc['on_propaganda']} "
              f"incoherent={tc['incoherent']}")
    print(f"\nSufficiency: max post-projection PC AUC across top-20 components:")
    for cls_name, aucs in sufficiency.items():
        print(f"  {cls_name}: max AUC = {max(aucs):.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
