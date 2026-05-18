"""d_zh extraction + 3D-vs-complement decomposition.

Within the ccp_other class only, the zh/en prompt groups are defined by
logit-lens top-1 language at tap 24 (the language-commitment tap). The
direction itself is extracted, decomposed, and steered in the writer band:
zh_mean and en_mean are taken at tap 20 (the language-empty valley where
the 3D writer signal is fully written), d_zh is the unit diff-of-means,
and it is split onto the QR-orthonormalised 3D writer basis (also at tap
20) and that basis's orthogonal complement. Steering each component alone
at L19 isolates whether the verdict-changing effect lives in the 3D
projection or the complement, i.e. whether the language axis and the
verdict are causally entangled or separable.
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
from qwc.hooks import make_steer_hook
from qwc.io import load_npz, write_json
from qwc.judge import judge_all
from qwc.logit_lens import cjk_token_mask, lm_head_vocab_size, top1_per_tap
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
    ap.add_argument("--alpha", type=float, default=30.0)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e22_dzh_decomposition.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    ccp_other = [it for it in items if it["class"] == "prc_sensitive"
                 and it["topic"] != "tiananmen" and it["id"] not in excl]

    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    ids = [str(x) for x in acts["ids"]]
    id_to_idx = {p: i for i, p in enumerate(ids)}

    # Tap that defines the zh/en split (logit-lens language commitment) vs the
    # writer-band tap where the direction is extracted and decomposed. They
    # differ by design: the language label comes from tap 24, the geometry
    # from tap 20 where the 3D writer signal is fully written but no specific
    # token has decoded yet.
    LANG_SPLIT_TAP = 24
    WRITER_TAP = 20
    STEER_LAYER = WRITER_TAP - 1  # hooking L19 writes the residual that becomes tap 20

    lm = model_mod.load_posttrain()
    cjk_mask = cjk_token_mask(lm.tokenizer, lm_head_vocab_size(lm))
    # Language groups: top-1 at the language-commitment tap, ccp_other only.
    sub_idx = [id_to_idx[it["id"]] for it in ccp_other]
    top1 = top1_per_tap(acts["hidden"][sub_idx], lm)  # [n, n_taps]
    cjk_split = cjk_mask[top1[:, LANG_SPLIT_TAP]]

    zh_ids = [it["id"] for it, is_cjk in zip(ccp_other, cjk_split) if is_cjk]
    en_ids = [it["id"] for it, is_cjk in zip(ccp_other, cjk_split) if not is_cjk]
    print(f"Within ccp_other at tap {LANG_SPLIT_TAP}: "
          f"Chinese top-1 = {len(zh_ids)}, English = {len(en_ids)}", flush=True)

    if not zh_ids or not en_ids:
        print("Insufficient data for d_zh extraction.")
        return

    # Extract the direction in the writer band.
    zh_mean = acts["hidden"][[id_to_idx[i] for i in zh_ids], WRITER_TAP].astype(np.float32).mean(0)
    en_mean = acts["hidden"][[id_to_idx[i] for i in en_ids], WRITER_TAP].astype(np.float32).mean(0)
    d_zh_raw = zh_mean - en_mean
    d_zh = d_zh_raw / max(np.linalg.norm(d_zh_raw), 1e-12)

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    B = qr_orthonormalize([dirs.d_prc[WRITER_TAP], dirs.d_refuse[WRITER_TAP],
                           dirs.d_style[WRITER_TAP]])  # [H, 3]
    coords_3 = B.T @ d_zh
    d_zh_3d = B @ coords_3
    d_zh_complement = d_zh - d_zh_3d

    norm_3 = float(np.linalg.norm(d_zh_3d))
    norm_c = float(np.linalg.norm(d_zh_complement))
    # Subspace share reported as a squared-norm (variance) fraction.
    denom = norm_3 ** 2 + norm_c ** 2
    var_frac_3d = norm_3 ** 2 / denom
    var_frac_complement = norm_c ** 2 / denom
    print(f"|d_zh| = 1.0; 3D-projection norm = {norm_3:.3f}; complement norm = {norm_c:.3f}")
    print(f"Variance split: 3D = {100*var_frac_3d:.0f}%; "
          f"complement = {100*var_frac_complement:.0f}%")

    steer_layer = STEER_LAYER
    judge_items = []
    for label, vec in [("full_dzh", d_zh), ("dzh_3d", d_zh_3d), ("dzh_complement", d_zh_complement)]:
        vec_unit = vec / max(np.linalg.norm(vec), 1e-12)
        d_t = torch.from_numpy(vec_unit.astype(np.float32)).to(lm.device)
        specs = ((lm.layers[steer_layer], "forward", make_steer_hook(d_t, args.alpha)),)
        prompts = [it["chat"] for it in ccp_other]
        texts = generate(lm, prompts, hook_specs=specs,
                         max_new_tokens=128, batch_size=8, verbose=False)
        for it, t in zip(ccp_other, texts):
            judge_items.append({"id": f"{label}__{it['id']}", "question": it["chat"], "response": t})

    judged = judge_all(judge_items)
    out = {
        "d_zh_decomposition": {
            "lang_split_tap": LANG_SPLIT_TAP,
            "writer_tap": WRITER_TAP,
            "steer_layer": STEER_LAYER,
            "alpha": args.alpha,
            "norm_3d": norm_3,
            "norm_complement": norm_c,
            "variance_fraction_3d": var_frac_3d,
            "variance_fraction_complement": var_frac_complement,
        }
    }
    rollouts = []
    for label in ["full_dzh", "dzh_3d", "dzh_complement"]:
        results = [judged[f"{label}__{it['id']}"] for it in ccp_other]
        offs = sum(1 for jr in results if jr.register != "prc_propaganda")
        out[label] = {
            "three_class": _three_class_breakdown(results),
            "legacy_binary_rate": offs / max(1, len(ccp_other)),
        }
        for it in ccp_other:
            jr = judged[f"{label}__{it['id']}"]
            rollouts.append({"id": f"{label}__{it['id']}", "condition": label,
                             "register": jr.register, "coherence": jr.coherence})
        tc = out[label]["three_class"]["counts"]
        print(f"  {label}: off={tc['off_propaganda']} on={tc['on_propaganda']} "
              f"incoherent={tc['incoherent']}")

    out["rollouts"] = rollouts
    write_json(args.out, out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
