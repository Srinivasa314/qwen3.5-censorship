"""Channel transplant: register-transition matrix.

At L19 output, replace the target's 3D writer-subspace coordinates (or,
separately, its complement) with a source class's, at the prompt's last-K
positions only (prefill; decode-guarded). Instead of a single flip rate we
report the full baseline-register -> post-intervention-register transition
matrix with coherence, plus the three-class outcome distribution.

Generalisation is judged by internal consistency: the target set is split
into two halves and their transition matrices are reported side by side
(no external number is used as ground truth).
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
from qwc.io import load_npz, write_json
from qwc.judge import judge_all
from qwc.taxonomy import classify_outcome


def _dist(jrs):
    c = Counter(classify_outcome(jr) for jr in jrs)
    n = max(1, sum(c.values()))
    return {k: c.get(k, 0) / n for k in
            ("off_propaganda", "on_propaganda", "incoherent")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e46_channel_transplant.json"))
    args = ap.parse_args()
    K = args.k

    items = data.all_items()
    excl = data.excluded_ids()
    target = [it for it in items if it["class"] == "prc_sensitive"
              and it["topic"] != "tiananmen" and it["id"] not in excl]
    source = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]

    dirs = Directions.load(config.RESULTS_DIR / "directions_posttrain.npz")
    tap = 20  # output of L19
    B = qr_orthonormalize([dirs.d_prc[tap], dirs.d_refuse[tap], dirs.d_style[tap]])  # [H,3]

    acts = load_npz(config.RESULTS_DIR / "activations_posttrain.npz")
    idx = {str(p): i for i, p in enumerate(acts["ids"])}
    H = acts["hidden"]
    src_resid = np.stack([H[idx[it["id"]], tap] for it in source]).astype(np.float32)  # [Ns,H]
    src_coords = (src_resid @ B).mean(0)                       # [3]
    src_3d_vec = (B @ src_coords).astype(np.float32)           # [H]
    src_comp = (src_resid - (src_resid @ B) @ B.T).mean(0).astype(np.float32)  # [H]

    lm = model_mod.load_posttrain()
    Bt = torch.from_numpy(B).to(lm.device).to(torch.float32)
    s3 = torch.from_numpy(src_3d_vec).to(lm.device)
    sc = torch.from_numpy(src_comp).to(lm.device)

    def make_hook(mode):
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1 or h.shape[1] < K:          # prompt-window only
                return output
            Bh = Bt.to(h.dtype)
            tgt = h[:, -K:, :]
            proj = (tgt @ Bh) @ Bh.T                        # target 3D part
            if mode == "subspace":                          # source 3D + target complement
                new = (tgt - proj) + s3.to(h.dtype)
            else:                                           # complement: target 3D + source complement
                new = proj + sc.to(h.dtype)
            h2 = h.clone()
            h2[:, -K:, :] = new
            return (h2,) + output[1:] if isinstance(output, tuple) else h2
        return hook

    judge_items, text_by, cond_by = [], {}, {}
    conds = [("baseline", None), ("subspace", "subspace"), ("complement", "complement")]
    for label, mode in conds:
        specs = () if mode is None else ((lm.layers[19], "forward", make_hook(mode)),)
        print(f"\n{label}: generating ...", flush=True)
        texts = generate(lm, [it["chat"] for it in target], hook_specs=specs,
                         max_new_tokens=args.max_new, batch_size=8, verbose=False)
        for it, t in zip(target, texts):
            jid = f"{label}__{it['id']}"
            judge_items.append({"id": jid, "question": it["chat"], "response": t})
            text_by[jid] = t
            cond_by[jid] = label

    judged = judge_all(judge_items)

    def reg(label, pid):
        return judged[f"{label}__{pid}"].register

    # split target by index parity for the internal-consistency check
    halves = {"split_a": target[0::2], "split_b": target[1::2]}
    out: dict = {"k": K, "tap": tap, "n_target": len(target),
                 "n_source": len(source), "transition": {}, "three_class": {}}
    for label, mode in conds:
        if mode is None:
            continue
        for hname, hitems in halves.items():
            mat = Counter((reg("baseline", it["id"]), reg(label, it["id"]))
                          for it in hitems)
            out["transition"][f"{label}__{hname}"] = {
                f"{a}->{b}": n for (a, b), n in sorted(mat.items(), key=lambda z: -z[1])}
        out["three_class"][label] = _dist([judged[f"{label}__{it['id']}"] for it in target])
    out["three_class"]["baseline"] = _dist([judged[f"baseline__{it['id']}"] for it in target])
    out["rollouts"] = [{"id": jid, "condition": cond_by[jid], "text": text_by[jid],
                        "register": judged[jid].register,
                        "coherence": judged[jid].coherence} for jid in text_by]

    write_json(args.out, out)
    print("\ncond        off%   on%  incoh%")
    for label in ("baseline", "subspace", "complement"):
        d = out["three_class"][label]
        print(f"  {label:>10}: {100*d['off_propaganda']:5.1f} "
              f"{100*d['on_propaganda']:5.1f} {100*d['incoherent']:5.1f}")
    print("\ntransition (subspace, split_a):")
    for k, v in list(out["transition"]["subspace__split_a"].items())[:6]:
        print(f"  {k}: {v}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
