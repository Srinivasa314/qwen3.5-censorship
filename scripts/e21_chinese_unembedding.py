"""Chinese-unembedding ablation (behavioural inertness test).

Subtract a large constant from every CJK token's logit at lm_head so Chinese
tokens cannot be emitted. Regenerate a baseline pass and measure how often
the verdict register stays the same. If it's largely unchanged, the
mid-stack Chinese template is behaviourally inert and the decision lives
upstream rather than in the language flavour of the readout.
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
from qwc.generate import generate
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.logit_lens import cjk_token_mask, lm_head_vocab_size
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
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e21_chinese_unembedding.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    keep = [it for it in items if it["id"] not in excl]

    lm = model_mod.load_posttrain()
    V = lm_head_vocab_size(lm)
    cjk_mask = cjk_token_mask(lm.tokenizer, V)
    cjk_idx = np.where(cjk_mask)[0]
    print(f"CJK tokens: {len(cjk_idx)} / {V}", flush=True)

    bias = torch.full((V,), 0.0, device=lm.device, dtype=torch.float32)
    bias[cjk_idx] = -1e9
    # Hook lm_head to subtract `bias`
    def hook(module, args_, output):
        return output + bias.to(output.dtype)
    specs = ((lm.model.lm_head, "forward", hook),)

    base_texts = generate(lm, [it["chat"] for it in keep], hook_specs=(),
                          max_new_tokens=128, batch_size=8, verbose=True)
    no_zh_texts = generate(lm, [it["chat"] for it in keep], hook_specs=specs,
                           max_new_tokens=128, batch_size=8, verbose=True)

    judge_items = []
    for it, t_base, t_noz in zip(keep, base_texts, no_zh_texts):
        judge_items.append({"id": f"base__{it['id']}", "question": it["chat"], "response": t_base})
        judge_items.append({"id": f"nozh__{it['id']}", "question": it["chat"], "response": t_noz})
    judged = judge_all(judge_items)

    same = 0
    rollouts = []
    for it in keep:
        jb = judged[f"base__{it['id']}"]
        jn = judged[f"nozh__{it['id']}"]
        rollouts.append({"id": it["id"], "class": it["class"], "topic": it["topic"],
                         "base_register": jb.register, "base_coherence": jb.coherence,
                         "nozh_register": jn.register, "nozh_coherence": jn.coherence})
        if jb.register == jn.register:
            same += 1
    same_register_rate = same / max(1, len(keep))
    three_class = _three_class_breakdown([judged[f"nozh__{it['id']}"] for it in keep])
    write_json(args.out, {
        "three_class": three_class,
        "same_register_rate": same_register_rate,
        "n": len(keep),
        "rollouts": rollouts,
    })
    tc = three_class["counts"]
    print(f"\n3-class after Chinese ban: off={tc['off_propaganda']} "
          f"on={tc['on_propaganda']} incoherent={tc['incoherent']} (n={three_class['n']})")
    print(f"Same-register rate after Chinese ban: {100*same_register_rate:.1f}%")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
