"""MLP-neuron ablation (600 neurons across L20-L31).

Identify the 50 most verdict-discriminative MLP neurons per reader-band
layer by probing on baseline-propaganda vs factual-answer residuals, then
zero their MLP-output channels at every position during generation.
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
from qwc.hooks import installed
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.model import render_chat_batch, tokenize_batch
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


def capture_mlp_outputs(lm, prompts: list[str], layer_indices: list[int]) -> dict[int, np.ndarray]:
    """Return {L: [N, H] MLP-output residual at last token} for each L."""
    out: dict[int, list[np.ndarray]] = {L: [] for L in layer_indices}

    def make_hook(L):
        def hook(module, args_, output):
            h = output[0] if isinstance(output, tuple) else output
            out[L].append(h[:, -1, :].detach().float().cpu().numpy())
            return output
        return hook

    BATCH = 8
    for bi in range(0, len(prompts), BATCH):
        chunk = prompts[bi : bi + BATCH]
        rendered = render_chat_batch(lm.tokenizer, chunk)
        enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
        specs = [(getattr(lm.layers[L], "mlp"), "forward", make_hook(L))
                 for L in layer_indices if hasattr(lm.layers[L], "mlp")]
        with installed(specs), torch.no_grad():
            lm.model(**enc, use_cache=False, return_dict=True)
    return {L: np.concatenate(arrs, axis=0) for L, arrs in out.items() if arrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e13_mlp_neuron_ablation.json"))
    ap.add_argument("--per-layer", type=int, default=50)
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(20, 32)))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    prop_items = [it for it in items if it["class"] == "prc_sensitive"
                  and it["topic"] != "tiananmen" and it["id"] not in excl]
    factual_items = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]

    lm = model_mod.load_posttrain()
    print("Capturing MLP outputs (propaganda) ...", flush=True)
    P = capture_mlp_outputs(lm, [it["chat"] for it in prop_items], args.layers)
    print("Capturing MLP outputs (factual) ...", flush=True)
    F = capture_mlp_outputs(lm, [it["chat"] for it in factual_items], args.layers)

    selected: dict[int, list[int]] = {}
    for L in args.layers:
        if L not in P or L not in F:
            continue
        mu_p = P[L].mean(0); mu_f = F[L].mean(0)
        sd = np.sqrt((P[L].var(0) + F[L].var(0)) / 2 + 1e-6)
        score = np.abs(mu_p - mu_f) / sd
        top = np.argsort(-score)[: args.per_layer]
        selected[L] = top.tolist()
    print(f"Selected {sum(len(v) for v in selected.values())} neurons across {len(selected)} layers")

    def mlp_zero_specs():
        specs = []
        for L, neurons in selected.items():
            mlp = lm.layers[L].mlp
            ns = list(neurons)
            def make_hook(ns_):
                def hook(module, args_, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    h[..., ns_] = 0.0
                    return (h,) + output[1:] if isinstance(output, tuple) else h
                return hook
            specs.append((mlp, "forward", make_hook(ns)))
        return specs

    target_items = prop_items
    texts = generate(lm, [it["chat"] for it in target_items],
                     hook_specs=mlp_zero_specs(),
                     max_new_tokens=128, batch_size=8, verbose=True)
    judge_items = [{"id": it["id"], "question": it["chat"], "response": t}
                   for it, t in zip(target_items, texts)]
    judged = judge_all(judge_items)
    offs = sum(1 for it in target_items if judged[it["id"]].register != "prc_propaganda")
    legacy_rate = offs / max(1, len(target_items))
    three_class = _three_class_breakdown([judged[it["id"]] for it in target_items])
    rollouts = [{"id": it["id"], "topic": it["topic"], "text": t,
                 "register": judged[it["id"]].register,
                 "coherence": judged[it["id"]].coherence}
                for it, t in zip(target_items, texts)]
    write_json(args.out, {
        "three_class": three_class,
        "legacy_binary_rate": legacy_rate,
        "selected_neurons": selected,
        "n_target": len(target_items),
        "rollouts": rollouts,
    })
    tc = three_class["counts"]
    print(f"3-class: off={tc['off_propaganda']} on={tc['on_propaganda']} "
          f"incoherent={tc['incoherent']} (n={three_class['n']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
