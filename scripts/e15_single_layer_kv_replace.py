"""Single-layer K/V replace + topic-token-only K/V zero.

(a) Sweep one full-attention layer at a time, replacing K/V with a fixed
    non-PRC reference source's projections; resume generation; measure
    off-propaganda. Tests whether any single full-attention layer's K/V
    is the load-bearing one.

(b) Topic-token-only K/V zero: zero K and V projections only at prompt
    positions carrying topic-bearing tokens.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.generate import generate
from qwc.hooks import installed
from qwc.io import write_json
from qwc.judge import judge_all
from qwc.model import render_chat_batch, tokenize_batch
from qwc.patching import KVCapture, make_kv_inject_specs, is_full_attention_layer, _attention_module
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


TOPIC_SUBSTRS = ["Tianan", "men", "Taiwan", "Xinjiang", "Hong", "Kong",
                 "Tibet", "Falun", "Xi", "Jinping"]


def topic_positions(tokenizer, text: str) -> set[int]:
    ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    decoded = tokenizer.batch_decode([[i] for i in ids])
    return {i for i, s in enumerate(decoded) if any(sub in s for sub in TOPIC_SUBSTRS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e15_single_layer_kv.json"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    target_items = [it for it in items if it["class"] == "prc_sensitive"
                    and it["topic"] != "tiananmen" and it["id"] not in excl]
    source_items = [it for it in items if it["class"] == "harmless" and it["id"] not in excl]
    # Use the longest harmless source so K/V injection truncates instead of
    # zero-pads (zero-padded K/V positions otherwise contaminate softmax).
    source = max(source_items, key=lambda it: len(it["chat"]))
    print(f"Targets: {len(target_items)}; source: {source['id']} (len {len(source['chat'])} chars)", flush=True)

    lm = model_mod.load_posttrain()
    full_idx = [i for i in range(lm.num_layers) if is_full_attention_layer(lm.layers[i])]

    # Capture K/V from one source pass at every full-attn layer at once.
    cap = KVCapture(layer_indices=full_idx)
    rendered = render_chat_batch(lm.tokenizer, [source["chat"]])
    enc = tokenize_batch(lm.tokenizer, rendered, lm.device)
    with installed(cap.install_specs(lm.layers)), torch.inference_mode():
        lm.model(**enc, use_cache=False, return_dict=True)

    judge_items = []
    log = {"single_layer": {}, "topic_token_zero_all": {}, "topic_token_zero_late": {}}

    # (a) one full-attn layer at a time
    for L in full_idx:
        inject_specs = make_kv_inject_specs(lm.layers, [L], cap.captured_k, cap.captured_v)
        texts = generate(lm, [it["chat"] for it in target_items], hook_specs=inject_specs,
                         max_new_tokens=96, batch_size=args.batch_size, verbose=False)
        for it, t in zip(target_items, texts):
            judge_items.append({"id": f"L{L}__{it['id']}", "question": it["chat"], "response": t})
        log["single_layer"][L] = {"n": len(target_items)}
        print(f"  single-layer K/V replace L={L}: queued", flush=True)

    # (b) topic-token zero — positions depend on the target text so we still
    #     iterate per-prompt, but at least each iteration is a single forward.
    def topic_zero_specs(layers_used: list[int], target_text: str):
        rendered_t = render_chat_batch(lm.tokenizer, [target_text])[0]
        positions = sorted(topic_positions(lm.tokenizer, rendered_t))
        specs = []
        for L in layers_used:
            attn = _attention_module(lm.layers[L])
            for proj_name in ("k_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is None:
                    continue
                def make_hook(pos):
                    def hook(module, args_, output):
                        out = output.clone()
                        for p in pos:
                            if p < out.shape[1]:
                                out[:, p, :] = 0.0
                        return out
                    return hook
                specs.append((proj, "forward", make_hook(positions)))
        return specs

    for variant_label, layers_used in [("topic_token_zero_all", full_idx),
                                       ("topic_token_zero_late", [L for L in full_idx if L >= 23])]:
        for it in target_items:
            specs = topic_zero_specs(layers_used, it["chat"])
            text = generate(lm, [it["chat"]], hook_specs=specs,
                            max_new_tokens=96, batch_size=1, verbose=False)[0]
            judge_items.append({"id": f"{variant_label}__{it['id']}", "question": it["chat"], "response": text})
        log[variant_label]["n"] = len(target_items)

    judged = judge_all(judge_items)
    rollouts = []
    for L in full_idx:
        results = [judged[f"L{L}__{it['id']}"] for it in target_items]
        offs = sum(1 for jr in results if jr.register != "prc_propaganda")
        log["single_layer"][L]["legacy_binary_rate"] = offs / max(1, len(target_items))
        log["single_layer"][L]["three_class"] = _three_class_breakdown(results)
        for it in target_items:
            jr = judged[f"L{L}__{it['id']}"]
            rollouts.append({"id": f"L{L}__{it['id']}", "condition": f"single_layer_L{L}",
                             "register": jr.register, "coherence": jr.coherence})
    for v in ("topic_token_zero_all", "topic_token_zero_late"):
        results = [judged[f"{v}__{it['id']}"] for it in target_items]
        offs = sum(1 for jr in results if jr.register != "prc_propaganda")
        log[v]["legacy_binary_rate"] = offs / max(1, len(target_items))
        log[v]["three_class"] = _three_class_breakdown(results)
        for it in target_items:
            jr = judged[f"{v}__{it['id']}"]
            rollouts.append({"id": f"{v}__{it['id']}", "condition": v,
                             "register": jr.register, "coherence": jr.coherence})

    log["rollouts"] = rollouts
    write_json(args.out, log)
    print("\nSingle-layer K/V replace (off/on/incoherent):")
    for L, info in log["single_layer"].items():
        tc = info["three_class"]["counts"]
        print(f"  L{L}: ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    for v in ("topic_token_zero_all", "topic_token_zero_late"):
        tc = log[v]["three_class"]["counts"]
        print(f"{v}: ({tc['off_propaganda']}/{tc['on_propaganda']}/{tc['incoherent']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
