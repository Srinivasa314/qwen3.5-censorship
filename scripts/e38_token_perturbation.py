"""Token-perturbation control.

Replace topic-bearing tokens ("Tiananmen", "Falun", "Taiwan", "Xinjiang",
"Hong Kong", "Tibet", "Xi") with the literal string "[X]". Re-run
generation and classify. The fraction that shifts off the prc verdict
measures how much of the topic detection is surface-token matching vs
semantic detection from the surrounding context.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.generate import generate
from qwc.io import write_json
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


TOPIC_WORDS = [
    "Tiananmen", "Tank Man", "Falun Gong", "Falun", "Taiwan", "Xinjiang",
    "Uyghur", "Uighur", "Hong Kong", "Tibet", "Xi Jinping", "Xi ",
    "June 4", "June 4th", "1989",
]
RE_TOPICS = re.compile("|".join(re.escape(w) for w in TOPIC_WORDS), re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "e38_token_perturbation.json"))
    args = ap.parse_args()

    items = data.all_items()
    excl = data.excluded_ids()
    prc = [it for it in items if it["class"] == "prc_sensitive" and it["id"] not in excl]
    perturbed = []
    for it in prc:
        text = RE_TOPICS.sub("[X]", it["chat"])
        perturbed.append({"id": it["id"], "chat": text, "topic": it["topic"]})
    print(f"Perturbing {len(prc)} PRC prompts; sample:", flush=True)
    for it in perturbed[:3]:
        print(f"  {it['id']}: {it['chat']!r}")

    lm = model_mod.load_posttrain()
    texts = generate(lm, [it["chat"] for it in perturbed],
                     max_new_tokens=128, batch_size=16, verbose=True)
    judge_items = [{"id": it["id"], "question": it["chat"], "response": t}
                   for it, t in zip(perturbed, texts)]
    judged = judge_all(judge_items)

    rollouts = []
    off = 0
    for it, t in zip(perturbed, texts):
        jr = judged[it["id"]]
        r = jr.register
        is_off = r not in ("prc_deflection", "prc_propaganda")
        if is_off:
            off += 1
        rollouts.append({"id": it["id"], "topic": it["topic"], "perturbed_chat": it["chat"],
                         "text": t, "register": r, "coherence": jr.coherence,
                         "legacy_off_prc": is_off})

    three_class = _three_class_breakdown([judged[it["id"]] for it in perturbed])
    out = {
        "three_class": three_class,
        "legacy_binary_rate": off / max(1, len(perturbed)),
        "n": len(perturbed),
        "rollouts": rollouts,
    }
    write_json(args.out, out)
    tc = three_class["counts"]
    print(f"\n3-class (anonymised prompts): off={tc['off_propaganda']} "
          f"on={tc['on_propaganda']} incoherent={tc['incoherent']} (n={three_class['n']})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
