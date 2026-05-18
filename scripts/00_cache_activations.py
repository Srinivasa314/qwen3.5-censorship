"""Cache last-prompt-token residuals at every tap, for all 200 prompts.

Output: results/activations_{label}.npz with
    hidden: float16 [N, L+1, H]
    ids:    array of N prompt IDs

Run for both posttrain and base checkpoints (the base run is used for the
checkpoint-comparison experiment).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwc import config, data, model as model_mod
from qwc.activations import cache_last_token_residuals
from qwc.io import save_npz


def collect_prompts():
    items = data.all_items()
    return items, [it["chat"] for it in items], [it["id"] for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=["posttrain", "base"], default="posttrain")
    ap.add_argument("--batch-size", type=int, default=25)
    args = ap.parse_args()

    path = config.MODEL_POSTTRAIN if args.checkpoint == "posttrain" else config.MODEL_BASE
    label = args.checkpoint

    items, chat_texts, ids = collect_prompts()
    print(f"Collected {len(items)} prompts. Loading {label} from {path} ...", flush=True)
    lm = model_mod.load(path)
    print(f"  loaded. num_layers={lm.num_layers} hidden={lm.hidden_size}", flush=True)

    residuals = cache_last_token_residuals(lm, chat_texts, batch_size=args.batch_size)
    out_path = config.RESULTS_DIR / f"activations_{label}.npz"
    save_npz(out_path, hidden=residuals, ids=np.array(ids))
    print(f"Saved {out_path}  shape={residuals.shape}")


if __name__ == "__main__":
    main()
