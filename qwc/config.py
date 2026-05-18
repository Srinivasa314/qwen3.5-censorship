"""Paths and architectural constants.

Override via env vars: QWC_ROOT, QWC_MODEL_POSTTRAIN, QWC_MODEL_BASE.
"""
import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default))


PROJECT_ROOT = _env_path("QWC_ROOT", str(Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

PROMPTS_PATH = DATA_DIR / "prompts.json"
GRID_PUBLIC_PATH = DATA_DIR / "steering_grid_public.json"
GRID_INPUTS_PATH = DATA_DIR / "steering_grid_inputs.json"

# Set these to local checkpoint paths via the env vars (see README).
MODEL_POSTTRAIN = os.environ.get("QWC_MODEL_POSTTRAIN", "Qwen3.5-9B")
MODEL_BASE = os.environ.get("QWC_MODEL_BASE", "Qwen3.5-9B-Base")

NUM_LAYERS = 32
HIDDEN_SIZE = 4096
N_TAPS = NUM_LAYERS + 1  # taps 0..32 inclusive

# Writer-band tap (where the direction is extracted) and steer-layer (which
# layer's forward output is hooked). The two are off-by-one by construction:
# hooking layer k writes the residual that becomes tap k+1.
DIRECTION_LAYOUT = {
    "d_prc":    {"tap": 14, "steer_layer": 13},
    "d_refuse": {"tap": 19, "steer_layer": 18},
    "d_style":  {"tap": 19, "steer_layer": 18},
}
