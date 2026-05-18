"""Small JSON / NPZ I/O helpers used across scripts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path, obj: Any, *, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_json(path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_npz(path, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path) -> dict[str, np.ndarray]:
    blob = np.load(path)
    return {k: blob[k] for k in blob.files}
