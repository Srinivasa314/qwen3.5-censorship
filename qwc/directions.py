"""Diff-of-means direction extraction.

Three directions, unit-normalised per tap:
    d_prc    = mean(prc_sensitive)       - mean(neutral_political)
    d_refuse = mean(harmful)             - mean(harmless)
    d_style  = mean(tia)                 - mean(prc_other)

Sign convention: positive d_style points toward the Tiananmen-deflection
register (the tia side); negative d_style points toward the propaganda
register. Steering at α<0 on a Tiananmen prompt therefore pushes it
toward the trained denial-propaganda template (not toward a factual
answer; factual answers are only reachable via d_prc).
The six overgeneralization/anomaly IDs are excluded from class means.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .data import class_means_groups
from .config import DIRECTION_LAYOUT


@dataclass
class Directions:
    """Three directions at every tap, unit-normalised."""
    d_prc:    np.ndarray   # [n_taps, H]
    d_refuse: np.ndarray
    d_style:  np.ndarray
    n_taps: int
    hidden: int

    def at_canonical(self, name: str) -> np.ndarray:
        """Return the named direction at its canonical writer-band tap."""
        tap = DIRECTION_LAYOUT[name]["tap"]
        return getattr(self, name)[tap]

    def steer_layer(self, name: str) -> int:
        return DIRECTION_LAYOUT[name]["steer_layer"]

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            d_prc=self.d_prc,
            d_refuse=self.d_refuse,
            d_style=self.d_style,
        )

    @classmethod
    def load(cls, path) -> "Directions":
        b = np.load(path)
        return cls(
            d_prc=b["d_prc"],
            d_refuse=b["d_refuse"],
            d_style=b["d_style"],
            n_taps=b["d_prc"].shape[0],
            hidden=b["d_prc"].shape[1],
        )


def _unit_per_tap(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise each tap-row to unit length. v shape [n_taps, H]."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norms, eps)


def diff_of_means(residuals: np.ndarray, ids: list[str],
                  pos_ids: list[str], neg_ids: list[str]) -> np.ndarray:
    """mean(pos) - mean(neg) at every tap. residuals: [N, n_taps, H]."""
    idx = {pid: i for i, pid in enumerate(ids)}
    pos_idx = [idx[i] for i in pos_ids]
    neg_idx = [idx[i] for i in neg_ids]
    pos_mean = residuals[pos_idx].astype(np.float32).mean(0)
    neg_mean = residuals[neg_idx].astype(np.float32).mean(0)
    return pos_mean - neg_mean


def extract_three_axes(residuals: np.ndarray, ids: list[str]) -> Directions:
    """Build d_prc, d_refuse, d_style at every tap, unit-normalised."""
    groups = class_means_groups()
    d_prc_raw    = diff_of_means(residuals, ids, groups["all_prc"], groups["neutral"])
    d_refuse_raw = diff_of_means(residuals, ids, groups["harmful"], groups["harmless"])
    d_style_raw  = diff_of_means(residuals, ids, groups["tiananmen"], groups["prc_other"])
    n_taps, H = d_prc_raw.shape
    return Directions(
        d_prc=_unit_per_tap(d_prc_raw),
        d_refuse=_unit_per_tap(d_refuse_raw),
        d_style=_unit_per_tap(d_style_raw),
        n_taps=n_taps,
        hidden=H,
    )


def project_onto(residuals: np.ndarray, direction: np.ndarray, tap: int) -> np.ndarray:
    """Scalar projection of every prompt's residual at one tap onto a direction."""
    r = residuals[:, tap, :].astype(np.float32)
    d = direction.astype(np.float32)
    d = d / max(np.linalg.norm(d), 1e-12)
    return r @ d


def per_class_stats(projections: np.ndarray, ids: list[str],
                    groups: dict[str, list[str]]) -> dict[str, dict]:
    """Per-group projection mean/std/min/max."""
    idx = {pid: i for i, pid in enumerate(ids)}
    out = {}
    for name, group_ids in groups.items():
        arr = projections[[idx[i] for i in group_ids if i in idx]]
        if arr.size == 0:
            continue
        out[name] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "n":    int(arr.size),
        }
    return out


def qr_orthonormalize(directions: list[np.ndarray]) -> np.ndarray:
    """QR-orthonormalise a list of vectors. Returns [H, k] with orthonormal cols."""
    mat = np.stack(directions, axis=1).astype(np.float32)
    Q, _ = np.linalg.qr(mat)
    return Q


def pairwise_cosines(vecs: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Symmetric pairwise cosine table for a dict of vectors."""
    out: dict[tuple[str, str], float] = {}
    names = list(vecs)
    for a in names:
        va = vecs[a] / (np.linalg.norm(vecs[a]) + 1e-12)
        for b in names:
            vb = vecs[b] / (np.linalg.norm(vecs[b]) + 1e-12)
            out[(a, b)] = float(np.dot(va, vb))
    return out
