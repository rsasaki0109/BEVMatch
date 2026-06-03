"""Scan-Context-style LiDAR descriptor (§9.5 baseline).

A polar (ring x sector) signature is built from the ground-projected points.
The per-ring marginal — the *ring key* — is rotation invariant and used for
Top-K retrieval; the full polar grid lets us recover a coarse yaw via column
shift (used as a seed and as retrieval evidence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanContextConfig:
    n_rings: int = 20
    n_sectors: int = 60
    max_range_m: float = 30.0
    smooth_passes: int = 2  # blur the polar grid for robustness to small jitter


_BLUR_KERNEL = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
_BLUR_KERNEL = _BLUR_KERNEL / _BLUR_KERNEL.sum()


def _smooth_polar(sc: np.ndarray, passes: int) -> np.ndarray:
    """Separable blur: edge-padded along rings, circular along sectors.

    Sparse scans make raw column-cosine distances brittle to a one-cell shift;
    smoothing lets neighbouring cells contribute and stabilises matching.
    """
    k = _BLUR_KERNEL
    pad = len(k) // 2
    out = sc
    for _ in range(passes):
        out = np.apply_along_axis(
            lambda m: np.convolve(np.pad(m, pad, mode="edge"), k, "valid"), 0, out
        )
        ext = np.concatenate([out[:, -pad:], out, out[:, :pad]], axis=1)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 1, ext)
    return out


def scan_context(points_xy: np.ndarray, config: ScanContextConfig) -> np.ndarray:
    """Return an ``(n_rings, n_sectors)`` (optionally smoothed) polar grid."""
    pts = np.asarray(points_xy, dtype=float)
    sc = np.zeros((config.n_rings, config.n_sectors), dtype=float)
    if pts.size == 0:
        return sc

    x, y = pts[:, 0], pts[:, 1]
    r = np.hypot(x, y)
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)
    keep = r < config.max_range_m
    if not np.any(keep):
        return sc

    ring = np.clip((r[keep] / config.max_range_m * config.n_rings).astype(int), 0, config.n_rings - 1)
    sector = np.clip((theta[keep] / (2 * np.pi) * config.n_sectors).astype(int), 0, config.n_sectors - 1)
    np.add.at(sc, (ring, sector), 1.0)
    if config.smooth_passes > 0:
        sc = _smooth_polar(sc, config.smooth_passes)
    return sc


def ring_key(sc: np.ndarray) -> np.ndarray:
    """Rotation-invariant per-ring marginal of a scan context."""
    return sc.mean(axis=1)


def ring_key_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between L1-normalised ring keys (scale invariant)."""
    a = a / (a.sum() + 1e-9)
    b = b / (b.sum() + 1e-9)
    return float(np.linalg.norm(a - b))


def _column_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-column cosine distance over columns occupied in both grids."""
    num = (a * b).sum(axis=0)
    da = np.linalg.norm(a, axis=0)
    db = np.linalg.norm(b, axis=0)
    valid = (da > 0) & (db > 0)
    if not np.any(valid):
        return 1.0
    cos = num[valid] / (da[valid] * db[valid])
    return float(1.0 - cos.mean())


def sc_alignment_distance(sc_query: np.ndarray, sc_ref: np.ndarray) -> tuple[float, int]:
    """Scan-Context distance: min mean-column-cosine distance over sector shifts.

    Returns ``(distance, best_shift)``. This uses the full angular structure, so
    it discriminates place identity far better than the ring key alone, while
    still being rotation-robust via the shift search.
    """
    n = sc_query.shape[1]
    best_dist, best_shift = np.inf, 0
    for shift in range(n):
        d = _column_cosine_distance(np.roll(sc_query, shift, axis=1), sc_ref)
        if d < best_dist:
            best_dist, best_shift = d, shift
    return float(best_dist), int(best_shift)


def shift_to_yaw(shift: int, config: ScanContextConfig) -> float:
    """Convert a sector shift to a yaw estimate (radians)."""
    return float(shift / config.n_sectors * 2 * np.pi)
