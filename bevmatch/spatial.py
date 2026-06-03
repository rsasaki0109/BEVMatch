"""Nearest-neighbor search that scales to dense point clouds.

The ICP aligners need per-point nearest neighbours. A naive ``(N, M)`` distance
matrix is O(N·M) memory and OOMs on real LiDAR (tens of thousands of points).
This uses ``scipy.spatial.cKDTree`` when available (fast), and otherwise falls
back to a chunked brute force that is memory-safe — so the core stays
numpy-only while real dense clouds still work (install ``scipy`` for speed).
"""

from __future__ import annotations

import numpy as np


def nearest_neighbors(
    query: np.ndarray,
    ref: np.ndarray,
    chunk: int = 2048,
    brute_threshold: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distances, indices)`` of each ``query`` point's nearest ``ref`` point.

    Small problems use a vectorised brute force (fastest, exact). Large ones use a
    KD-tree if scipy is available, else a memory-safe chunked brute force.
    """
    query = np.asarray(query, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n, m = len(query), len(ref)
    if n == 0 or m == 0:
        return np.empty(n), np.zeros(n, dtype=int)

    # Small: a single vectorised distance matrix is fast and fits in memory.
    if n * m <= brute_threshold:
        d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        idx = d2.argmin(axis=1)
        return np.sqrt(d2[np.arange(n), idx]), idx

    # Large: KD-tree (fast) when scipy is present.
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(ref).query(query, workers=-1)
        return np.asarray(dist), np.asarray(idx, dtype=int)
    except ImportError:
        pass

    # Large without scipy: memory-safe chunked brute force.
    idx = np.empty(n, dtype=int)
    dist = np.empty(n, dtype=float)
    for s in range(0, n, chunk):
        q = query[s:s + chunk]
        d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        j = d2.argmin(axis=1)
        idx[s:s + chunk] = j
        dist[s:s + chunk] = np.sqrt(d2[np.arange(len(q)), j])
    return dist, idx
