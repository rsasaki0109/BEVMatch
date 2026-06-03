"""Shared binary-grid helpers (dilation, connected components).

Used by the change-detection and map-validation layers so the rasterised
geometry operations stay consistent.
"""

from __future__ import annotations

import numpy as np


def dilate(mask: np.ndarray, passes: int = 1) -> np.ndarray:
    """3x3 binary dilation, applied ``passes`` times (no scipy)."""
    out = mask.copy()
    for _ in range(passes):
        grown = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= np.roll(np.roll(out, dr, axis=0), dc, axis=1)
        out = grown
    return out


def connected_components(mask: np.ndarray, min_cells: int) -> list[np.ndarray]:
    """8-connected components as arrays of (row, col), filtered by size."""
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    rows, cols = np.nonzero(mask)
    for r0, c0 in zip(rows, cols):
        if visited[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        visited[r0, c0] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < mask.shape[0]
                        and 0 <= nc < mask.shape[1]
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
        if len(cells) >= min_cells:
            comps.append(np.array(cells))
    return comps
