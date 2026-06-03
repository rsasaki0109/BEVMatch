"""BEV occupancy representation (§1.3, §5.4).

BEV is one shared comparison representation, not a required input. A ground-
projected point set is rasterised onto a square top-down grid centred on the
sensor origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEVConfig:
    range_m: float = 30.0  # half-extent: grid covers [-range_m, +range_m] on each axis
    resolution_m: float = 0.5  # metres per cell

    @property
    def size(self) -> int:
        return int(round(2 * self.range_m / self.resolution_m))

    @property
    def center(self) -> int:
        return self.size // 2


@dataclass
class BEVOccupancy:
    """A top-down occupancy grid. ``grid[row, col]`` indexes (y, x)."""

    grid: np.ndarray  # (size, size) float counts
    config: BEVConfig

    def occupied(self, threshold: float = 0.5) -> np.ndarray:
        return self.grid >= threshold

    def cell_to_xy(self, row: int, col: int) -> tuple[float, float]:
        res = self.config.resolution_m
        x = (col - self.config.center) * res
        y = (row - self.config.center) * res
        return float(x), float(y)


def points_to_bev(points_xy: np.ndarray, config: BEVConfig) -> BEVOccupancy:
    """Rasterise ``(N, 2)`` points into a BEV occupancy (count) grid."""
    pts = np.asarray(points_xy, dtype=float)
    size = config.size
    grid = np.zeros((size, size), dtype=float)
    if pts.size == 0:
        return BEVOccupancy(grid, config)

    res = config.resolution_m
    cols = np.round(config.center + pts[:, 0] / res).astype(int)
    rows = np.round(config.center + pts[:, 1] / res).astype(int)
    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    np.add.at(grid, (rows[valid], cols[valid]), 1.0)
    return BEVOccupancy(grid, config)
