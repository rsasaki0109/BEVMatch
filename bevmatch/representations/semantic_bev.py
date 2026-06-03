"""Semantic BEV representation (§5.4, §11.5 semantic-level).

A top-down grid where each observed cell carries a semantic class (the most
frequent label of the points falling in it). Enables semantic-aware comparison
on top of the geometric occupancy grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.representations.bev import BEVConfig

UNLABELLED = -1


@dataclass
class SemanticBEV:
    label_grid: np.ndarray  # (size, size) int class id, -1 where unobserved
    config: BEVConfig
    n_classes: int

    def known(self) -> np.ndarray:
        return self.label_grid >= 0

    def cell_to_xy(self, row: int, col: int) -> tuple[float, float]:
        res = self.config.resolution_m
        return float((col - self.config.center) * res), float((row - self.config.center) * res)


def points_to_semantic_bev(
    points_xy: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    config: BEVConfig | None = None,
) -> SemanticBEV:
    """Rasterise labelled points into a per-cell majority-class grid."""
    config = config or BEVConfig()
    pts = np.asarray(points_xy, dtype=float)
    lab = np.asarray(labels, dtype=int)
    size = config.size
    # votes[r, c, k] = number of class-k points in cell (r, c)
    votes = np.zeros((size, size, n_classes), dtype=np.int32)
    if pts.size:
        res = config.resolution_m
        cols = np.round(config.center + pts[:, 0] / res).astype(int)
        rows = np.round(config.center + pts[:, 1] / res).astype(int)
        valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size) & (lab >= 0) & (lab < n_classes)
        np.add.at(votes, (rows[valid], cols[valid], lab[valid]), 1)
    total = votes.sum(axis=2)
    label_grid = np.where(total > 0, votes.argmax(axis=2), UNLABELLED)
    return SemanticBEV(label_grid=label_grid, config=config, n_classes=n_classes)


def semantic_change_mask(query: SemanticBEV, reference: SemanticBEV) -> np.ndarray:
    """Cells observed in both whose class differs (semantic change candidate)."""
    both = query.known() & reference.known()
    return both & (query.label_grid != reference.label_grid)
