"""Geometry-level BEV occupancy diff (§11.2, §11.5).

After alignment, the query is transformed into the reference frame and both are
rasterised. Within the *comparable region* (the observed disk), cells occupied
in one scene but free in the other become ``added`` / ``removed`` change
hypotheses. Change detection is alignment-gated (Principle 3): the caller only
runs this when alignment succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.core.datamodel import ChangeHypothesis, Pose2D
from bevmatch.representations.bev import BEVConfig, BEVOccupancy, points_to_bev


@dataclass(frozen=True)
class ChangeConfig:
    bev: BEVConfig = field(default_factory=BEVConfig)
    occ_threshold: float = 0.5
    min_cells: int = 2  # discard specks smaller than this (noise floor)
    range_margin_m: float = 1.0  # shrink comparable disk to avoid edge artefacts
    suppress_passes: int = 1  # tolerance (in cells) to structure in the other scene


def _dilate(mask: np.ndarray, passes: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(passes):
        grown = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= np.roll(np.roll(out, dr, axis=0), dc, axis=1)
        out = grown
    return out


def _comparable_disk(config: BEVConfig, margin_m: float) -> np.ndarray:
    size = config.size
    yy, xx = np.mgrid[0:size, 0:size]
    res = config.resolution_m
    x = (xx - config.center) * res
    y = (yy - config.center) * res
    return np.hypot(x, y) <= (config.range_m - margin_m)


def _connected_components(mask: np.ndarray, min_cells: int) -> list[np.ndarray]:
    """8-connected components as arrays of (row, col), filtered by size."""
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    rows, cols = np.nonzero(mask)
    for r0, c0 in zip(rows, cols):
        if visited[r0, c0]:
            continue
        stack = [(r0, c0)]
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


def _components_to_changes(
    comps: list[np.ndarray],
    category: str,
    bev: BEVOccupancy,
    cell_area: float,
    confidence_scale: float,
) -> list[ChangeHypothesis]:
    changes: list[ChangeHypothesis] = []
    for cells in comps:
        xs, ys = [], []
        for r, c in cells:
            x, y = bev.cell_to_xy(int(r), int(c))
            xs.append(x)
            ys.append(y)
        xs_a, ys_a = np.array(xs), np.array(ys)
        n = len(cells)
        conf = confidence_scale * (n / (n + 4.0))
        changes.append(
            ChangeHypothesis(
                category=category,
                centroid_xy=(float(xs_a.mean()), float(ys_a.mean())),
                area_m2=float(n * cell_area),
                num_cells=int(n),
                confidence=float(np.clip(conf, 0.0, 1.0)),
                bbox_xy=(
                    float(xs_a.min()),
                    float(ys_a.min()),
                    float(xs_a.max()),
                    float(ys_a.max()),
                ),
            )
        )
    return changes


def detect_changes(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    relative_pose: Pose2D,
    config: ChangeConfig | None = None,
    align_overlap: float = 1.0,
) -> list[ChangeHypothesis]:
    """Detect added/removed regions between an aligned query/reference pair."""
    cfg = config or ChangeConfig()
    bev = cfg.bev

    q_in_ref = relative_pose.transform(np.asarray(query_xy, dtype=float)[:, :2])
    q_grid = points_to_bev(q_in_ref, bev)
    r_grid = points_to_bev(np.asarray(ref_xy, dtype=float)[:, :2], bev)

    q_occ = q_grid.occupied(cfg.occ_threshold)
    r_occ = r_grid.occupied(cfg.occ_threshold)
    observed = _comparable_disk(bev, cfg.range_margin_m)

    # "added" = occupied in the query but not near any reference structure.
    added_mask = q_occ & ~_dilate(r_occ, cfg.suppress_passes) & observed
    removed_mask = r_occ & ~_dilate(q_occ, cfg.suppress_passes) & observed

    cell_area = bev.resolution_m ** 2
    conf_scale = float(np.clip(align_overlap, 0.0, 1.0))

    changes = _components_to_changes(
        _connected_components(added_mask, cfg.min_cells), "added", q_grid, cell_area, conf_scale
    )
    changes += _components_to_changes(
        _connected_components(removed_mask, cfg.min_cells), "removed", r_grid, cell_area, conf_scale
    )
    changes.sort(key=lambda c: c.area_m2, reverse=True)
    return changes
