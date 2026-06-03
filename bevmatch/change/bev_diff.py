"""Geometry-level BEV occupancy diff (§11.2, §11.5).

After alignment, the query is transformed into the reference frame and both are
rasterised. Within the *comparable region* (observed by both scenes, §11.2),
cells occupied in one scene but free in the other become ``added`` / ``removed``
change hypotheses. Change detection is alignment-gated (Principle 3): the caller
only runs this when alignment succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.change.comparable import comparable_region, observability
from bevmatch.core.datamodel import ChangeHypothesis, Pose2D
from bevmatch.grid_utils import connected_components as _connected_components
from bevmatch.grid_utils import dilate as _dilate
from bevmatch.representations.bev import BEVConfig, BEVOccupancy, points_to_bev


@dataclass(frozen=True)
class ChangeConfig:
    bev: BEVConfig = field(default_factory=BEVConfig)
    occ_threshold: float = 0.5
    min_cells: int = 2  # discard specks smaller than this (noise floor)
    range_margin_m: float = 1.0  # shrink comparable disk to avoid edge artefacts
    suppress_passes: int = 1  # tolerance (in cells) to structure in the other scene
    # Occlusion gating (§11.2) needs dense/structured observations; opt in when
    # the scene has real occluders. On sparse point scenes leave it off.
    use_occlusion: bool = False


@dataclass
class ChangeResult:
    """Change hypotheses plus the comparable/occlusion evidence behind them."""

    changes: list[ChangeHypothesis] = field(default_factory=list)
    comparable_ratio: float = 1.0
    occluded_ratio: float = 0.0

    def added(self) -> list[ChangeHypothesis]:
        return [c for c in self.changes if c.category == "added"]

    def removed(self) -> list[ChangeHypothesis]:
        return [c for c in self.changes if c.category == "removed"]


def _comparable_disk(config: BEVConfig, margin_m: float) -> np.ndarray:
    size = config.size
    yy, xx = np.mgrid[0:size, 0:size]
    res = config.resolution_m
    x = (xx - config.center) * res
    y = (yy - config.center) * res
    return np.hypot(x, y) <= (config.range_m - margin_m)


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


def detect_changes_detailed(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    relative_pose: Pose2D,
    config: ChangeConfig | None = None,
    align_overlap: float = 1.0,
) -> ChangeResult:
    """Detect added/removed regions plus comparable/occlusion evidence."""
    cfg = config or ChangeConfig()
    bev = cfg.bev

    q_xy = np.asarray(query_xy, dtype=float)[:, :2]
    q_in_ref = relative_pose.transform(q_xy)
    r_xy = np.asarray(ref_xy, dtype=float)[:, :2]
    q_grid = points_to_bev(q_in_ref, bev)
    r_grid = points_to_bev(r_xy, bev)

    q_occ = q_grid.occupied(cfg.occ_threshold)
    r_occ = r_grid.occupied(cfg.occ_threshold)

    if cfg.use_occlusion:
        # Observability is ray-cast from each sensor's origin (the query origin
        # lands at the alignment translation once expressed in the ref frame).
        q_origin = relative_pose.transform(np.zeros((1, 2)))[0]
        q_obs = observability(q_in_ref, q_origin, bev, occ_threshold=cfg.occ_threshold)
        r_obs = observability(r_xy, np.zeros(2), bev, occ_threshold=cfg.occ_threshold)
        region = comparable_region(q_obs, r_obs)
        comparable = region.comparable
        comparable_ratio, occluded_ratio = region.comparable_ratio, region.occluded_ratio
    else:
        comparable = _comparable_disk(bev, cfg.range_margin_m)
        comparable_ratio, occluded_ratio = 1.0, 0.0

    # "added" = occupied in the query but not near any reference structure,
    # within the region both scenes actually observed.
    added_mask = q_occ & ~_dilate(r_occ, cfg.suppress_passes) & comparable
    removed_mask = r_occ & ~_dilate(q_occ, cfg.suppress_passes) & comparable

    cell_area = bev.resolution_m ** 2
    conf_scale = float(np.clip(align_overlap, 0.0, 1.0))

    changes = _components_to_changes(
        _connected_components(added_mask, cfg.min_cells), "added", q_grid, cell_area, conf_scale
    )
    changes += _components_to_changes(
        _connected_components(removed_mask, cfg.min_cells), "removed", r_grid, cell_area, conf_scale
    )
    changes.sort(key=lambda c: c.area_m2, reverse=True)
    return ChangeResult(
        changes=changes,
        comparable_ratio=comparable_ratio,
        occluded_ratio=occluded_ratio,
    )


def detect_changes(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    relative_pose: Pose2D,
    config: ChangeConfig | None = None,
    align_overlap: float = 1.0,
) -> list[ChangeHypothesis]:
    """Convenience wrapper returning just the change list (see ``*_detailed``)."""
    return detect_changes_detailed(query_xy, ref_xy, relative_pose, config, align_overlap).changes
