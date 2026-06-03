"""Comparable-region & occlusion estimation (§11.2, §2.2 occlusion evidence).

A difference is only meaningful where *both* scenes actually observed the world.
Each scene's observability is estimated by polar ray-casting from its sensor
origin: along each azimuth, everything past the first return is in shadow
(occluded / unknown). The comparable region is the intersection of the two
observed masks — change detection is restricted to it so that structure hidden
behind an occluder is never reported as "removed".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.representations.bev import BEVConfig, points_to_bev


@dataclass
class Observability:
    observed: np.ndarray  # bool grid: within range AND not in shadow
    occluded: np.ndarray  # bool grid: within range BUT behind the first return
    occupied: np.ndarray  # bool grid: occupied cells


def _suppress_thin_occluders(hit: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only wide occluders: zero-out shadows from runs shorter than ``min_run``.

    A small isolated return (one blob) casts a negligible shadow you would see
    around; only an angularly wide, contiguous occluder (a wall) should shadow
    what is behind it. Runs are measured circularly.
    """
    finite = np.isfinite(hit)
    if finite.all() or not finite.any():
        return hit
    n = len(hit)
    start = int(np.argmin(finite))  # rotate so index 0 is a gap -> no wraparound run
    order = np.roll(np.arange(n), -start)
    f = finite[order]
    out = hit.copy()
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if (j - i) < min_run:
                out[order[i:j]] = np.inf
            i = j
        else:
            i += 1
    return out


def _cell_polar(bev: BEVConfig, origin_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell range and azimuth relative to ``origin_xy`` (in the BEV frame)."""
    size = bev.size
    rows, cols = np.mgrid[0:size, 0:size]
    x = (cols - bev.center) * bev.resolution_m - origin_xy[0]
    y = (rows - bev.center) * bev.resolution_m - origin_xy[1]
    return np.hypot(x, y), np.arctan2(y, x)


def observability(
    points_xy: np.ndarray,
    origin_xy: np.ndarray,
    bev: BEVConfig,
    n_azimuth: int = 360,
    occlusion_margin_m: float = 1.0,
    occ_threshold: float = 0.5,
    azimuth_spread: int = 2,
    occluder_min_run: int = 10,
) -> Observability:
    """Estimate observed / occluded / occupied masks for one scene.

    ``azimuth_spread`` widens each return's shadow by a few azimuth bins so a
    sparse but contiguous occluder (whose points don't fall in every bin) still
    casts a continuous shadow rather than leaking through the gaps.
    ``occluder_min_run`` requires an occluder to span that many azimuth bins
    before it casts a shadow, so isolated small objects don't hide what is behind
    them while walls / large structures do.
    """
    pts = np.asarray(points_xy, dtype=float)[:, :2]
    origin = np.asarray(origin_xy, dtype=float)
    occupied = points_to_bev(pts, bev).occupied(occ_threshold)

    # First-return range per azimuth bin (the occlusion boundary).
    rel = pts - origin
    pr = np.hypot(rel[:, 0], rel[:, 1])
    pa = np.arctan2(rel[:, 1], rel[:, 0])
    pbin = np.clip(((pa + np.pi) / (2 * np.pi) * n_azimuth).astype(int), 0, n_azimuth - 1)
    hit = np.full(n_azimuth, np.inf)
    np.minimum.at(hit, pbin, pr)
    # Each return shadows a thin angular wedge: fill bin gaps with neighbour mins.
    for s in range(1, azimuth_spread + 1):
        hit = np.minimum(hit, np.minimum(np.roll(hit, s), np.roll(hit, -s)))
    # Only wide, contiguous occluders cast shadows.
    hit = _suppress_thin_occluders(hit, occluder_min_run)

    cell_r, cell_a = _cell_polar(bev, origin)
    cbin = np.clip(((cell_a + np.pi) / (2 * np.pi) * n_azimuth).astype(int), 0, n_azimuth - 1)
    boundary = hit[cbin] + occlusion_margin_m

    in_range = cell_r <= bev.range_m
    observed = in_range & (cell_r <= boundary)
    occluded = in_range & (cell_r > boundary)
    return Observability(observed=observed, occluded=occluded, occupied=occupied)


@dataclass
class ComparableRegion:
    comparable: np.ndarray  # observed in both scenes
    occluded: np.ndarray  # observed in one but occluded in the other
    comparable_ratio: float  # comparable / in-range union
    occluded_ratio: float


def comparable_region(query_obs: Observability, ref_obs: Observability) -> ComparableRegion:
    comparable = query_obs.observed & ref_obs.observed
    occluded = (query_obs.occluded & ref_obs.observed) | (ref_obs.occluded & query_obs.observed)
    union = query_obs.observed | ref_obs.observed
    n_union = max(1, int(np.count_nonzero(union)))
    return ComparableRegion(
        comparable=comparable,
        occluded=occluded,
        comparable_ratio=int(np.count_nonzero(comparable)) / n_union,
        occluded_ratio=int(np.count_nonzero(occluded)) / n_union,
    )
