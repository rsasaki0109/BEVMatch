"""Synthetic same-place toy benchmark (§14.3 BEVMatch Mini Benchmark).

A small route of distinct "places", each a cluster of object-like point blobs.
One place is revisited as the query, observed from a different SE2 pose, with a
few objects added and removed. This exercises the full pipeline end-to-end with
no dataset download (Principle 5: offline-first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.core.datamodel import Observation, Pose2D, Scene

MODALITY = "lidar_bev_points"


def _object_blob(rng: np.random.Generator, center: np.ndarray, n_points: int = 12, radius: float = 0.35) -> np.ndarray:
    offs = rng.normal(scale=radius, size=(n_points, 2))
    return center[None, :] + offs


def _place_centers(rng: np.random.Generator, n_objects: int, extent: float = 20.0, keepout: float = 3.0) -> np.ndarray:
    centers = []
    while len(centers) < n_objects:
        c = rng.uniform(-extent, extent, size=2)
        if np.hypot(*c) < keepout:
            continue  # keep a clear area near the sensor origin
        centers.append(c)
    return np.array(centers)


def _points_from_centers(rng: np.random.Generator, centers: np.ndarray) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros((0, 2))
    return np.vstack([_object_blob(rng, c) for c in centers])


@dataclass
class SyntheticSamePlace:
    """A generated dataset: historical route + one query revisit with changes."""

    historical: list[Scene]
    query: Scene
    gt_place_id: str
    gt_relative_pose: Pose2D  # maps query frame -> historical frame
    added_centers: np.ndarray  # in the historical frame
    removed_centers: np.ndarray  # in the historical frame


@dataclass
class RouteQuery:
    """One query revisit with its ground truth, for retrieval evaluation."""

    scene: Scene
    gt_place_id: str
    gt_relative_pose: Pose2D


@dataclass
class SyntheticRoute:
    """A historical route plus many query revisits (retrieval benchmark, §14.3)."""

    historical: list[Scene]
    queries: list[RouteQuery]


def _build_query_scene(
    rng: np.random.Generator,
    world_centers: np.ndarray,
    place_id: str,
    scene_id: str,
    timestamp: float,
    n_added: int,
    n_removed: int,
    yaw_range_deg: float = 40.0,
    trans_range_m: float = 3.0,
) -> tuple[Scene, Pose2D, np.ndarray, np.ndarray]:
    keep_mask = np.ones(len(world_centers), dtype=bool)
    if n_removed > 0:
        removed_idx = rng.choice(len(world_centers), size=min(n_removed, len(world_centers)), replace=False)
        keep_mask[removed_idx] = False
        removed_centers = world_centers[removed_idx]
    else:
        removed_centers = np.zeros((0, 2))
    kept_centers = world_centers[keep_mask]

    added_centers = _place_centers(rng, n_added, extent=18.0, keepout=5.0) if n_added > 0 else np.zeros((0, 2))

    query_world_centers = np.vstack([kept_centers, added_centers]) if len(added_centers) else kept_centers
    query_world_pts = _points_from_centers(rng, query_world_centers)

    t_wq = Pose2D(
        x=float(rng.uniform(-trans_range_m, trans_range_m)),
        y=float(rng.uniform(-trans_range_m, trans_range_m)),
        yaw=float(np.deg2rad(rng.uniform(-yaw_range_deg, yaw_range_deg))),
    )
    query_local_pts = t_wq.inverse().transform(query_world_pts)

    scene = Scene(
        scene_id=scene_id,
        place_id=place_id,
        timestamp=timestamp,
        pose=t_wq,
        observations={MODALITY: Observation(MODALITY, query_local_pts)},
        metadata={"role": "query"},
    )
    return scene, t_wq, added_centers, removed_centers


def make_synthetic_route(
    seed: int = 0,
    n_places: int = 12,
    n_queries: int = 24,
    n_added: int = 2,
    n_removed: int = 2,
) -> SyntheticRoute:
    """Generate a historical route and many query revisits for evaluation."""
    rng = np.random.default_rng(seed)

    historical: list[Scene] = []
    place_centers: list[np.ndarray] = []
    for p in range(n_places):
        n_obj = int(rng.integers(18, 26))
        centers = _place_centers(rng, n_obj)
        place_centers.append(centers)
        historical.append(
            Scene(
                scene_id=f"hist_{p}",
                place_id=f"place_{p}",
                timestamp=float(p),
                pose=Pose2D(),
                observations={MODALITY: Observation(MODALITY, _points_from_centers(rng, centers))},
                metadata={"role": "historical"},
            )
        )

    queries: list[RouteQuery] = []
    for q in range(n_queries):
        p = int(rng.integers(0, n_places))
        scene, t_wq, _, _ = _build_query_scene(
            rng,
            place_centers[p],
            place_id=f"place_{p}",
            scene_id=f"query_{q}",
            timestamp=float(n_places + q),
            n_added=n_added,
            n_removed=n_removed,
        )
        queries.append(RouteQuery(scene=scene, gt_place_id=f"place_{p}", gt_relative_pose=t_wq))

    return SyntheticRoute(historical=historical, queries=queries)


def make_synthetic_same_place(
    seed: int = 7,
    n_places: int = 6,
    revisit_index: int = 2,
    n_added: int = 2,
    n_removed: int = 2,
) -> SyntheticSamePlace:
    rng = np.random.default_rng(seed)

    historical: list[Scene] = []
    place_centers: list[np.ndarray] = []
    for p in range(n_places):
        n_obj = int(rng.integers(18, 26))
        centers = _place_centers(rng, n_obj)
        place_centers.append(centers)
        pts = _points_from_centers(rng, centers)
        historical.append(
            Scene(
                scene_id=f"hist_{p}",
                place_id=f"place_{p}",
                timestamp=float(p),
                pose=Pose2D(),
                observations={MODALITY: Observation(MODALITY, pts)},
                metadata={"role": "historical"},
            )
        )

    # Build the query as a revisit of one place, from a different SE2 pose.
    world_centers = place_centers[revisit_index]
    keep_mask = np.ones(len(world_centers), dtype=bool)
    removed_idx = rng.choice(len(world_centers), size=n_removed, replace=False)
    keep_mask[removed_idx] = False
    removed_centers = world_centers[removed_idx]
    kept_centers = world_centers[keep_mask]

    added_centers = _place_centers(rng, n_added, extent=18.0, keepout=5.0)

    query_world_centers = np.vstack([kept_centers, added_centers])
    query_world_pts = _points_from_centers(rng, query_world_centers)

    # T_wq maps query-local points into the world/historical frame.
    t_wq = Pose2D(
        x=float(rng.uniform(-3.0, 3.0)),
        y=float(rng.uniform(-3.0, 3.0)),
        yaw=float(np.deg2rad(rng.uniform(-40.0, 40.0))),
    )
    query_local_pts = t_wq.inverse().transform(query_world_pts)

    query = Scene(
        scene_id="query_0",
        place_id=f"place_{revisit_index}",
        timestamp=float(n_places + 10),
        pose=t_wq,
        observations={MODALITY: Observation(MODALITY, query_local_pts)},
        metadata={"role": "query"},
    )

    return SyntheticSamePlace(
        historical=historical,
        query=query,
        gt_place_id=f"place_{revisit_index}",
        gt_relative_pose=t_wq,
        added_centers=added_centers,
        removed_centers=removed_centers,
    )
