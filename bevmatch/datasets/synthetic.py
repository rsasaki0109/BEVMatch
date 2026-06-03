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


@dataclass
class ChangeCase:
    """A reference scene + a burst of query frames with stable and dynamic changes."""

    reference: Scene
    query_frames: list[Scene]
    gt_relative_pose: Pose2D
    added_centers: np.ndarray  # stable, actionable (world frame)
    removed_centers: np.ndarray  # stable, actionable (world frame)
    dynamic_centers_per_frame: list[np.ndarray]  # moving / transient (world frame)


@dataclass
class OcclusionCase:
    """A pair where the query has an occluder hiding objects the reference saw."""

    reference: Scene
    query: Scene
    gt_relative_pose: Pose2D
    removed_centers: np.ndarray  # truly removed -> should be reported
    occluded_centers: np.ndarray  # hidden behind the occluder -> must NOT be reported


def make_synthetic_change_case(
    seed: int = 3,
    n_frames: int = 4,
    n_added: int = 2,
    n_removed: int = 2,
    n_dynamic: int = 2,
) -> ChangeCase:
    """Reference + multi-frame query burst: stable changes vs moving dynamics."""
    rng = np.random.default_rng(seed)
    world = _place_centers(rng, int(rng.integers(18, 24)))

    removed_idx = rng.choice(len(world), size=n_removed, replace=False)
    removed_centers = world[removed_idx]
    keep = np.ones(len(world), dtype=bool)
    keep[removed_idx] = False
    kept = world[keep]

    added_centers = _place_centers(rng, n_added, extent=16.0, keepout=6.0)

    reference = Scene(
        scene_id="ref",
        place_id="place_x",
        timestamp=0.0,
        pose=Pose2D(),
        observations={MODALITY: Observation(MODALITY, _points_from_centers(rng, world))},
        metadata={"role": "historical"},
    )

    t_wq = Pose2D(x=float(rng.uniform(-2, 2)), y=float(rng.uniform(-2, 2)),
                  yaw=float(np.deg2rad(rng.uniform(-20, 20))))

    # Dynamic objects must stay clear of the static layout, else a moving object
    # landing on a removed/added spot would corrupt that stable detection.
    static_centers = np.vstack([world, added_centers])

    def _dynamic_centers() -> np.ndarray:
        picks: list[np.ndarray] = []
        while len(picks) < n_dynamic:
            c = rng.uniform(-15.0, 15.0, size=2)
            if np.hypot(*c) < 6.0:
                continue
            if len(static_centers) and np.min(np.linalg.norm(static_centers - c, axis=1)) < 4.0:
                continue
            if picks and np.min(np.linalg.norm(np.array(picks) - c, axis=1)) < 4.0:
                continue
            picks.append(c)
        return np.array(picks)

    query_frames: list[Scene] = []
    dynamic_per_frame: list[np.ndarray] = []
    for f in range(n_frames):
        dynamic = _dynamic_centers()
        dynamic_per_frame.append(dynamic)
        world_pts = _points_from_centers(rng, np.vstack([kept, added_centers, dynamic]))
        local = t_wq.inverse().transform(world_pts)
        query_frames.append(
            Scene(
                scene_id=f"query_f{f}",
                place_id="place_x",
                timestamp=float(10 + f),
                pose=t_wq,
                observations={MODALITY: Observation(MODALITY, local)},
                metadata={"role": "query", "frame": f},
            )
        )

    return ChangeCase(
        reference=reference,
        query_frames=query_frames,
        gt_relative_pose=t_wq,
        added_centers=added_centers,
        removed_centers=removed_centers,
        dynamic_centers_per_frame=dynamic_per_frame,
    )


def _wall_points(rng: np.random.Generator, x: float, y0: float, y1: float, step: float = 0.15) -> np.ndarray:
    ys = np.arange(y0, y1, step)
    xs = np.full_like(ys, x)
    pts = np.stack([xs, ys], axis=1)
    return pts + rng.normal(scale=0.05, size=pts.shape)


def make_occlusion_case(seed: int = 5) -> OcclusionCase:
    """Query gains an occluder that hides reference objects (occlusion vs removal)."""
    rng = np.random.default_rng(seed)

    # Common objects, kept clear of the occluder shadow (azimuth +/-45 deg, far).
    common = []
    while len(common) < 16:
        c = rng.uniform(-18, 18, size=2)
        r, az = np.hypot(*c), abs(np.degrees(np.arctan2(c[1], c[0])))
        if r < 4:
            continue
        if az < 45 and c[0] > 4:  # would be in the occluder shadow
            continue
        common.append(c)
    common = np.array(common)

    # Behind-wall objects: only the reference sees them.
    occluded_centers = np.array([[13.0, -1.0], [15.0, 1.2]])
    # A genuinely removed object, out of the shadow.
    removed_centers = np.array([[-10.0, 6.0]])

    ref_centers = np.vstack([common, removed_centers, occluded_centers])
    ref_pts = _points_from_centers(rng, ref_centers)

    wall = _wall_points(rng, x=4.0, y0=-3.0, y1=3.0)
    query_pts = np.vstack([_points_from_centers(rng, common), wall])

    reference = Scene(
        scene_id="occ_ref", place_id="place_occ", timestamp=0.0, pose=Pose2D(),
        observations={MODALITY: Observation(MODALITY, ref_pts)}, metadata={"role": "historical"},
    )
    query = Scene(
        scene_id="occ_query", place_id="place_occ", timestamp=1.0, pose=Pose2D(),
        observations={MODALITY: Observation(MODALITY, query_pts)}, metadata={"role": "query"},
    )
    return OcclusionCase(
        reference=reference,
        query=query,
        gt_relative_pose=Pose2D(),
        removed_centers=removed_centers,
        occluded_centers=occluded_centers,
    )


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
