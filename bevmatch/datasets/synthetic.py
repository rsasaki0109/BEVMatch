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


@dataclass
class MapValidationCase:
    """A point cloud map + current observation frames, with ground-truth issues."""

    pcd_map: object  # bevmatch.maps.PointCloudMap (typed lazily to avoid import cycle)
    occ_map: object  # bevmatch.maps.OccupancyMap
    vmap: object  # bevmatch.maps.VectorMap
    current_frames: list[Scene]  # current observation burst (current/local frame)
    current_map_frame: Scene  # frame 0 expressed in the map frame (already localized)
    gt_relative_pose: Pose2D
    gt_issues: list[tuple[str, np.ndarray]]  # (issue_type, location in map frame)


def make_map_validation_case(
    seed: int = 1,
    changed: bool = True,
    n_frames: int = 3,
    n_dynamic: int = 0,
) -> MapValidationCase:
    """Build a map + current observations where the world may have changed.

    ``n_dynamic`` adds moving objects per frame (filtered by persistence); the
    default 0 keeps the map-validation story deterministic. Dynamic filtering is
    demonstrated separately by the change-detection case.
    """
    from bevmatch.maps.datamodel import MapElement, OccupancyMap, PointCloudMap, VectorMap

    rng = np.random.default_rng(seed)
    map_world = _place_centers(rng, int(rng.integers(18, 24)))

    def _far_center(avoid: np.ndarray, min_dist: float = 5.0) -> np.ndarray:
        while True:
            c = rng.uniform(-15.0, 15.0, size=2)
            if np.hypot(*c) < 6.0:
                continue
            if len(avoid) and np.min(np.linalg.norm(avoid - c, axis=1)) < min_dist:
                continue
            return c

    gt_issues: list[tuple[str, np.ndarray]] = []
    removed_center = None
    added_center = None
    if changed:
        # Remove the most isolated object so its disappearance is unambiguous
        # (no close neighbour to keep the vector element / region "supported").
        d = np.linalg.norm(map_world[:, None, :] - map_world[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        ridx = int(np.argmax(d.min(axis=1)))
        removed_center = map_world[ridx].copy()
        keep = np.ones(len(map_world), dtype=bool)
        keep[ridx] = False
        kept = map_world[keep]
        added_center = _far_center(map_world, min_dist=5.0)
        gt_issues = [
            ("new_static_obstacle", added_center),
            ("missing_static_structure", removed_center),
        ]
    else:
        kept = map_world

    pcd_map = PointCloudMap(points=_points_from_centers(rng, map_world), map_id="map_a", version="v1")
    occ_map = OccupancyMap.from_points(pcd_map.xy(), map_id="map_a", version="v1")

    static_now = kept if added_center is None else np.vstack([kept, added_center[None, :]])
    t_wq = Pose2D(x=float(rng.uniform(-1.5, 1.5)), y=float(rng.uniform(-1.5, 1.5)),
                  yaw=float(np.deg2rad(rng.uniform(-15, 15))))

    static_clear = np.vstack([map_world, added_center[None, :]]) if added_center is not None else map_world

    def _dyn() -> np.ndarray:
        picks: list[np.ndarray] = []
        while len(picks) < n_dynamic:
            c = rng.uniform(-15, 15, size=2)
            if np.hypot(*c) < 6:
                continue
            if np.min(np.linalg.norm(static_clear - c, axis=1)) < 4.0:
                continue
            picks.append(c)
        return np.array(picks).reshape(-1, 2)

    current_frames: list[Scene] = []
    frame0_map_pts = None
    for f in range(n_frames):
        world_pts = _points_from_centers(rng, np.vstack([static_now, _dyn()]))
        if f == 0:
            frame0_map_pts = world_pts
        local = t_wq.inverse().transform(world_pts)
        current_frames.append(
            Scene(scene_id=f"cur_f{f}", place_id="map_a", timestamp=float(f),
                  pose=t_wq, observations={MODALITY: Observation(MODALITY, local)},
                  metadata={"role": "current"}),
        )

    current_map_frame = Scene(
        scene_id="cur_map_frame", place_id="map_a", timestamp=0.0, pose=Pose2D(),
        observations={MODALITY: Observation(MODALITY, frame0_map_pts)}, metadata={"role": "current"},
    )

    # Vector map: a supported element anchored on a real object, and (if changed)
    # one over the removed area that the current observation no longer supports.
    base = kept[0]
    elements = [MapElement("E1", "lane_boundary", np.array([base + [-1.0, 0.0], base + [1.0, 0.0]]))]
    if removed_center is not None:
        elements.append(
            MapElement("E2", "stop_line", np.array([removed_center - [0.5, 0], removed_center + [0.5, 0]]))
        )
        gt_issues.append(("map_element_unobserved", removed_center))
    vmap = VectorMap(elements=elements, map_id="map_a", version="v1")

    return MapValidationCase(
        pcd_map=pcd_map,
        occ_map=occ_map,
        vmap=vmap,
        current_frames=current_frames,
        current_map_frame=current_map_frame,
        gt_relative_pose=t_wq,
        gt_issues=gt_issues,
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
