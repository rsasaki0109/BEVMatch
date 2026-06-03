"""Real-data loaders (§14): turn real point clouds into BEVMatch Scenes.

Pure-numpy helpers (voxel downsampling, ground removal, scene construction) plus
file readers for common formats. Heavy/optional readers (open3d for PCD, laspy
for LAS) are imported lazily so the core stays numpy-only. Dense clouds should be
voxel-downsampled before retrieval/alignment (the pipeline targets ~1-5k points).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bevmatch.core.datamodel import Observation, Pose2D, Scene


def voxel_downsample(points: np.ndarray, voxel: float = 0.7) -> np.ndarray:
    """Keep one representative point per voxel (grid) cell. Works for 2D or 3D."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts
    keys = np.round(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def remove_ground(points_xyz: np.ndarray, margin_m: float = 0.6, top_m: float = 25.0,
                  ground_percentile: float = 3.0) -> np.ndarray:
    """Drop ground/near-ground returns, keeping a structure band above ground.

    Ground height is estimated as a low z-percentile (robust to outliers).
    Returns the surviving points (same dimensionality as input).
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[1] < 3:
        return pts
    g = np.percentile(pts[:, 2], ground_percentile)
    keep = (pts[:, 2] > g + margin_m) & (pts[:, 2] < g + top_m)
    return pts[keep]


def scene_from_points(
    points: np.ndarray,
    scene_id: str,
    modality: str = "lidar",
    place_id: str | None = None,
    pose: Pose2D | None = None,
    voxel: float | None = 0.7,
    drop_ground: bool = False,
    recenter: bool = False,
) -> Scene:
    """Build a Scene from a real point cloud (optionally downsampled / ground-removed)."""
    pts = np.asarray(points, dtype=float)
    if drop_ground and pts.shape[1] >= 3:
        pts = remove_ground(pts)
    xy = pts[:, :2]
    if recenter and len(xy):
        xy = xy - xy.mean(axis=0)
    if voxel:
        xy = voxel_downsample(xy, voxel)
    return Scene(
        scene_id=scene_id, place_id=place_id, pose=pose,
        observations={modality: Observation(modality, xy)},
    )


# --- file readers (optional dependencies, lazily imported) ---
def load_kitti_bin(path: str | Path) -> np.ndarray:
    """Load a KITTI Velodyne ``.bin`` scan as an ``(N, 4)`` array (x, y, z, intensity)."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def load_pcd(path: str | Path) -> np.ndarray:
    """Load a ``.pcd`` file as ``(N, 3)`` points (requires open3d)."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_pcd requires open3d (pip install open3d)") from exc
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def load_las_tile(
    path: str | Path,
    center_xy: tuple[float, float] | None = None,
    half_m: float = 40.0,
    chunk: int = 5_000_000,
) -> np.ndarray:
    """Stream a ``.las``/``.laz`` file and return points within a tile (requires laspy).

    Without ``center_xy`` the whole cloud is returned (may be large).
    """
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_las_tile requires laspy (pip install laspy)") from exc
    out = []
    with laspy.open(str(path)) as f:
        for chunk_pts in f.chunk_iterator(chunk):
            x = np.asarray(chunk_pts.x); y = np.asarray(chunk_pts.y); z = np.asarray(chunk_pts.z)
            if center_xy is None:
                out.append(np.stack([x, y, z], axis=1))
                continue
            cx, cy = center_xy
            m = (x > cx - half_m) & (x < cx + half_m) & (y > cy - half_m) & (y < cy + half_m)
            if m.any():
                out.append(np.stack([x[m], y[m], z[m]], axis=1))
    return np.vstack(out) if out else np.zeros((0, 3))
