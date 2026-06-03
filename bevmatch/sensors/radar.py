"""Radar modality: project radar targets to the shared BEV (§3.1 MulRan, §1.3).

Radar returns are sparser and noisier than LiDAR but live on the same ground
plane, so they project to the same BEV occupancy / Scan-Context representation —
LiDAR retrieval and alignment plugins work on radar unchanged (Principle 2).
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Observation, Scene

MODALITY = "radar_points"


def radar_to_points(targets_xy: np.ndarray) -> np.ndarray:
    """Return radar target points as ground-plane (N, 2) coordinates."""
    pts = np.asarray(targets_xy, dtype=float)
    return pts[:, :2] if pts.ndim == 2 and pts.shape[1] >= 2 else pts.reshape(-1, 2)


def radar_scene(scene_id: str, targets_xy: np.ndarray, place_id: str | None = None, **kw) -> Scene:
    """Build a Scene from radar targets (projected to the BEV ground plane)."""
    obs = Observation(MODALITY, points=radar_to_points(targets_xy))
    return Scene(scene_id=scene_id, observations={MODALITY: obs}, place_id=place_id, **kw)
