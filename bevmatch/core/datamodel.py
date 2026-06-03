"""Core data model for BEVMatch (architecture.md §5, §8).

The v0.1 MVP works on a ground-robot assumption, so the pose lives in SE2
(x, y, yaw). The model keeps the §5 vocabulary — Scene / Observation /
Candidate / Alignment Hypothesis / Change Hypothesis — so later versions can
extend to SE3 and richer representations without renaming the entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Pose2D:
    """A rigid SE2 transform / pose: rotate by ``yaw`` then translate by (x, y)."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def rotation(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s], [s, c]], dtype=float)

    def transform(self, points_xy: np.ndarray) -> np.ndarray:
        """Apply the pose to an ``(N, 2)`` array: ``R @ p + t``."""
        pts = np.asarray(points_xy, dtype=float)
        if pts.size == 0:
            return pts.reshape(-1, 2)
        return pts @ self.rotation().T + np.array([self.x, self.y])

    def inverse(self) -> "Pose2D":
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        # R^-1 @ (-t)
        ix = -(c * self.x + s * self.y)
        iy = -(-s * self.x + c * self.y)
        return Pose2D(ix, iy, _wrap_angle(-self.yaw))

    def compose(self, other: "Pose2D") -> "Pose2D":
        """Return ``self ∘ other`` (apply ``other`` first, then ``self``)."""
        R = self.rotation()
        t = R @ np.array([other.x, other.y]) + np.array([self.x, self.y])
        return Pose2D(float(t[0]), float(t[1]), _wrap_angle(self.yaw + other.yaw))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


def _wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Observation:
    """A single-modality (near-)raw observation belonging to a Scene (§5.3)."""

    modality: str
    points: np.ndarray  # (N, 2) ground-projected points for the v0.1 LiDAR-BEV path
    metadata: dict[str, Any] = field(default_factory=dict)

    def xy(self) -> np.ndarray:
        pts = np.asarray(self.points, dtype=float)
        return pts[:, :2] if pts.ndim == 2 and pts.shape[1] >= 2 else pts.reshape(-1, 2)


@dataclass
class Scene:
    """The minimal unit of comparison (§5.2): one time, one robot, one rig."""

    scene_id: str
    observations: dict[str, Observation] = field(default_factory=dict)
    timestamp: float | None = None
    pose: Pose2D | None = None  # approximate / ground-truth pose if known
    place_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def primary(self) -> Observation:
        """Return the primary observation (first inserted) for the v0.1 path."""
        if not self.observations:
            raise ValueError(f"Scene {self.scene_id} has no observations")
        return next(iter(self.observations.values()))


@dataclass
class Candidate:
    """A retrieval candidate to be compared against the query (§5.5)."""

    scene_id: str
    score: float
    descriptor_type: str
    reason: str = ""
    place_id: str | None = None
    temporal_gap: float | None = None
    expected_yaw: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "place_id": self.place_id,
            "score": round(float(self.score), 6),
            "descriptor_type": self.descriptor_type,
            "reason": self.reason,
            "temporal_gap": self.temporal_gap,
            "expected_yaw": self.expected_yaw,
        }


@dataclass
class AlignmentHypothesis:
    """A transform that makes query and candidate comparable (§5.6, §10.4).

    ``relative_pose`` maps query-frame points into the candidate (historical)
    frame: ``p_hist = relative_pose.transform(p_query)``.
    """

    relative_pose: Pose2D
    overlap_ratio: float
    inlier_ratio: float
    success: bool
    score: float = 0.0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_pose": self.relative_pose.to_dict(),
            "overlap_ratio": round(float(self.overlap_ratio), 4),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "score": round(float(self.score), 4),
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ChangeHypothesis:
    """A candidate change derived from an aligned comparison (§5.7, §11.4)."""

    category: str  # "added" | "removed" (v0.1 geometry-level)
    centroid_xy: tuple[float, float]  # in the historical frame, metres
    area_m2: float
    num_cells: int
    confidence: float
    bbox_xy: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "centroid_xy": [round(float(c), 3) for c in self.centroid_xy],
            "area_m2": round(float(self.area_m2), 3),
            "num_cells": int(self.num_cells),
            "confidence": round(float(self.confidence), 4),
            "bbox_xy": [round(float(b), 3) for b in self.bbox_xy] if self.bbox_xy else None,
        }
