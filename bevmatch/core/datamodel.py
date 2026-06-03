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
class PoseSE3:
    """A rigid SE3 pose with ZYX (yaw-pitch-roll) Euler rotation (§10.3)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def rotation(self) -> np.ndarray:
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def transform(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=float)
        if pts.size == 0:
            return pts.reshape(-1, 3)
        return pts @ self.rotation().T + self.translation()

    def inverse(self) -> "PoseSE3":
        R = self.rotation()
        t = -R.T @ self.translation()
        return PoseSE3.from_matrix(R.T, t)

    def project_se2(self) -> Pose2D:
        return Pose2D(self.x, self.y, _wrap_angle(self.yaw))

    @staticmethod
    def from_matrix(R: np.ndarray, t: np.ndarray) -> "PoseSE3":
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return PoseSE3(float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw)

    @staticmethod
    def from_se2(pose: Pose2D) -> "PoseSE3":
        return PoseSE3(pose.x, pose.y, 0.0, 0.0, 0.0, pose.yaw)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw,
        }


@dataclass
class Observation:
    """A single-modality (near-)raw observation belonging to a Scene (§5.3)."""

    modality: str
    points: np.ndarray  # (N, 2) ground-projected points for the v0.1 LiDAR-BEV path
    metadata: dict[str, Any] = field(default_factory=dict)

    def xy(self) -> np.ndarray:
        pts = np.asarray(self.points, dtype=float)
        return pts[:, :2] if pts.ndim == 2 and pts.shape[1] >= 2 else pts.reshape(-1, 2)

    def xyz(self) -> np.ndarray:
        """Return ``(N, 3)`` points, padding z=0 when the data is 2D."""
        pts = np.asarray(self.points, dtype=float)
        if pts.ndim != 2:
            pts = pts.reshape(-1, pts.shape[-1] if pts.ndim else 1)
        if pts.shape[1] >= 3:
            return pts[:, :3]
        return np.hstack([pts[:, :2], np.zeros((len(pts), 1))])


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
    failure_class: str | None = None  # §10.5 classified failure mode
    num_correspondences: int = 0
    rmse_m: float = 0.0  # residual RMSE over inlier correspondences
    comparable_area_ratio: float = 0.0
    degeneracy: dict[str, Any] = field(default_factory=dict)  # observability flags
    pose_se3: dict[str, float] | None = None  # full 6-DoF estimate if available

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_pose": self.relative_pose.to_dict(),
            "pose_se3": self.pose_se3,
            "overlap_ratio": round(float(self.overlap_ratio), 4),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "comparable_area_ratio": round(float(self.comparable_area_ratio), 4),
            "num_correspondences": int(self.num_correspondences),
            "rmse_m": round(float(self.rmse_m), 4),
            "score": round(float(self.score), 4),
            "success": self.success,
            "failure_reason": self.failure_reason,
            "failure_class": self.failure_class,
            "degeneracy": self.degeneracy,
        }


@dataclass
class ChangeHypothesis:
    """A candidate change derived from an aligned comparison (§5.7, §11.4).

    ``category`` is "added" / "removed" for actionable geometry changes, or
    "dynamic" for transient detections filtered by temporal persistence (§11.3).
    ``persistence`` is the fraction of observations supporting the change.
    """

    category: str
    centroid_xy: tuple[float, float]  # in the historical frame, metres
    area_m2: float
    num_cells: int
    confidence: float
    bbox_xy: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax)
    persistence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return self.category in ("added", "removed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "centroid_xy": [round(float(c), 3) for c in self.centroid_xy],
            "area_m2": round(float(self.area_m2), 3),
            "num_cells": int(self.num_cells),
            "confidence": round(float(self.confidence), 4),
            "persistence": round(float(self.persistence), 4),
            "bbox_xy": [round(float(b), 3) for b in self.bbox_xy] if self.bbox_xy else None,
            "evidence": self.evidence,
        }
