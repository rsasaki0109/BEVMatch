"""Global relocalization → initial pose candidates (§17.2 Pattern A, §18.2 A).

Retrieve Top-K historical/map places, align the current observation to each, and
turn the result into ranked initial-pose candidates with covariance — exactly
what an Autoware/Nav2 localizer needs to initialise. The current robot pose in
the map frame is ``ref.pose ∘ relative_pose`` (the historical scene's map pose
composed with the estimated query→reference transform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.se2 import SE2Aligner
from bevmatch.core.datamodel import AlignmentHypothesis, Pose2D, Scene
from bevmatch.retrieval.retriever import SceneDatabase

_BIG = 1e6  # variance for unobservable z / roll / pitch in an SE2 estimate


def covariance_from_alignment(alignment: AlignmentHypothesis) -> list[float]:
    """Return a 6x6 row-major pose covariance (x, y, z, roll, pitch, yaw).

    Translation/yaw variances grow as inliers drop and residual rises; the
    out-of-plane DOF are marked effectively unobservable.
    """
    q = max(0.05, 1.0 - alignment.inlier_ratio)
    sigma_xy = 0.1 + 2.0 * q + alignment.rmse_m
    sigma_yaw = math.radians(1.0 + 10.0 * q)
    diag = [sigma_xy ** 2, sigma_xy ** 2, _BIG, _BIG, _BIG, sigma_yaw ** 2]
    cov = [0.0] * 36
    for i in range(6):
        cov[i * 6 + i] = diag[i]
    return cov


@dataclass
class InitialPoseCandidate:
    """A ranked initial-pose hypothesis for localizer initialisation."""

    pose: Pose2D  # current robot pose in the map frame
    covariance: list[float]  # 6x6 row-major
    score: float
    place_id: str | None
    source_scene: str
    overlap_ratio: float
    failure_class: str | None = None

    def to_dict(self) -> dict:
        return {
            "pose": self.pose.to_dict(),
            "score": round(float(self.score), 4),
            "place_id": self.place_id,
            "source_scene": self.source_scene,
            "overlap_ratio": round(float(self.overlap_ratio), 4),
            "covariance_diag": [round(self.covariance[i * 6 + i], 5) for i in range(6)],
        }


def relocalization_candidates(
    current: Scene,
    database: SceneDatabase,
    aligner: Aligner | None = None,
    top_k: int = 3,
) -> list[InitialPoseCandidate]:
    """Return ranked initial-pose candidates for the current observation."""
    aligner = aligner or SE2Aligner()
    candidates: list[InitialPoseCandidate] = []
    for cand in database.query(current, top_k=top_k):
        ref = database.get_scene(cand.scene_id)
        alignment = aligner.align(current, ref)
        if not alignment.success:
            continue
        ref_pose = ref.pose or Pose2D()
        pose_map = ref_pose.compose(alignment.relative_pose)
        candidates.append(
            InitialPoseCandidate(
                pose=pose_map,
                covariance=covariance_from_alignment(alignment),
                score=float(cand.score * alignment.overlap_ratio),
                place_id=cand.place_id,
                source_scene=cand.scene_id,
                overlap_ratio=alignment.overlap_ratio,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
