"""Autoware integration adapter (§17).

BEVMatch complements Autoware localization (§17.1): it supplies global
localization candidates, monitors localization health by same-place consistency,
detects point cloud map staleness, and checks Lanelet2 observation consistency.
It does not replace NDT/EKF or the Lanelet2 file-syntax validator.
"""

from __future__ import annotations

from dataclasses import dataclass

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.se2 import SE2Aligner
from bevmatch.core.datamodel import Pose2D, Scene
from bevmatch.integrations.relocalization import (
    InitialPoseCandidate,
    relocalization_candidates,
)
from bevmatch.maps.datamodel import MapValidationIssue, PointCloudMap, VectorMap
from bevmatch.maps.report import MapValidationReport
from bevmatch.maps.validators import PointCloudMapValidator, VectorMapValidator
from bevmatch.retrieval.retriever import SceneDatabase
from bevmatch.ros.diagnostics import DiagnosticLevel


@dataclass
class LocalizationHealth:
    """Same-place consistency check on a reported localization pose (§17.2 B)."""

    level: DiagnosticLevel
    message: str
    trans_error_m: float
    rot_error_deg: float
    reference_place: str | None
    confidence: float

    def to_dict(self) -> dict:
        return {
            "level": self.level.name,
            "message": self.message,
            "trans_error_m": round(self.trans_error_m, 4),
            "rot_error_deg": round(self.rot_error_deg, 4),
            "reference_place": self.reference_place,
            "confidence": round(self.confidence, 4),
        }


class AutowareAdapter:
    def __init__(self, aligner: Aligner | None = None) -> None:
        self.aligner = aligner or SE2Aligner()

    # Pattern A: initial pose assistance (feeds NDT Monte-Carlo init).
    def initial_pose_candidates(
        self, current: Scene, database: SceneDatabase, top_k: int = 3
    ) -> list[InitialPoseCandidate]:
        return relocalization_candidates(current, database, self.aligner, top_k=top_k)

    # Pattern B: localization health monitoring.
    def localization_health(
        self,
        current: Scene,
        database: SceneDatabase,
        reported_pose: Pose2D,
        trans_tol_m: float = 1.0,
        rot_tol_deg: float = 5.0,
    ) -> LocalizationHealth:
        import numpy as np

        from bevmatch.core.datamodel import _wrap_angle

        cands = relocalization_candidates(current, database, self.aligner, top_k=1)
        if not cands:
            return LocalizationHealth(
                DiagnosticLevel.ERROR, "no same-place reference could be established",
                float("inf"), float("inf"), None, 0.0,
            )
        best = cands[0]
        t_err = float(np.hypot(best.pose.x - reported_pose.x, best.pose.y - reported_pose.y))
        r_err = float(abs(np.rad2deg(_wrap_angle(best.pose.yaw - reported_pose.yaw))))

        if t_err <= trans_tol_m and r_err <= rot_tol_deg:
            level, msg = DiagnosticLevel.OK, "reported pose agrees with same-place evidence"
        elif t_err <= 3 * trans_tol_m and r_err <= 3 * rot_tol_deg:
            level, msg = DiagnosticLevel.WARN, "reported pose drifting from same-place evidence"
        else:
            level, msg = DiagnosticLevel.ERROR, "reported pose disagrees with same-place evidence"
        return LocalizationHealth(level, msg, t_err, r_err, best.place_id, best.overlap_ratio)

    # Pattern C: point cloud map freshness.
    def pointcloud_map_freshness(
        self, current_frames: list[Scene], pcd_map: PointCloudMap
    ) -> MapValidationReport:
        report = MapValidationReport(
            map_id=pcd_map.map_id, map_version=pcd_map.version,
            scene_id=current_frames[0].scene_id if current_frames else "",
            provenance={"workflow": "autoware/pointcloud_map_freshness"},
        )
        report.add(PointCloudMapValidator(self.aligner).validate(current_frames, pcd_map))
        return report

    # Pattern D: Lanelet2 observation consistency.
    def lanelet2_consistency(
        self, current: Scene, vmap: VectorMap, pose: Pose2D | None = None
    ) -> list[MapValidationIssue]:
        return VectorMapValidator().validate(current, vmap, pose)
