"""Point cloud map validator (§12.3, §12.4, §17.2 Pattern C).

Aligns current observation(s) to the point cloud map, then turns persistent
geometry changes into issues:
  - occupied now but not in the map  -> ``new_static_obstacle``
  - in the map but no longer observed -> ``missing_static_structure``
If the observation cannot be aligned to the map, that itself is reported as a
``localization_risk_region`` (the map may be stale or the place mismatched).
Multi-frame input uses temporal persistence so dynamics don't become issues.
"""

from __future__ import annotations

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.se2 import SE2Aligner
from bevmatch.change.bev_diff import ChangeConfig
from bevmatch.change.persistence import PersistenceConfig
from bevmatch.change.sequence import detect_persistent_changes
from bevmatch.core.datamodel import Observation, Scene
from bevmatch.maps.datamodel import MapValidationIssue, PointCloudMap
from bevmatch.maps.severity import assess_severity

_CATEGORY_TO_ISSUE = {
    "added": "new_static_obstacle",
    "removed": "missing_static_structure",
}


class PointCloudMapValidator:
    name = "pointcloud-map-validator"

    def __init__(
        self,
        aligner: Aligner | None = None,
        change_config: ChangeConfig | None = None,
        persistence_config: PersistenceConfig | None = None,
    ) -> None:
        self.aligner = aligner or SE2Aligner()
        self.change_config = change_config or ChangeConfig(use_occlusion=False)
        self.persistence_config = persistence_config or PersistenceConfig()

    def _map_scene(self, pcd_map: PointCloudMap) -> Scene:
        modality = "lidar_bev_points"
        return Scene(
            scene_id=f"{pcd_map.map_id}:{pcd_map.version}",
            observations={modality: Observation(modality, pcd_map.xy())},
        )

    def validate(self, current_frames: list[Scene], pcd_map: PointCloudMap) -> list[MapValidationIssue]:
        if not current_frames:
            return []
        map_scene = self._map_scene(pcd_map)

        align0 = self.aligner.align(current_frames[0], map_scene)
        if not align0.success:
            return [
                MapValidationIssue(
                    issue_type="localization_risk_region",
                    severity=assess_severity("localization_risk_region", confidence=0.5),
                    centroid_xy=(0.0, 0.0),
                    confidence=0.5,
                    evidence={
                        "reason": "observation could not be aligned to map",
                        "alignment_failure_class": align0.failure_class,
                        "overlap_ratio": round(align0.overlap_ratio, 3),
                    },
                )
            ]

        changes = detect_persistent_changes(
            current_frames, map_scene, self.aligner, self.change_config, self.persistence_config
        )

        issues: list[MapValidationIssue] = []
        for ch in changes:
            if not ch.actionable:
                continue
            issue_type = _CATEGORY_TO_ISSUE[ch.category]
            issues.append(
                MapValidationIssue(
                    issue_type=issue_type,
                    severity=assess_severity(issue_type, ch.area_m2, ch.confidence, ch.persistence),
                    centroid_xy=ch.centroid_xy,
                    area_m2=ch.area_m2,
                    confidence=ch.confidence,
                    persistence=ch.persistence,
                    bbox_xy=ch.bbox_xy,
                    evidence={
                        "source": "pointcloud-map-diff",
                        "change_category": ch.category,
                        "map_id": pcd_map.map_id,
                        "map_version": pcd_map.version,
                    },
                )
            )
        return issues
