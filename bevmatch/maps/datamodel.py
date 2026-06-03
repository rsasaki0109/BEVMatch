"""Map artifacts and the Map Validation Issue (§8, §12.2, §12.4, §5.8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bevmatch.maps.severity import Severity, recommended_action
from bevmatch.representations.bev import BEVConfig, points_to_bev


@dataclass
class PointCloudMap:
    """A point cloud localization / structure map (§12.2)."""

    points: np.ndarray  # (N, 2/3) in the map frame
    map_id: str = "map"
    version: str = "v1"

    def xy(self) -> np.ndarray:
        pts = np.asarray(self.points, dtype=float)
        return pts[:, :2] if pts.ndim == 2 and pts.shape[1] >= 2 else pts.reshape(-1, 2)


@dataclass
class OccupancyMap:
    """A 2D occupancy map: occupied / free / unknown over a BEV grid (§12.2)."""

    occupied: np.ndarray  # bool grid
    known: np.ndarray  # bool grid (False = unknown / unobserved)
    bev: BEVConfig
    map_id: str = "map"
    version: str = "v1"

    @staticmethod
    def from_points(points_xy: np.ndarray, bev: BEVConfig | None = None, **kw) -> "OccupancyMap":
        bev = bev or BEVConfig()
        occ = points_to_bev(points_xy, bev).occupied(0.5)
        size = bev.size
        yy, xx = np.mgrid[0:size, 0:size]
        rng = np.hypot((xx - bev.center) * bev.resolution_m, (yy - bev.center) * bev.resolution_m)
        known = rng <= bev.range_m
        return OccupancyMap(occupied=occ, known=known, bev=bev, **kw)


@dataclass
class MapElement:
    """A vector-map element: a typed polyline (lane boundary, stop line, ...)."""

    element_id: str
    element_type: str
    polyline: np.ndarray  # (M, 2)


@dataclass
class VectorMap:
    """A lightweight vector / Lanelet2-style map (§12.2)."""

    elements: list[MapElement] = field(default_factory=list)
    map_id: str = "map"
    version: str = "v1"


@dataclass
class MapValidationIssue:
    """An operational issue: the map disagrees with the current world (§12.4)."""

    issue_type: str
    severity: Severity
    centroid_xy: tuple[float, float]
    area_m2: float = 0.0
    confidence: float = 1.0
    persistence: float = 1.0
    bbox_xy: tuple[float, float, float, float] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    review_status: str = "pending"
    issue_id: str = ""

    def __post_init__(self) -> None:
        if not self.recommended_action:
            self.recommended_action = recommended_action(self.issue_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "severity": self.severity.label,
            "centroid_xy": [round(float(c), 3) for c in self.centroid_xy],
            "area_m2": round(float(self.area_m2), 3),
            "confidence": round(float(self.confidence), 4),
            "persistence": round(float(self.persistence), 4),
            "bbox_xy": [round(float(b), 3) for b in self.bbox_xy] if self.bbox_xy else None,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "review_status": self.review_status,
        }
