"""Nav2 integration adapter (§18).

BEVMatch adds long-term place memory and map-freshness checking around Nav2: it
provides AMCL relocalization candidates, detects static (occupancy) map
staleness, and annotates changed/blocked areas for operator review. It does not
replace AMCL or the Map Server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.se2 import SE2Aligner
from bevmatch.core.datamodel import Pose2D, Scene
from bevmatch.integrations.relocalization import (
    InitialPoseCandidate,
    relocalization_candidates,
)
from bevmatch.maps.datamodel import MapValidationIssue, OccupancyMap
from bevmatch.maps.validators import OccupancyMapValidator
from bevmatch.representations.bev import BEVConfig
from bevmatch.retrieval.retriever import SceneDatabase

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


@dataclass
class OccupancyGrid:
    """A nav_msgs/OccupancyGrid-style 2D map (data in [-1, 0..100])."""

    data: np.ndarray  # (height, width) int8: -1 unknown, 0 free, 100 occupied
    resolution: float
    origin: Pose2D  # pose of cell (0,0) corner in the map frame
    frame_id: str = "map"

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    def occupied_mask(self, threshold: int = 50) -> np.ndarray:
        return self.data >= threshold

    def known_mask(self) -> np.ndarray:
        return self.data >= 0

    @staticmethod
    def from_occupancy_map(occ_map: OccupancyMap) -> "OccupancyGrid":
        """Convert a centred BEV occupancy map to a corner-origin grid."""
        bev = occ_map.bev
        data = np.full(occ_map.occupied.shape, UNKNOWN, dtype=np.int16)
        data[occ_map.known] = FREE
        data[occ_map.occupied] = OCCUPIED
        origin = Pose2D(-bev.center * bev.resolution_m, -bev.center * bev.resolution_m, 0.0)
        return OccupancyGrid(data=data, resolution=bev.resolution_m, origin=origin)

    def to_occupancy_map(self, map_id: str = "nav2_map", version: str = "v1") -> OccupancyMap:
        """Convert back to a centred BEV occupancy map (assumes a centred grid)."""
        size = self.width
        bev = BEVConfig(range_m=size * self.resolution / 2.0, resolution_m=self.resolution)
        return OccupancyMap(
            occupied=self.occupied_mask(), known=self.known_mask(), bev=bev,
            map_id=map_id, version=version,
        )


class Nav2Adapter:
    def __init__(self, aligner: Aligner | None = None) -> None:
        self.aligner = aligner or SE2Aligner()

    # Use Case A: relocalization assistance (AMCL initial pose).
    def relocalization(
        self, current: Scene, database: SceneDatabase, top_k: int = 3
    ) -> list[InitialPoseCandidate]:
        return relocalization_candidates(current, database, self.aligner, top_k=top_k)

    # Use Case B: static map staleness.
    def occupancy_staleness(
        self, current: Scene, occ_grid: OccupancyGrid, pose: Pose2D | None = None
    ) -> list[MapValidationIssue]:
        occ_map = occ_grid.to_occupancy_map()
        return OccupancyMapValidator().validate(current, occ_map, pose)

    # Use Case C: changed-area annotation (blocked / new obstacles for review).
    def changed_area_annotations(
        self, issues: list[MapValidationIssue]
    ) -> list[MapValidationIssue]:
        return [i for i in issues if i.issue_type == "new_static_obstacle"]
