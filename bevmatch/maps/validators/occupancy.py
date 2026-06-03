"""Occupancy map validator (§12.2, §12.4, §18.2 Use Case B).

Compares the current observation's occupancy against a stored occupancy map in
the map frame:
  - occupied now, free & known in the map -> ``new_static_obstacle``
  - occupied in the map, free now (and observed) -> ``map_stale_region``
Only the known region of the map is compared; unknown cells are skipped.
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Pose2D, Scene
from bevmatch.grid_utils import connected_components, dilate
from bevmatch.maps.datamodel import MapValidationIssue, OccupancyMap
from bevmatch.maps.severity import assess_severity
from bevmatch.representations.bev import points_to_bev


class OccupancyMapValidator:
    name = "occupancy-map-validator"

    def __init__(self, min_cells: int = 3, suppress_passes: int = 1) -> None:
        self.min_cells = min_cells
        self.suppress_passes = suppress_passes

    def validate(
        self,
        current: Scene,
        occ_map: OccupancyMap,
        pose: Pose2D | None = None,
    ) -> list[MapValidationIssue]:
        """``pose`` maps the current scene into the map frame (default identity)."""
        bev = occ_map.bev
        pose = pose or Pose2D()
        cur_xy = pose.transform(current.primary().xy())
        cur_occ = points_to_bev(cur_xy, bev).occupied(0.5)

        # Observed-now region = within range disk (the sensor's footprint).
        size = bev.size
        yy, xx = np.mgrid[0:size, 0:size]
        rng = np.hypot((xx - bev.center) * bev.resolution_m, (yy - bev.center) * bev.resolution_m)
        observed_now = rng <= bev.range_m
        compare = occ_map.known & observed_now

        new_mask = cur_occ & ~dilate(occ_map.occupied, self.suppress_passes) & compare
        stale_mask = occ_map.occupied & ~dilate(cur_occ, self.suppress_passes) & compare

        issues: list[MapValidationIssue] = []
        issues += self._mask_to_issues(new_mask, "new_static_obstacle", occ_map)
        issues += self._mask_to_issues(stale_mask, "map_stale_region", occ_map)
        return issues

    def _mask_to_issues(self, mask, issue_type, occ_map) -> list[MapValidationIssue]:
        bev = occ_map.bev
        cell_area = bev.resolution_m ** 2
        out: list[MapValidationIssue] = []
        for cells in connected_components(mask, self.min_cells):
            xs = (cells[:, 1] - bev.center) * bev.resolution_m
            ys = (cells[:, 0] - bev.center) * bev.resolution_m
            n = len(cells)
            area = n * cell_area
            conf = n / (n + 4.0)
            out.append(
                MapValidationIssue(
                    issue_type=issue_type,
                    severity=assess_severity(issue_type, area, conf),
                    centroid_xy=(float(xs.mean()), float(ys.mean())),
                    area_m2=float(area),
                    confidence=float(conf),
                    bbox_xy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                    evidence={"source": "occupancy-map-diff", "map_id": occ_map.map_id,
                              "map_version": occ_map.version},
                )
            )
        return out
