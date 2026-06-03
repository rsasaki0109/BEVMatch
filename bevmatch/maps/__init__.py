"""Map validation layer (§12).

Turns same-place comparison evidence into operational Map Validation Issues:
"does this map still match the current world?" (not file-syntax validation, §12.5).
"""

from bevmatch.maps.datamodel import (
    MapElement,
    MapValidationIssue,
    OccupancyMap,
    PointCloudMap,
    VectorMap,
)
from bevmatch.maps.report import MapValidationReport
from bevmatch.maps.severity import Severity, assess_severity, recommended_action
from bevmatch.maps.validators import (
    OccupancyMapValidator,
    PointCloudMapValidator,
    VectorMapValidator,
)

__all__ = [
    "PointCloudMap",
    "OccupancyMap",
    "VectorMap",
    "MapElement",
    "MapValidationIssue",
    "Severity",
    "assess_severity",
    "recommended_action",
    "PointCloudMapValidator",
    "OccupancyMapValidator",
    "VectorMapValidator",
    "MapValidationReport",
]
