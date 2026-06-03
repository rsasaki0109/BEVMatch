"""Robotics-ecosystem adapters (§17 Autoware, §18 Nav2).

BEVMatch does not replace Autoware/Nav2 localization; it provides global
localization candidates, map-freshness evidence, and localization-health signals
to those stacks. These adapters orchestrate the retrieval / alignment / map
layers into the artifacts each ecosystem consumes.
"""

from bevmatch.integrations.autoware import AutowareAdapter, LocalizationHealth
from bevmatch.integrations.nav2 import Nav2Adapter, OccupancyGrid
from bevmatch.integrations.relocalization import (
    InitialPoseCandidate,
    covariance_from_alignment,
    relocalization_candidates,
)

__all__ = [
    "InitialPoseCandidate",
    "covariance_from_alignment",
    "relocalization_candidates",
    "AutowareAdapter",
    "LocalizationHealth",
    "Nav2Adapter",
    "OccupancyGrid",
]
