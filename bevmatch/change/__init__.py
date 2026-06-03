"""Change detection layer (§11): occlusion-aware BEV occupancy diff + persistence."""

from bevmatch.change.bev_diff import (
    ChangeConfig,
    ChangeResult,
    detect_changes,
    detect_changes_detailed,
)
from bevmatch.change.comparable import ComparableRegion, Observability, comparable_region, observability
from bevmatch.change.persistence import PersistenceConfig, consolidate_changes
from bevmatch.change.sequence import detect_persistent_changes

__all__ = [
    "ChangeConfig",
    "ChangeResult",
    "detect_changes",
    "detect_changes_detailed",
    "Observability",
    "ComparableRegion",
    "observability",
    "comparable_region",
    "PersistenceConfig",
    "consolidate_changes",
    "detect_persistent_changes",
]
