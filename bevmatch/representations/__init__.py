"""Representation layer (§5.4, §6.2): derive comparison representations from a Scene."""

from bevmatch.representations.bev import BEVConfig, BEVOccupancy, points_to_bev
from bevmatch.representations.semantic_bev import (
    SemanticBEV,
    points_to_semantic_bev,
    semantic_change_mask,
)

__all__ = [
    "BEVConfig",
    "BEVOccupancy",
    "points_to_bev",
    "SemanticBEV",
    "points_to_semantic_bev",
    "semantic_change_mask",
]
