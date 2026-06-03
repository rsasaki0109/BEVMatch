"""Dataset adapters (§14). v0.1 ships a synthetic same-place toy benchmark."""

from bevmatch.datasets.synthetic import (
    ChangeCase,
    MapValidationCase,
    MultiModalPlaces,
    ObjectChangeCase,
    OcclusionCase,
    RouteQuery,
    SyntheticRoute,
    SyntheticSamePlace,
    make_map_validation_case,
    make_multimodal_places,
    make_object_change_case,
    make_occlusion_case,
    make_synthetic_change_case,
    make_synthetic_route,
    make_synthetic_same_place,
)

__all__ = [
    "SyntheticSamePlace",
    "make_synthetic_same_place",
    "SyntheticRoute",
    "RouteQuery",
    "make_synthetic_route",
    "ChangeCase",
    "make_synthetic_change_case",
    "OcclusionCase",
    "make_occlusion_case",
    "MapValidationCase",
    "make_map_validation_case",
    "MultiModalPlaces",
    "make_multimodal_places",
    "ObjectChangeCase",
    "make_object_change_case",
]
