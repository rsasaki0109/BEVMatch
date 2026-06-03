"""Dataset adapters (§14). v0.1 ships a synthetic same-place toy benchmark."""

from bevmatch.datasets.synthetic import (
    RouteQuery,
    SyntheticRoute,
    SyntheticSamePlace,
    make_synthetic_route,
    make_synthetic_same_place,
)

__all__ = [
    "SyntheticSamePlace",
    "make_synthetic_same_place",
    "SyntheticRoute",
    "RouteQuery",
    "make_synthetic_route",
]
