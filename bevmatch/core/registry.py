"""Minimal plugin registry (§7).

v0.1 keeps this lightweight: plugins are registered by ``(category, name)`` so
later versions can swap descriptors, aligners, change detectors, dataset
adapters, etc. without touching the pipeline wiring.
"""

from __future__ import annotations

from typing import Any, Callable

# category -> name -> factory/callable
_REGISTRY: dict[str, dict[str, Any]] = {}

# Plugin categories from architecture.md §7.2.
CATEGORIES = (
    "data_source",
    "scene_normalizer",
    "representation",
    "descriptor",
    "index_backend",
    "retriever",
    "reranker",
    "alignment",
    "overlap",
    "change_detector",
    "dynamic_filter",
    "map_validator",
    "evaluation",
    "visualization_exporter",
    "confidence",
)


def register(category: str, name: str) -> Callable[[Any], Any]:
    """Decorator: register a plugin under ``category``/``name``."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown plugin category: {category!r}. Known: {CATEGORIES}")

    def _wrap(obj: Any) -> Any:
        _REGISTRY.setdefault(category, {})[name] = obj
        return obj

    return _wrap


def get(category: str, name: str) -> Any:
    try:
        return _REGISTRY[category][name]
    except KeyError as exc:
        raise KeyError(f"No plugin {category!r}/{name!r} registered") from exc


def available(category: str) -> list[str]:
    return sorted(_REGISTRY.get(category, {}).keys())
