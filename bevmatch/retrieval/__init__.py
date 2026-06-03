"""Retrieval layer (§9): descriptor + Top-K candidate search."""

from bevmatch.retrieval.descriptor import (
    ScanContextConfig,
    ring_key,
    ring_key_distance,
    scan_context,
)
from bevmatch.retrieval.retriever import SceneDatabase

__all__ = [
    "ScanContextConfig",
    "scan_context",
    "ring_key",
    "ring_key_distance",
    "SceneDatabase",
]
