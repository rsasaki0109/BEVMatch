"""Retrieval layer (§9): pluggable descriptor + index backend + Top-K search."""

from bevmatch.retrieval.base import DescriptorCode, GlobalDescriptor
from bevmatch.retrieval.descriptor import (
    ScanContextConfig,
    ring_key,
    ring_key_distance,
    scan_context,
    sc_alignment_distance,
)
from bevmatch.retrieval.descriptors import BEVGridDescriptor, ScanContextDescriptor
from bevmatch.retrieval.index import BruteForceIndex, IndexBackend, make_index
from bevmatch.retrieval.retriever import SceneDatabase

__all__ = [
    "GlobalDescriptor",
    "DescriptorCode",
    "ScanContextDescriptor",
    "BEVGridDescriptor",
    "IndexBackend",
    "BruteForceIndex",
    "make_index",
    "SceneDatabase",
    "ScanContextConfig",
    "scan_context",
    "ring_key",
    "ring_key_distance",
    "sc_alignment_distance",
]
