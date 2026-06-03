"""Benchmark suite (§13, §0.8 Benchmark Suite).

Pipeline-level evaluation across retrieval, alignment, change and map validation,
with dataset cards, reproducible split manifests, and leaderboard-ready output —
so a new descriptor / aligner / detector plugin can be added and compared on a
shared protocol (§20.5 community strategy).
"""

from bevmatch.benchmarks.cards import CARDS, DatasetCard, get_card
from bevmatch.benchmarks.leaderboard import (
    SubmissionEntry,
    format_full_leaderboard,
    format_leaderboard,
    leaderboard_rows,
)
from bevmatch.benchmarks.splits import SplitManifest, dataset_fingerprint, make_manifest
from bevmatch.benchmarks.suite import (
    BenchmarkResult,
    BenchmarkSuite,
    MethodResult,
    run_alignment_benchmark,
    run_change_benchmark,
    run_map_benchmark,
    run_retrieval_benchmark,
)

__all__ = [
    "DatasetCard",
    "CARDS",
    "get_card",
    "SplitManifest",
    "make_manifest",
    "dataset_fingerprint",
    "MethodResult",
    "BenchmarkResult",
    "BenchmarkSuite",
    "run_retrieval_benchmark",
    "run_alignment_benchmark",
    "run_change_benchmark",
    "run_map_benchmark",
    "format_leaderboard",
    "format_full_leaderboard",
    "leaderboard_rows",
    "SubmissionEntry",
]
