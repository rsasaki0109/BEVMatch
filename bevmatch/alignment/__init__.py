"""Alignment layer (§10): make a query/candidate pair comparable."""

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.failure import classify_alignment_failure
from bevmatch.alignment.se2 import SE2AlignConfig, SE2Aligner, align_se2, bev_overlap
from bevmatch.alignment.se3 import SE3AlignConfig, SE3Aligner, align_se3

__all__ = [
    "Aligner",
    "SE2Aligner",
    "SE2AlignConfig",
    "align_se2",
    "bev_overlap",
    "SE3Aligner",
    "SE3AlignConfig",
    "align_se3",
    "classify_alignment_failure",
]
