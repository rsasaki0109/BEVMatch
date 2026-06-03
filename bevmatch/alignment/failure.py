"""Alignment failure classification (§10.5).

Alignment failure is a first-class result. Rather than returning a silently
wrong pose, the aligner labels *why* it failed so operators and downstream
gating can react. v0.3 uses a small, explainable rule set over the alignment
diagnostics.
"""

from __future__ import annotations

# Known failure classes (a subset of the §10.5 list implemented in v0.3).
FAILURE_CLASSES = (
    "insufficient_constraints",  # too few points / correspondences
    "overlap_insufficient",  # comparable region too small
    "high_residual",  # converged but residual large (likely wrong / repetitive)
    "ambiguous",  # marginal: neither clearly good nor clearly explained
)


def classify_alignment_failure(
    *,
    overlap_ratio: float,
    inlier_ratio: float,
    rmse_m: float,
    num_correspondences: int,
    success: bool,
    min_overlap_ratio: float,
    min_correspondences: int,
    rmse_fail_m: float,
) -> str | None:
    """Return a failure class, or ``None`` when alignment succeeded."""
    if success:
        return None
    if num_correspondences < min_correspondences:
        return "insufficient_constraints"
    if overlap_ratio < min_overlap_ratio:
        return "overlap_insufficient"
    if rmse_m > rmse_fail_m:
        return "high_residual"
    return "ambiguous"
