"""Multi-frame change detection with persistence (§11.1 stages 1-3).

Aligns each query frame to the reference, runs the per-frame diff, then
consolidates across frames so that only persistent changes are actionable and
transient ones are labelled ``dynamic``.
"""

from __future__ import annotations

from bevmatch.alignment.base import Aligner
from bevmatch.change.bev_diff import ChangeConfig, detect_changes_detailed
from bevmatch.change.persistence import PersistenceConfig, consolidate_changes
from bevmatch.core.datamodel import ChangeHypothesis, Scene


def detect_persistent_changes(
    query_frames: list[Scene],
    reference: Scene,
    aligner: Aligner,
    change_config: ChangeConfig | None = None,
    persistence_config: PersistenceConfig | None = None,
) -> list[ChangeHypothesis]:
    """Return persistence-consolidated changes over a query burst."""
    ref_xy = reference.primary().xy()
    per_frame: list[list[ChangeHypothesis]] = []
    for frame in query_frames:
        alignment = aligner.align(frame, reference)
        if not alignment.success:
            per_frame.append([])  # alignment-gated: contribute nothing this frame
            continue
        result = detect_changes_detailed(
            frame.primary().xy(),
            ref_xy,
            alignment.relative_pose,
            change_config,
            align_overlap=alignment.overlap_ratio,
        )
        per_frame.append(result.changes)
    return consolidate_changes(per_frame, persistence_config)
