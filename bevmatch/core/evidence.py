"""Comparison Evidence Bundle — BEVMatch's central artifact (§2.2, §8.2).

Place Recognition returns a candidate; BEVMatch returns *decision evidence*:
which place, at what pose, how confidently aligned, what changed, and the
provenance/uncertainty behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bevmatch.core.datamodel import (
    AlignmentHypothesis,
    Candidate,
    ChangeHypothesis,
)

SCHEMA_VERSION = "0.1"


@dataclass
class ComparisonEvidenceBundle:
    """One evidence package for a single query scene (§2.2)."""

    query_scene_id: str
    candidates: list[Candidate] = field(default_factory=list)
    best_candidate: Candidate | None = None
    alignment: AlignmentHypothesis | None = None
    changes: list[ChangeHypothesis] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)

    def added(self) -> list[ChangeHypothesis]:
        return [c for c in self.changes if c.category == "added"]

    def removed(self) -> list[ChangeHypothesis]:
        return [c for c in self.changes if c.category == "removed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "query_scene_id": self.query_scene_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "alignment": self.alignment.to_dict() if self.alignment else None,
            "changes": [c.to_dict() for c in self.changes],
            "provenance": self.provenance,
            "uncertainty": self.uncertainty,
        }

    def summary(self) -> str:
        """A short, human-readable evidence summary (Principle 1: evidence-first)."""
        lines = [f"Query scene: {self.query_scene_id}"]
        if self.best_candidate is not None:
            bc = self.best_candidate
            lines.append(
                f"Best match: {bc.scene_id} (place={bc.place_id}, "
                f"{bc.descriptor_type}, score={bc.score:.4f})"
            )
        else:
            lines.append("Best match: none")
        if self.alignment is not None:
            a = self.alignment
            if a.success:
                p = a.relative_pose
                lines.append(
                    f"Alignment: x={p.x:+.2f} m, y={p.y:+.2f} m, "
                    f"yaw={p.yaw * 57.29578:+.1f} deg, "
                    f"overlap={a.overlap_ratio:.0%}, inliers={a.inlier_ratio:.0%}"
                )
            else:
                lines.append(f"Alignment: FAILED ({a.failure_reason})")
        lines.append(
            f"Changes: {len(self.added())} added, {len(self.removed())} removed"
        )
        return "\n".join(lines)
