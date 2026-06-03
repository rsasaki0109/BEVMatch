"""Diagnostics (§10.4, §16.2 diagnostics-first).

Pure-Python diagnostic statuses (mirroring diagnostic_msgs/DiagnosticStatus)
derived from BEVMatch evidence. The rclpy node converts these to ROS messages;
offline tools read them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class DiagnosticLevel(IntEnum):
    OK = 0
    WARN = 1
    ERROR = 2
    STALE = 3


@dataclass
class DiagnosticStatus:
    level: DiagnosticLevel
    name: str
    message: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_label": self.level.name,
            "name": self.name,
            "message": self.message,
            "values": {k: str(v) for k, v in self.values.items()},
        }


def diagnostics_from_bundle(bundle, retrieval_min_score: float = 0.5) -> list[DiagnosticStatus]:
    """Per-stage diagnostics for one comparison evidence bundle."""
    out: list[DiagnosticStatus] = []

    bc = bundle.best_candidate
    if bc is None:
        out.append(DiagnosticStatus(DiagnosticLevel.ERROR, "bevmatch/retrieval", "no candidates"))
    else:
        lvl = DiagnosticLevel.OK if bc.score >= retrieval_min_score else DiagnosticLevel.WARN
        out.append(DiagnosticStatus(lvl, "bevmatch/retrieval", f"matched {bc.place_id}",
                                    {"place_id": bc.place_id, "score": round(bc.score, 4)}))

    a = bundle.alignment
    if a is None:
        out.append(DiagnosticStatus(DiagnosticLevel.WARN, "bevmatch/alignment", "not attempted"))
    elif a.success:
        out.append(DiagnosticStatus(
            DiagnosticLevel.OK, "bevmatch/alignment", "aligned",
            {"overlap": round(a.overlap_ratio, 3), "inliers": round(a.inlier_ratio, 3),
             "rmse_m": round(a.rmse_m, 3)}))
    else:
        out.append(DiagnosticStatus(
            DiagnosticLevel.ERROR, "bevmatch/alignment", f"failed: {a.failure_class}",
            {"failure_class": a.failure_class, "overlap": round(a.overlap_ratio, 3)}))

    n_changes = len(bundle.changes)
    lvl = DiagnosticLevel.OK if n_changes == 0 else DiagnosticLevel.WARN
    out.append(DiagnosticStatus(lvl, "bevmatch/change", f"{n_changes} change(s)",
                                {"added": len(bundle.added()), "removed": len(bundle.removed())}))
    return out


def diagnostics_from_map_report(report) -> DiagnosticStatus:
    """A single rolled-up diagnostic from a map validation report (§12)."""
    from bevmatch.maps.severity import Severity

    if not report.issues:
        return DiagnosticStatus(DiagnosticLevel.OK, "bevmatch/map_validation", "map matches world",
                                {"n_issues": 0})
    max_sev = max(i.severity for i in report.issues)
    level = DiagnosticLevel.ERROR if max_sev >= Severity.HIGH else DiagnosticLevel.WARN
    return DiagnosticStatus(
        level, "bevmatch/map_validation", f"{len(report.issues)} issue(s); max {max_sev.label}",
        {"n_issues": len(report.issues), "max_severity": max_sev.label, **report.severity_counts()},
    )
