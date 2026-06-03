"""Map validation report + human-review export (§12.3, §12 deliverables).

Aggregates issues from one or more validators, assigns stable IDs, prioritises by
severity, and exports a JSON artifact and a human-readable markdown review sheet
a mapping engineer can triage.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bevmatch.maps.datamodel import MapValidationIssue
from bevmatch.maps.severity import Severity


@dataclass
class MapValidationReport:
    map_id: str
    map_version: str
    scene_id: str
    issues: list[MapValidationIssue] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._assign_ids()

    def _assign_ids(self) -> None:
        for i, issue in enumerate(self.prioritized()):
            issue.issue_id = f"{self.map_id}-ISSUE-{i:04d}"

    def add(self, issues: list[MapValidationIssue]) -> None:
        self.issues.extend(issues)
        self._assign_ids()

    def prioritized(self) -> list[MapValidationIssue]:
        return sorted(self.issues, key=lambda i: (int(i.severity), i.area_m2), reverse=True)

    def severity_counts(self) -> dict[str, int]:
        c = Counter(i.severity.label for i in self.issues)
        return {s.label: c.get(s.label, 0) for s in Severity}

    def stale_regions(self) -> list[MapValidationIssue]:
        """Issues indicating the map no longer matches the world (stale)."""
        stale_types = {"missing_static_structure", "map_stale_region", "localization_risk_region"}
        return [i for i in self.prioritized() if i.issue_type in stale_types]

    def to_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "map_version": self.map_version,
            "scene_id": self.scene_id,
            "n_issues": len(self.issues),
            "severity_counts": self.severity_counts(),
            "issues": [i.to_dict() for i in self.prioritized()],
            "provenance": self.provenance,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        lines = [
            f"# Map Validation Review — {self.map_id} ({self.map_version})",
            "",
            f"Query scene: `{self.scene_id}`  |  Issues: {len(self.issues)}",
            "",
            "| ID | Severity | Type | Location (x, y) | Area m² | Conf | Action | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for i in self.prioritized():
            x, y = i.centroid_xy
            lines.append(
                f"| {i.issue_id} | **{i.severity.label}** | {i.issue_type} | "
                f"({x:+.1f}, {y:+.1f}) | {i.area_m2:.1f} | {i.confidence:.2f} | "
                f"{i.recommended_action} | {i.review_status} |"
            )
        return "\n".join(lines)

    def save_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path
