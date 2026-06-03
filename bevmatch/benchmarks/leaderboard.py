"""Leaderboard rendering and submission schema (§0.8, §20.6).

Renders per-task leaderboards (markdown / rows) ranked by a primary metric, and
defines a submission entry so an external plugin's results can be merged into the
same board on the shared protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# task -> (primary metric, higher_is_better, displayed metric columns)
_TASK_SPEC = {
    "retrieval": ("recall@1", True, ["recall@1", "recall@5", "mrr"]),
    "alignment": ("within_tol_rate", True, ["success_rate", "within_tol_rate", "trans_err_m", "rot_err_deg"]),
    "change": ("removed_recall", True, ["added_precision", "added_recall", "removed_precision", "removed_recall", "false_changes"]),
    "map_validation": ("issue_recall", True, ["issue_precision", "issue_recall", "false_issues_fresh", "review_burden"]),
}


@dataclass
class SubmissionEntry:
    """An external result to merge into the leaderboard (§20.6 contribution)."""

    task: str
    method: str
    dataset: str
    metrics: dict[str, float]
    manifest: dict[str, Any] = field(default_factory=dict)  # plugin manifest / fingerprint

    def to_result(self):
        from bevmatch.benchmarks.suite import MethodResult

        return MethodResult(self.task, self.method, self.dataset, self.metrics)


def leaderboard_rows(results, task: str) -> list:
    """Return results for ``task`` sorted by the task's primary metric."""
    primary, higher, _ = _TASK_SPEC[task]
    rows = [r for r in results if r.task == task]
    rows.sort(key=lambda r: r.metrics.get(primary, 0.0), reverse=higher)
    return rows


def format_leaderboard(results, task: str) -> str:
    """Render a markdown leaderboard table for one task."""
    primary, _, cols = _TASK_SPEC[task]
    rows = leaderboard_rows(results, task)
    header = "| rank | method | " + " | ".join(cols) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in cols) + " |"
    lines = [f"### {task}  (ranked by {primary})", "", header, sep]
    for i, r in enumerate(rows, 1):
        cells = " | ".join(f"{r.metrics.get(c, float('nan')):.3f}" for c in cols)
        lines.append(f"| {i} | {r.method} | {cells} |")
    return "\n".join(lines)


def format_full_leaderboard(result) -> str:
    tasks = ["retrieval", "alignment", "change", "map_validation"]
    present = [t for t in tasks if result.by_task(t)]
    return "\n\n".join(format_leaderboard(result.results, t) for t in present)
