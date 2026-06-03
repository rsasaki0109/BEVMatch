"""Map validation metrics (§13.5).

Issue precision/recall (a reported issue is a true positive if it matches a
ground-truth issue of the same type within a radius) and human-review burden
(issues surfaced per scene).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.maps.datamodel import MapValidationIssue


def match_issues(
    reported: list[MapValidationIssue],
    gt: list[tuple[str, np.ndarray]],
    radius: float = 3.0,
) -> tuple[int, int, int]:
    """Greedy type-aware matching. Returns ``(tp, fp, fn)``."""
    used: set[int] = set()
    tp = 0
    for gtype, gloc in gt:
        gloc = np.asarray(gloc, dtype=float)
        best, best_d = None, radius
        for i, issue in enumerate(reported):
            if i in used or issue.issue_type != gtype:
                continue
            d = float(np.linalg.norm(np.array(issue.centroid_xy) - gloc))
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(reported) - tp, len(gt) - tp


@dataclass
class IssuePRF:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4)}


def issue_prf(reported, gt, radius: float = 3.0) -> IssuePRF:
    tp, fp, fn = match_issues(reported, gt, radius)
    return IssuePRF(tp=tp, fp=fp, fn=fn)


def review_burden(n_issues: int, n_scenes: int) -> float:
    """Issues surfaced per scene (§13.5 human review burden proxy)."""
    return n_issues / n_scenes if n_scenes else 0.0
