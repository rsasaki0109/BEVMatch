"""Change-detection metrics (§13.4).

Instance-level precision/recall by matching reported change centroids to ground-
truth object centers within a radius, plus a count of false changes. Operates on
``ChangeHypothesis`` lists so it works for single-frame or persistence-
consolidated output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.core.datamodel import ChangeHypothesis


def _centroids(changes: list[ChangeHypothesis]) -> np.ndarray:
    if not changes:
        return np.zeros((0, 2))
    return np.array([c.centroid_xy for c in changes], dtype=float)


def greedy_match(reported: np.ndarray, gt: np.ndarray, radius: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns ``(tp, fp, fn)``."""
    used: set[int] = set()
    tp = 0
    for g in gt:
        best, best_d = None, radius
        for i, r in enumerate(reported):
            if i in used:
                continue
            d = float(np.hypot(r[0] - g[0], r[1] - g[1]))
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(reported) - tp, len(gt) - tp


@dataclass
class ChangePRF:
    category: str
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
        return {
            "category": self.category,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def change_prf(
    changes: list[ChangeHypothesis],
    gt_centers: np.ndarray,
    category: str,
    radius: float = 2.5,
) -> ChangePRF:
    """Precision/recall/F1 for one change category vs ground-truth centers."""
    reported = _centroids([c for c in changes if c.category == category])
    gt = np.asarray(gt_centers, dtype=float).reshape(-1, 2)
    tp, fp, fn = greedy_match(reported, gt, radius)
    return ChangePRF(category=category, tp=tp, fp=fp, fn=fn)


def false_changes(
    changes: list[ChangeHypothesis],
    gt_added: np.ndarray,
    gt_removed: np.ndarray,
    radius: float = 2.5,
) -> int:
    """Count actionable changes that match no ground-truth object (§13.4)."""
    add = change_prf(changes, gt_added, "added", radius)
    rem = change_prf(changes, gt_removed, "removed", radius)
    return add.fp + rem.fp
