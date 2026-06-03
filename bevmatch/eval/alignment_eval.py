"""Alignment evaluation harness and metrics (§13.3).

Each query is aligned against its ground-truth place scene (isolating alignment
quality from retrieval), and the estimated relative pose is compared to the
known ground-truth transform. Reports translation/rotation error, success rate,
overlap, and the distribution of failure classes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from bevmatch.alignment.base import Aligner
from bevmatch.core.datamodel import Pose2D, _wrap_angle
from bevmatch.datasets.synthetic import RouteQuery
from bevmatch.retrieval.retriever import SceneDatabase


def pose_errors(est: Pose2D, gt: Pose2D) -> tuple[float, float]:
    """Return ``(translation_error_m, rotation_error_deg)`` between two SE2 poses."""
    trans = float(np.hypot(est.x - gt.x, est.y - gt.y))
    rot = float(abs(np.rad2deg(_wrap_angle(est.yaw - gt.yaw))))
    return trans, rot


@dataclass
class AlignmentReport:
    aligner: str
    n_queries: int
    success_rate: float = 0.0
    within_tol_rate: float = 0.0  # success AND pose within tolerance
    mean_trans_err_m: float = 0.0
    median_trans_err_m: float = 0.0
    mean_rot_err_deg: float = 0.0
    mean_overlap: float = 0.0
    failure_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "aligner": self.aligner,
            "n_queries": self.n_queries,
            "success_rate": round(self.success_rate, 4),
            "within_tol_rate": round(self.within_tol_rate, 4),
            "mean_trans_err_m": round(self.mean_trans_err_m, 4),
            "median_trans_err_m": round(self.median_trans_err_m, 4),
            "mean_rot_err_deg": round(self.mean_rot_err_deg, 4),
            "mean_overlap": round(self.mean_overlap, 4),
            "failure_counts": self.failure_counts,
        }

    def as_row(self) -> str:
        return "  ".join(
            [
                f"{self.aligner:<12}",
                f"{self.success_rate:.3f}",
                f"{self.within_tol_rate:.3f}",
                f"{self.mean_trans_err_m:.3f}",
                f"{self.mean_rot_err_deg:.2f}",
                f"{self.mean_overlap:.3f}",
            ]
        )


def evaluate_alignment(
    database: SceneDatabase,
    queries: list[RouteQuery],
    aligner: Aligner,
    trans_tol_m: float = 1.0,
    rot_tol_deg: float = 5.0,
) -> AlignmentReport:
    trans_errs: list[float] = []
    rot_errs: list[float] = []
    overlaps: list[float] = []
    n_success = 0
    n_within = 0
    failures: Counter[str] = Counter()

    for rq in queries:
        ref = database.scene_by_place(rq.gt_place_id)
        hyp = aligner.align(rq.scene, ref)
        overlaps.append(hyp.overlap_ratio)
        if not hyp.success:
            failures[hyp.failure_class or "unknown"] += 1
            continue
        n_success += 1
        t_err, r_err = pose_errors(hyp.relative_pose, rq.gt_relative_pose)
        trans_errs.append(t_err)
        rot_errs.append(r_err)
        if t_err <= trans_tol_m and r_err <= rot_tol_deg:
            n_within += 1

    n = len(queries)
    report = AlignmentReport(aligner=aligner.name, n_queries=n)
    report.success_rate = n_success / n if n else 0.0
    report.within_tol_rate = n_within / n if n else 0.0
    report.mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    if trans_errs:
        report.mean_trans_err_m = float(np.mean(trans_errs))
        report.median_trans_err_m = float(np.median(trans_errs))
        report.mean_rot_err_deg = float(np.mean(rot_errs))
    report.failure_counts = dict(failures)
    return report


def format_alignment_table(reports: list[AlignmentReport]) -> str:
    header = ["aligner".ljust(12), "succ", "wtol", "t_err", "r_err", "ovlp"]
    lines = ["  ".join(header), "-" * 52]
    lines += [r.as_row() for r in reports]
    return "\n".join(lines)
