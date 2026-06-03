"""Benchmark suite runner (§13).

Runs each task benchmark on its dataset card and returns ``MethodResult``s
(one per method/plugin per task) with a flat metrics dict — ready for the
leaderboard. Aggregated tasks (change, map) average over the card's seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.alignment import SE2Aligner, SE3Aligner
from bevmatch.alignment.base import Aligner
from bevmatch.benchmarks.cards import CARDS, DatasetCard
from bevmatch.change import ChangeConfig, detect_persistent_changes
from bevmatch.datasets import (
    make_map_validation_case,
    make_synthetic_change_case,
    make_synthetic_route,
)
from bevmatch.eval import (
    change_prf,
    evaluate_alignment,
    evaluate_retrieval,
    false_changes,
    issue_prf,
    review_burden,
)
from bevmatch.maps import PointCloudMapValidator, VectorMapValidator
from bevmatch.retrieval import BEVGridDescriptor, ScanContextDescriptor, SceneDatabase
from bevmatch.retrieval.base import GlobalDescriptor


@dataclass
class MethodResult:
    task: str
    method: str
    dataset: str
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return {"task": self.task, "method": self.method, "dataset": self.dataset,
                "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()}}


def run_retrieval_benchmark(
    descriptors: list[GlobalDescriptor] | None = None,
    card: DatasetCard | None = None,
    ks: tuple[int, ...] = (1, 5),
) -> list[MethodResult]:
    card = card or CARDS["mini-route"]
    descriptors = descriptors or [ScanContextDescriptor(), BEVGridDescriptor()]
    route = make_synthetic_route(seed=card.seed, n_places=card.n_places, n_queries=card.n_queries)
    results = []
    for desc in descriptors:
        db = SceneDatabase(descriptor=desc)
        db.add_all(route.historical)
        rep = evaluate_retrieval(db, route.queries, ks=ks)
        metrics = {f"recall@{k}": rep.recall_at_k[k] for k in ks}
        metrics["mrr"] = rep.mrr
        results.append(MethodResult("retrieval", desc.name, card.name, metrics))
    return results


def run_alignment_benchmark(
    aligners: list[Aligner] | None = None,
    card: DatasetCard | None = None,
) -> list[MethodResult]:
    card = card or CARDS["mini-route"]
    aligners = aligners or [SE2Aligner(), SE3Aligner()]
    route = make_synthetic_route(seed=card.seed, n_places=card.n_places, n_queries=card.n_queries)
    db = SceneDatabase(descriptor=ScanContextDescriptor())
    db.add_all(route.historical)
    results = []
    for aligner in aligners:
        rep = evaluate_alignment(db, route.queries, aligner)
        results.append(MethodResult("alignment", aligner.name, card.name, {
            "success_rate": rep.success_rate,
            "within_tol_rate": rep.within_tol_rate,
            "trans_err_m": rep.mean_trans_err_m,
            "rot_err_deg": rep.mean_rot_err_deg,
        }))
    return results


def run_change_benchmark(
    card: DatasetCard | None = None,
    method_name: str = "bev-diff+persistence",
) -> list[MethodResult]:
    card = card or CARDS["mini-change"]
    aligner = SE2Aligner()
    cfg = ChangeConfig(use_occlusion=False)
    p_add, r_add, p_rem, r_rem, fc = [], [], [], [], []
    for s in (card.seeds or (card.seed,)):
        case = make_synthetic_change_case(seed=s)
        changes = detect_persistent_changes(case.query_frames, case.reference, aligner, cfg)
        actionable = [c for c in changes if c.actionable]
        a = change_prf(actionable, case.added_centers, "added")
        r = change_prf(actionable, case.removed_centers, "removed")
        p_add.append(a.precision); r_add.append(a.recall)
        p_rem.append(r.precision); r_rem.append(r.recall)
        fc.append(false_changes(actionable, case.added_centers, case.removed_centers))
    metrics = {
        "added_precision": float(np.mean(p_add)), "added_recall": float(np.mean(r_add)),
        "removed_precision": float(np.mean(p_rem)), "removed_recall": float(np.mean(r_rem)),
        "false_changes": float(np.mean(fc)),
    }
    return [MethodResult("change", method_name, card.name, metrics)]


def run_map_benchmark(
    card: DatasetCard | None = None,
    method_name: str = "pointcloud+vector-validator",
) -> list[MethodResult]:
    card = card or CARDS["mini-map"]
    precisions, recalls, fresh_fp, burdens = [], [], [], []
    for s in (card.seeds or (card.seed,)):
        changed = make_map_validation_case(seed=s, changed=True)
        issues = (PointCloudMapValidator().validate(changed.current_frames, changed.pcd_map)
                  + VectorMapValidator().validate(changed.current_map_frame, changed.vmap))
        prf = issue_prf(issues, changed.gt_issues)
        precisions.append(prf.precision); recalls.append(prf.recall)
        burdens.append(review_burden(len(issues), 1))

        fresh = make_map_validation_case(seed=s, changed=False)
        fresh_issues = (PointCloudMapValidator().validate(fresh.current_frames, fresh.pcd_map)
                        + VectorMapValidator().validate(fresh.current_map_frame, fresh.vmap))
        fresh_fp.append(len(fresh_issues))
    metrics = {
        "issue_precision": float(np.mean(precisions)), "issue_recall": float(np.mean(recalls)),
        "false_issues_fresh": float(np.mean(fresh_fp)), "review_burden": float(np.mean(burdens)),
    }
    return [MethodResult("map_validation", method_name, card.name, metrics)]


@dataclass
class BenchmarkResult:
    results: list[MethodResult] = field(default_factory=list)

    def by_task(self, task: str) -> list[MethodResult]:
        return [r for r in self.results if r.task == task]

    def to_dict(self) -> dict:
        return {"results": [r.to_dict() for r in self.results]}


@dataclass
class BenchmarkSuite:
    """Runs the full suite across the registered tasks."""

    def run(
        self,
        descriptors: list[GlobalDescriptor] | None = None,
        aligners: list[Aligner] | None = None,
    ) -> BenchmarkResult:
        results: list[MethodResult] = []
        results += run_retrieval_benchmark(descriptors)
        results += run_alignment_benchmark(aligners)
        results += run_change_benchmark()
        results += run_map_benchmark()
        return BenchmarkResult(results=results)
