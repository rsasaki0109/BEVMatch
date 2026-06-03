"""v0.3 alignment framework: SE2/SE3 aligners, evidence, failure, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bevmatch.alignment import SE2Aligner, SE3Aligner
from bevmatch.alignment.failure import classify_alignment_failure
from bevmatch.core.datamodel import Pose2D, PoseSE3
from bevmatch.datasets import make_synthetic_route
from bevmatch.eval.alignment_eval import evaluate_alignment, pose_errors
from bevmatch.retrieval import ScanContextDescriptor, SceneDatabase


def _db_route(seed=0, n_places=12, n_queries=24):
    route = make_synthetic_route(seed=seed, n_places=n_places, n_queries=n_queries)
    db = SceneDatabase(descriptor=ScanContextDescriptor())
    db.add_all(route.historical)
    return db, route


def test_pose_se3_inverse_roundtrip():
    p = PoseSE3(1.0, -2.0, 0.5, 0.1, -0.2, 0.3)
    pts = np.random.default_rng(0).normal(size=(50, 3))
    back = p.inverse().transform(p.transform(pts))
    assert np.allclose(back, pts, atol=1e-9)


def test_pose_se3_projects_to_se2():
    p = PoseSE3(1.0, 2.0, 3.0, 0.0, 0.0, 0.5)
    se2 = p.project_se2()
    assert (se2.x, se2.y) == (1.0, 2.0)
    assert se2.yaw == pytest.approx(0.5)


def test_failure_classifier():
    assert classify_alignment_failure(
        overlap_ratio=0.8, inlier_ratio=0.9, rmse_m=0.1, num_correspondences=100,
        success=True, min_overlap_ratio=0.45, min_correspondences=20, rmse_fail_m=1.0,
    ) is None
    assert classify_alignment_failure(
        overlap_ratio=0.1, inlier_ratio=0.2, rmse_m=0.5, num_correspondences=100,
        success=False, min_overlap_ratio=0.45, min_correspondences=20, rmse_fail_m=1.0,
    ) == "overlap_insufficient"
    assert classify_alignment_failure(
        overlap_ratio=0.5, inlier_ratio=0.5, rmse_m=0.5, num_correspondences=5,
        success=False, min_overlap_ratio=0.45, min_correspondences=20, rmse_fail_m=1.0,
    ) == "insufficient_constraints"


@pytest.mark.parametrize("aligner_cls", [SE2Aligner, SE3Aligner])
def test_aligner_recovers_pose(aligner_cls):
    db, route = _db_route()
    report = evaluate_alignment(db, route.queries, aligner_cls())
    assert report.success_rate == pytest.approx(1.0)
    assert report.mean_trans_err_m < 0.5
    assert report.mean_rot_err_deg < 2.0
    assert report.within_tol_rate >= 0.95


def test_wrong_place_alignment_fails():
    db, route = _db_route()
    q0 = route.queries[0]
    wrong_place = "place_0" if q0.gt_place_id != "place_0" else "place_1"
    hyp = SE2Aligner().align(q0.scene, db.scene_by_place(wrong_place))
    assert not hyp.success
    assert hyp.failure_class == "overlap_insufficient"


def test_se3_reports_planar_degeneracy():
    db, route = _db_route()
    q0 = route.queries[0]
    hyp = SE3Aligner().align(q0.scene, db.scene_by_place(q0.gt_place_id))
    assert hyp.degeneracy.get("planar_scene") is True
    assert set(hyp.degeneracy.get("unobservable", [])) == {"z", "roll", "pitch"}


def test_alignment_evidence_serialises():
    db, route = _db_route()
    q0 = route.queries[0]
    hyp = SE2Aligner().align(q0.scene, db.scene_by_place(q0.gt_place_id))
    d = hyp.to_dict()
    for key in ("rmse_m", "num_correspondences", "comparable_area_ratio", "failure_class", "degeneracy"):
        assert key in d
    t, r = pose_errors(hyp.relative_pose, q0.gt_relative_pose)
    assert t < 0.5 and r < 2.0
