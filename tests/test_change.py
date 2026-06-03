"""v0.4 change detection: comparable region, persistence, metrics."""

from __future__ import annotations

import numpy as np

from bevmatch.alignment import SE2Aligner
from bevmatch.change import ChangeConfig, detect_changes_detailed, detect_persistent_changes
from bevmatch.datasets import (
    make_occlusion_case,
    make_synthetic_change_case,
    make_synthetic_same_place,
)
from bevmatch.eval.change_eval import change_prf, false_changes, greedy_match
from bevmatch.retrieval import SceneDatabase
from bevmatch import SamePlaceComparisonPipeline


def test_greedy_match():
    reported = np.array([[0.0, 0.0], [10.0, 10.0]])
    gt = np.array([[0.1, 0.0], [10.0, 9.9], [20.0, 20.0]])
    tp, fp, fn = greedy_match(reported, gt, radius=1.0)
    assert (tp, fp, fn) == (2, 0, 1)


def test_persistence_filters_dynamic():
    case = make_synthetic_change_case(seed=3)
    changes = detect_persistent_changes(
        case.query_frames, case.reference, SE2Aligner(), ChangeConfig(use_occlusion=False)
    )
    actionable = [c for c in changes if c.actionable]
    dynamic = [c for c in changes if c.category == "dynamic"]

    assert change_prf(actionable, case.added_centers, "added").recall == 1.0
    assert change_prf(actionable, case.removed_centers, "removed").recall == 1.0
    assert false_changes(actionable, case.added_centers, case.removed_centers) == 0
    assert len(dynamic) >= 1  # moving objects are caught and filtered

    # No moving object should leak into the actionable set.
    act_xy = np.array([c.centroid_xy for c in actionable]) if actionable else np.zeros((0, 2))
    for frame_dyn in case.dynamic_centers_per_frame:
        for d in frame_dyn:
            if len(act_xy):
                assert np.min(np.linalg.norm(act_xy - d, axis=1)) > 2.0


def test_occlusion_excludes_hidden_objects():
    occ = make_occlusion_case(seed=5)
    a = SE2Aligner().align(occ.query, occ.reference)
    args = (occ.query.primary().xy(), occ.reference.primary().xy(), a.relative_pose)

    naive = detect_changes_detailed(*args, ChangeConfig(use_occlusion=False))
    aware = detect_changes_detailed(*args, ChangeConfig(use_occlusion=True))

    # Naive diff mis-reports the hidden objects as removed; occlusion-aware does not.
    naive_occluded = change_prf(naive.removed(), occ.occluded_centers, "removed").tp
    aware_occluded = change_prf(aware.removed(), occ.occluded_centers, "removed").tp
    assert naive_occluded >= 1
    assert aware_occluded == 0

    # The genuinely removed object is still detected either way.
    assert change_prf(aware.removed(), occ.removed_centers, "removed").recall == 1.0
    assert aware.occluded_ratio > 0.0


def test_change_evidence_in_bundle():
    data = make_synthetic_same_place(seed=7)
    db = SceneDatabase()
    db.add_all(data.historical)
    bundle = SamePlaceComparisonPipeline(database=db).run(data.query)
    assert "comparable_ratio" in bundle.uncertainty
    assert "occluded_ratio" in bundle.uncertainty
    d = bundle.changes[0].to_dict()
    assert "persistence" in d and "evidence" in d
