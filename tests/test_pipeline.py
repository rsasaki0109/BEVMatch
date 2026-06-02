"""End-to-end pipeline tests for the v0.1 MVP."""

from __future__ import annotations

import numpy as np

from bevmatch import SamePlaceComparisonPipeline, SceneDatabase
from bevmatch.core.datamodel import _wrap_angle
from bevmatch.datasets import make_synthetic_same_place


def _run(seed: int):
    data = make_synthetic_same_place(seed=seed)
    db = SceneDatabase()
    db.add_all(data.historical)
    pipeline = SamePlaceComparisonPipeline(database=db, top_k=5)
    return data, pipeline.run(data.query)


def test_retrieval_finds_correct_place():
    data, bundle = _run(seed=7)
    assert bundle.best_candidate is not None
    assert bundle.best_candidate.place_id == data.gt_place_id


def test_alignment_recovers_pose():
    data, bundle = _run(seed=7)
    a = bundle.alignment
    assert a is not None and a.success
    gt = data.gt_relative_pose
    est = a.relative_pose
    assert abs(est.x - gt.x) < 1.0
    assert abs(est.y - gt.y) < 1.0
    assert abs(np.rad2deg(_wrap_angle(est.yaw - gt.yaw))) < 4.0
    assert a.overlap_ratio > 0.4


def test_change_detection_finds_added_and_removed():
    data, bundle = _run(seed=7)
    added = bundle.added()
    removed = bundle.removed()
    assert len(added) >= 1
    assert len(removed) >= 1

    # Each injected added object should be near some reported "added" centroid.
    for c in data.added_centers:
        d = [np.hypot(c[0] - ch.centroid_xy[0], c[1] - ch.centroid_xy[1]) for ch in added]
        assert min(d) < 2.5, f"added object {c} not matched (min dist {min(d):.2f})"

    for c in data.removed_centers:
        d = [np.hypot(c[0] - ch.centroid_xy[0], c[1] - ch.centroid_xy[1]) for ch in removed]
        assert min(d) < 2.5, f"removed object {c} not matched (min dist {min(d):.2f})"


def test_evidence_bundle_serialises():
    _, bundle = _run(seed=7)
    d = bundle.to_dict()
    assert d["schema_version"] == "0.1"
    assert d["best_candidate"] is not None
    assert "alignment" in d and d["alignment"]["success"] is True


def test_robust_across_seeds():
    # Retrieval should pick the right place across several random layouts.
    for seed in (1, 2, 3, 11, 23):
        data, bundle = _run(seed=seed)
        assert bundle.best_candidate.place_id == data.gt_place_id
