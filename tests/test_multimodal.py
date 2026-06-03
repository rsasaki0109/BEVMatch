"""v0.9 multi-modal expansion: sensors, semantic BEV, object change, scene graph, NL."""

from __future__ import annotations

import numpy as np
import pytest

from bevmatch.alignment import SE2Aligner
from bevmatch.change import detect_object_changes
from bevmatch.core.datamodel import Pose2D
from bevmatch.datasets import make_multimodal_places, make_object_change_case
from bevmatch.nl import summarize_map_report, summarize_object_changes
from bevmatch.representations import points_to_semantic_bev, semantic_change_mask
from bevmatch.representations.bev import BEVConfig
from bevmatch.retrieval import ScanContextDescriptor, SceneDatabase
from bevmatch.scene_graph import ObjectInstance, build_scene_graph
from bevmatch.sensors import CameraEmbeddingDescriptor


@pytest.mark.parametrize("modality", ["lidar", "radar", "camera"])
def test_modality_agnostic_retrieval(modality):
    mm = make_multimodal_places(seed=0)
    hist, query, desc = {
        "lidar": (mm.lidar_hist, mm.lidar_query, ScanContextDescriptor()),
        "radar": (mm.radar_hist, mm.radar_query, ScanContextDescriptor()),
        "camera": (mm.camera_hist, mm.camera_query, CameraEmbeddingDescriptor()),
    }[modality]
    db = SceneDatabase(descriptor=desc)
    db.add_all(hist)
    assert db.query(query, top_k=1)[0].place_id == mm.gt_place_id


def test_object_level_change_recovers_all_categories():
    for seed in (4, 5, 6, 7):
        case = make_object_change_case(seed=seed)
        a = SE2Aligner().align(case.query_lidar, case.ref_lidar)
        changes = detect_object_changes(case.query_objects, case.ref_objects, a.relative_pose)
        cats = sorted(c.category for c in changes)
        assert cats == ["added", "class_changed", "moved", "removed"]
        # locations match ground truth
        for gcat, _gcls, gloc in case.gt_object_changes:
            ch = next(c for c in changes if c.category == gcat)
            assert np.hypot(*(np.array(ch.location()) - gloc)) < 2.5


def test_semantic_bev():
    bev = BEVConfig(range_m=10, resolution_m=0.5)
    # two clusters of distinct classes
    pa = np.random.default_rng(0).normal(loc=[3, 3], scale=0.2, size=(30, 2))
    pb = np.random.default_rng(1).normal(loc=[-3, -3], scale=0.2, size=(30, 2))
    pts = np.vstack([pa, pb])
    labels = np.array([0] * 30 + [1] * 30)
    sem = points_to_semantic_bev(pts, labels, n_classes=2, config=bev)
    ca = bev.size // 2 + int(round(3 / 0.5))
    assert sem.label_grid[ca, ca] == 0  # cluster A cell -> class 0
    # changing labels yields a semantic change mask
    sem2 = points_to_semantic_bev(pts, 1 - labels, n_classes=2, config=bev)
    assert semantic_change_mask(sem, sem2).any()


def test_scene_graph():
    objs = [ObjectInstance("a", "pole", (0, 0)), ObjectInstance("b", "vehicle", (3, 0)),
            ObjectInstance("c", "vehicle", (0, 4))]
    g = build_scene_graph(objs, k_near=2)
    assert g.class_counts() == {"pole": 1, "vehicle": 2}
    assert len(g.edges) >= 3
    moved = g.transformed(Pose2D(10, 0, 0))
    assert moved.objects[0].xy == (10.0, 0.0)


def test_nl_summaries():
    case = make_object_change_case(seed=4)
    a = SE2Aligner().align(case.query_lidar, case.ref_lidar)
    text = summarize_object_changes(detect_object_changes(case.query_objects, case.ref_objects, a.relative_pose))
    assert isinstance(text, str) and len(text) > 0
    assert any(w in text for w in ("appeared", "gone", "moved", "changed"))

    from bevmatch.datasets import make_map_validation_case
    from bevmatch.maps import MapValidationReport, PointCloudMapValidator

    mc = make_map_validation_case(seed=1, changed=True)
    rep = MapValidationReport(map_id="map_a", map_version="v1", scene_id="cur")
    rep.add(PointCloudMapValidator().validate(mc.current_frames, mc.pcd_map))
    summary = summarize_map_report(rep)
    assert "map_a" in summary and "issue" in summary
