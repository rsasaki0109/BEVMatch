"""v0.7 Autoware / Nav2 adapters."""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Pose2D
from bevmatch.datasets import make_map_validation_case, make_synthetic_route
from bevmatch.eval.alignment_eval import pose_errors
from bevmatch.integrations import (
    AutowareAdapter,
    Nav2Adapter,
    OccupancyGrid,
    covariance_from_alignment,
    relocalization_candidates,
)
from bevmatch.maps import OccupancyMap
from bevmatch.ros.diagnostics import DiagnosticLevel
from bevmatch.retrieval import SceneDatabase


def _route_db(seed=0, n_places=10, n_queries=5):
    route = make_synthetic_route(seed=seed, n_places=n_places, n_queries=n_queries)
    db = SceneDatabase()
    db.add_all(route.historical)
    return route, db


def test_initial_pose_candidates_recover_map_pose():
    route, db = _route_db()
    for q in route.queries[:3]:
        cands = relocalization_candidates(q.scene, db, top_k=3)
        assert cands and cands[0].place_id == q.gt_place_id
        t, r = pose_errors(cands[0].pose, q.gt_relative_pose)
        assert t < 1.0 and r < 4.0
        # candidates are ranked by score
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)


def test_covariance_marks_out_of_plane_unobservable():
    route, db = _route_db()
    from bevmatch.alignment import SE2Aligner

    a = SE2Aligner().align(route.queries[0].scene, db.scene_by_place(route.queries[0].gt_place_id))
    cov = covariance_from_alignment(a)
    assert len(cov) == 36
    assert cov[0] > 0 and cov[7] > 0 and cov[35] > 0  # x, y, yaw
    assert cov[14] > 1e5 and cov[21] > 1e5 and cov[28] > 1e5  # z, roll, pitch huge


def test_localization_health():
    route, db = _route_db()
    aw = AutowareAdapter()
    q = route.queries[0]
    good = aw.localization_health(q.scene, db, q.gt_relative_pose)
    bad = aw.localization_health(q.scene, db, q.gt_relative_pose.compose(Pose2D(10, 10, 0)))
    assert good.level == DiagnosticLevel.OK
    assert bad.level == DiagnosticLevel.ERROR
    assert bad.trans_error_m > good.trans_error_m


def test_autoware_map_freshness():
    aw = AutowareAdapter()
    case = make_map_validation_case(seed=1, changed=True)
    report = aw.pointcloud_map_freshness(case.current_frames, case.pcd_map)
    types = {i.issue_type for i in report.issues}
    assert "new_static_obstacle" in types
    assert "missing_static_structure" in types
    # a fresh map yields no freshness issues
    fresh = make_map_validation_case(seed=1, changed=False)
    assert aw.pointcloud_map_freshness(fresh.current_frames, fresh.pcd_map).issues == []


def test_occupancy_grid_roundtrip():
    case = make_map_validation_case(seed=1, changed=True)
    grid = OccupancyGrid.from_occupancy_map(case.occ_map)
    assert grid.width == case.occ_map.bev.size
    assert grid.origin.x == -case.occ_map.bev.center * case.occ_map.bev.resolution_m
    back = grid.to_occupancy_map()
    assert np.array_equal(back.occupied, case.occ_map.occupied)
    assert np.array_equal(back.known, case.occ_map.known)


def test_nav2_occupancy_staleness_and_annotations():
    nav = Nav2Adapter()
    case = make_map_validation_case(seed=1, changed=True)
    grid = OccupancyGrid.from_occupancy_map(case.occ_map)
    issues = nav.occupancy_staleness(case.current_map_frame, grid)
    types = {i.issue_type for i in issues}
    assert "new_static_obstacle" in types
    assert "map_stale_region" in types
    blocked = nav.changed_area_annotations(issues)
    assert all(i.issue_type == "new_static_obstacle" for i in blocked)
    assert len(blocked) >= 1
