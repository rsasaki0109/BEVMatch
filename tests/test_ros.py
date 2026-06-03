"""v0.6 ROS2 integration — pure-Python bridge (no rclpy required)."""

from __future__ import annotations

import numpy as np
import pytest

from bevmatch import SamePlaceComparisonPipeline, SceneDatabase
from bevmatch.core.datamodel import Pose2D
from bevmatch.datasets import make_map_validation_case, make_synthetic_route
from bevmatch.maps import MapValidationReport, PointCloudMapValidator
from bevmatch.ros import (
    BagReplayPipeline,
    DiagnosticLevel,
    LifecycleState,
    TFTree,
    change_markers,
    diagnostics_from_bundle,
    diagnostics_from_map_report,
    issue_markers,
)


def test_tf_tree_chain():
    tf = TFTree()
    tf.set_transform("map", "odom", Pose2D(10, 0, 0))
    tf.set_transform("odom", "base_link", Pose2D(0, 5, np.deg2rad(90)))
    tf.set_transform("base_link", "sensor", Pose2D(1, 0, 0))

    # sensor origin -> map frame
    origin_in_map = tf.transform_points(np.array([[0.0, 0.0]]), "sensor", "map")[0]
    assert np.allclose(origin_in_map, [10.0, 6.0], atol=1e-6)

    # lookup inverse round-trips
    fwd = tf.lookup("map", "sensor")
    back = fwd.inverse()
    pts = np.array([[3.0, -2.0], [0.0, 1.0]])
    assert np.allclose(back.transform(fwd.transform(pts)), pts, atol=1e-9)


def test_tf_transform_scene():
    route = make_synthetic_route(seed=0, n_places=4, n_queries=1)
    scene = route.queries[0].scene
    tf = TFTree()
    tf.set_transform("map", "sensor", Pose2D(5, 0, 0))
    moved = tf.transform_scene(scene, "sensor", "map")
    assert moved.metadata["frame_id"] == "map"
    assert np.allclose(moved.primary().xy(), scene.primary().xy() + [5, 0])


def _bundle():
    route = make_synthetic_route(seed=0, n_places=8, n_queries=1)
    db = SceneDatabase()
    db.add_all(route.historical)
    return SamePlaceComparisonPipeline(database=db).run(route.queries[0].scene)


def test_diagnostics_from_bundle():
    diags = diagnostics_from_bundle(_bundle())
    names = {d.name for d in diags}
    assert {"bevmatch/retrieval", "bevmatch/alignment", "bevmatch/change"} <= names
    align = next(d for d in diags if d.name == "bevmatch/alignment")
    assert align.level == DiagnosticLevel.OK


def test_diagnostics_from_map_report():
    changed = make_map_validation_case(seed=1, changed=True)
    report = MapValidationReport(map_id="map_a", map_version="v1", scene_id="cur")
    report.add(PointCloudMapValidator().validate(changed.current_frames, changed.pcd_map))
    status = diagnostics_from_map_report(report)
    assert status.level in (DiagnosticLevel.WARN, DiagnosticLevel.ERROR)

    fresh = make_map_validation_case(seed=1, changed=False)
    empty = MapValidationReport(map_id="map_a", map_version="v1", scene_id="cur")
    empty.add(PointCloudMapValidator().validate(fresh.current_frames, fresh.pcd_map))
    assert diagnostics_from_map_report(empty).level == DiagnosticLevel.OK


def test_markers():
    bundle = _bundle()
    markers = change_markers(bundle.changes, frame_id="map")
    assert len(markers) == len(bundle.changes)
    for m in markers:
        assert m["frame_id"] == "map"
        assert set(m["color"]) == {"r", "g", "b", "a"}

    case = make_map_validation_case(seed=1, changed=True)
    issues = PointCloudMapValidator().validate(case.current_frames, case.pcd_map)
    im = issue_markers(issues, frame_id="map")
    assert len(im) == len(issues)
    assert all(m["type"] == "CUBE" for m in im)


def test_replay_lifecycle():
    route = make_synthetic_route(seed=0, n_places=6, n_queries=3)
    db = SceneDatabase()
    db.add_all(route.historical)
    rp = BagReplayPipeline(database=db)

    assert rp.state == LifecycleState.UNCONFIGURED
    with pytest.raises(RuntimeError):
        rp.process(route.queries[0].scene)  # not active yet

    rp.configure()
    assert rp.state == LifecycleState.INACTIVE
    rp.activate()
    assert rp.state == LifecycleState.ACTIVE

    out = rp.process(route.queries[0].scene)
    assert out.scene_id == route.queries[0].scene.scene_id
    assert len(out.diagnostics) == 3

    rp.deactivate()
    rp.shutdown()
    assert rp.state == LifecycleState.FINALIZED


def test_replay_auto_configures():
    route = make_synthetic_route(seed=0, n_places=6, n_queries=4)
    db = SceneDatabase()
    db.add_all(route.historical)
    outputs = BagReplayPipeline(database=db).replay([q.scene for q in route.queries])
    assert len(outputs) == 4
    assert all(o.bundle.best_candidate is not None for o in outputs)
