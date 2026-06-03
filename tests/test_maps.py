"""v0.5 map validation: validators, severity, report export, metrics."""

from __future__ import annotations

import json

import numpy as np

from bevmatch.datasets import make_map_validation_case
from bevmatch.eval import issue_prf, match_issues, review_burden
from bevmatch.maps import (
    MapValidationReport,
    OccupancyMapValidator,
    PointCloudMapValidator,
    Severity,
    VectorMapValidator,
    assess_severity,
)


def test_severity_schema():
    assert assess_severity("new_static_obstacle") == Severity.HIGH
    # weak evidence demotes
    assert assess_severity("new_static_obstacle", confidence=0.2) == Severity.MEDIUM
    # large, confident, persistent evidence promotes
    assert assess_severity("new_static_obstacle", area_m2=10, confidence=0.9, persistence=1.0) == Severity.CRITICAL
    assert Severity.CRITICAL > Severity.HIGH > Severity.MEDIUM


def test_pointcloud_validator_changed():
    case = make_map_validation_case(seed=1, changed=True)
    issues = PointCloudMapValidator().validate(case.current_frames, case.pcd_map)
    types = {i.issue_type for i in issues}
    assert "new_static_obstacle" in types
    assert "missing_static_structure" in types
    # every issue matches a ground-truth change of the same type
    prf = issue_prf(issues, [g for g in case.gt_issues if g[0] != "map_element_unobserved"])
    assert prf.precision == 1.0 and prf.recall == 1.0


def test_pointcloud_validator_fresh_is_clean():
    case = make_map_validation_case(seed=1, changed=False)
    issues = PointCloudMapValidator().validate(case.current_frames, case.pcd_map)
    assert issues == []


def test_pointcloud_validator_alignment_failure():
    a = make_map_validation_case(seed=1)
    b = make_map_validation_case(seed=99)
    issues = PointCloudMapValidator().validate(a.current_frames, b.pcd_map)
    assert len(issues) == 1
    assert issues[0].issue_type == "localization_risk_region"
    assert issues[0].severity == Severity.HIGH


def test_occupancy_validator():
    case = make_map_validation_case(seed=1, changed=True)
    issues = OccupancyMapValidator().validate(case.current_map_frame, case.occ_map)
    types = {i.issue_type for i in issues}
    assert "new_static_obstacle" in types
    assert "map_stale_region" in types


def test_vector_validator():
    changed = make_map_validation_case(seed=1, changed=True)
    issues = VectorMapValidator().validate(changed.current_map_frame, changed.vmap)
    assert any(i.issue_type == "map_element_unobserved" for i in issues)
    # the supported lane-boundary element must NOT be flagged
    fresh = make_map_validation_case(seed=1, changed=False)
    assert VectorMapValidator().validate(fresh.current_map_frame, fresh.vmap) == []


def test_report_export_and_priority(tmp_path):
    case = make_map_validation_case(seed=1, changed=True)
    report = MapValidationReport(map_id="map_a", map_version="v1", scene_id="cur_f0")
    report.add(PointCloudMapValidator().validate(case.current_frames, case.pcd_map))
    report.add(VectorMapValidator().validate(case.current_map_frame, case.vmap))

    prioritized = report.prioritized()
    sev = [int(i.severity) for i in prioritized]
    assert sev == sorted(sev, reverse=True)  # severity-descending
    assert all(i.issue_id for i in report.issues)  # IDs assigned

    d = report.to_dict()
    assert d["n_issues"] == len(report.issues)
    assert "severity_counts" in d
    md = report.to_markdown()
    assert "Map Validation Review" in md

    jp = report.save_json(tmp_path / "r.json")
    mp = report.save_markdown(tmp_path / "r.md")
    assert json.loads(jp.read_text())["map_id"] == "map_a"
    assert mp.read_text().startswith("# Map Validation Review")


def test_map_metrics():
    from bevmatch.maps import MapValidationIssue

    reported = [
        MapValidationIssue("new_static_obstacle", Severity.HIGH, (0.0, 0.0)),
        MapValidationIssue("missing_static_structure", Severity.MEDIUM, (10.0, 0.0)),
    ]
    gt = [("new_static_obstacle", np.array([0.2, 0.0])), ("missing_static_structure", np.array([20.0, 0.0]))]
    tp, fp, fn = match_issues(reported, gt, radius=2.0)
    assert (tp, fp, fn) == (1, 1, 1)
    assert review_burden(4, 2) == 2.0
