"""v1.0 stable artifact schema."""

from __future__ import annotations

import pytest

from bevmatch import SamePlaceComparisonPipeline, SceneDatabase, __version__
from bevmatch.datasets import make_synthetic_same_place
from bevmatch.schema import (
    ARTIFACT_SCHEMA_VERSION,
    KNOWN_ARTIFACTS,
    envelope,
    is_compatible,
    require_compatible,
    validate_artifact,
)


def test_versions():
    assert __version__.startswith("1.")  # package version (>= 1.0)
    assert ARTIFACT_SCHEMA_VERSION == "1.0"  # artifact schema is the stable contract


def test_is_compatible_major():
    assert is_compatible("1.0")
    assert is_compatible("1.7")  # additive within major
    assert not is_compatible("2.0")
    assert not is_compatible("0.9")


def test_require_compatible_raises():
    require_compatible("1.3")  # ok
    with pytest.raises(ValueError):
        require_compatible("2.0")


def test_validate_artifact():
    assert validate_artifact("change_hypothesis",
                             {"category": "added", "centroid_xy": [0, 0],
                              "confidence": 1.0, "persistence": 1.0}) == []
    problems = validate_artifact("change_hypothesis", {"category": "added"})
    assert problems and any("centroid_xy" in p for p in problems)
    assert validate_artifact("nope", {}) == ["unknown artifact 'nope'"]


def test_envelope():
    env = envelope("change_hypothesis", {"category": "added"})
    assert env["artifact"] == "change_hypothesis"
    assert env["schema_version"] == ARTIFACT_SCHEMA_VERSION
    with pytest.raises(ValueError):
        envelope("unknown", {})


def test_real_bundle_is_valid():
    data = make_synthetic_same_place(seed=7)
    db = SceneDatabase()
    db.add_all(data.historical)
    bundle = SamePlaceComparisonPipeline(database=db).run(data.query)
    d = bundle.to_dict()
    assert d["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert validate_artifact("comparison_evidence_bundle", d) == []
    assert "comparison_evidence_bundle" in KNOWN_ARTIFACTS
