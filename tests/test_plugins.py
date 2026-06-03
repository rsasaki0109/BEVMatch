"""v1.0 plugin manifests (§7.3)."""

from __future__ import annotations

import pytest

from bevmatch.core.registry import CATEGORIES
from bevmatch.plugins import (
    MANIFEST_VERSION,
    PluginManifest,
    get_manifest,
    list_manifests,
    register_manifest,
)


def test_builtin_manifests_present():
    names = {m.name for m in list_manifests()}
    assert {"scan-context", "se2-bev-xcorr", "bev-occupancy-diff",
            "pointcloud-map-validator", "camera-embedding"} <= names


def test_manifest_fields_and_categories():
    for m in list_manifests():
        assert m.category in CATEGORIES, f"{m.name} has bad category {m.category}"
        d = m.to_dict()
        assert d["manifest_version"] == MANIFEST_VERSION
        assert d["name"] == m.name
        assert isinstance(d["input_modality"], list)


def test_scan_context_manifest_declares_rotation_invariance():
    m = get_manifest("scan-context")
    assert "rotation" in m.invariance
    assert "lidar" in m.input_modality
    assert m.uncertainty_support == "score_only"


def test_register_custom_manifest():
    register_manifest(PluginManifest(
        name="test-descriptor", category="descriptor", output_artifact="descriptor",
        input_modality=("lidar",), invariance=("rotation",),
    ))
    assert get_manifest("test-descriptor").category == "descriptor"


def test_unknown_manifest_raises():
    with pytest.raises(KeyError):
        get_manifest("does-not-exist")
