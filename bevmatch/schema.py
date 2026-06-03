"""Stable artifact schema and compatibility policy (§21 v1.0, §20.6).

BEVMatch's stability contract is at the *artifact* level: the JSON shape of the
Comparison Evidence Bundle and the other artifacts. Versioning is semantic on
the schema: same major = compatible (fields may be added), major bump = breaking.
Consumers should check ``is_compatible`` before relying on an artifact.
"""

from __future__ import annotations

ARTIFACT_SCHEMA_VERSION = "1.0"

# Required top-level keys per artifact (additive changes keep the major version).
_REQUIRED_KEYS: dict[str, set[str]] = {
    "comparison_evidence_bundle": {
        "schema_version", "query_scene_id", "candidates", "best_candidate",
        "alignment", "changes", "provenance", "uncertainty",
    },
    "alignment_hypothesis": {
        "relative_pose", "overlap_ratio", "inlier_ratio", "success",
        "failure_class", "rmse_m",
    },
    "change_hypothesis": {"category", "centroid_xy", "confidence", "persistence"},
    "map_validation_issue": {
        "issue_type", "severity", "centroid_xy", "recommended_action", "review_status",
    },
    "map_validation_report": {"map_id", "map_version", "issues", "severity_counts"},
    "initial_pose_candidate": {"pose", "score", "covariance_diag"},
}

KNOWN_ARTIFACTS = frozenset(_REQUIRED_KEYS)


def schema_version() -> str:
    return ARTIFACT_SCHEMA_VERSION


def is_compatible(version: str, against: str = ARTIFACT_SCHEMA_VERSION) -> bool:
    """Same major version = compatible (additive-only changes within a major)."""
    try:
        return version.split(".")[0] == against.split(".")[0]
    except (AttributeError, IndexError):
        return False


def require_compatible(version: str) -> None:
    if not is_compatible(version):
        raise ValueError(
            f"artifact schema {version!r} is incompatible with "
            f"{ARTIFACT_SCHEMA_VERSION!r} (major version mismatch)"
        )


def envelope(artifact: str, payload: dict) -> dict:
    """Wrap a payload in a self-describing, versioned envelope."""
    if artifact not in KNOWN_ARTIFACTS:
        raise ValueError(f"unknown artifact {artifact!r}; known: {sorted(KNOWN_ARTIFACTS)}")
    return {"artifact": artifact, "schema_version": ARTIFACT_SCHEMA_VERSION, "payload": payload}


def validate_artifact(artifact: str, payload: dict) -> list[str]:
    """Return a list of schema problems (empty list = valid)."""
    if artifact not in _REQUIRED_KEYS:
        return [f"unknown artifact {artifact!r}"]
    missing = _REQUIRED_KEYS[artifact] - set(payload)
    return [f"missing required key: {k}" for k in sorted(missing)]
