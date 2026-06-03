"""Issue severity schema and recommended actions (§12.4).

Severity reflects operational impact, scaled by how confident and how large the
evidence is. It is deliberately explainable (a small rule set) so map engineers
can trust and tune it.
"""

from __future__ import annotations

from enum import IntEnum


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()


# Base severity per issue type (before evidence scaling).
_BASE = {
    "new_static_obstacle": Severity.HIGH,  # something blocks a previously-clear map
    "missing_static_structure": Severity.MEDIUM,  # localization feature lost
    "map_stale_region": Severity.MEDIUM,
    "localization_risk_region": Severity.HIGH,  # cannot align observation to map
    "map_element_unobserved": Severity.MEDIUM,  # vector element not supported by obs
    "lane_geometry_mismatch": Severity.HIGH,
}

_RECOMMENDED = {
    "new_static_obstacle": "Inspect area; add obstacle to map or re-survey if persistent.",
    "missing_static_structure": "Verify structure removal; update point cloud / feature map.",
    "map_stale_region": "Flag region for re-mapping; reduce localization trust here.",
    "localization_risk_region": "Manual check: map may be outdated or place mismatched.",
    "map_element_unobserved": "Confirm vector element still exists; correct or remove from map.",
    "lane_geometry_mismatch": "Re-survey lane geometry; update Lanelet2 / vector map.",
}


def recommended_action(issue_type: str) -> str:
    return _RECOMMENDED.get(issue_type, "Manual review required.")


def assess_severity(
    issue_type: str,
    area_m2: float = 0.0,
    confidence: float = 1.0,
    persistence: float = 1.0,
) -> Severity:
    """Assign severity from the base level, scaled by evidence strength."""
    base = _BASE.get(issue_type, Severity.MEDIUM)
    level = int(base)

    # Weak evidence demotes; strong, large, persistent evidence promotes.
    if confidence < 0.3 or persistence < 0.5:
        level -= 1
    if area_m2 >= 6.0 and confidence >= 0.6 and persistence >= 0.8:
        level += 1

    return Severity(int(max(Severity.INFO, min(Severity.CRITICAL, level))))
