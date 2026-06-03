"""Natural-language report summaries (§0.9 NL report, §19.2 Natural Language Map QA).

Template-based summaries of evidence — no model dependency, but structured so a
VLM/LLM can be swapped in to ground answers on BEVMatch artifacts (§19.3).
"""

from __future__ import annotations

import numpy as np


def _bearing(xy) -> str:
    angle = np.degrees(np.arctan2(xy[1], xy[0]))
    dirs = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    idx = int(((angle + 22.5) % 360) // 45)
    rng = float(np.hypot(xy[0], xy[1]))
    return f"{rng:.0f} m to the {dirs[idx]}"


def summarize_bundle(bundle) -> str:
    """One-paragraph summary of a comparison evidence bundle."""
    bc = bundle.best_candidate
    if bc is None:
        return "No matching place was retrieved for the query."
    parts = [f"The query matches {bc.place_id} (score {bc.score:.2f})."]
    a = bundle.alignment
    if a is not None and a.success:
        parts.append(
            f"It aligned with {a.overlap_ratio:.0%} overlap "
            f"({a.relative_pose.x:+.1f} m, {a.relative_pose.y:+.1f} m, "
            f"{np.degrees(a.relative_pose.yaw):+.0f}°)."
        )
    elif a is not None:
        return parts[0] + f" Alignment failed ({a.failure_class}); changes are not asserted."
    n_add, n_rem = len(bundle.added()), len(bundle.removed())
    if n_add or n_rem:
        parts.append(f"{n_add} object(s) appeared and {n_rem} disappeared since the reference.")
    else:
        parts.append("No changes were detected in the comparable region.")
    return " ".join(parts)


def summarize_object_changes(changes) -> str:
    """Summarise object-level changes in plain language."""
    if not changes:
        return "No object-level changes were detected."
    sentences = []
    for c in changes:
        loc = _bearing(c.location())
        if c.category == "added":
            sentences.append(f"A new {c.object_class} appeared {loc}.")
        elif c.category == "removed":
            sentences.append(f"A {c.object_class} is gone ({loc}).")
        elif c.category == "moved":
            sentences.append(f"A {c.object_class} moved {c.displacement_m:.1f} m ({loc}).")
        elif c.category == "class_changed":
            sentences.append(f"An object {loc} changed from {c.from_class} to {c.to_class}.")
    return " ".join(sentences)


def summarize_map_report(report) -> str:
    """Summarise a map validation report for an operator."""
    if not report.issues:
        return f"Map {report.map_id} ({report.map_version}) still matches the current world; no issues."
    counts = report.severity_counts()
    high = counts.get("high", 0) + counts.get("critical", 0)
    head = (f"Map {report.map_id} has {len(report.issues)} issue(s) "
            f"({high} high/critical).")
    top = report.prioritized()[0]
    return head + (f" Most urgent: {top.issue_type} ({top.severity.label}) "
                   f"{_bearing(top.centroid_xy)} — {top.recommended_action}")
