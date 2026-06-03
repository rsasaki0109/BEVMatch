"""Visualization markers (§15.3 ROS visualization, §15.4 Foxglove).

Pure-Python marker dicts (mirroring visualization_msgs/Marker) for change
hypotheses and map issues. The rclpy node converts them to a MarkerArray; the
offline path writes them as JSON for a Foxglove/web viewer.
"""

from __future__ import annotations

from typing import Any

_CHANGE_COLOR = {
    "added": (0.9, 0.1, 0.1, 0.9),    # red
    "removed": (0.1, 0.3, 0.9, 0.9),  # blue
    "dynamic": (1.0, 0.6, 0.0, 0.7),  # orange
}

# Severity -> RGBA (info..critical)
_SEVERITY_COLOR = {
    "info": (0.6, 0.6, 0.6, 0.7),
    "low": (0.3, 0.7, 0.3, 0.8),
    "medium": (1.0, 0.7, 0.0, 0.9),
    "high": (0.95, 0.3, 0.1, 0.95),
    "critical": (0.8, 0.0, 0.0, 1.0),
}


def _marker(ns: str, mid: int, frame_id: str, x: float, y: float, size: float,
            color: tuple[float, float, float, float], text: str, mtype: str = "SPHERE") -> dict[str, Any]:
    return {
        "ns": ns,
        "id": mid,
        "frame_id": frame_id,
        "type": mtype,
        "action": "ADD",
        "pose": {"position": {"x": float(x), "y": float(y), "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "scale": {"x": float(size), "y": float(size), "z": float(size)},
        "color": {"r": color[0], "g": color[1], "b": color[2], "a": color[3]},
        "text": text,
    }


def _diameter(area_m2: float, floor: float = 1.0) -> float:
    return max(floor, (area_m2 ** 0.5))


def change_markers(changes, frame_id: str = "map", ns: str = "bevmatch/changes") -> list[dict[str, Any]]:
    markers = []
    for i, ch in enumerate(changes):
        color = _CHANGE_COLOR.get(ch.category, (1.0, 0.0, 1.0, 0.8))
        text = f"{ch.category} ({ch.confidence:.2f})"
        markers.append(_marker(ns, i, frame_id, ch.centroid_xy[0], ch.centroid_xy[1],
                               _diameter(ch.area_m2), color, text))
    return markers


def issue_markers(issues, frame_id: str = "map", ns: str = "bevmatch/map_issues") -> list[dict[str, Any]]:
    markers = []
    for i, issue in enumerate(issues):
        color = _SEVERITY_COLOR.get(issue.severity.label, (1.0, 0.0, 1.0, 0.8))
        text = f"{issue.issue_type} [{issue.severity.label}]"
        markers.append(_marker(ns, i, frame_id, issue.centroid_xy[0], issue.centroid_xy[1],
                               _diameter(issue.area_m2), color, text, mtype="CUBE"))
    return markers
