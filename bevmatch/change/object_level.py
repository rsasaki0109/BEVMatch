"""Object-level change detection (§11.3, §11.5 semantic/object-level).

When object instances (class + position) are available, changes are reasoned at
the object level: added / removed / moved / class-changed — richer and more
operationally meaningful than geometry-level occupancy diffs. Query objects are
transformed into the reference frame via the alignment, then greedily matched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.core.datamodel import Pose2D
from bevmatch.scene_graph import ObjectInstance


@dataclass
class ObjectChange:
    category: str  # "added" | "removed" | "moved" | "class_changed"
    object_class: str
    ref_xy: tuple[float, float] | None
    query_xy: tuple[float, float] | None
    displacement_m: float = 0.0
    from_class: str | None = None
    to_class: str | None = None

    def location(self) -> tuple[float, float]:
        return self.ref_xy if self.ref_xy is not None else self.query_xy  # type: ignore

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "object_class": self.object_class,
            "ref_xy": list(self.ref_xy) if self.ref_xy else None,
            "query_xy": list(self.query_xy) if self.query_xy else None,
            "displacement_m": round(self.displacement_m, 3),
            "from_class": self.from_class,
            "to_class": self.to_class,
        }


def detect_object_changes(
    query_objects: list[ObjectInstance],
    reference_objects: list[ObjectInstance],
    relative_pose: Pose2D | None = None,
    match_radius_m: float = 2.5,
    move_threshold_m: float = 1.5,
) -> list[ObjectChange]:
    """Compare object instances; ``relative_pose`` maps query into the ref frame."""
    pose = relative_pose or Pose2D()
    q_moved = [
        ObjectInstance(o.object_id, o.object_class,
                       tuple(pose.transform(np.array([o.xy]))[0]), o.size)
        for o in query_objects
    ]

    matched_ref: set[int] = set()
    changes: list[ObjectChange] = []

    for q in q_moved:
        best, best_d = None, match_radius_m
        for j, r in enumerate(reference_objects):
            if j in matched_ref:
                continue
            d = float(np.hypot(q.xy[0] - r.xy[0], q.xy[1] - r.xy[1]))
            if d < best_d:
                best, best_d = j, d
        if best is None:
            changes.append(ObjectChange("added", q.object_class, None, q.xy))
            continue
        matched_ref.add(best)
        r = reference_objects[best]
        if q.object_class != r.object_class:
            changes.append(ObjectChange("class_changed", r.object_class, r.xy, q.xy,
                                        displacement_m=best_d, from_class=r.object_class,
                                        to_class=q.object_class))
        elif best_d > move_threshold_m:
            changes.append(ObjectChange("moved", r.object_class, r.xy, q.xy, displacement_m=best_d))
        # else: unchanged

    for j, r in enumerate(reference_objects):
        if j not in matched_ref:
            changes.append(ObjectChange("removed", r.object_class, r.xy, None))

    return changes
