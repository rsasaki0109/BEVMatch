"""Object-level scene graph (§5.4 object-level scene graph, §0.9 prototype).

A Scene (or map) reduced to typed object nodes with spatial-relation edges. This
is a structured representation that object-level change reasoning and
natural-language summaries build on, and a step toward map graphs / VLM grounding
(§19).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.core.datamodel import Pose2D


@dataclass
class ObjectInstance:
    object_id: str
    object_class: str
    xy: tuple[float, float]
    size: float = 1.0

    def to_dict(self) -> dict:
        return {"object_id": self.object_id, "object_class": self.object_class,
                "xy": [round(self.xy[0], 3), round(self.xy[1], 3)], "size": round(self.size, 3)}


@dataclass
class SceneEdge:
    source: str
    target: str
    relation: str  # "near"
    distance: float


@dataclass
class SceneGraph:
    objects: list[ObjectInstance] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)

    def transformed(self, pose: Pose2D) -> "SceneGraph":
        """Return a copy with object positions mapped by ``pose``."""
        if not self.objects:
            return SceneGraph([], [])
        xy = pose.transform(np.array([o.xy for o in self.objects]))
        objs = [ObjectInstance(o.object_id, o.object_class, (float(p[0]), float(p[1])), o.size)
                for o, p in zip(self.objects, xy)]
        return SceneGraph(objs, list(self.edges))

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.object_class] = counts.get(o.object_class, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "edges": [{"source": e.source, "target": e.target, "relation": e.relation,
                       "distance": round(e.distance, 3)} for e in self.edges],
        }


def build_scene_graph(objects: list[ObjectInstance], k_near: int = 3) -> SceneGraph:
    """Build a graph with k-nearest 'near' relations between objects."""
    edges: list[SceneEdge] = []
    if len(objects) >= 2:
        pos = np.array([o.xy for o in objects])
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        for i, obj in enumerate(objects):
            nearest = np.argsort(d[i])[:k_near]
            for j in nearest:
                if np.isfinite(d[i, j]):
                    edges.append(SceneEdge(obj.object_id, objects[j].object_id, "near", float(d[i, j])))
    return SceneGraph(objects=list(objects), edges=edges)
