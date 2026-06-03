"""TF tree handling (§8.3, §16.2 TF-aware).

A minimal SE2 transform tree matching the ROS TF convention
``map -> odom -> base_link -> sensor``. Each edge stores ``T_parent_child`` (the
child frame expressed in the parent); a lookup composes along the tree to give
``T_target_source`` mapping points from the source frame into the target frame.
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Pose2D, Scene


class TFTree:
    """An SE2 transform tree (parent -> child edges)."""

    def __init__(self) -> None:
        # child -> (parent, T_parent_child)
        self._edges: dict[str, tuple[str, Pose2D]] = {}

    def set_transform(self, parent: str, child: str, t_parent_child: Pose2D) -> None:
        """Register the transform mapping ``child``-frame points into ``parent``."""
        self._edges[child] = (parent, t_parent_child)

    def _to_root(self, frame: str) -> Pose2D:
        """Transform mapping ``frame`` points up to the tree root."""
        t = Pose2D()
        seen = set()
        while frame in self._edges:
            if frame in seen:
                raise ValueError(f"cycle in TF tree at {frame!r}")
            seen.add(frame)
            parent, t_parent_child = self._edges[frame]
            t = t_parent_child.compose(t)  # T_root_child = T_root_parent ∘ T_parent_child
            frame = parent
        return t

    def lookup(self, target: str, source: str) -> Pose2D:
        """Return ``T_target_source`` mapping source-frame points into target."""
        return self._to_root(target).inverse().compose(self._to_root(source))

    def transform_points(self, points_xy: np.ndarray, source: str, target: str) -> np.ndarray:
        return self.lookup(target, source).transform(points_xy)

    def transform_scene(self, scene: Scene, source: str, target: str) -> Scene:
        """Return a copy of ``scene`` with its primary observation in ``target``."""
        from bevmatch.core.datamodel import Observation

        obs = scene.primary()
        moved = self.transform_points(obs.xy(), source, target)
        modality = next(iter(scene.observations))
        return Scene(
            scene_id=scene.scene_id,
            observations={modality: Observation(obs.modality, moved, dict(obs.metadata))},
            timestamp=scene.timestamp,
            pose=scene.pose,
            place_id=scene.place_id,
            metadata={**scene.metadata, "frame_id": target},
        )
