"""Top-K scene retriever (§9.2, §9.4).

Two-stage retrieval: an index backend prefilters by the descriptor's
rotation-invariant vector, then the descriptor's own distance rescores the
shortlist (and may yield a yaw estimate as retrieval evidence). Descriptor and
index are pluggable (§7.2); defaults reproduce the v0.1 Scan-Context behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.core.datamodel import Candidate, Scene
from bevmatch.retrieval.base import DescriptorCode, GlobalDescriptor
from bevmatch.retrieval.descriptors import ScanContextDescriptor
from bevmatch.retrieval.index import BruteForceIndex, IndexBackend


@dataclass
class _Entry:
    scene: Scene
    code: DescriptorCode


@dataclass
class SceneDatabase:
    """Holds historical scenes and answers Top-K same-place queries."""

    descriptor: GlobalDescriptor = field(default_factory=ScanContextDescriptor)
    index: IndexBackend = field(default_factory=BruteForceIndex)
    prefilter: int | None = None  # shortlist size for stage 1 (None = auto)
    _entries: list[_Entry] = field(default_factory=list)
    _dirty: bool = False

    def add(self, scene: Scene) -> None:
        self._entries.append(_Entry(scene=scene, code=self.descriptor.extract(scene)))
        self._dirty = True

    def add_all(self, scenes: list[Scene]) -> None:
        for s in scenes:
            self.add(s)

    def __len__(self) -> int:
        return len(self._entries)

    def get_scene(self, scene_id: str) -> Scene:
        for e in self._entries:
            if e.scene.scene_id == scene_id:
                return e.scene
        raise KeyError(scene_id)

    def scene_by_place(self, place_id: str) -> Scene:
        """Return the first scene registered for ``place_id`` (eval helper)."""
        for e in self._entries:
            if e.scene.place_id == place_id:
                return e.scene
        raise KeyError(place_id)

    def _ensure_index(self) -> None:
        if self._dirty or not self._entries:
            vectors = np.array([e.code.vector for e in self._entries], dtype=float)
            self.index.build(vectors)
            self._dirty = False

    def query(self, query_scene: Scene, top_k: int = 5) -> list[Candidate]:
        if not self._entries:
            return []
        self._ensure_index()

        q_code = self.descriptor.extract(query_scene)

        # Stage 1: index prefilter on the rotation-invariant vector.
        prefilter = self.prefilter or max(10, top_k * 4)
        shortlist_idx = self.index.search(q_code.vector, prefilter)
        if not shortlist_idx:
            shortlist_idx = list(range(len(self._entries)))

        # Stage 2: descriptor distance rescoring on the shortlist.
        rescored = []
        for i in shortlist_idx:
            e = self._entries[i]
            dist, yaw = self.descriptor.distance(q_code, e.code)
            rescored.append((dist, yaw, e))
        rescored.sort(key=lambda t: t[0])

        candidates: list[Candidate] = []
        q_t = query_scene.timestamp
        for dist, yaw, e in rescored[:top_k]:
            t = e.scene.timestamp
            gap = abs(q_t - t) if (q_t is not None and t is not None) else None
            candidates.append(
                Candidate(
                    scene_id=e.scene.scene_id,
                    place_id=e.scene.place_id,
                    score=1.0 / (1.0 + dist),
                    descriptor_type=self.descriptor.name,
                    reason=f"{self.descriptor.name} distance={dist:.4f}",
                    temporal_gap=gap,
                    expected_yaw=yaw,
                )
            )
        return candidates
