"""Top-K scene retriever (§9.4).

A simple in-memory descriptor database. v0.1 uses a brute-force ring-key
nearest-neighbour search; FAISS or a spatial index can drop in later behind the
same interface (Index Backend Plugin, §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.core.datamodel import Candidate, Scene
from bevmatch.retrieval.descriptor import (
    ScanContextConfig,
    ring_key,
    ring_key_distance,
    scan_context,
    sc_alignment_distance,
    shift_to_yaw,
)

DESCRIPTOR_TYPE = "scan-context"


@dataclass
class _Entry:
    scene: Scene
    sc: np.ndarray
    key: np.ndarray


@dataclass
class SceneDatabase:
    """Holds historical scenes and answers Top-K same-place queries."""

    sc_config: ScanContextConfig = field(default_factory=ScanContextConfig)
    _entries: list[_Entry] = field(default_factory=list)

    def add(self, scene: Scene) -> None:
        sc = scan_context(scene.primary().xy(), self.sc_config)
        self._entries.append(_Entry(scene=scene, sc=sc, key=ring_key(sc)))

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

    def query(self, query_scene: Scene, top_k: int = 5, prefilter: int | None = None) -> list[Candidate]:
        """Return Top-K candidates.

        Two-stage (§9.2): a rotation-invariant ring-key prefilter narrows the
        database, then the full Scan-Context column-shift distance scores and
        ranks the shortlist (and yields a coarse yaw as retrieval evidence).
        """
        if not self._entries:
            return []
        q_sc = scan_context(query_scene.primary().xy(), self.sc_config)
        q_key = ring_key(q_sc)

        # Stage 1: ring-key prefilter (cheap, rotation invariant).
        prefilter = prefilter or max(10, top_k * 4)
        coarse = sorted(self._entries, key=lambda e: ring_key_distance(q_key, e.key))
        shortlist = coarse[:prefilter]

        # Stage 2: full Scan-Context distance (discriminative, rotation robust).
        rescored = []
        for e in shortlist:
            dist, shift = sc_alignment_distance(q_sc, e.sc)
            rescored.append((dist, shift, e))
        rescored.sort(key=lambda t: t[0])

        candidates: list[Candidate] = []
        q_t = query_scene.timestamp
        for dist, shift, e in rescored[:top_k]:
            t = e.scene.timestamp
            gap = abs(q_t - t) if (q_t is not None and t is not None) else None
            candidates.append(
                Candidate(
                    scene_id=e.scene.scene_id,
                    place_id=e.scene.place_id,
                    score=1.0 / (1.0 + dist),  # higher is better
                    descriptor_type=DESCRIPTOR_TYPE,
                    reason=f"scan-context column distance={dist:.4f}",
                    temporal_gap=gap,
                    expected_yaw=shift_to_yaw(shift, self.sc_config),
                )
            )
        return candidates
