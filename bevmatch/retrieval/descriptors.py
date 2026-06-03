"""Descriptor baselines (§9.5).

- ``ScanContextDescriptor``: rotation-invariant ring-key prefilter + full
  Scan-Context column-shift distance (rotation robust, yields a yaw estimate).
- ``BEVGridDescriptor``: a coarse flattened BEV occupancy grid with cosine
  distance. It is *not* rotation invariant — a deliberate contrast baseline that
  shows, in the benchmark, why rotation handling matters under revisit yaw.
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Scene
from bevmatch.representations.bev import BEVConfig, points_to_bev
from bevmatch.retrieval.base import DescriptorCode, GlobalDescriptor
from bevmatch.retrieval.descriptor import (
    ScanContextConfig,
    ring_key,
    scan_context,
    sc_alignment_distance,
    shift_to_yaw,
)


class ScanContextDescriptor(GlobalDescriptor):
    name = "scan-context"

    def __init__(self, config: ScanContextConfig | None = None) -> None:
        self.config = config or ScanContextConfig()

    def extract(self, scene: Scene) -> DescriptorCode:
        sc = scan_context(scene.primary().xy(), self.config)
        return DescriptorCode(vector=ring_key(sc), payload=sc)

    def distance(self, query: DescriptorCode, ref: DescriptorCode) -> tuple[float, float | None]:
        dist, shift = sc_alignment_distance(query.payload, ref.payload)
        return dist, shift_to_yaw(shift, self.config)


class BEVGridDescriptor(GlobalDescriptor):
    name = "bev-grid"

    def __init__(self, range_m: float = 30.0, cells: int = 24) -> None:
        # coarse grid: resolution chosen so the grid is ``cells`` x ``cells``
        self.config = BEVConfig(range_m=range_m, resolution_m=2 * range_m / cells)

    def _vector(self, scene: Scene) -> np.ndarray:
        grid = points_to_bev(scene.primary().xy(), self.config).occupied(0.5).astype(float)
        return grid.flatten()

    def extract(self, scene: Scene) -> DescriptorCode:
        v = self._vector(scene)
        return DescriptorCode(vector=v, payload=v)

    def distance(self, query: DescriptorCode, ref: DescriptorCode) -> tuple[float, float | None]:
        a, b = query.vector, ref.vector
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        cos = float(a @ b) / denom
        return 1.0 - cos, None
