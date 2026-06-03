"""Dataset cards (§14.4, §20.6) — metadata describing each benchmark dataset.

A card documents what a benchmark contains and how it is generated so results
are interpretable and reproducible. v0.8 ships synthetic mini-benchmarks; real
dataset adapters (KITTI, MulRan, ...) register cards with the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetCard:
    name: str
    description: str
    modality: str  # "lidar", "camera", ...
    tasks: tuple[str, ...]  # retrieval / alignment / change / map_validation
    seed: int
    conditions: tuple[str, ...] = ()
    n_places: int = 0
    n_queries: int = 0
    seeds: tuple[int, ...] = ()  # extra seeds for aggregated (change/map) benchmarks
    license: str = "synthetic (BEVMatch, Apache-2.0)"
    source: str = "bevmatch.datasets.synthetic"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "modality": self.modality,
            "tasks": list(self.tasks),
            "seed": self.seed,
            "seeds": list(self.seeds),
            "n_places": self.n_places,
            "n_queries": self.n_queries,
            "conditions": list(self.conditions),
            "license": self.license,
            "source": self.source,
        }


CARDS: dict[str, DatasetCard] = {
    "mini-route": DatasetCard(
        name="mini-route",
        description="A historical route of distinct places revisited as queries "
                    "under viewpoint yaw, translation and small scene changes.",
        modality="lidar",
        tasks=("retrieval", "alignment"),
        seed=0,
        n_places=12,
        n_queries=24,
        conditions=("yaw<=40deg", "translation<=3m", "2 added", "2 removed"),
    ),
    "mini-change": DatasetCard(
        name="mini-change",
        description="Multi-frame query bursts over one place with stable added/"
                    "removed objects and moving dynamic objects.",
        modality="lidar",
        tasks=("change",),
        seed=3,
        seeds=(3, 7, 8, 11),
        conditions=("4-frame burst", "2 added", "2 removed", "2 dynamic/frame"),
    ),
    "mini-map": DatasetCard(
        name="mini-map",
        description="A point cloud / occupancy / vector map validated against "
                    "current observations (changed vs fresh world).",
        modality="lidar",
        tasks=("map_validation",),
        seed=1,
        seeds=(1, 2, 3, 5),
        conditions=("1 new obstacle", "1 missing structure", "1 unobserved element"),
    ),
}


def get_card(name: str) -> DatasetCard:
    if name not in CARDS:
        raise KeyError(f"unknown dataset card {name!r}; known: {sorted(CARDS)}")
    return CARDS[name]
