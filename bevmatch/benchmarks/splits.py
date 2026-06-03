"""Reproducible split manifests (§13.6 reproducibility).

Each benchmark dataset is generated deterministically from seeds. A manifest
records the seeds, sizes, and a content fingerprint (a hash over the canonical
ground truth) so a re-run on any machine can be verified identical — the basis
for a trustworthy leaderboard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from bevmatch.benchmarks.cards import CARDS, DatasetCard
from bevmatch.datasets import (
    make_map_validation_case,
    make_synthetic_change_case,
    make_synthetic_route,
)


def _round(arr) -> list:
    return np.round(np.asarray(arr, dtype=float), 3).tolist()


def _route_fingerprint_obj(card: DatasetCard) -> list:
    route = make_synthetic_route(seed=card.seed, n_places=card.n_places, n_queries=card.n_queries)
    return [[q.gt_place_id, _round([q.gt_relative_pose.x, q.gt_relative_pose.y, q.gt_relative_pose.yaw])]
            for q in route.queries]


def _change_fingerprint_obj(card: DatasetCard) -> list:
    obj = []
    for s in (card.seeds or (card.seed,)):
        case = make_synthetic_change_case(seed=s)
        obj.append([s, _round(case.added_centers), _round(case.removed_centers)])
    return obj


def _map_fingerprint_obj(card: DatasetCard) -> list:
    obj = []
    for s in (card.seeds or (card.seed,)):
        for changed in (True, False):
            case = make_map_validation_case(seed=s, changed=changed)
            obj.append([s, changed, [[t, _round(c)] for t, c in case.gt_issues]])
    return obj


_FINGERPRINTERS = {
    "mini-route": _route_fingerprint_obj,
    "mini-change": _change_fingerprint_obj,
    "mini-map": _map_fingerprint_obj,
}


def dataset_fingerprint(card_name: str) -> str:
    """Return a short content hash of the deterministic ground truth."""
    card = CARDS[card_name]
    obj = _FINGERPRINTERS[card_name](card)
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class SplitManifest:
    card: str
    role: str
    seeds: list[int]
    fingerprint: str

    def to_dict(self) -> dict:
        return {"card": self.card, "role": self.role, "seeds": self.seeds,
                "fingerprint": self.fingerprint}


def make_manifest(card_name: str, role: str = "test") -> SplitManifest:
    card = CARDS[card_name]
    seeds = list(card.seeds) if card.seeds else [card.seed]
    return SplitManifest(card=card_name, role=role, seeds=seeds,
                         fingerprint=dataset_fingerprint(card_name))
