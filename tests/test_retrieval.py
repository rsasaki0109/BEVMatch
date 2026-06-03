"""v0.2 retrieval framework: metrics, descriptor plugins, index, evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from bevmatch.datasets import make_synthetic_route
from bevmatch.eval import mean_reciprocal_rank, precision_at_k, recall_at_k
from bevmatch.eval.retrieval_eval import evaluate_retrieval
from bevmatch.retrieval import (
    BEVGridDescriptor,
    BruteForceIndex,
    ScanContextDescriptor,
    SceneDatabase,
)

RANKED = [["a", "b", "c"], ["b", "a", "c"], ["c", "c", "a"]]
GT = ["a", "b", "a"]


def test_recall_at_k():
    assert recall_at_k(RANKED, GT, 1) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, GT, 2) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, GT, 3) == pytest.approx(1.0)


def test_precision_at_k():
    assert precision_at_k(RANKED, GT, 2) == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_mrr():
    assert mean_reciprocal_rank(RANKED, GT) == pytest.approx((1 + 1 + 1 / 3) / 3)


def test_brute_force_index():
    idx = BruteForceIndex()
    idx.build(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    out = idx.search(np.array([1.0, 0.1]), k=2)
    assert out[0] == 0  # closest by cosine
    assert len(out) == 2


def test_descriptor_plugins_swap():
    route = make_synthetic_route(seed=1, n_places=6, n_queries=4)
    for desc in (ScanContextDescriptor(), BEVGridDescriptor()):
        db = SceneDatabase(descriptor=desc)
        db.add_all(route.historical)
        cands = db.query(route.queries[0].scene, top_k=3)
        assert cands and cands[0].descriptor_type == desc.name
        assert all(c.place_id is not None for c in cands)


def test_scancontext_outperforms_bevgrid_under_rotation():
    route = make_synthetic_route(seed=0, n_places=12, n_queries=24)

    sc_db = SceneDatabase(descriptor=ScanContextDescriptor())
    sc_db.add_all(route.historical)
    sc = evaluate_retrieval(sc_db, route.queries, ks=(1, 5))

    bev_db = SceneDatabase(descriptor=BEVGridDescriptor())
    bev_db.add_all(route.historical)
    bev = evaluate_retrieval(bev_db, route.queries, ks=(1, 5))

    assert sc.recall_at_k[5] >= 0.9
    assert sc.recall_at_k[1] > bev.recall_at_k[1]
    assert sc.mrr > bev.mrr
