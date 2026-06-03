"""Retrieval metrics (§13.2).

Each query has a list of ranked predicted labels (e.g. place_ids, best first)
and a ground-truth label. v0.2 assumes a single relevant place per query.
"""

from __future__ import annotations


def _hit_rank(ranked: list[str], gt: str) -> int | None:
    """1-based rank of the first correct prediction, or None if absent."""
    for i, label in enumerate(ranked):
        if label == gt:
            return i + 1
    return None


def recall_at_k(ranked_per_query: list[list[str]], gt_per_query: list[str], k: int) -> float:
    """Fraction of queries whose ground truth appears within the top-k."""
    if not ranked_per_query:
        return 0.0
    hits = 0
    for ranked, gt in zip(ranked_per_query, gt_per_query):
        rank = _hit_rank(ranked[:k], gt)
        hits += int(rank is not None)
    return hits / len(ranked_per_query)


def precision_at_k(ranked_per_query: list[list[str]], gt_per_query: list[str], k: int) -> float:
    """Mean fraction of correct labels among the top-k (single relevant place)."""
    if not ranked_per_query:
        return 0.0
    total = 0.0
    for ranked, gt in zip(ranked_per_query, gt_per_query):
        correct = sum(1 for label in ranked[:k] if label == gt)
        total += correct / k
    return total / len(ranked_per_query)


def mean_reciprocal_rank(ranked_per_query: list[list[str]], gt_per_query: list[str]) -> float:
    """Mean of 1/rank of the first correct prediction (0 if not retrieved)."""
    if not ranked_per_query:
        return 0.0
    total = 0.0
    for ranked, gt in zip(ranked_per_query, gt_per_query):
        rank = _hit_rank(ranked, gt)
        total += 1.0 / rank if rank else 0.0
    return total / len(ranked_per_query)
