"""Retrieval evaluation harness (§13.1, §13.2).

Runs a set of labelled queries against a SceneDatabase and reports Recall@K,
Precision@K and MRR — the same recipe a KITTI/MulRan-style adapter plugs into
(§9, §14.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bevmatch.datasets.synthetic import RouteQuery
from bevmatch.eval.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from bevmatch.retrieval.retriever import SceneDatabase


@dataclass
class RetrievalReport:
    descriptor: str
    index: str
    n_queries: int
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "descriptor": self.descriptor,
            "index": self.index,
            "n_queries": self.n_queries,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "mrr": round(self.mrr, 4),
        }

    def as_row(self, ks: tuple[int, ...]) -> str:
        cells = [f"{self.descriptor:<14}", f"{self.index:<12}"]
        cells += [f"{self.recall_at_k.get(k, 0.0):.3f}" for k in ks]
        cells.append(f"{self.mrr:.3f}")
        return "  ".join(cells)


def evaluate_retrieval(
    database: SceneDatabase,
    queries: list[RouteQuery],
    ks: tuple[int, ...] = (1, 5),
) -> RetrievalReport:
    """Evaluate a built database over labelled route queries."""
    top_k = max(ks)
    ranked_per_query: list[list[str]] = []
    gt_per_query: list[str] = []
    for rq in queries:
        candidates = database.query(rq.scene, top_k=top_k)
        ranked_per_query.append([c.place_id for c in candidates])
        gt_per_query.append(rq.gt_place_id)

    report = RetrievalReport(
        descriptor=database.descriptor.name,
        index=database.index.name,
        n_queries=len(queries),
        mrr=mean_reciprocal_rank(ranked_per_query, gt_per_query),
    )
    for k in ks:
        report.recall_at_k[k] = recall_at_k(ranked_per_query, gt_per_query, k)
        report.precision_at_k[k] = precision_at_k(ranked_per_query, gt_per_query, k)
    return report


def format_table(reports: list[RetrievalReport], ks: tuple[int, ...] = (1, 5)) -> str:
    """Render reports as a leaderboard-style text table."""
    header = ["descriptor".ljust(14), "index".ljust(12)]
    header += [f"R@{k}" for k in ks]
    header.append("MRR")
    lines = ["  ".join(header), "-" * (len(header) * 8)]
    lines += [r.as_row(ks) for r in reports]
    return "\n".join(lines)
