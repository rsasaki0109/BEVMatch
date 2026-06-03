"""Retrieval benchmark (§9, §13.2): compare descriptors on the same route.

    python examples/run_retrieval_eval.py

Builds one synthetic route, then evaluates each descriptor plugin under the same
index backend. Demonstrates the framework value: swapping a descriptor changes
Recall@K/MRR on a shared benchmark — and why rotation invariance matters under
revisit yaw.
"""

from __future__ import annotations

import json
from pathlib import Path

from bevmatch.datasets import make_synthetic_route
from bevmatch.eval.retrieval_eval import evaluate_retrieval, format_table
from bevmatch.retrieval import (
    BEVGridDescriptor,
    BruteForceIndex,
    ScanContextDescriptor,
    SceneDatabase,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "out"
KS = (1, 5)


def main() -> None:
    route = make_synthetic_route(seed=0, n_places=12, n_queries=24)
    print(f"Route: {len(route.historical)} historical places, {len(route.queries)} queries\n")

    descriptors = [ScanContextDescriptor(), BEVGridDescriptor()]
    reports = []
    for desc in descriptors:
        db = SceneDatabase(descriptor=desc, index=BruteForceIndex())
        db.add_all(route.historical)
        reports.append(evaluate_retrieval(db, route.queries, ks=KS))

    print(format_table(reports, ks=KS))

    out = OUT_DIR / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
