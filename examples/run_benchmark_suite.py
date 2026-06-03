"""BEVMatch benchmark suite (§0.8, §13).

    python examples/run_benchmark_suite.py

Runs the pipeline-level benchmark across retrieval / alignment / change / map
validation, prints per-task leaderboards, records reproducible split manifests,
and writes a leaderboard JSON. Adding a descriptor/aligner plugin and re-running
yields a comparable leaderboard entry (§20.5).
"""

from __future__ import annotations

import json
from pathlib import Path

from bevmatch.benchmarks import (
    CARDS,
    BenchmarkSuite,
    SubmissionEntry,
    format_full_leaderboard,
    leaderboard_rows,
    make_manifest,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    print("=== Dataset cards ===")
    for name, card in CARDS.items():
        m = make_manifest(name)
        print(f"  {name:<12} tasks={list(card.tasks)} seeds={m.seeds} fingerprint={m.fingerprint}")

    result = BenchmarkSuite().run()
    print("\n" + format_full_leaderboard(result))

    # Example: merging an external submission onto the retrieval board.
    external = SubmissionEntry(
        task="retrieval", method="my-paper-descriptor", dataset="mini-route",
        metrics={"recall@1": 0.90, "recall@5": 0.99, "mrr": 0.93},
        manifest={"plugin": "external", "fingerprint": make_manifest("mini-route").fingerprint},
    )
    merged = result.results + [external.to_result()]
    print("\n=== retrieval board with external submission ===")
    for i, r in enumerate(leaderboard_rows(merged, "retrieval"), 1):
        print(f"  {i}. {r.method:<22} R@1={r.metrics['recall@1']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "cards": {n: CARDS[n].to_dict() for n in CARDS},
        "manifests": {n: make_manifest(n).to_dict() for n in CARDS},
        "results": result.to_dict()["results"],
    }
    path = OUT_DIR / "benchmark_leaderboard.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLeaderboard written to: {path}")


if __name__ == "__main__":
    main()
