"""v0.8 benchmark suite: cards, reproducible splits, suite, leaderboard."""

from __future__ import annotations

import pytest

from bevmatch.benchmarks import (
    CARDS,
    BenchmarkSuite,
    SubmissionEntry,
    dataset_fingerprint,
    format_leaderboard,
    get_card,
    leaderboard_rows,
    make_manifest,
    run_retrieval_benchmark,
)


def test_cards_registered():
    assert {"mini-route", "mini-change", "mini-map"} <= set(CARDS)
    card = get_card("mini-route")
    assert "retrieval" in card.tasks and card.to_dict()["name"] == "mini-route"


def test_fingerprint_is_deterministic():
    for name in CARDS:
        assert dataset_fingerprint(name) == dataset_fingerprint(name)
    # distinct datasets -> distinct fingerprints
    fps = {dataset_fingerprint(n) for n in CARDS}
    assert len(fps) == len(CARDS)


def test_manifest():
    m = make_manifest("mini-change")
    assert m.card == "mini-change"
    assert m.seeds == list(CARDS["mini-change"].seeds)
    assert len(m.fingerprint) == 16


def test_retrieval_benchmark_ranks_methods():
    results = run_retrieval_benchmark()
    ranked = leaderboard_rows(results, "retrieval")
    assert ranked[0].metrics["recall@1"] >= ranked[-1].metrics["recall@1"]
    assert ranked[0].method == "scan-context"  # rotation-invariant wins


def test_full_suite_covers_all_tasks():
    result = BenchmarkSuite().run()
    tasks = {r.task for r in result.results}
    assert tasks == {"retrieval", "alignment", "change", "map_validation"}
    # every result carries a metrics dict and serialises
    d = result.to_dict()
    assert all("metrics" in r for r in d["results"])


def test_leaderboard_markdown_and_submission_merge():
    results = run_retrieval_benchmark()
    md = format_leaderboard(results, "retrieval")
    assert "retrieval" in md and "recall@1" in md

    external = SubmissionEntry(
        task="retrieval", method="ext", dataset="mini-route",
        metrics={"recall@1": 0.99, "recall@5": 1.0, "mrr": 0.99},
    )
    merged = results + [external.to_result()]
    assert leaderboard_rows(merged, "retrieval")[0].method == "ext"  # tops the board


def test_change_and_map_benchmarks_are_perfect_on_synthetic():
    from bevmatch.benchmarks import run_change_benchmark, run_map_benchmark

    change = run_change_benchmark()[0].metrics
    assert change["added_recall"] == pytest.approx(1.0)
    assert change["removed_recall"] == pytest.approx(1.0)
    assert change["false_changes"] == pytest.approx(0.0)

    mp = run_map_benchmark()[0].metrics
    assert mp["issue_precision"] == pytest.approx(1.0)
    assert mp["issue_recall"] == pytest.approx(1.0)
    assert mp["false_issues_fresh"] == pytest.approx(0.0)
