"""Alignment benchmark (§10, §13.3): compare aligners; show failure & residuals.

    python examples/run_alignment_eval.py

- Evaluates SE2 and SE3 aligners against ground-truth relative poses.
- Demonstrates failure classification by aligning a query to a wrong place.
- Saves a residual/overlay figure (if matplotlib is installed).
"""

from __future__ import annotations

import json
from pathlib import Path

from bevmatch.alignment import SE2Aligner, SE3Aligner
from bevmatch.datasets import make_synthetic_route
from bevmatch.eval.alignment_eval import evaluate_alignment, format_alignment_table
from bevmatch.retrieval import ScanContextDescriptor, SceneDatabase
from bevmatch.viz import save_alignment_figure

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    route = make_synthetic_route(seed=0, n_places=12, n_queries=24)
    db = SceneDatabase(descriptor=ScanContextDescriptor())
    db.add_all(route.historical)

    reports = [
        evaluate_alignment(db, route.queries, SE2Aligner()),
        evaluate_alignment(db, route.queries, SE3Aligner()),
    ]
    print(format_alignment_table(reports))
    print("\nfailure classes:", {r.aligner: r.failure_counts for r in reports})

    # Failure demonstration: align a query against a deliberately wrong place.
    q0 = route.queries[0]
    wrong_place = "place_0" if q0.gt_place_id != "place_0" else "place_1"
    wrong = db.scene_by_place(wrong_place)
    bad = SE2Aligner().align(q0.scene, wrong)
    print(
        f"\nWrong-place alignment ({q0.gt_place_id} vs {wrong_place}): "
        f"success={bad.success}, class={bad.failure_class}, overlap={bad.overlap_ratio:.2f}"
    )

    # SE3 degeneracy on planar data.
    se3 = SE3Aligner().align(q0.scene, db.scene_by_place(q0.gt_place_id))
    print(f"SE3 degeneracy (planar scene): {se3.degeneracy}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "alignment_eval.json").write_text(
        json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8"
    )

    ref = db.scene_by_place(q0.gt_place_id)
    good = SE2Aligner().align(q0.scene, ref)
    fig = save_alignment_figure(
        q0.scene.primary().xy(), ref.primary().xy(), good, OUT_DIR / "alignment_residual.png"
    )
    print(f"\nReports written to: {OUT_DIR / 'alignment_eval.json'}")
    if fig:
        print(f"Residual figure written to: {fig}")


if __name__ == "__main__":
    main()
