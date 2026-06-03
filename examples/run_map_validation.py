"""Map validation benchmark (§12, §13.5).

    python examples/run_map_validation.py

Validates a point cloud / occupancy / vector map against current observations:
"does the map still match the world?" (not file-syntax validation). Emits a
prioritised, human-review-ready report (JSON + markdown) and issue precision/
recall / review-burden metrics on a changed and a fresh map.
"""

from __future__ import annotations

from pathlib import Path

from bevmatch.datasets import make_map_validation_case
from bevmatch.eval import issue_prf, review_burden
from bevmatch.maps import (
    MapValidationReport,
    OccupancyMapValidator,
    PointCloudMapValidator,
    VectorMapValidator,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def _validate(case) -> MapValidationReport:
    report = MapValidationReport(
        map_id=case.pcd_map.map_id, map_version=case.pcd_map.version,
        scene_id=case.current_frames[0].scene_id,
        provenance={"validators": ["pointcloud", "occupancy", "vector"]},
    )
    report.add(PointCloudMapValidator().validate(case.current_frames, case.pcd_map))
    report.add(OccupancyMapValidator().validate(case.current_map_frame, case.occ_map))
    report.add(VectorMapValidator().validate(case.current_map_frame, case.vmap))
    return report


def main() -> None:
    print("=== Changed map (world has moved on) ===")
    changed = make_map_validation_case(seed=1, changed=True)
    report = _validate(changed)
    print(report.to_markdown())
    print("\nseverity counts:", report.severity_counts())
    print("stale regions:", [i.issue_id for i in report.stale_regions()])
    pc = PointCloudMapValidator().validate(changed.current_frames, changed.pcd_map)
    vm = VectorMapValidator().validate(changed.current_map_frame, changed.vmap)
    prf = issue_prf(pc + vm, changed.gt_issues)
    print(f"issue P/R/F1 = {prf.precision:.2f}/{prf.recall:.2f}/{prf.f1:.2f}  "
          f"review burden = {review_burden(len(report.issues), 1):.1f} issues/scene")

    print("\n=== Fresh map (still matches the world) ===")
    fresh = make_map_validation_case(seed=1, changed=False)
    fresh_report = _validate(fresh)
    print("severity counts:", fresh_report.severity_counts())
    print(f"point-cloud/vector issues = "
          f"{len(PointCloudMapValidator().validate(fresh.current_frames, fresh.pcd_map))} "
          f"(expected 0 for an up-to-date map)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = report.save_json(OUT_DIR / "map_validation_report.json")
    md_path = report.save_markdown(OUT_DIR / "map_validation_review.md")
    print(f"\nReport written to: {json_path}")
    print(f"Human-review sheet: {md_path}")


if __name__ == "__main__":
    main()
