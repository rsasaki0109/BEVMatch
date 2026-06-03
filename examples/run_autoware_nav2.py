"""Autoware / Nav2 adapter demo (§17, §18).

    python examples/run_autoware_nav2.py

Shows how BEVMatch plugs into existing stacks: Autoware initial-pose assistance,
localization health monitoring, point cloud map freshness, Lanelet2 consistency;
Nav2 relocalization, occupancy-map staleness, changed-area annotation.
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Pose2D
from bevmatch.datasets import make_map_validation_case, make_synthetic_route
from bevmatch.eval.alignment_eval import pose_errors
from bevmatch.integrations import AutowareAdapter, Nav2Adapter, OccupancyGrid
from bevmatch.retrieval import SceneDatabase


def main() -> None:
    route = make_synthetic_route(seed=0, n_places=10, n_queries=5)
    db = SceneDatabase()
    db.add_all(route.historical)
    aw = AutowareAdapter()
    q = route.queries[0]

    print("=== Autoware: initial pose assistance (Pattern A) ===")
    for c in aw.initial_pose_candidates(q.scene, db, top_k=3):
        print(f"  place={c.place_id:<9} pose=({c.pose.x:+.2f},{c.pose.y:+.2f},"
              f"{np.rad2deg(c.pose.yaw):+.1f}deg) score={c.score:.3f} overlap={c.overlap_ratio:.2f}")
    top = aw.initial_pose_candidates(q.scene, db, top_k=1)[0]
    t, r = pose_errors(top.pose, q.gt_relative_pose)
    print(f"  best vs ground truth: {t:.2f} m / {r:.2f} deg; cov(x,y,yaw) diag = "
          f"{top.covariance[0]:.3f}, {top.covariance[7]:.3f}, {top.covariance[35]:.4f}")

    print("\n=== Autoware: localization health monitoring (Pattern B) ===")
    good = aw.localization_health(q.scene, db, q.gt_relative_pose)
    drifted = aw.localization_health(q.scene, db, q.gt_relative_pose.compose(Pose2D(8, 8, 0)))
    print(f"  reported≈truth : {good.level.name}  ({good.message}, {good.trans_error_m:.2f} m)")
    print(f"  reported drifted: {drifted.level.name}  ({drifted.message}, {drifted.trans_error_m:.2f} m)")

    print("\n=== Autoware: point cloud map freshness (Pattern C) ===")
    case = make_map_validation_case(seed=1, changed=True)
    report = aw.pointcloud_map_freshness(case.current_frames, case.pcd_map)
    print(f"  {len(report.issues)} issue(s); stale regions: "
          f"{[(i.issue_type, i.severity.label) for i in report.stale_regions()]}")

    print("\n=== Autoware: Lanelet2 observation consistency (Pattern D) ===")
    ll = aw.lanelet2_consistency(case.current_map_frame, case.vmap)
    print(f"  vector issues: {[(i.evidence.get('element_id'), i.issue_type) for i in ll]}")

    print("\n=== Nav2: relocalization + occupancy staleness (Use Cases A/B/C) ===")
    nav = Nav2Adapter()
    grid = OccupancyGrid.from_occupancy_map(case.occ_map)
    print(f"  OccupancyGrid {grid.width}x{grid.height} @ {grid.resolution} m, origin "
          f"({grid.origin.x:.0f},{grid.origin.y:.0f})")
    issues = nav.occupancy_staleness(case.current_map_frame, grid)
    blocked = nav.changed_area_annotations(issues)
    print(f"  staleness issues: {sorted({i.issue_type for i in issues})}")
    print(f"  changed/blocked areas for operator review: {len(blocked)}")
    cands = nav.relocalization(q.scene, db, top_k=1)
    if cands:
        print(f"  AMCL initial pose: ({cands[0].pose.x:+.2f},{cands[0].pose.y:+.2f},"
              f"{np.rad2deg(cands[0].pose.yaw):+.1f}deg)")


if __name__ == "__main__":
    main()
