"""BEVMatch v0.1 end-to-end demo (§22 MVP).

    python examples/run_demo.py

Builds a synthetic same-place dataset, runs the comparison pipeline, prints the
evidence summary, writes a Comparison Evidence Bundle to JSON, and — if
matplotlib is installed — saves a 4-panel same-place comparison figure
(§15.2: current / historical / aligned overlay / change evidence).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bevmatch import SamePlaceComparisonPipeline, SceneDatabase
from bevmatch.datasets import make_synthetic_same_place
from bevmatch.io import save_bundle
from bevmatch.representations.bev import BEVConfig, points_to_bev

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    data = make_synthetic_same_place(seed=7)

    db = SceneDatabase()
    db.add_all(data.historical)
    pipeline = SamePlaceComparisonPipeline(database=db, top_k=5)

    bundle = pipeline.run(data.query)

    print("=" * 60)
    print(bundle.summary())
    print("=" * 60)
    print(f"Ground-truth place : {data.gt_place_id}")
    gt = data.gt_relative_pose
    print(f"Ground-truth pose  : x={gt.x:+.2f} y={gt.y:+.2f} yaw={np.rad2deg(gt.yaw):+.1f} deg")
    print(f"GT added objects   : {len(data.added_centers)}")
    print(f"GT removed objects : {len(data.removed_centers)}")

    out_json = save_bundle(bundle, OUT_DIR / "evidence_bundle.json")
    print(f"\nEvidence bundle written to: {out_json}")

    _maybe_plot(data, bundle)


def _maybe_plot(data, bundle) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping figure)")
        return

    cfg = BEVConfig()
    ref = data.historical[int(data.gt_place_id.split("_")[1])]
    q_xy = data.query.primary().xy()
    r_xy = ref.primary().xy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    extent = [-cfg.range_m, cfg.range_m, -cfg.range_m, cfg.range_m]

    axes[0, 0].imshow(points_to_bev(q_xy, cfg).grid > 0, origin="lower", extent=extent, cmap="Blues")
    axes[0, 0].set_title("Current scene (query)")

    axes[0, 1].imshow(points_to_bev(r_xy, cfg).grid > 0, origin="lower", extent=extent, cmap="Greens")
    axes[0, 1].set_title("Historical scene (retrieved)")

    ax = axes[1, 0]
    ax.imshow(points_to_bev(r_xy, cfg).grid > 0, origin="lower", extent=extent, cmap="Greens", alpha=0.6)
    if bundle.alignment and bundle.alignment.success:
        q_in_ref = bundle.alignment.relative_pose.transform(q_xy)
        ax.imshow(points_to_bev(q_in_ref, cfg).grid > 0, origin="lower", extent=extent, cmap="Blues", alpha=0.5)
        ax.set_title("Aligned overlay (green=hist, blue=query)")
    else:
        ax.set_title("Aligned overlay (alignment failed)")

    ax = axes[1, 1]
    ax.imshow(points_to_bev(r_xy, cfg).grid > 0, origin="lower", extent=extent, cmap="Greys", alpha=0.3)
    for ch in bundle.added():
        ax.scatter(*ch.centroid_xy, c="red", s=80, marker="+", linewidths=2)
    for ch in bundle.removed():
        ax.scatter(*ch.centroid_xy, c="blue", s=80, marker="x", linewidths=2)
    ax.set_xlim(-cfg.range_m, cfg.range_m)
    ax.set_ylim(-cfg.range_m, cfg.range_m)
    ax.set_title(f"Change evidence (+added={len(bundle.added())}, xremoved={len(bundle.removed())})")

    fig.tight_layout()
    out_png = OUT_DIR / "same_place_comparison.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print(f"Figure written to: {out_png}")


if __name__ == "__main__":
    main()
