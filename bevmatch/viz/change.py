"""Before/after change viewer (§15.2 Same-Place Comparison Viewer).

Four panels: before (reference), after (aligned query), comparable/occlusion
region, and change evidence (added / removed / dynamic). No-op without
matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bevmatch.core.datamodel import ChangeHypothesis, Pose2D
from bevmatch.representations.bev import BEVConfig, points_to_bev

_COLORS = {"added": "red", "removed": "blue", "dynamic": "orange"}
_MARKERS = {"added": "+", "removed": "x", "dynamic": "o"}


def save_change_figure(
    reference_xy: np.ndarray,
    query_xy: np.ndarray,
    relative_pose: Pose2D,
    changes: list[ChangeHypothesis],
    path: str | Path,
    bev: BEVConfig | None = None,
    comparable_mask: np.ndarray | None = None,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    bev = bev or BEVConfig()
    extent = [-bev.range_m, bev.range_m, -bev.range_m, bev.range_m]
    r = np.asarray(reference_xy, dtype=float)[:, :2]
    q_in_ref = relative_pose.transform(np.asarray(query_xy, dtype=float)[:, :2])

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes[0, 0].imshow(points_to_bev(r, bev).grid > 0, origin="lower", extent=extent, cmap="Greens")
    axes[0, 0].set_title("Before (reference)")
    axes[0, 1].imshow(points_to_bev(q_in_ref, bev).grid > 0, origin="lower", extent=extent, cmap="Blues")
    axes[0, 1].set_title("After (aligned query)")

    ax = axes[1, 0]
    if comparable_mask is not None:
        ax.imshow(comparable_mask, origin="lower", extent=extent, cmap="Greys", alpha=0.8)
        ax.set_title("Comparable region (white = comparable)")
    else:
        ax.imshow(points_to_bev(r, bev).grid > 0, origin="lower", extent=extent, cmap="Greens", alpha=0.4)
        ax.imshow(points_to_bev(q_in_ref, bev).grid > 0, origin="lower", extent=extent, cmap="Blues", alpha=0.4)
        ax.set_title("Aligned overlay")

    ax = axes[1, 1]
    ax.imshow(points_to_bev(r, bev).grid > 0, origin="lower", extent=extent, cmap="Greys", alpha=0.25)
    seen = set()
    for ch in changes:
        c = _COLORS.get(ch.category, "magenta")
        m = _MARKERS.get(ch.category, "s")
        label = ch.category if ch.category not in seen else None
        seen.add(ch.category)
        ax.scatter(*ch.centroid_xy, c=c, s=90, marker=m, linewidths=2, label=label)
    ax.set_xlim(-bev.range_m, bev.range_m)
    ax.set_ylim(-bev.range_m, bev.range_m)
    ax.set_aspect("equal")
    if seen:
        ax.legend(loc="upper right")
    counts = {k: sum(c.category == k for c in changes) for k in ("added", "removed", "dynamic")}
    ax.set_title(f"Change evidence  +{counts['added']} added  x{counts['removed']} removed  o{counts['dynamic']} dynamic")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
