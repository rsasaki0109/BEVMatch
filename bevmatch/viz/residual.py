"""Alignment residual / overlay visualization (§10.4, §15.2).

Renders the aligned overlay and a per-point residual map so an operator can see
*where* the alignment is trusted and where it is not. Returns ``None`` (no-op)
when matplotlib is unavailable, keeping the core dependency-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bevmatch.core.datamodel import AlignmentHypothesis


def _nearest_residuals(moved: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d2 = ((moved[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def save_alignment_figure(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    hypothesis: AlignmentHypothesis,
    path: str | Path,
) -> Path | None:
    """Save a 2-panel aligned-overlay + residual figure. No-op without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    q = np.asarray(query_xy, dtype=float)[:, :2]
    r = np.asarray(ref_xy, dtype=float)[:, :2]
    moved = hypothesis.relative_pose.transform(q)
    residuals = _nearest_residuals(moved, r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(r[:, 0], r[:, 1], s=8, c="green", label="reference")
    axes[0].scatter(moved[:, 0], moved[:, 1], s=8, c="blue", alpha=0.6, label="aligned query")
    axes[0].set_title(
        f"Aligned overlay ({hypothesis.relative_pose.x:+.2f}, "
        f"{hypothesis.relative_pose.y:+.2f}, {np.rad2deg(hypothesis.relative_pose.yaw):+.1f}°)\n"
        f"overlap={hypothesis.overlap_ratio:.0%} inliers={hypothesis.inlier_ratio:.0%} "
        f"rmse={hypothesis.rmse_m:.2f} m"
    )
    axes[0].legend(loc="upper right")
    axes[0].set_aspect("equal")

    sc = axes[1].scatter(moved[:, 0], moved[:, 1], s=12, c=np.clip(residuals, 0, 2.0), cmap="inferno_r")
    axes[1].set_title("Per-point residual (m) — bright = comparable, dark = uncertain")
    axes[1].set_aspect("equal")
    fig.colorbar(sc, ax=axes[1], shrink=0.8)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
