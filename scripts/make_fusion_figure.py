"""One-figure summary of the late-fusion result (Finding 3).

Reads docs/assets/kitti_fusion_results.json (no recompute) and renders a grouped
bar chart of Recall@1 @ 5 m across the KITTI loop sequences for the two single
modalities and the three fusion strategies, making the headline visible at a
glance: score-level fusion (RRF, confidence-gate) does not beat the better
sensor and collapses on the reverse loop, while **geometry-verified** fusion wins
on every sequence and fully recovers seq 08.

    python scripts/make_fusion_figure.py   # -> docs/assets/bevmatch_fusion_summary.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "docs" / "assets"
OUT = A / "bevmatch_fusion_summary.png"

SEQS = ["00", "05", "06", "07", "08"]
BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"; GRID = "#21262d"
LIDAR = "#22d3ee"; CAM = "#6b7685"; RRF = "#f0a020"; GATED = "#a78bfa"; VER = "#34d399"; RED = "#f87171"

SERIES = [
    ("LiDAR · Scan-Context", "lidar", LIDAR),
    ("Camera · EigenPlaces", "camera_eigenplaces", CAM),
    ("fusion · naive RRF", "fusion_rrf", RRF),
    ("fusion · confidence-gated", "fusion_gated", GATED),
    ("fusion · geometry-verified", "fusion_verified", VER),
]


def main() -> None:
    data = {r["sequence"]: r for r in json.loads((A / "kitti_fusion_results.json").read_text())}
    r1 = {s: {k: data[s]["by_distance"]["5m"][k]["recall@1"] for _, k, _ in SERIES} for s in SEQS}
    means = {k: np.mean([r1[s][k] for s in SEQS]) for _, k, _ in SERIES}

    x = np.arange(len(SEQS))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11.6, 5.8), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for i, (label, key, color) in enumerate(SERIES):
        vals = [r1[s][key] for s in SEQS]
        off = (i - 2) * w
        hi = key == "fusion_verified"
        bars = ax.bar(x + off, vals, w, color=color, edgecolor=(VER if hi else BG),
                      linewidth=(1.4 if hi else 0.5), zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", color=color, fontsize=6.6,
                    family="monospace", fontweight=("bold" if hi else "normal"), zorder=4)

    # callouts on the decisive reverse loop (seq 08, x = 4)
    ax.annotate("naive fusion\ncollapses", xy=(4 + 0 * w, 0.081), xytext=(3.55, 0.46),
                color=RED, fontsize=8.0, family="monospace", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.annotate("geometry-verified\nrecovers LiDAR (0.34)", xy=(4 + 2 * w, 0.343),
                xytext=(4.05, 0.62), color=VER, fontsize=8.2, family="monospace",
                ha="center", va="center", arrowprops=dict(arrowstyle="->", color=VER, lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"seq {s}" + ("\n(reverse)" if s == "08" else "") for s in SEQS],
                       color=FG, fontsize=9.5, family="monospace")
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)], color=MUTED, fontsize=8.5)
    ax.set_ylabel("Recall@1  @ 5 m", color=FG, fontsize=10.5, family="monospace")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    fig.text(0.060, 0.955, "BEVMatch", color=FG, fontsize=17, fontweight="bold", family="monospace")
    fig.text(0.198, 0.957, "· fusing LiDAR + camera: score-level fails, geometry-verified wins",
             color=MUTED, fontsize=10.3, family="monospace")
    fig.text(0.060, 0.913, f"mean R@1 @ 5 m — LiDAR {means['lidar']:.3f} · camera "
             f"{means['camera_eigenplaces']:.3f} · naive RRF {means['fusion_rrf']:.3f} · "
             f"confidence-gate {means['fusion_gated']:.3f} · "
             f"geometry-verified {means['fusion_verified']:.3f}",
             color="#6b7685", fontsize=8.2, family="monospace")
    fig.text(0.060, 0.890, "verify the camera's place with LiDAR geometry, keep it only when it aligns: "
             "best on every sequence, and the blind reverse loop fully recovered. No ground truth.",
             color="#6b7685", fontsize=8.0, family="monospace")

    handles = [Patch(facecolor=c, label=l, edgecolor=(VER if k == "fusion_verified" else c),
                     linewidth=(1.4 if k == "fusion_verified" else 0)) for l, k, c in SERIES]
    leg = ax.legend(handles=handles, loc="upper center", ncol=5, fontsize=7.8,
                    facecolor="#161b22", edgecolor=GRID, labelcolor=FG,
                    framealpha=0.95, bbox_to_anchor=(0.5, -0.09))
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.060, right=0.985, top=0.85, bottom=0.17)
    fig.savefig(OUT, facecolor=BG)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
