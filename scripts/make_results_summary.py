"""One-figure summary of BEVMatch's real-data place-recognition benchmarks.

Reads the benchmark JSONs (no recompute) and renders a grouped bar chart of
Recall@1 @ 5 m across the KITTI loop sequences for three descriptors that all
run behind BEVMatch's one retrieval interface:

  * LiDAR Scan-Context           (hand-crafted, training-free)
  * Camera ResNet-18 (ImageNet)  (generic features, baseline)
  * Camera EigenPlaces           (learned VPR SOTA, KITTI held out)

The story the figure tells, all from measured numbers:
  - a learned SOTA descriptor lifts the forward cases over the baseline;
  - on seq 08 (reverse-direction revisits) BOTH camera descriptors collapse to
    ~0.015 — a viewpoint wall that learning does not break;
  - the 360 deg LiDAR keeps 0.34 there and a pure config swap recovers it to
    0.765 (drawn as a ghost bar) — the asymmetry that motivates multi-modality.

    python scripts/make_results_summary.py   # -> docs/assets/bevmatch_results_summary.png
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
OUT = A / "bevmatch_results_summary.png"

SEQS = ["00", "05", "06", "07", "08"]
BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"; GRID = "#21262d"
LIDAR = "#22d3ee"; BASE = "#6b7685"; LEARN = "#34d399"; ACCENT = "#f0a020"; RED = "#f87171"


def _r1(path: Path) -> dict:
    """seq -> Recall@1 @ 5 m from a per-sequence benchmark JSON."""
    data = json.loads(path.read_text())
    return {r["sequence"]: r["by_distance"]["5m"]["recall@1"] for r in data}


def main() -> None:
    lidar = _r1(A / "kitti_lidar_results.json")
    base = _r1(A / "kitti_vpr_results.json")
    learn = _r1(A / "kitti_vpr_learned_results.json")
    cfg = json.loads((A / "kitti_scancontext_config.json").read_text())
    wide = {s: cfg[s]["wide (40x120, 80m)"]["recall@1"] for s in cfg}  # 00, 08

    series = [("LiDAR · Scan-Context (hand-crafted)", lidar, LIDAR),
              ("Camera · ResNet-18 (ImageNet baseline)", base, BASE),
              ("Camera · EigenPlaces (learned VPR SOTA)", learn, LEARN)]

    x = np.arange(len(SEQS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for i, (label, d, color) in enumerate(series):
        vals = [d.get(s, 0.0) for s in SEQS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                      edgecolor=BG, linewidth=0.6, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", color=color, fontsize=7.6,
                    family="monospace", zorder=4)

    # ghost bar: LiDAR-wide recovery on seq 08 (and 00), pure config swap
    for s, xi in zip(SEQS, x):
        if s in wide:
            ax.bar(xi - 1 * w, wide[s], w, facecolor="none", edgecolor=LIDAR,
                   linewidth=1.6, linestyle=(0, (2, 1.5)), zorder=5)
    ax.annotate(f"config swap\n0.34 → {wide['08']:.2f}", xy=(4 - w, wide["08"]),
                xytext=(3.05, 0.86), color=LIDAR, fontsize=8.4, family="monospace",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=LIDAR, lw=1.3))
    ax.annotate("viewpoint wall:\nlearning doesn't help", xy=(4 + 0.03, 0.05),
                xytext=(4.28, 0.50), color=RED, fontsize=8.4, family="monospace",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"seq {s}" + ("\n(reverse)" if s == "08" else
                        ("\n(few revisits)" if s == "07" else "")) for s in SEQS],
                       color=FG, fontsize=9.5, family="monospace")
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)], color=MUTED, fontsize=8.5)
    ax.set_ylabel("Recall@1  @ 5 m", color=FG, fontsize=10.5, family="monospace")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    fig.text(0.065, 0.955, "BEVMatch", color=FG, fontsize=17, fontweight="bold", family="monospace")
    fig.text(0.205, 0.957, "· real KITTI place recognition — three descriptors, one interface",
             color=MUTED, fontsize=10.5, family="monospace")
    fig.text(0.065, 0.915, "Recall@1 @ 5 m across the loop sequences. A learned SOTA descriptor "
             "lifts the forward cases — but on reverse-direction seq 08 both camera "
             "descriptors", color="#6b7685", fontsize=8.0, family="monospace")
    fig.text(0.065, 0.892, "collapse to ~0.02 (a viewpoint wall), while the 360 deg LiDAR holds 0.34 "
             "and recovers to 0.76 by a config swap alone (dashed). Measured, not staged.",
             color="#6b7685", fontsize=8.0, family="monospace")

    handles = [Patch(facecolor=c, label=l) for l, _, c in series]
    handles.append(Patch(facecolor="none", edgecolor=LIDAR, linestyle="--",
                         label="LiDAR · wide config (plugin swap)"))
    leg = ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.6,
                    facecolor="#161b22", edgecolor=GRID, labelcolor=FG,
                    framealpha=0.95, bbox_to_anchor=(0.5, -0.10))
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.065, right=0.98, top=0.86, bottom=0.20)
    fig.savefig(OUT, facecolor=BG)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
