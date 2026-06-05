"""One-figure summary of the cross-session long-term result (Finding 5).

Reads docs/assets/nclt_cross_session_{wide,default}.json (no recompute) and renders
Recall@1 for the winter map (2012-01-08) queried by a summer traverse (2012-08-04,
209 days later), against the same-day within-session baseline. Headline config is
the wide descriptor (40x120, 80 m) that Finding 4 already showed the campus needs;
the gap at the tight 5 m radius is the cost of seven months and a full change of
season.

    python scripts/make_cross_session_figure.py   # -> docs/assets/bevmatch_cross_session_summary.png
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
OUT = A / "bevmatch_cross_session_summary.png"

DS = ["5m", "10m", "25m"]
BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"; GRID = "#21262d"
WITHIN = "#6b7685"; CROSS = "#22d3ee"; ACCENT = "#34d399"


def main() -> None:
    wide = json.loads((A / "nclt_cross_session_wide.json").read_text())
    dflt = json.loads((A / "nclt_cross_session_default.json").read_text())
    win = [wide["within_session"][d]["recall@1"] for d in DS]
    crs = [wide["cross_session"][d]["recall@1"] for d in DS]

    x = np.arange(len(DS))
    w = 0.34
    fig, ax = plt.subplots(figsize=(10.4, 5.6), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for off, vals, color, lab in [(-w / 2, win, WITHIN, "within-session (same day, |dt|>30 s)"),
                                  (+w / 2, crs, CROSS, "cross-session (209 days, winter→summer)")]:
        hi = color == CROSS
        bars = ax.bar(x + off, vals, w, color=color, edgecolor=(ACCENT if hi else BG),
                      linewidth=(1.4 if hi else 0.5), zorder=3, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.013, f"{v:.2f}", ha="center",
                    va="bottom", color=color, fontsize=8.4, family="monospace",
                    fontweight=("bold" if hi else "normal"), zorder=4)

    # the headline gap at 5 m
    gap = win[0] - crs[0]
    ax.annotate(f"-{gap:.2f}\nthe cost of\n7 months", xy=(0 + w / 2, crs[0]),
                xytext=(0.66, 0.40), color=ACCENT, fontsize=9.0, family="monospace",
                ha="center", va="center", arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"positive radius {d}" for d in DS], color=FG, fontsize=10, family="monospace")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 0.91, 0.2))
    ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 0.91, 0.2)], color=MUTED, fontsize=8.5)
    ax.set_ylabel("Recall@1", color=FG, fontsize=10.5, family="monospace")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    fig.text(0.060, 0.955, "BEVMatch", color=FG, fontsize=17, fontweight="bold", family="monospace")
    fig.text(0.198, 0.957, "· NCLT long-term: a winter map still localises a summer drive",
             color=MUTED, fontsize=10.3, family="monospace")
    fig.text(0.060, 0.912,
             f"map 2012-01-08 (winter) · query 2012-08-04 (summer) · HDL-32E · wide descriptor "
             f"(40x120, 80 m) · R@1@5 m {crs[0]:.2f} vs same-day {win[0]:.2f}",
             color="#6b7685", fontsize=8.2, family="monospace")
    fig.text(0.060, 0.889,
             f"default descriptor (20x60, 30 m): cross {dflt['cross_session']['5m']['recall@1']:.2f} "
             f"vs same-day {dflt['within_session']['5m']['recall@1']:.2f} — wide wins across "
             "sessions too, as on KITTI. LiDAR geometry is appearance-blind.",
             color="#6b7685", fontsize=8.0, family="monospace")

    handles = [Patch(facecolor=WITHIN, label="within-session (same day, |dt|>30 s)"),
               Patch(facecolor=CROSS, edgecolor=ACCENT, linewidth=1.4,
                     label="cross-session (209 days, winter→summer)")]
    leg = ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.6,
                    facecolor="#161b22", edgecolor=GRID, labelcolor=FG,
                    framealpha=0.95, bbox_to_anchor=(0.5, -0.10))
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.060, right=0.985, top=0.85, bottom=0.17)
    fig.savefig(OUT, facecolor=BG)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
