"""Generate the README hero GIF from REAL LiDAR map data + real BEVMatch output.

    python scripts/make_hero_gif.py                # docs/assets/bevmatch_hero.gif
    python scripts/make_hero_gif.py --preview 80   # single PNG frame to inspect

Six tiles cropped from a real 109M-point survey LiDAR map (docs/assets/
real_map_tiles.npz, voxel-downsampled) are the place database. A real observation
(one tile, presented at an unknown pose) is localized by BEVMatch: it retrieves
the matching place among the real candidates and recovers the SE2 pose with
covariance — the actual pipeline result, on real data. (The maps held no genuine
map-to-map change, so this shows the retrieval+localization use case rather than
a fabricated change.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from bevmatch.alignment import SE2Aligner  # noqa: E402
from bevmatch.core.datamodel import Observation, Pose2D, Scene  # noqa: E402
from bevmatch.integrations.relocalization import covariance_from_alignment  # noqa: E402
from bevmatch.retrieval import SceneDatabase  # noqa: E402

TILES_NPZ = ROOT / "docs" / "assets" / "real_map_tiles.npz"
OUT = ROOT / "docs" / "assets" / "bevmatch_hero.gif"

BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"
CYAN = "#22d3ee"; GREEN = "#34d399"; ACCENT = "#f0a020"; BLUE = "#58a6ff"
GT = 5  # query place (densest tile, retrieves cleanly)


def load():
    d = np.load(TILES_NPZ)
    tiles = [d[f"t{k}"].astype(float) for k in range(6)]
    db = SceneDatabase()
    for k, xy in enumerate(tiles):
        db.add(Scene(f"tile_{k}", place_id=f"place_{k}", pose=Pose2D(),
                     observations={"l": Observation("l", xy)}))
    t_wq = Pose2D(4.0, -3.0, np.deg2rad(28.0))
    q_local = t_wq.inverse().transform(tiles[GT])
    query = Scene("obs", observations={"l": Observation("l", q_local)})
    cand = db.query(query, top_k=1)[0]
    ref = db.get_scene(cand.scene_id)
    align = SE2Aligner().align(query, ref)
    return tiles, q_local, ref.primary().xy(), cand, align


def _ease(t):
    return t * t * (3 - 2 * t)


def _scatter(ax, pts, color, base=4, glow=16, a=0.9, ag=0.07):
    ax.scatter(pts[:, 0], pts[:, 1], s=glow, c=color, alpha=ag, linewidths=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=base, c=color, alpha=a, linewidths=0)


def build(preview_frame=None):
    tiles, q_xy, r_xy, cand, align = load()
    rel = align.relative_pose
    cov = covariance_from_alignment(align)
    sx, sy, syaw = cov[0] ** 0.5, cov[7] ** 0.5, np.degrees(cov[35] ** 0.5)
    matched = int(cand.place_id.split("_")[1])

    N = 100
    fig = plt.figure(figsize=(11, 5.4), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 20, left=0.015, right=0.985, top=0.86, bottom=0.04,
                          height_ratios=[3.1, 1.0], hspace=0.18, wspace=0.0)
    ax = fig.add_subplot(gs[0, :13]); ax_hud = fig.add_subplot(gs[0, 13:])
    ax_gal = fig.add_subplot(gs[1, :])

    fig.text(0.018, 0.93, "BEVMatch", color=FG, fontsize=21, fontweight="bold", family="monospace")
    fig.text(0.142, 0.935, "· localization on a real LiDAR map", color=MUTED, fontsize=11, family="monospace")
    fig.text(0.018, 0.895, "real survey map · 109M points · 6 place tiles", color="#5a6675",
             fontsize=8.5, family="monospace")
    caption = fig.text(0.018, 0.008, "", color=BLUE, fontsize=11, family="monospace")

    # precompute gallery thumbnails (downsampled, normalised into unit boxes)
    thumbs = []
    for t in tiles:
        s = t[:: max(1, len(t) // 350)]
        s = s - s.mean(0)
        sc = 0.9 / (np.abs(s).max() + 1e-6)
        thumbs.append(s * sc)

    def style(a):
        a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    def draw(frame):
        for a in (ax, ax_hud, ax_gal):
            a.cla(); style(a)
        ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_aspect("equal")
        ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)
        ax_gal.set_xlim(0, 6); ax_gal.set_ylim(0, 1)

        p_obs = np.clip(frame / 14, 0, 1)
        p_retr = np.clip((frame - 16) / 22, 0, 1)
        p_align = np.clip((frame - 44) / 34, 0, 1)

        ax.add_patch(plt.Circle((0, 0), 41, fill=False, ec=MUTED, lw=0.7, ls=(0, (4, 4)), alpha=0.16))

        # reference (matched real map tile) fades in at retrieval lock
        if p_retr > 0.4:
            _scatter(ax, r_xy, GREEN, base=6, glow=20,
                     a=0.85 * min(1, (p_retr - 0.4) / 0.6), ag=0.06)

        # observation, animating from as-observed pose to the recovered pose;
        # fade as it locks so the green map stays visible underneath (= localized)
        n_show = int(len(q_xy) * _ease(p_obs))
        if p_align <= 0:
            shown = q_xy[:n_show]
            q_alpha = 0.85
        else:
            e = _ease(p_align)
            shown = Pose2D(rel.x * e, rel.y * e, rel.yaw * e).transform(q_xy)
            q_alpha = 0.85 - 0.45 * e
        _scatter(ax, shown, CYAN, a=q_alpha, ag=0.06)

        # --- gallery ---
        ax_gal.text(0.02, 1.18, "place database (real map tiles)", color=MUTED,
                    fontsize=8.5, family="monospace", transform=ax_gal.transAxes)
        scan = int(frame * 0.5) % 6 if 0 < p_retr < 1 else matched
        for k, th in enumerate(thumbs):
            cx = k + 0.5
            ax_gal.scatter(cx + th[:, 0] * 0.42, 0.5 + th[:, 1] * 0.42, s=2,
                           c=GREEN if k == matched else "#3b4654",
                           alpha=0.9 if (p_retr > 0 and k == matched) else 0.5, linewidths=0)
            highlight = (p_retr >= 1 and k == matched) or (0 < p_retr < 1 and k == scan)
            if highlight:
                col = ACCENT if k == matched else BLUE
                ax_gal.add_patch(plt.Rectangle((k + 0.06, 0.04), 0.88, 0.92, fill=False,
                                               ec=col, lw=2.0))
            ax_gal.text(cx, 0.02, f"place_{k}", color=MUTED, fontsize=7, ha="center",
                        family="monospace")
        if p_retr >= 1:
            ax_gal.text(matched + 0.5, 0.9, "match", color=ACCENT, fontsize=8.5, ha="center",
                        family="monospace", fontweight="bold")

        # --- HUD ---
        ax_hud.text(0.04, 0.95, "Localization evidence", color=FG, fontsize=10.5,
                    family="monospace", fontweight="bold")
        ax_hud.plot([0.04, 0.96], [0.90, 0.90], color="#21262d", lw=1)
        if p_retr > 0.3:
            ax_hud.text(0.04, 0.80, "retrieval", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.5, 0.80, cand.place_id, color=GREEN, fontsize=9.5, family="monospace")
            ax_hud.text(0.5, 0.74, f"score {cand.score:.2f}", color=FG, fontsize=8.5, family="monospace")
        if p_align > 0:
            e = _ease(p_align)
            ax_hud.text(0.04, 0.60, "pose (map)", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.5, 0.60, f"x {rel.x*e:+.2f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.5, 0.545, f"y {rel.y*e:+.2f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.5, 0.49, f"yaw {np.degrees(rel.yaw)*e:+.1f}°", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.04, 0.40, "overlap", color=MUTED, fontsize=9, family="monospace")
            ax_hud.add_patch(plt.Rectangle((0.5, 0.385), 0.42, 0.035, color="#21262d"))
            ax_hud.add_patch(plt.Rectangle((0.5, 0.385), 0.42 * align.overlap_ratio * e, 0.035, color=BLUE))
            if e > 0.6:
                ax_hud.text(0.04, 0.26, "± cov", color=MUTED, fontsize=9, family="monospace")
                ax_hud.text(0.5, 0.26, f"{sx:.2f} m / {syaw:.1f}°", color="#9aa4b2",
                            fontsize=8.5, family="monospace")
                ax_hud.text(0.04, 0.12, "→ initial pose for", color="#5a6675", fontsize=8,
                            family="monospace")
                ax_hud.text(0.04, 0.07, "  Autoware / Nav2", color="#5a6675", fontsize=8, family="monospace")

        if p_align > 0:
            caption.set_text("2/2  aligning  ->  6-DoF initial pose")
        elif p_retr > 0:
            caption.set_text("1/2  retrieving the place in the real map")
        else:
            caption.set_text("real LiDAR observation  ·  unknown pose")
        return []

    if preview_frame is not None:
        draw(preview_frame)
        png = OUT.with_name(f"preview_{preview_frame}.png")
        fig.savefig(png, facecolor=BG, dpi=100)
        print("preview ->", png)
        return
    anim = FuncAnimation(fig, draw, frames=N, interval=55, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=18))
    print("gif ->", OUT, f"({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        build(preview_frame=int(sys.argv[2]))
    else:
        build()
