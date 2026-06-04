"""Generate the README hero GIF from real BEVMatch pipeline output.

    python scripts/make_hero_gif.py            # writes docs/assets/bevmatch_hero.gif
    python scripts/make_hero_gif.py --preview 70   # writes a single PNG frame to inspect

A structured "street" scene (building facades + poles) is revisited from a new
viewpoint with a construction barrier added and a building removed. BEVMatch
retrieves the place, aligns it (SE2), and detects the changes — the animation
shows the actual pipeline result, not a mock-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bevmatch import SamePlaceComparisonPipeline, SceneDatabase  # noqa: E402
from bevmatch.change import ChangeConfig  # noqa: E402
from bevmatch.core.datamodel import Observation, Pose2D, Scene  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "bevmatch_hero.gif"

# palette (GitHub dark)
BG = "#0d1117"
FG = "#c9d1d9"
MUTED = "#8b949e"
CYAN = "#22d3ee"     # current / query
GREEN = "#34d399"    # historical / reference
ADDED = "#fb7185"    # added change
REMOVED = "#c084fc"  # removed change
ACCENT = "#58a6ff"


# --------------------------------------------------------------------------- #
# scene generation: a street as building facades + curb poles
# --------------------------------------------------------------------------- #
def _facade(rng, x0, x1, y, jitter=0.05, step=0.45):
    xs = np.arange(x0, x1, step)
    pts = np.stack([xs, np.full_like(xs, y)], axis=1)
    return pts + rng.normal(scale=jitter, size=pts.shape)


def _block(rng, x0, x1, y_near, depth, side):
    """A building footprint: road-facing facade + two short side returns."""
    pts = [_facade(rng, x0, x1, y_near)]
    y_far = y_near + depth * side
    pts.append(_facade(rng, x0, x1, y_far))
    for x in (x0, x1):
        ys = np.arange(min(y_near, y_far), max(y_near, y_far), 0.45)
        seg = np.stack([np.full_like(ys, x), ys], axis=1)
        pts.append(seg + rng.normal(scale=0.05, size=seg.shape))
    return np.vstack(pts)


def _poles(rng, xs, y):
    out = []
    for x in xs:
        out.append(rng.normal(loc=[x, y], scale=0.18, size=(10, 2)))
    return np.vstack(out)


def _buildings_side(rng, y_near, side):
    """Randomly placed building footprints along one side (distinct per seed)."""
    objs = []
    x = rng.uniform(-25, -20)
    while x < 22:
        w = rng.uniform(5.5, 9.5)
        if rng.random() < 0.78:  # sometimes a gap (side street)
            yn = y_near + rng.uniform(-0.8, 0.8)
            pts = _block(rng, x, x + w, y_near=yn, depth=rng.uniform(4, 6), side=side)
            objs.append((pts, f"building_{'n' if side > 0 else 's'}"))
        x += w + rng.uniform(2.5, 6.0)
    return objs


def street_objects(seed: int):
    """Return a list of (points, kind) objects forming a distinct street layout."""
    rng = np.random.default_rng(seed)
    objs = []
    objs += _buildings_side(rng, y_near=7.5, side=+1)
    objs += _buildings_side(rng, y_near=-7.5, side=-1)
    objs.append((_poles(rng, np.arange(-22, 24, rng.uniform(5, 7)), 5.5), "poles"))
    objs.append((_poles(rng, np.arange(-20, 24, rng.uniform(5, 7)), -5.5), "poles"))
    return objs


def street_scene(seed: int) -> np.ndarray:
    return np.vstack([pts for pts, _ in street_objects(seed)])


def make_case():
    """Build a small place DB + a revisit query with real changes."""
    db_scenes = []
    for p in range(5):
        pts = street_scene(seed=10 + p)
        db_scenes.append(Scene(f"hist_{p}", place_id=f"place_{p}", pose=Pose2D(), timestamp=float(p),
                               observations={"lidar": Observation("lidar", pts)}))

    revisit = 2
    objs = street_objects(seed=10 + revisit)
    # remove the north building whose centre is closest to x = +4
    north = [(i, pts) for i, (pts, kind) in enumerate(objs) if kind == "building_n"]
    rm_idx = min(north, key=lambda t: abs(t[1][:, 0].mean() - 4.0))[0]
    removed_center = objs[rm_idx][0].mean(axis=0)
    kept = [pts for i, (pts, _) in enumerate(objs) if i != rm_idx]
    # add a construction barrier in the road
    rng = np.random.default_rng(99)
    barrier_c = np.array([-7.0, 0.0])
    barrier = rng.normal(loc=barrier_c, scale=[1.7, 0.6], size=(46, 2))
    world_now = np.vstack(kept + [barrier])
    # observe from a new viewpoint
    t_wq = Pose2D(x=2.2, y=-1.4, yaw=np.deg2rad(17.0))
    query_local = t_wq.inverse().transform(world_now)
    query = Scene("query", place_id=f"place_{revisit}", pose=t_wq, timestamp=99.0,
                  observations={"lidar": Observation("lidar", query_local)})

    db = SceneDatabase()
    db.add_all(db_scenes)
    pipeline = SamePlaceComparisonPipeline(
        database=db, change_config=ChangeConfig(min_cells=5, suppress_passes=2))
    bundle = pipeline.run(query)
    ref = db.get_scene(bundle.best_candidate.scene_id)
    return query, ref, bundle, removed_center, barrier_c


# --------------------------------------------------------------------------- #
# animation
# --------------------------------------------------------------------------- #
def _ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def _scatter(ax, pts, color, base=5, glow=22, a_core=0.9, a_glow=0.08):
    ax.scatter(pts[:, 0], pts[:, 1], s=glow, c=color, alpha=a_glow, linewidths=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=base, c=color, alpha=a_core, linewidths=0)


def _merge(changes, radius=7.0):
    """Group nearby same-category change components into single markers."""
    centers = []
    for ch in sorted(changes, key=lambda c: c.area_m2, reverse=True):
        c = np.array(ch.centroid_xy)
        if all(np.hypot(*(c - np.array(m))) > radius for m in centers):
            centers.append((float(c[0]), float(c[1])))
    return centers


def build(preview_frame=None):
    query, ref, bundle, removed_center, barrier_c = make_case()
    q_xy = query.primary().xy()
    r_xy = ref.primary().xy()
    rel = bundle.alignment.relative_pose
    overlap = bundle.alignment.overlap_ratio
    added = _merge(bundle.added())
    removed = _merge(bundle.removed())

    N = 96
    fig = plt.figure(figsize=(11, 4.6), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 20, left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.0)
    ax = fig.add_subplot(gs[0, :13]); ax_hud = fig.add_subplot(gs[0, 13:])
    for a in (ax, ax_hud):
        a.set_facecolor(BG)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_xticks([]); a.set_yticks([])
    ax.set_xlim(-27, 27); ax.set_ylim(-15, 15); ax.set_aspect("equal")
    ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)

    fig.text(0.03, 0.93, "BEVMatch", color=FG, fontsize=22, fontweight="bold", family="monospace")
    fig.text(0.157, 0.935, "· same-place comparison", color=MUTED, fontsize=11, family="monospace")
    caption = fig.text(0.03, 0.025, "", color=ACCENT, fontsize=11, family="monospace")

    def hud_line(y, label, value, color=FG):
        ax_hud.text(0.04, y, label, color=MUTED, fontsize=9.5, family="monospace", va="center")
        return ax_hud.text(0.42, y, value, color=color, fontsize=9.5, family="monospace", va="center")

    def draw(frame):
        ax.cla(); ax_hud.cla()
        for a in (ax, ax_hud):
            a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
        ax.set_xlim(-27, 27); ax.set_ylim(-15, 15); ax.set_aspect("equal")
        ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)

        # grounding: road band, sensor range ring, ego marker
        ax.axhspan(-5.6, 5.6, color="#10161f", zorder=0)
        ax.add_patch(plt.Circle((0, 0), 26, fill=False, ec=MUTED, lw=0.8, ls=(0, (4, 4)),
                                alpha=0.18, zorder=0))
        ax.scatter(0, 0, s=90, marker="^", c="white", zorder=6)
        ax.text(0, -2.3, "ego", color=MUTED, fontsize=8, ha="center", family="monospace")

        t = frame / (N - 1)
        # phase schedule
        p_current = np.clip(frame / 12, 0, 1)
        p_retr = np.clip((frame - 12) / 14, 0, 1)
        p_align = np.clip((frame - 30) / 30, 0, 1)
        p_change = np.clip((frame - 64) / 18, 0, 1)

        ax_hud.text(0.04, 0.93, "Comparison Evidence", color=FG, fontsize=10.5,
                    family="monospace", fontweight="bold")
        ax_hud.plot([0.04, 0.96], [0.88, 0.88], color="#21262d", lw=1)

        # reference (historical) fades in during retrieval
        if p_retr > 0:
            _scatter(ax, r_xy, GREEN, a_core=0.55 * p_retr, a_glow=0.05 * p_retr)

        # current/query, animating from unaligned (identity) to recovered pose
        n_show = int(len(q_xy) * _ease(p_current))
        if p_align <= 0:
            shown = q_xy[:n_show]
        else:
            e = _ease(p_align)
            inter = Pose2D(rel.x * e, rel.y * e, rel.yaw * e)
            shown = inter.transform(q_xy)
        _scatter(ax, shown, CYAN, a_core=0.9, a_glow=0.08)

        # change markers
        if p_change > 0:
            e = _ease(p_change)
            pulse = 130 + 60 * np.sin(frame * 0.5)
            for (cx, cy) in added:
                ax.scatter(cx, cy, s=pulse * e, c=ADDED, marker="P",
                           edgecolors="white", linewidths=1.3, zorder=5)
                ax.text(cx, cy + 2.6, "added", color=ADDED, fontsize=9.5, ha="center",
                        family="monospace", alpha=e, fontweight="bold")
            for (cx, cy) in removed:
                ax.scatter(cx, cy, s=pulse * e, c=REMOVED, marker="X",
                           edgecolors="white", linewidths=1.3, zorder=5)
                ax.text(cx, cy + 2.6, "removed", color=REMOVED, fontsize=9.5, ha="center",
                        family="monospace", alpha=e, fontweight="bold")

        # HUD content
        y = 0.78
        if p_retr > 0:
            score = bundle.best_candidate.score
            ax_hud.text(0.04, y, "retrieval", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.5, y, f"{bundle.best_candidate.place_id}", color=GREEN,
                        fontsize=9.5, family="monospace")
            ax_hud.text(0.5, y - 0.07, f"score {score:.2f}", color=FG, fontsize=8.5, family="monospace")
        y = 0.58
        if p_align > 0:
            e = _ease(p_align)
            ax_hud.text(0.04, y, "alignment", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.5, y, f"x {rel.x*e:+.2f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.5, y - 0.06, f"y {rel.y*e:+.2f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.5, y - 0.12, f"yaw {np.rad2deg(rel.yaw)*e:+.1f}°", color=CYAN,
                        fontsize=8.5, family="monospace")
            # overlap bar
            ax_hud.text(0.04, y - 0.20, "overlap", color=MUTED, fontsize=9, family="monospace")
            ax_hud.add_patch(plt.Rectangle((0.5, y - 0.225), 0.42, 0.035, color="#21262d"))
            ax_hud.add_patch(plt.Rectangle((0.5, y - 0.225), 0.42 * overlap * e, 0.035, color=ACCENT))
        y = 0.20
        if p_change > 0:
            ax_hud.text(0.04, y, "changes", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.5, y, f"{len(added)} added", color=ADDED, fontsize=9, family="monospace")
            ax_hud.text(0.5, y - 0.07, f"{len(removed)} removed", color=REMOVED, fontsize=9, family="monospace")

        # captions
        if p_change > 0:
            cap = "3/3  comparing  ->  map-update evidence"
        elif p_align > 0:
            cap = "2/3  aligning  (SE2 BEV + ICP)"
        elif p_retr > 0:
            cap = "1/3  retrieving the same place"
        else:
            cap = "current observation"
        caption.set_text(cap)
        return []

    if preview_frame is not None:
        draw(preview_frame)
        png = OUT.with_name(f"preview_{preview_frame}.png")
        fig.savefig(png, facecolor=BG, dpi=100)
        print("preview ->", png)
        return

    anim = FuncAnimation(fig, draw, frames=N, interval=55, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=18))
    print("gif ->", OUT, f"({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        build(preview_frame=int(sys.argv[2]))
    else:
        build()
