"""Cross-modal failure-mode GIF on real KITTI seq 08 (reverse-direction loops).

The headline finding, made visual: on a reverse revisit the forward-facing
**camera** retrieves the WRONG place (it sees the opposite view), while the
360-degree rotation-invariant **LiDAR** Scan-Context retrieves the CORRECT
revisited place — same query, same database, two sensors, opposite outcomes.

We pick query frames where camera's top-1 is far from the query's true pose
(> 25 m, a wrong place) but LiDAR's top-1 is the genuine revisit (< 5 m), and
animate the four-panel comparison. All real data; retrieval is BEVMatch's.

    python scripts/make_crossmodal_gif.py            # docs/assets/bevmatch_crossmodal.gif
    python scripts/make_crossmodal_gif.py --list     # just print the chosen frames
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bevmatch.datasets.loaders import load_kitti_bin  # noqa: E402
from bevmatch.retrieval.descriptor import sc_alignment_distance  # noqa: E402

DS = Path("/home/sasaki/datasets")
IMG = DS / "kitti_seq08_image0"
VELO = DS / "kitti_seq08_velodyne"
POSES = DS / "kitti_odometry_training_subsets/seq08/poses_08.txt"
TIMES = DS / "kitti_odometry_training_subsets/seq08/times.txt"
CAM_EMB = Path("/tmp/kitti_vpr_emb/seq08_resnet18.npy")
SC_WIDE = Path("/tmp/kitti_seq08_sc_wide.npz")
OUT = ROOT / "docs/assets/bevmatch_crossmodal.gif"

TIME_EXCLUDE = 30.0
BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"
GREEN = "#34d399"; RED = "#f87171"; CYAN = "#22d3ee"; ACCENT = "#f0a020"


def retrievals():
    pos = np.loadtxt(POSES)[:, [3, 7, 11]]
    times = np.loadtxt(TIMES)
    n = min(len(pos), len(times), len(list(IMG.glob("*.png"))))
    pos, times = pos[:n], times[:n]
    far = np.abs(times[:, None] - times[None, :]) > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    # Camera: cosine top-1 over retrievable candidates
    emb = np.load(CAM_EMB)[:n]
    nrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = nrm @ nrm.T
    sims[~far] = -2.0
    cam_top1 = sims.argmax(axis=1)

    # LiDAR (wide Scan-Context): ring-key prefilter then SC rerank, top-1
    d = np.load(SC_WIDE)
    rks, scs = d["rk"][:n], d["sc"][:n]
    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T), 0)
    rk_order = np.argsort(rk_d2, axis=1)

    positive = (pd <= 5.0) & far
    revisits = np.where(positive.any(axis=1))[0]
    # candidates: camera top-1 clearly wrong place
    cand = [q for q in revisits if pd[q, cam_top1[q]] > 25.0]

    rows = []
    for q in cand:
        shortlist = [j for j in rk_order[q] if far[q, j]][:25]
        lidar_top1 = min(shortlist, key=lambda j: sc_alignment_distance(scs[q], scs[j])[0])
        if pd[q, lidar_top1] < 5.0:  # LiDAR right, camera wrong
            rows.append((q, int(cam_top1[q]), pd[q, cam_top1[q]],
                         int(lidar_top1), pd[q, lidar_top1]))
    # spread across the sequence
    rows.sort()
    if len(rows) > 8:
        idx = np.linspace(0, len(rows) - 1, 8).astype(int)
        rows = [rows[i] for i in idx]
    return rows


def _img(frame, w=460):
    im = Image.open(IMG / f"{frame:06d}.png").convert("L")
    im = im.resize((w, int(im.height * w / im.width)))
    return np.asarray(im)


def _bev(frame, rng=50.0):
    pts = load_kitti_bin(VELO / f"{frame:06d}.bin")
    m = (np.abs(pts[:, 0]) < rng) & (np.abs(pts[:, 1]) < rng) & (pts[:, 2] > -2.5)
    return pts[m, 0], pts[m, 1]


def build(list_only=False):
    rows = retrievals()
    print(f"found {len(rows)} 'camera wrong / LiDAR right' reverse-revisit frames:")
    for q, cm, cd, lm, ld in rows:
        print(f"  q={q:4d}  camera->{cm:4d} (GT {cd:5.1f} m, wrong)   "
              f"LiDAR->{lm:4d} (GT {ld:4.1f} m, correct)")
    if list_only or not rows:
        return

    fig = plt.figure(figsize=(9.2, 7.0), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.86, bottom=0.06,
                          hspace=0.18, wspace=0.10)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    fig.text(0.05, 0.955, "BEVMatch", color=FG, fontsize=19, fontweight="bold", family="monospace")
    fig.text(0.255, 0.957, "· same reverse revisit, two sensors, opposite outcomes",
             color=MUTED, fontsize=10.5, family="monospace")
    fig.text(0.05, 0.915, "real KITTI seq 08 (reverse-direction loops) · camera retrieves the wrong "
             "place · LiDAR retrieves the right one", color="#6b7685", fontsize=8.2, family="monospace")

    HOLD = 6
    N = len(rows) * HOLD

    def draw(f):
        q, cm, cd, lm, ld = rows[min(f // HOLD, len(rows) - 1)]
        for r in range(2):
            for c in range(2):
                a = axes[r][c]; a.cla(); a.set_facecolor(BG)
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
        # row 0: camera
        axes[0][0].imshow(_img(q), cmap="gray", aspect="auto")
        axes[0][0].add_patch(plt.Rectangle((0, 0), _img(q).shape[1]-1, _img(q).shape[0]-1, fill=False, ec=CYAN, lw=2.5))
        axes[0][0].set_title(f"current camera · frame {q}", color=CYAN, fontsize=10,
                             family="monospace", loc="left", pad=4)
        axes[0][1].imshow(_img(cm), cmap="gray", aspect="auto")
        axes[0][1].add_patch(plt.Rectangle((0, 0), _img(cm).shape[1]-1, _img(cm).shape[0]-1, fill=False, ec=RED, lw=2.5))
        axes[0][1].set_title(f"camera VPR best · frame {cm} · GT {cd:.0f} m away  ✗ wrong place",
                             color=RED, fontsize=10, family="monospace", loc="left", pad=4)
        # row 1: LiDAR BEV
        for ax, fr, ec, ttl in ((axes[1][0], q, CYAN, f"current LiDAR (BEV) · frame {q}"),
                                (axes[1][1], lm, GREEN, f"LiDAR Scan-Context best · frame {lm} · GT {ld:.1f} m  ✓ same place")):
            x, y = _bev(fr)
            ax.scatter(y, x, s=0.4, c=np.hypot(x, y), cmap="viridis", alpha=0.7, linewidths=0)
            ax.set_xlim(50, -50); ax.set_ylim(-50, 50); ax.set_aspect("equal")
            ax.add_patch(plt.Rectangle((-49.5, -49.5), 99, 99, fill=False, ec=ec, lw=2.0, transform=ax.transData))
            ax.set_title(ttl, color=ec, fontsize=10, family="monospace", loc="left", pad=4)
        return []

    if list_only:
        return
    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, draw, frames=N, interval=130, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=7))
    print(f"gif -> {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    build(list_only="--list" in sys.argv)
