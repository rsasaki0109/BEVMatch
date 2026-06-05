"""Generate a camera->BEV place-recognition GIF from REAL KITTI stereo.

    python scripts/make_camera_bev_gif.py             # docs/assets/bevmatch_camera_bev.gif
    python scripts/make_camera_bev_gif.py --preview 30  # single PNG frame to inspect

Real KITTI seq 00 (a drive that revisits its start). For each frame we build a
*camera* bird's-eye view: stereo disparity (left image_0 + right image_1) ->
depth -> 3D points -> top-down BEV. First-visit frames form the place database;
revisit frames (~4500) are queries. BEVMatch retrieves the matching past frame
(ResNet-18 image embeddings) and then SE2-aligns the two camera BEVs -> the query
BEV slides onto the matched map BEV and overlaps = the same place, recognized in
a camera-derived bird's-eye view (Principle 2: modality=camera, representation=BEV).

Data: KITTI odometry seq 00 (Geiger et al.), grayscale stereo. Research demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bevmatch.alignment import SE2Aligner  # noqa: E402
from bevmatch.core.datamodel import Observation, Scene, Pose2D  # noqa: E402
from bevmatch.retrieval import SceneDatabase  # noqa: E402
from bevmatch.sensors.camera import CameraEmbeddingDescriptor, camera_scene  # noqa: E402

L_DIR = Path("/home/sasaki/datasets/kitti_seq00_full/image_0")
R_DIR = Path("/home/sasaki/datasets/kitti_seq00_full/image_1")
EMB_CACHE = Path("/tmp/kitti_emb.npz")
OUT = ROOT / "docs" / "assets" / "bevmatch_camera_bev.gif"

# KITTI seq 00 calibration (calib.txt): fx, cx, cy and stereo baseline (P1 tx / fx)
FX = 718.856
CX = 607.1928
CY = 185.2157
BASELINE = 386.1448 / FX  # ~0.537 m

DB_FRAMES = list(range(0, 1600, 4))   # matches /tmp/kitti_emb.npz layout
Q_FRAMES = list(range(4500, 4531, 2))

BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"
CYAN = "#22d3ee"; GREEN = "#34d399"; ACCENT = "#f0a020"; BLUE = "#58a6ff"
MAPC = "#fbbf24"  # matched map BEV — gold, complementary to the cyan query

_SGBM = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5,
    P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)


def _disparity(frame):
    left = cv2.imread(str(L_DIR / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(R_DIR / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return _SGBM.compute(left, right).astype(np.float32) / 16.0


def camera_bev(frame, disp=None):
    """Stereo -> depth -> top-down BEV point cloud (x lateral, z forward, h up)."""
    if disp is None:
        disp = _disparity(frame)
    h, w = disp.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    m = disp > 1.0
    z = FX * BASELINE / disp[m]
    x = (us[m] - CX) * z / FX
    y = (vs[m] - CY) * z / FX            # camera y is DOWN
    keep = (z > 3) & (z < 35) & (np.abs(x) < 17) & (y < 1.3) & (y > -3.0)
    pts = np.stack([x[keep], z[keep], -y[keep]], axis=1)  # height = -y (up)
    return _voxel(pts, 0.3)


def _voxel(pts, vox):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts[:, :2] / vox).astype(np.int64)
    _, idx = np.unique(keys[:, 0] * 100003 + keys[:, 1], return_index=True)
    return pts[idx]


def retrieve():
    d = np.load(EMB_CACHE)
    db_emb, q_emb = d["db"], d["q"]
    database = SceneDatabase(descriptor=CameraEmbeddingDescriptor())
    for f, e in zip(DB_FRAMES, db_emb):
        database.add(camera_scene(f"db_{f}", e, place_id=f"{f}"))
    res = []
    for qf, qe in zip(Q_FRAMES, q_emb):
        cand = database.query(camera_scene(f"q_{qf}", qe), top_k=1)[0]
        res.append((qf, int(cand.place_id), float(cand.score)))
    return res


def _cam_img(frame, w=420):
    im = Image.open(L_DIR / f"{frame:06d}.png").convert("L")
    return np.asarray(im.resize((w, int(im.height * w / im.width))))


def _scatter(ax, pts, color, base=5, glow=18, a=0.9, ag=0.06):
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=glow, c=color, alpha=ag, linewidths=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=base, c=color, alpha=a, linewidths=0)


def _ease(t):
    return t * t * (3 - 2 * t)


def build(preview_frame=None):
    results = retrieve()
    # choose the query with the best (highest) retrieval score for a clean demo
    results.sort(key=lambda r: -r[2])
    qf, mf, sim = results[0]

    disp_q = _disparity(qf)
    q_bev = camera_bev(qf, disp_q)
    m_bev = camera_bev(mf)
    cam_q = _cam_img(qf)
    depth_q = np.where(disp_q > 1.0, disp_q, np.nan)  # for display

    q_scene = Scene("q", observations={"l": Observation("l", q_bev[:, :2])})
    m_scene = Scene("m", observations={"l": Observation("l", m_bev[:, :2])})
    align = SE2Aligner().align(q_scene, m_scene)
    rel = align.relative_pose
    overlap = align.overlap_ratio

    N = 90
    fig = plt.figure(figsize=(11, 5.4), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 24, left=0.015, right=0.985, top=0.85, bottom=0.06,
                          height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.0)
    ax_cam = fig.add_subplot(gs[0, :8])
    ax_depth = fig.add_subplot(gs[1, :8])
    ax_bev = fig.add_subplot(gs[:, 9:17])
    ax_hud = fig.add_subplot(gs[:, 18:])

    fig.text(0.018, 0.93, "BEVMatch", color=FG, fontsize=21, fontweight="bold", family="monospace")
    fig.text(0.142, 0.935, "· place recognition in a camera bird's-eye view", color=MUTED,
             fontsize=10.5, family="monospace")
    fig.text(0.018, 0.895, "real KITTI seq 00 · stereo depth -> BEV · loop closure", color="#5a6675",
             fontsize=8.5, family="monospace")
    caption = fig.text(0.018, 0.012, "", color=BLUE, fontsize=11, family="monospace")

    def style(a):
        a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    def draw(frame):
        for a in (ax_cam, ax_depth, ax_hud, ax_bev):
            a.cla(); style(a)

        p_retr = np.clip((frame - 14) / 18, 0, 1)
        p_align = np.clip((frame - 40) / 36, 0, 1)

        # --- current camera image ---
        ax_cam.imshow(cam_q, cmap="gray", aspect="auto")
        ax_cam.add_patch(plt.Rectangle((0, 0), cam_q.shape[1] - 1, cam_q.shape[0] - 1,
                                       fill=False, ec=CYAN, lw=2.0))
        ax_cam.text(8, 22, f"current camera · {qf:04d}", color=CYAN, fontsize=9.5,
                    family="monospace", fontweight="bold",
                    bbox=dict(fc="#0d1117cc", ec="none", pad=2))

        # --- stereo depth ---
        ax_depth.imshow(depth_q, cmap="magma", aspect="auto")
        ax_depth.text(8, 22, "stereo depth", color="#f0a020", fontsize=9.5,
                      family="monospace", fontweight="bold",
                      bbox=dict(fc="#0d1117cc", ec="none", pad=2))

        # --- BEV panel ---
        ax_bev.set_xlim(-17, 17); ax_bev.set_ylim(0, 35); ax_bev.set_aspect("equal")
        ax_bev.text(-16.3, 33.2, "camera bird's-eye view", color=MUTED, fontsize=9,
                    family="monospace")
        if p_retr > 0.5:
            ax_bev.text(-16.3, 1.6, "■ matched map", color=MAPC, fontsize=8, family="monospace")
        ax_bev.text(-16.3, 0.2, "■ current", color=CYAN, fontsize=8, family="monospace")
        # matched map BEV fades in at retrieval lock
        if p_retr > 0.3:
            _scatter(ax_bev, m_bev[:, :2], MAPC, base=7, glow=22,
                     a=0.9 * min(1, (p_retr - 0.3) / 0.7), ag=0.06)
        # query BEV slides from as-seen to the aligned pose, then fades to reveal overlap
        if p_align <= 0:
            shown = q_bev[:, :2]
            q_alpha = 0.85
        else:
            e = _ease(p_align)
            shown = Pose2D(rel.x * e, rel.y * e, rel.yaw * e).transform(q_bev[:, :2])
            q_alpha = 0.85 - 0.4 * e
        _scatter(ax_bev, shown, CYAN, a=q_alpha, ag=0.05)
        # ego marker
        ax_bev.scatter([0], [0], marker="^", s=70, c=FG, zorder=5)

        # --- HUD ---
        ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)
        ax_hud.text(0.04, 0.95, "Match evidence", color=FG, fontsize=10.5,
                    family="monospace", fontweight="bold")
        ax_hud.plot([0.04, 0.96], [0.90, 0.90], color="#21262d", lw=1)
        if p_retr > 0.3:
            ax_hud.text(0.04, 0.80, "retrieval", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.46, 0.80, f"frame {mf:04d}", color=MAPC, fontsize=9.5, family="monospace")
            ax_hud.text(0.46, 0.74, f"score {sim:.2f}", color=FG, fontsize=8.5, family="monospace")
            ax_hud.text(0.04, 0.665, "(first-visit)", color="#5a6675", fontsize=8, family="monospace")
        if p_align > 0:
            e = _ease(p_align)
            ax_hud.text(0.04, 0.55, "BEV align", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.46, 0.55, f"x {rel.x * e:+.1f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.46, 0.495, f"y {rel.y * e:+.1f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.46, 0.44, f"yaw {np.degrees(rel.yaw) * e:+.1f}°", color=CYAN,
                        fontsize=8.5, family="monospace")
            ax_hud.text(0.04, 0.35, "overlap", color=MUTED, fontsize=9, family="monospace")
            ax_hud.add_patch(plt.Rectangle((0.46, 0.335), 0.40, 0.035, color="#21262d"))
            ax_hud.add_patch(plt.Rectangle((0.46, 0.335), 0.40 * overlap * e, 0.035, color=BLUE))

        if p_align > 0:
            caption.set_text("3/3  aligning the two camera BEVs  ->  same place")
        elif p_retr > 0:
            caption.set_text("2/3  retrieving the place from the database")
        else:
            caption.set_text("1/3  camera  ->  stereo depth  ->  bird's-eye view")
        return []

    if preview_frame is not None:
        draw(preview_frame)
        png = OUT.with_name(f"cambev_preview_{preview_frame}.png")
        fig.savefig(png, facecolor=BG, dpi=100)
        print("preview ->", png)
        print(f"q={qf} m={mf} sim={sim:.3f} | align x={rel.x:.2f} y={rel.y:.2f} "
              f"yaw={np.degrees(rel.yaw):.1f} overlap={overlap:.2f} "
              f"| q_pts={len(q_bev)} m_pts={len(m_bev)}")
        return

    # Render each frame ourselves and assemble with PIL -> deterministic frame
    # count, duration, and palette size (PillowWriter mangled both here).
    frames = []
    for k in range(N):
        draw(k)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(buf).quantize(colors=64, method=Image.MEDIANCUT,
                                                     dither=Image.Dither.NONE))
    frames[0].save(OUT, save_all=True, append_images=frames[1:], loop=0,
                   duration=60, optimize=True, disposal=2)
    print("gif ->", OUT, f"({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        build(preview_frame=int(sys.argv[2]))
    else:
        build()
