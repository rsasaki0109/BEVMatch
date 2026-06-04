"""Large-scale camera->BEV map matching from REAL KITTI stereo + poses.

    python scripts/make_camera_bev_map_gif.py             # docs/assets/bevmatch_camera_bev_map.gif
    python scripts/make_camera_bev_map_gif.py --preview 60

Real KITTI seq 00 closes a loop: the car returns to its start (frame ~56)
~4500 frames later (frame ~4502) at the same spot, same heading. We accumulate
a band of stereo frames around each visit into a dense *camera* bird's-eye map
(stereo disparity -> depth -> 3D points -> pose-compounded top-down BEV):

  * first-visit band  -> the map (gold)
  * revisit band      -> a query observation, presented at an unknown pose (cyan)

BEVMatch SE2-aligns the query band onto the map band -> the revisit map snaps
onto the first-visit map and ~50 m of street overlaps = the same place,
localized in a large camera-derived BEV (Principle 2: modality=camera, repr=BEV).

Data: KITTI odometry seq 00 (Geiger et al.), grayscale stereo + GT poses.
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

DATA = Path("$HOME/datasets/kitti_seq00_full")
L_DIR = DATA / "image_0"
R_DIR = DATA / "image_1"
POSES = DATA / "poses_00.txt"
OUT = ROOT / "docs" / "assets" / "bevmatch_camera_bev_map.gif"

FX = 718.856
CX = 607.1928
CY = 185.2157
BASELINE = 386.1448 / FX  # ~0.537 m

MAP_CENTER = 56      # first-visit frame (the map)
QRY_CENTER = 4502    # revisit frame (the query observation)
BAND = 22            # frames accumulated each side of a center (-> ~45 frames, ~50 m)
STEP = 2             # use every STEP-th frame in the band
VOX = 0.35

BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"
CYAN = "#22d3ee"; ACCENT = "#f0a020"; BLUE = "#58a6ff"
MAPC = "#fbbf24"  # matched map BEV — gold, complementary to the cyan query

_SGBM = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5,
    P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)


def _poses():
    return np.loadtxt(POSES).reshape(-1, 3, 4)


def _points3d(frame):
    """Stereo -> 3D points in the camera frame (X right, Y down, Z forward)."""
    left = cv2.imread(str(L_DIR / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(R_DIR / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    disp = _SGBM.compute(left, right).astype(np.float32) / 16.0
    h, w = disp.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    m = disp > 1.0
    z = FX * BASELINE / disp[m]
    x = (us[m] - CX) * z / FX
    y = (vs[m] - CY) * z / FX
    keep = (z > 3) & (z < 32) & (np.abs(x) < 16) & (y < 1.3) & (y > -3.0)
    return np.stack([x[keep], y[keep], z[keep]], axis=1)


def _voxel(pts, vox):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts[:, :2] / vox).astype(np.int64)
    _, idx = np.unique(keys[:, 0] * 100003 + keys[:, 1], return_index=True)
    return pts[idx]


def accumulate(center, poses):
    """Accumulate a band of stereo frames into the center frame's local BEV (x, z)."""
    Rc, tc = poses[center][:, :3], poses[center][:, 3]
    out = []
    for f in range(center - BAND, center + BAND + 1, STEP):
        if f < 0:
            continue
        p = _points3d(f)
        if len(p) == 0:
            continue
        Ri, ti = poses[f][:, :3], poses[f][:, 3]
        world = p @ Ri.T + ti               # cam_f -> world
        local = (world - tc) @ Rc           # world -> center-local
        out.append(local)
    pts = np.concatenate(out)
    bev = np.stack([pts[:, 0], pts[:, 2]], axis=1)  # x right, z forward
    return _voxel(bev, VOX)


def _cam_img(frame, w=420):
    im = Image.open(L_DIR / f"{frame:06d}.png").convert("L")
    return np.asarray(im.resize((w, int(im.height * w / im.width))))


def _scatter(ax, pts, color, base=4, glow=14, a=0.9, ag=0.05):
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=glow, c=color, alpha=ag, linewidths=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=base, c=color, alpha=a, linewidths=0)


def _ease(t):
    return t * t * (3 - 2 * t)


def build(preview_frame=None):
    poses = _poses()
    map_bev = accumulate(MAP_CENTER, poses)    # gold
    qry_bev = accumulate(QRY_CENTER, poses)    # cyan

    # real alignment of the two accumulated camera maps
    align = SE2Aligner().align(
        Scene("q", observations={"l": Observation("l", qry_bev)}),
        Scene("m", observations={"l": Observation("l", map_bev)}))
    rel = align.relative_pose
    overlap = align.overlap_ratio

    # present the query at an "unknown" pose, recovered by alignment (localization framing)
    unknown = Pose2D(13.0, -7.0, np.deg2rad(24.0))
    qry_unknown = unknown.transform(qry_bev)
    cam_q = _cam_img(QRY_CENTER)

    # view extent
    allpts = np.concatenate([map_bev, qry_unknown])
    cx, cz = allpts[:, 0].mean(), allpts[:, 1].mean()
    span = 1.05 * max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1])) / 2

    N = 95
    fig = plt.figure(figsize=(11, 5.6), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 24, left=0.015, right=0.985, top=0.85, bottom=0.05,
                          height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.0)
    ax_cam = fig.add_subplot(gs[0, :7])
    ax_bev = fig.add_subplot(gs[:, 8:17])
    ax_hud = fig.add_subplot(gs[:, 18:])

    fig.text(0.018, 0.93, "BEVMatch", color=FG, fontsize=21, fontweight="bold", family="monospace")
    fig.text(0.142, 0.937, "· localization in a large camera BEV map", color=MUTED,
             fontsize=10.5, family="monospace")
    fig.text(0.018, 0.9, f"real KITTI seq 00 · stereo depth -> BEV · ~{2 * BAND} frames accumulated each side",
             color="#5a6675", fontsize=8.5, family="monospace")
    caption = fig.text(0.018, 0.012, "", color=BLUE, fontsize=11, family="monospace")

    def style(a):
        a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    def draw(frame):
        for a in (ax_cam, ax_bev, ax_hud):
            a.cla(); style(a)

        p_retr = np.clip((frame - 16) / 20, 0, 1)
        p_align = np.clip((frame - 44) / 40, 0, 1)

        ax_cam.imshow(cam_q, cmap="gray", aspect="auto")
        ax_cam.add_patch(plt.Rectangle((0, 0), cam_q.shape[1] - 1, cam_q.shape[0] - 1,
                                       fill=False, ec=CYAN, lw=2.0))
        ax_cam.text(8, 22, f"revisit · frame {QRY_CENTER}", color=CYAN, fontsize=9.5,
                    family="monospace", fontweight="bold",
                    bbox=dict(fc="#0d1117cc", ec="none", pad=2))

        ax_bev.set_xlim(cx - span, cx + span); ax_bev.set_ylim(cz - span, cz + span)
        ax_bev.set_aspect("equal")
        ax_bev.text(0.02, 0.965, "camera BEV map", color=MUTED, fontsize=9,
                    family="monospace", transform=ax_bev.transAxes)
        if p_retr > 0.5:
            ax_bev.text(0.02, 0.04, "■ first-visit map", color=MAPC, fontsize=8,
                        family="monospace", transform=ax_bev.transAxes)
        ax_bev.text(0.78, 0.04, "■ revisit", color=CYAN, fontsize=8,
                    family="monospace", transform=ax_bev.transAxes)

        if p_retr > 0.25:
            _scatter(ax_bev, map_bev, MAPC, base=5, glow=16,
                     a=0.9 * min(1, (p_retr - 0.25) / 0.75), ag=0.06)

        if p_align <= 0:
            shown = qry_unknown
            q_alpha = 0.85
        else:
            e = _ease(p_align)
            # interpolate the displayed pose from the unknown pose to the recovered fit
            dx = unknown.x + (rel.x - unknown.x) * e
            dy = unknown.y + (rel.y - unknown.y) * e
            dyaw = unknown.yaw + (rel.yaw - unknown.yaw) * e
            shown = Pose2D(dx, dy, dyaw).transform(qry_bev)
            q_alpha = 0.85 - 0.4 * e
        _scatter(ax_bev, shown, CYAN, a=q_alpha, ag=0.06)

        # HUD
        ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)
        ax_hud.text(0.04, 0.95, "Localization evidence", color=FG, fontsize=10.5,
                    family="monospace", fontweight="bold")
        ax_hud.plot([0.04, 0.96], [0.90, 0.90], color="#21262d", lw=1)
        ax_hud.text(0.04, 0.81, "map points", color=MUTED, fontsize=9, family="monospace")
        ax_hud.text(0.5, 0.81, f"{len(map_bev):,}", color=MAPC, fontsize=9, family="monospace")
        ax_hud.text(0.04, 0.75, "query points", color=MUTED, fontsize=9, family="monospace")
        ax_hud.text(0.5, 0.75, f"{len(qry_bev):,}", color=CYAN, fontsize=9, family="monospace")
        if p_align > 0:
            e = _ease(p_align)
            ax_hud.text(0.04, 0.60, "recovered pose", color=MUTED, fontsize=9, family="monospace")
            ax_hud.text(0.5, 0.60, f"x {rel.x * e:+.1f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.5, 0.545, f"y {rel.y * e:+.1f} m", color=CYAN, fontsize=8.5, family="monospace")
            ax_hud.text(0.5, 0.49, f"yaw {np.degrees(rel.yaw) * e:+.1f}°", color=CYAN,
                        fontsize=8.5, family="monospace")
            ax_hud.text(0.04, 0.40, "overlap", color=MUTED, fontsize=9, family="monospace")
            ax_hud.add_patch(plt.Rectangle((0.5, 0.385), 0.40, 0.035, color="#21262d"))
            ax_hud.add_patch(plt.Rectangle((0.5, 0.385), 0.40 * overlap * e, 0.035, color=BLUE))
            ax_hud.text(0.5, 0.30, f"{overlap * e * 100:.0f}% of ~50 m street", color="#9aa4b2",
                        fontsize=8, family="monospace")

        if p_align > 0:
            caption.set_text("3/3  aligning the revisit map onto the first-visit map  ->  localized")
        elif p_retr > 0:
            caption.set_text("2/3  retrieving the first-visit map for this place")
        else:
            caption.set_text("1/3  revisit observation at an unknown pose")
        return []

    if preview_frame is not None:
        draw(preview_frame)
        png = OUT.with_name(f"cbmap_preview_{preview_frame}.png")
        fig.savefig(png, facecolor=BG, dpi=100)
        print("preview ->", png)
        print(f"map_pts={len(map_bev)} qry_pts={len(qry_bev)} | align x={rel.x:.2f} "
              f"y={rel.y:.2f} yaw={np.degrees(rel.yaw):.1f} overlap={overlap:.2f}")
        return

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
