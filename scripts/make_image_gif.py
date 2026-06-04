"""Generate a camera (image) hero GIF: real visual place recognition on KITTI.

    python scripts/make_image_gif.py             # docs/assets/bevmatch_camera.gif
    python scripts/make_image_gif.py --preview 20

Real KITTI sequence 00 (a drive that revisits its start). Frames from the first
visit form a camera place database; frames from the revisit (~4500) are queries.
BEVMatch's CameraEmbeddingDescriptor (ResNet-18 image embeddings) retrieves the
matching past frame for each current frame — a real loop closure, the actual
camera-modality pipeline (Principle 2: same retrieval framework, image input).

Data: KITTI odometry seq 00 (Geiger et al.). Images shown for research demo.
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
from bevmatch.retrieval import SceneDatabase  # noqa: E402
from bevmatch.sensors.camera import CameraEmbeddingDescriptor, camera_scene  # noqa: E402

IMG_DIR = Path("/home/sasaki/datasets/kitti_seq00_full/image_0")
EMB_CACHE = Path("/tmp/kitti_emb.npz")
OUT = ROOT / "docs" / "assets" / "bevmatch_camera.gif"

BG = "#0d1117"; FG = "#c9d1d9"; MUTED = "#8b949e"; GREEN = "#34d399"; CYAN = "#22d3ee"; ACCENT = "#f0a020"
DB_FRAMES = list(range(0, 1600, 4))
Q_FRAMES = list(range(4500, 4531, 2))


def _embed(frames):
    import torch, torchvision as tv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); net.eval().to(dev)
    tf = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)), tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), 64):
            batch = [tf(Image.open(IMG_DIR / f"{f:06d}.png").convert("RGB")) for f in frames[i:i + 64]]
            out.append(net(torch.stack(batch).to(dev)).cpu().numpy())
    return np.concatenate(out)


def embeddings():
    if EMB_CACHE.exists():
        d = np.load(EMB_CACHE)
        return d["db"], d["q"]
    db, q = _embed(DB_FRAMES), _embed(Q_FRAMES)
    np.savez(EMB_CACHE, db=db, q=q)
    return db, q


def match():
    db_emb, q_emb = embeddings()
    database = SceneDatabase(descriptor=CameraEmbeddingDescriptor())
    for f, e in zip(DB_FRAMES, db_emb):
        database.add(camera_scene(f"db_{f}", e, place_id=f"{f}"))
    results = []
    for qf, qe in zip(Q_FRAMES, q_emb):
        cand = database.query(camera_scene(f"q_{qf}", qe), top_k=1)[0]
        results.append((qf, int(cand.place_id), cand.score))
    return results


def _img(frame, w=520):
    im = Image.open(IMG_DIR / f"{frame:06d}.png").convert("L")
    im = im.resize((w, int(im.height * w / im.width)))
    return np.asarray(im)


def build(preview_frame=None):
    results = match()
    # preload images
    q_imgs = {qf: _img(qf) for qf, _, _ in results}
    m_imgs = {mf: _img(mf) for _, mf, _ in results}
    # filmstrip db thumbnails
    strip_frames = DB_FRAMES[::6]
    thumbs = {f: _img(f, 120) for f in strip_frames}

    HOLD = 5
    N = len(results) * HOLD
    fig = plt.figure(figsize=(8.6, 6.2), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 3.0, 1.1], left=0.04, right=0.96,
                          top=0.9, bottom=0.05, hspace=0.28)
    ax_q = fig.add_subplot(gs[0]); ax_m = fig.add_subplot(gs[1]); ax_s = fig.add_subplot(gs[2])

    fig.text(0.04, 0.95, "BEVMatch", color=FG, fontsize=18, fontweight="bold", family="monospace")
    fig.text(0.215, 0.953, "· visual place recognition (camera)", color=MUTED, fontsize=10.5, family="monospace")
    fig.text(0.04, 0.918, "real KITTI seq 00 · loop closure · ResNet-18 embeddings", color="#5a6675",
             fontsize=8, family="monospace")

    def draw(frame):
        idx = min(frame // HOLD, len(results) - 1)
        qf, mf, sim = results[idx]
        for a in (ax_q, ax_m, ax_s):
            a.cla(); a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)

        ax_q.imshow(q_imgs[qf], cmap="gray", aspect="auto")
        ax_q.add_patch(plt.Rectangle((0, 0), q_imgs[qf].shape[1] - 1, q_imgs[qf].shape[0] - 1,
                                     fill=False, ec=CYAN, lw=2.5))
        ax_q.text(8, 22, f"current camera  ·  frame {qf:04d}  (revisit)", color=CYAN,
                  fontsize=10, family="monospace", fontweight="bold",
                  bbox=dict(fc="#0d1117cc", ec="none", pad=2))

        ax_m.imshow(m_imgs[mf], cmap="gray", aspect="auto")
        ax_m.add_patch(plt.Rectangle((0, 0), m_imgs[mf].shape[1] - 1, m_imgs[mf].shape[0] - 1,
                                     fill=False, ec=GREEN, lw=2.5))
        ax_m.text(8, 22, f"BEVMatch match  ·  frame {mf:04d}  (first visit)  ·  sim {sim:.2f}",
                  color=GREEN, fontsize=10, family="monospace", fontweight="bold",
                  bbox=dict(fc="#0d1117cc", ec="none", pad=2))

        # filmstrip with matched highlighted
        ax_s.set_xlim(0, len(strip_frames)); ax_s.set_ylim(0, 1)
        ax_s.text(0.0, 1.28, "camera place database (first-visit frames)", color=MUTED,
                  fontsize=8, family="monospace", transform=ax_s.transAxes)
        nearest = min(range(len(strip_frames)), key=lambda i: abs(strip_frames[i] - mf))
        for i, f in enumerate(strip_frames):
            ax_s.imshow(thumbs[f], cmap="gray", extent=(i + 0.04, i + 0.96, 0.08, 0.92), aspect="auto", zorder=1)
            if i == nearest:
                ax_s.add_patch(plt.Rectangle((i + 0.02, 0.05), 0.96, 0.9, fill=False, ec=ACCENT, lw=2.5, zorder=3))
        ax_s.set_xlim(0, len(strip_frames)); ax_s.set_ylim(0, 1)
        return []

    if preview_frame is not None:
        draw(preview_frame)
        png = OUT.with_name(f"cam_preview_{preview_frame}.png")
        fig.savefig(png, facecolor=BG, dpi=100); print("preview ->", png); return

    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, draw, frames=N, interval=120, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=8))
    print("gif ->", OUT, f"({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        build(preview_frame=int(sys.argv[2]))
    else:
        build()
