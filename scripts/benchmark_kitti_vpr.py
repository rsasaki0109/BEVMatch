"""Real Visual Place Recognition benchmark on KITTI odometry (Recall@K).

Standard VPR protocol — the same one used by Patch-NetVLAD, OverlapNet, etc.:

  * database  = every frame of the sequence
  * positives = frames whose ground-truth pose is within ``D`` metres of the
    query AND separated in time by more than ``T`` seconds (so trivial
    same-pass neighbours never count as a place revisit)
  * a query is any frame that has >= 1 positive (i.e. an actual revisit)
  * at retrieval time we exclude the temporal window |t_q - t| <= T, rank the
    rest by the descriptor's distance, and a query is a hit@K if any positive
    is in the top K.

The distance is exactly ``CameraEmbeddingDescriptor``'s cosine distance — this
measures BEVMatch's own retrieval, not a side implementation. We assert that
the framework's ``SceneDatabase`` reproduces the ranking on a sample.

Data: KITTI odometry (Geiger et al., CVPR 2012), public. seq 00 is the classic
loop-closure-rich sequence. Images + ground-truth poses already on disk.

    python scripts/benchmark_kitti_vpr.py            # -> docs/assets/kitti_vpr_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bevmatch.retrieval import SceneDatabase  # noqa: E402
from bevmatch.sensors.camera import CameraEmbeddingDescriptor, camera_scene  # noqa: E402

# (label, image dir, poses file, times file)
SEQUENCES = [
    ("00",
     Path("$HOME/datasets/kitti_seq00_full/image_0"),
     Path("$HOME/datasets/kitti_seq00_subset/poses_00.txt"),
     Path("$HOME/datasets/kitti_seq00_full/times.txt")),
]
DISTANCES = [5.0, 10.0, 25.0]   # positive radius (m)
KS = [1, 5, 10, 20]
TIME_EXCLUDE = 30.0             # s: exclude this temporal window from retrieval
CACHE_DIR = Path("/tmp/kitti_vpr_emb")
OUT = ROOT / "docs" / "assets" / "kitti_vpr_results.json"


def load_poses(path: Path) -> np.ndarray:
    """KITTI poses: each line is a 3x4 [R|t] row-major. Return (N,3) translations."""
    m = np.loadtxt(path)
    return m[:, [3, 7, 11]]


def embed_sequence(label: str, img_dir: Path, n: int) -> np.ndarray:
    cache = CACHE_DIR / f"seq{label}_resnet18.npy"
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == n:
            return emb
    import torch
    import torch.nn.functional as F
    import torchvision as tv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    # Preprocess on the GPU (decode-only on CPU): the host is CPU-contended, so
    # torchvision's CPU Resize/Normalize would dominate. Resize + normalise on GPU.
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, n, 64):
            arrs = [np.asarray(Image.open(img_dir / f"{f:06d}.png").convert("RGB"))
                    for f in range(i, min(i + 64, n))]
            t = torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2).float().div_(255)
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            t = (t - mean) / std
            out.append(net(t).cpu().numpy())
            print(f"  embed seq{label}: {min(i + 64, n)}/{n}", end="\r", flush=True)
    emb = np.concatenate(out).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    print()
    return emb


def verify_framework(emb: np.ndarray, n_sample: int = 20) -> None:
    """Assert BEVMatch's SceneDatabase reproduces the cosine ranking we evaluate."""
    db = SceneDatabase(descriptor=CameraEmbeddingDescriptor())
    db.prefilter = len(emb)  # rescore every entry — no approximate prefilter
    for i, e in enumerate(emb):
        db.add(camera_scene(f"f{i}", e, place_id=str(i)))
    nrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    for qi in range(0, len(emb), len(emb) // n_sample):
        cand = db.query(camera_scene(f"q{qi}", emb[qi]), top_k=3)
        # framework's top-1 (excluding self) vs vectorised cosine top-1
        sims = nrm @ nrm[qi]
        sims[qi] = -2
        ref = int(sims.argmax())
        got = [int(c.place_id) for c in cand if int(c.place_id) != qi]
        assert got and got[0] == ref, f"framework/cosine mismatch at {qi}: {got[0]} vs {ref}"
    print("  framework check: SceneDatabase ranking == evaluated cosine  (OK)")


def evaluate(label: str, img_dir: Path, poses_path: Path, times_path: Path) -> dict:
    pos = load_poses(poses_path)
    times = np.loadtxt(times_path)
    n = min(len(pos), len(times), len(list(img_dir.glob("*.png"))))
    pos, times = pos[:n], times[:n]
    print(f"seq{label}: {n} frames, {times[-1]:.0f}s")

    emb = embed_sequence(label, img_dir, n)
    emb = emb[:n]
    verify_framework(emb)

    nrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = nrm @ nrm.T                       # cosine similarity, full matrix
    dt = np.abs(times[:, None] - times[None, :])
    far = dt > TIME_EXCLUDE                   # retrievable (outside temporal window)

    # pairwise GT pose distance
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    results = {"sequence": label, "frames": int(n), "time_exclude_s": TIME_EXCLUDE,
               "descriptor": "CameraEmbeddingDescriptor (ResNet-18, ImageNet)",
               "protocol": "Recall@K over revisits; positive = pose<=D & |dt|>T",
               "by_distance": {}}

    order = np.argsort(-sims, axis=1)         # candidates ranked by similarity, per query

    for D in DISTANCES:
        positive = (pd <= D) & far            # GT positive AND retrievable
        has_pos = positive.any(axis=1)
        queries = np.where(has_pos)[0]
        recall = {f"recall@{k}": 0 for k in KS}
        for q in queries:
            ranked = order[q][far[q][order[q]]]   # drop temporally-excluded candidates
            hit_rank = None
            posset = positive[q]
            for rank, c in enumerate(ranked[:max(KS)]):
                if posset[c]:
                    hit_rank = rank
                    break
            for k in KS:
                if hit_rank is not None and hit_rank < k:
                    recall[f"recall@{k}"] += 1
        nq = len(queries)
        results["by_distance"][f"{D:.0f}m"] = {
            "n_queries": int(nq),
            **{k: round(v / nq, 4) for k, v in recall.items()},
        }
        r = results["by_distance"][f"{D:.0f}m"]
        print(f"  D={D:>4.0f}m  queries={nq:5d}  "
              f"R@1={r['recall@1']:.3f}  R@5={r['recall@5']:.3f}  "
              f"R@10={r['recall@10']:.3f}  R@20={r['recall@20']:.3f}")
    return results


def main() -> None:
    all_results = []
    for label, img_dir, poses, times in SEQUENCES:
        if not img_dir.exists() or not poses.exists():
            print(f"skip seq{label}: missing data")
            continue
        all_results.append(evaluate(label, img_dir, poses, times))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
