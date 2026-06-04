"""Learned-descriptor VPR benchmark: EigenPlaces vs the ResNet-18 baseline on KITTI.

A SOTA reference point for BEVMatch's *camera* retrieval, and a demonstration of
the descriptor-plugin value: swap the generic ImageNet ResNet-18 features for a
descriptor *purpose-trained for place recognition* and watch recall move.

Descriptor: **EigenPlaces** (Berton et al., ICCV 2023, MIT). A ResNet + GeM +
fully-connected head trained on San Francisco eXtra Large (SF-XL) street-view
imagery and distributed via torch.hub. Crucially, **SF-XL is disjoint from
KITTI**, so KITTI is a genuine held-out domain for *every* sequence — unlike a
KITTI-trained LiDAR model, there is no train-on-test caveat anywhere here, and
the full 00/05/06/07/08 comparison is fair.

Same protocol as scripts/benchmark_kitti_vpr.py (positive = pose <= D m AND
|dt| > 30 s; temporal window excluded; Recall@K over revisits; the framework's
SceneDatabase is asserted to reproduce the cosine ranking), so the numbers are
directly comparable to the ResNet-18 baseline table.

Honest caveats: we only have KITTI grayscale image_0 (replicated to 3 channels),
while EigenPlaces trained on RGB; that domain gap is real and reported, not hidden.

    python scripts/benchmark_kitti_vpr_learned.py                  # all sequences, ResNet50/2048
    python scripts/benchmark_kitti_vpr_learned.py 00 05            # a subset
    python scripts/benchmark_kitti_vpr_learned.py --backbone ResNet18 --dim 512

License: EigenPlaces is MIT and loaded at runtime via torch.hub; nothing is
vendored into this Apache-2.0 project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.benchmark_kitti_vpr import (  # noqa: E402  (identical protocol)
    DISTANCES, KS, SEQ_IDS, TIME_EXCLUDE, _seq_paths, load_poses, verify_framework,
)

CACHE_DIR = Path("/tmp/kitti_vpr_emb")
OUT = ROOT / "docs" / "assets" / "kitti_vpr_learned_results.json"
INPUT_HW = (384, 384)  # square resize fed to the fully-convolutional backbone


def embed_sequence(label: str, img_dir: Path, n: int, backbone: str, dim: int) -> np.ndarray:
    cache = CACHE_DIR / f"seq{label}_eigenplaces_{backbone}_{dim}.npy"
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == n:
            return emb
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # EigenPlaces (MIT), trained on SF-XL; KITTI is a held-out domain.
    net = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                         backbone=backbone, fc_output_dim=dim)
    net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, n, 64):
            arrs = [np.asarray(Image.open(img_dir / f"{f:06d}.png").convert("RGB"))
                    for f in range(i, min(i + 64, n))]
            t = torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2).float().div_(255)
            t = F.interpolate(t, size=INPUT_HW, mode="bilinear", align_corners=False)
            t = (t - mean) / std
            out.append(net(t).cpu().numpy())
            print(f"  embed seq{label}: {min(i + 64, n)}/{n}", end="\r", flush=True)
    emb = np.concatenate(out).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    print()
    return emb


def evaluate(label: str, img_dir: Path, poses_path: Path, times_path: Path,
             backbone: str, dim: int) -> dict:
    pos = load_poses(poses_path)
    times = np.loadtxt(times_path)
    n = min(len(pos), len(times), len(list(img_dir.glob("*.png"))))
    pos, times = pos[:n], times[:n]
    print(f"seq{label}: {n} frames, {times[-1]:.0f}s")

    emb = embed_sequence(label, img_dir, n, backbone, dim)[:n]
    verify_framework(emb)   # the learned descriptor plugs into BEVMatch's SceneDatabase too

    nrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = nrm @ nrm.T
    dt = np.abs(times[:, None] - times[None, :])
    far = dt > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    results = {"sequence": label, "frames": int(n), "time_exclude_s": TIME_EXCLUDE,
               "descriptor": f"EigenPlaces ({backbone}, {dim}-d, SF-XL pretrained — KITTI held out)",
               "source": "gmberton/eigenplaces (MIT)",
               "protocol": "Recall@K over revisits; positive = pose<=D & |dt|>T",
               "by_distance": {}}
    order = np.argsort(-sims, axis=1)
    for D in DISTANCES:
        positive = (pd <= D) & far
        queries = np.where(positive.any(axis=1))[0]
        recall = {f"recall@{k}": 0 for k in KS}
        for q in queries:
            ranked = order[q][far[q][order[q]]]
            posset = positive[q]
            hit_rank = next((r for r, c in enumerate(ranked[:max(KS)]) if posset[c]), None)
            for k in KS:
                if hit_rank is not None and hit_rank < k:
                    recall[f"recall@{k}"] += 1
        nq = len(queries)
        results["by_distance"][f"{D:.0f}m"] = {"n_queries": int(nq),
            **{k: round(v / nq, 4) for k, v in recall.items()}}
        r = results["by_distance"][f"{D:.0f}m"]
        print(f"  D={D:>4.0f}m  queries={nq:5d}  R@1={r['recall@1']:.3f}  "
              f"R@5={r['recall@5']:.3f}  R@10={r['recall@10']:.3f}  R@20={r['recall@20']:.3f}")
    return results


def _print_sidebyside(learned: list[dict]) -> None:
    base_json = ROOT / "docs" / "assets" / "kitti_vpr_results.json"
    if not base_json.exists():
        return
    base = {r["sequence"]: r for r in json.loads(base_json.read_text())}
    print("\n=== camera R@1 @ 5 m: ResNet-18 (ImageNet) baseline vs EigenPlaces (learned VPR) ===")
    print(f"  {'seq':>4}  {'ResNet-18 baseline':>18}  {'EigenPlaces':>12}")
    for r in learned:
        s = r["sequence"]
        b = base.get(s, {}).get("by_distance", {}).get("5m", {}).get("recall@1")
        o = r["by_distance"]["5m"]["recall@1"]
        if b is not None:
            print(f"  {s:>4}  {b:>18.3f}  {o:>12.3f}")


def main() -> None:
    argv = sys.argv[1:]
    backbone, dim = "ResNet50", 2048
    if "--backbone" in argv:
        backbone = argv[argv.index("--backbone") + 1]
    if "--dim" in argv:
        dim = int(argv[argv.index("--dim") + 1])
    want = [a for a in argv if a in SEQ_IDS] or SEQ_IDS
    all_results = []
    for s in want:
        label, img_dir, poses, times = _seq_paths(s)
        if not img_dir.exists() or not poses.exists():
            print(f"skip seq{label}: missing data")
            continue
        all_results.append(evaluate(label, img_dir, poses, times, backbone, dim))
    if all_results:
        for D in DISTANCES:
            key = f"{D:.0f}m"
            vals = {k: np.mean([r["by_distance"][key][f"recall@{k}"] for r in all_results])
                    for k in KS}
            print(f"\n=== mean @ {key} over {len(all_results)} sequences ===  "
                  + "  ".join(f"R@{k}={vals[k]:.3f}" for k in KS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT}")
    _print_sidebyside(all_results)


if __name__ == "__main__":
    main()
