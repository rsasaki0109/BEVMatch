"""Real LiDAR place-recognition benchmark on KITTI odometry (Recall@K).

BEVMatch's Scan-Context retrieval, exactly as designed (and as the original
Scan-Context paper does it): a rotation-invariant **ring-key** KNN prefilter,
then a full **Scan-Context column-shift** distance rerank on the survivors.

Same place-recognition protocol as the camera benchmark:

  * database  = every velodyne scan of the sequence
  * positive  = pose within D metres AND |dt| > T seconds
  * the |dt| <= T temporal window is excluded from candidates BEFORE reranking
    (so trivial same-pass neighbours never crowd out the genuine revisit)
  * hit@K if a positive is in the top K of the final (reranked) order.

Stages use BEVMatch's own descriptor functions (``ring_key``,
``sc_alignment_distance``) — same code the live pipeline runs.

Data: KITTI odometry seq 00 velodyne (Geiger et al.), public. Extract with
scripts/_fetch_kitti_velodyne_seq00.py / _extract_velodyne_from_head.py.

    python scripts/benchmark_kitti_lidar.py     # -> docs/assets/kitti_lidar_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bevmatch.datasets.loaders import load_kitti_bin, scene_from_points  # noqa: E402
from bevmatch.retrieval.descriptor import (  # noqa: E402
    ScanContextConfig, ring_key, scan_context, sc_alignment_distance,
)

VELO_DIR = Path("$HOME/datasets/kitti_seq00_velodyne")
POSES = Path("$HOME/datasets/kitti_seq00_subset/poses_00.txt")
TIMES = Path("$HOME/datasets/kitti_seq00_full/times.txt")
DISTANCES = [5.0, 10.0, 25.0]
KS = [1, 5, 10, 20]
TIME_EXCLUDE = 30.0     # s
RERANK = 25             # ring-key survivors reranked by full Scan-Context distance
VOXEL = 0.5
CACHE = Path("/tmp/kitti_seq00_sc_cache.npz")
OUT = ROOT / "docs" / "assets" / "kitti_lidar_results.json"
CFG = ScanContextConfig()


def load_poses(path: Path) -> np.ndarray:
    return np.loadtxt(path)[:, [3, 7, 11]]


def encode_all(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ring keys (N, n_rings) and SC grids (N, n_rings, n_sectors)."""
    if CACHE.exists():
        d = np.load(CACHE)
        if len(d["rk"]) == n:
            print("  (loaded SC cache)")
            return d["rk"], d["sc"]
    rks = np.zeros((n, CFG.n_rings), dtype=np.float32)
    scs = np.zeros((n, CFG.n_rings, CFG.n_sectors), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        pts = load_kitti_bin(VELO_DIR / f"{i:06d}.bin")[:, :3]
        scene = scene_from_points(pts, f"f{i}", voxel=VOXEL, drop_ground=True)
        sc = scan_context(scene.primary().xy(), CFG)
        scs[i] = sc
        rks[i] = ring_key(sc)
        if i % 200 == 0 or i == n - 1:
            print(f"  encode {i + 1}/{n}  ({(time.time() - t0):.0f}s)", end="\r", flush=True)
    print()
    np.savez(CACHE, rk=rks, sc=scs)
    return rks, scs


def main() -> None:
    bins = sorted(VELO_DIR.glob("*.bin"))
    pos = load_poses(POSES)
    times = np.loadtxt(TIMES)
    n = min(len(bins), len(pos), len(times))
    pos, times = pos[:n], times[:n]
    print(f"seq00 LiDAR: {n} scans, {times[-1]:.0f}s")

    rks, scs = encode_all(n)
    rks, scs = rks[:n], scs[:n]

    # Stage 1: rotation-invariant ring-key distance, full pairwise (vectorised).
    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    # L2 distance matrix via (a-b)^2 expansion
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T)
    np.maximum(rk_d2, 0, out=rk_d2)
    rk_order = np.argsort(rk_d2, axis=1)   # nearest ring-key first

    dt = np.abs(times[:, None] - times[None, :])
    far = dt > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    # Stage 2: rerank the top-RERANK ring-key survivors (outside temporal window)
    # by the full Scan-Context column-shift distance. Cache the final order.
    print("  reranking (Scan-Context)...", flush=True)
    final_order = [None] * n
    t0 = time.time()
    for q in range(n):
        cand = [j for j in rk_order[q] if far[q, j]][:RERANK]
        scq = scs[q]
        scored = sorted(cand, key=lambda j: sc_alignment_distance(scq, scs[j])[0])
        final_order[q] = scored
        if q % 200 == 0 or q == n - 1:
            print(f"  rerank {q + 1}/{n}  ({(time.time() - t0):.0f}s)", end="\r", flush=True)
    print()

    results = {"sequence": "00", "scans": int(n), "time_exclude_s": TIME_EXCLUDE,
               "descriptor": "ScanContextDescriptor (ring-key KNN prefilter + SC column-shift rerank)",
               "config": {"n_rings": CFG.n_rings, "n_sectors": CFG.n_sectors,
                          "max_range_m": CFG.max_range_m, "voxel_m": VOXEL, "rerank": RERANK},
               "protocol": "Recall@K over revisits; positive = pose<=D & |dt|>T",
               "by_distance": {}}

    for D in DISTANCES:
        positive = (pd <= D) & far
        queries = np.where(positive.any(axis=1))[0]
        recall = {f"recall@{k}": 0 for k in KS}
        for q in queries:
            posset = positive[q]
            hit = next((r for r, j in enumerate(final_order[q][:max(KS)]) if posset[j]), None)
            for k in KS:
                if hit is not None and hit < k:
                    recall[f"recall@{k}"] += 1
        nq = len(queries)
        results["by_distance"][f"{D:.0f}m"] = {
            "n_queries": int(nq),
            **{k: round(v / nq, 4) for k, v in recall.items()},
        }
        r = results["by_distance"][f"{D:.0f}m"]
        print(f"  D={D:>4.0f}m  queries={nq:5d}  R@1={r['recall@1']:.3f}  "
              f"R@5={r['recall@5']:.3f}  R@10={r['recall@10']:.3f}  R@20={r['recall@20']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([results], indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
