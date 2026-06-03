"""Real LiDAR place-recognition benchmark on KITTI odometry (Recall@K).

BEVMatch's Scan-Context retrieval, exactly as designed (and as the original
Scan-Context paper does it): a rotation-invariant **ring-key** KNN prefilter,
then a full **Scan-Context column-shift** distance rerank on the survivors.

Standard place-recognition protocol, evaluated across the KITTI loop sequences:

  * database  = every velodyne scan of the sequence
  * positive  = pose within D metres AND |dt| > T seconds
  * the |dt| <= T temporal window is excluded from candidates BEFORE reranking
    (so trivial same-pass neighbours never crowd out the genuine revisit)
  * hit@K if a positive is in the top K of the final (reranked) order.

Stages use BEVMatch's own descriptor functions (``ring_key``,
``sc_alignment_distance``) — same code the live pipeline runs.

Data: KITTI odometry velodyne (Geiger et al.), public. Sequences 00/05/06/07/08
are the loop-closure-rich drives.

    python scripts/benchmark_kitti_lidar.py            # all available sequences
    python scripts/benchmark_kitti_lidar.py 00 05      # a subset
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

DATASETS = "$HOME/datasets"
SUBSETS = f"{DATASETS}/kitti_odometry_training_subsets"

# velodyne dir, poses file, times file per sequence
def _paths(seq: str) -> dict:
    velo = f"{DATASETS}/kitti_seq{seq}_velodyne"
    if seq == "00":
        return {"velo": velo, "poses": f"{DATASETS}/kitti_seq00_subset/poses_00.txt",
                "times": f"{DATASETS}/kitti_seq00_full/times.txt"}
    return {"velo": velo, "poses": f"{SUBSETS}/seq{seq}/poses_{seq}.txt",
            "times": f"{SUBSETS}/seq{seq}/times.txt"}


SEQUENCES = ["00", "05", "06", "07", "08"]
DISTANCES = [5.0, 10.0, 25.0]
KS = [1, 5, 10, 20]
TIME_EXCLUDE = 30.0
RERANK = 25
VOXEL = 0.5
OUT = ROOT / "docs" / "assets" / "kitti_lidar_results.json"
CFG = ScanContextConfig()


def load_poses(path: str) -> np.ndarray:
    return np.loadtxt(path)[:, [3, 7, 11]]


def encode_all(seq: str, velo: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(f"/tmp/kitti_seq{seq}_sc_cache.npz")
    if cache.exists():
        d = np.load(cache)
        if len(d["rk"]) == n:
            print("  (loaded SC cache)")
            return d["rk"], d["sc"]
    rks = np.zeros((n, CFG.n_rings), dtype=np.float32)
    scs = np.zeros((n, CFG.n_rings, CFG.n_sectors), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        pts = load_kitti_bin(velo / f"{i:06d}.bin")[:, :3]
        scene = scene_from_points(pts, f"f{i}", voxel=VOXEL, drop_ground=True)
        sc = scan_context(scene.primary().xy(), CFG)
        scs[i] = sc
        rks[i] = ring_key(sc)
        if i % 200 == 0 or i == n - 1:
            print(f"  encode {i + 1}/{n}  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print()
    np.savez(cache, rk=rks, sc=scs)
    return rks, scs


def evaluate(seq: str) -> dict | None:
    p = _paths(seq)
    velo = Path(p["velo"])
    bins = sorted(velo.glob("*.bin"))
    if not bins:
        print(f"seq{seq}: no velodyne, skip")
        return None
    pos = load_poses(p["poses"])
    times = np.loadtxt(p["times"])
    n = min(len(bins), len(pos), len(times))
    pos, times = pos[:n], times[:n]
    print(f"seq{seq} LiDAR: {n} scans, {times[-1]:.0f}s")

    rks, scs = encode_all(seq, velo, n)
    rks, scs = rks[:n], scs[:n]

    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T)
    np.maximum(rk_d2, 0, out=rk_d2)
    rk_order = np.argsort(rk_d2, axis=1)

    dt = np.abs(times[:, None] - times[None, :])
    far = dt > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    print("  reranking (Scan-Context)...", flush=True)
    final_order = [None] * n
    t0 = time.time()
    for q in range(n):
        cand = [j for j in rk_order[q] if far[q, j]][:RERANK]
        scq = scs[q]
        final_order[q] = sorted(cand, key=lambda j: sc_alignment_distance(scq, scs[j])[0])
        if q % 200 == 0 or q == n - 1:
            print(f"  rerank {q + 1}/{n}  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print()

    res = {"sequence": seq, "scans": int(n), "time_exclude_s": TIME_EXCLUDE,
           "descriptor": "ScanContextDescriptor (ring-key KNN prefilter + SC column-shift rerank)",
           "config": {"n_rings": CFG.n_rings, "n_sectors": CFG.n_sectors,
                      "max_range_m": CFG.max_range_m, "voxel_m": VOXEL, "rerank": RERANK},
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
        res["by_distance"][f"{D:.0f}m"] = {"n_queries": int(nq),
                                           **{k: round(v / nq, 4) for k, v in recall.items()}}
        r = res["by_distance"][f"{D:.0f}m"]
        print(f"  D={D:>4.0f}m  queries={nq:5d}  R@1={r['recall@1']:.3f}  "
              f"R@5={r['recall@5']:.3f}  R@10={r['recall@10']:.3f}  R@20={r['recall@20']:.3f}")
    return res


def main() -> None:
    seqs = sys.argv[1:] or SEQUENCES
    results = [r for r in (evaluate(s) for s in seqs) if r]
    # mean across sequences at 5 m (the headline radius)
    if results:
        for D in DISTANCES:
            key = f"{D:.0f}m"
            print(f"\n=== mean @ {key} over {len(results)} sequences ===")
            for k in KS:
                vals = [r["by_distance"][key][f"recall@{k}"] for r in results]
                print(f"  mean recall@{k} = {np.mean(vals):.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
