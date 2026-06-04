"""Geometric-verification fusion with BEVMatch's *real* alignment stage (full ICP).

Finding 3's winning fusion verifies the camera's proposed place geometrically. In
benchmark_kitti_fusion.py that verifier is a cheap Scan-Context alignment-distance
*proxy*. This experiment swaps in BEVMatch's actual SE2 aligner — a 360° BEV
cross-correlation seed refined by ICP, with the framework's own success/failure
verdict (`AlignmentHypothesis.success`, overlap ≥ 0.45) — making the
retrieve → align → evidence claim literal: we accept the camera's top-1 only if
**the alignment stage says the two LiDAR scans actually align**, else fall back to
LiDAR's ranking.

We report, per sequence, R@1 @ 5 m for the two single modalities and two
verified-fusion variants — the SC-distance proxy (α = 1.3) and the full SE2-ICP
verifier — so they sit side by side. No ground truth; same protocol as the other
benchmarks. ICP runs only on the revisit query frames (it is needed only there).

    python scripts/experiment_icp_verification.py            # all sequences
    python scripts/experiment_icp_verification.py 08 06      # a subset

Prereq caches: /tmp/kitti_seq{SEQ}_sc_cache.npz (benchmark_kitti_lidar) and
/tmp/kitti_vpr_emb/seq{SEQ}_eigenplaces_ResNet50_2048.npy (benchmark_kitti_vpr_learned),
plus the velodyne scans the LiDAR benchmark uses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.benchmark_kitti_lidar import (  # noqa: E402
    DISTANCES, KS, RERANK, TIME_EXCLUDE, _paths, load_poses,
)
from bevmatch.retrieval.descriptor import sc_alignment_distance  # noqa: E402
from bevmatch.datasets.loaders import load_kitti_bin, remove_ground, voxel_downsample  # noqa: E402
from bevmatch.alignment.se2 import SE2AlignConfig, align_se2  # noqa: E402

ALPHA = 1.3        # SC-proxy acceptance factor (matches benchmark_kitti_fusion)
VOXEL = 1.0        # downsample for the ICP verifier (BEV verifier covers ±30 m)
# A binary accept/reject verdict needs only a coarse yaw search; 6 deg keeps the
# real aligner (BEV cross-correlation + ICP, overlap>=0.45 success) ~3x faster.
VCFG = SE2AlignConfig(yaw_step_deg=6.0)
LIDAR_CACHE = "/tmp/kitti_seq{seq}_sc_cache.npz"
CAM_CACHE = "/tmp/kitti_vpr_emb/seq{seq}_eigenplaces_ResNet50_2048.npy"
OUT = ROOT / "docs" / "assets" / "kitti_icp_verification_results.json"


def _rank_matrix(score_desc: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-score_desc, axis=1), axis=1).astype(np.float32)


def _xy(velo: Path, i: int) -> np.ndarray:
    pts = load_kitti_bin(velo / f"{i:06d}.bin")[:, :3]
    pts = remove_ground(pts)
    return voxel_downsample(pts[:, :2], VOXEL)


def evaluate(seq: str) -> dict | None:
    lc, cc = Path(LIDAR_CACHE.format(seq=seq)), Path(CAM_CACHE.format(seq=seq))
    if not lc.exists() or not cc.exists():
        print(f"seq{seq}: missing cache, skip")
        return None
    d = np.load(lc)
    rks, scs = d["rk"], d["sc"]
    emb = np.load(cc)
    p = _paths(seq)
    velo = Path(p["velo"])
    pos = load_poses(p["poses"])
    times = np.loadtxt(p["times"])
    n = min(len(rks), len(emb), len(pos), len(times))
    rks, scs, emb, pos, times = rks[:n], scs[:n], emb[:n], pos[:n], times[:n]
    print(f"seq{seq} ICP-verify: {n} frames")

    far = np.abs(times[:, None] - times[None, :]) > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    cnrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    csim = cnrm @ cnrm.T
    csim[~far] = -np.inf
    cam_rank = _rank_matrix(csim)
    cam_top1 = np.argmax(csim, axis=1)

    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T)
    np.maximum(rk_d2, 0, out=rk_d2)
    rk_d2[~far] = np.inf
    rk_order = np.argsort(rk_d2, axis=1)
    lidar_rank = _rank_matrix(-rk_d2) + RERANK
    d_lid = np.full(n, np.inf)
    d_cam = np.full(n, np.inf)
    t0 = time.time()
    for q in range(n):
        surv = rk_order[q, :RERANK]
        surv = surv[far[q, surv]]
        dists = np.array([sc_alignment_distance(scs[q], scs[j])[0] for j in surv])
        sd = np.argsort(dists)
        for i, si in enumerate(sd):
            lidar_rank[q, surv[si]] = i
        if len(sd):
            d_lid[q] = dists[sd[0]]
        if far[q, cam_top1[q]]:
            d_cam[q] = sc_alignment_distance(scs[q], scs[cam_top1[q]])[0]
        if q % 500 == 0 or q == n - 1:
            print(f"  rerank {q + 1}/{n} ({time.time()-t0:.0f}s)", end="\r", flush=True)
    print()

    # SC-proxy verified ranking (the fusion-script method)
    accept_sc = d_cam <= ALPHA * d_lid
    verified_sc = np.where(accept_sc[:, None], cam_rank, lidar_rank)

    # full SE2-ICP verified ranking: accept the camera only where BEVMatch's
    # alignment stage reports success. ICP is run on the headline (5 m) query set;
    # verified[ICP] is reported at 5 m, the radius the comparison turns on.
    query_frames = np.where(((pd <= 5.0) & far).any(axis=1))[0]
    xy_cache: dict[int, np.ndarray] = {}

    def xy(i: int) -> np.ndarray:
        if i not in xy_cache:
            xy_cache[i] = _xy(velo, i)
        return xy_cache[i]

    accept_icp = np.zeros(n, dtype=bool)
    t0 = time.time()
    for k, q in enumerate(query_frames):
        hyp = align_se2(xy(int(q)), xy(int(cam_top1[q])), VCFG)
        accept_icp[q] = bool(hyp.success)
        if k % 200 == 0 or k == len(query_frames) - 1:
            print(f"  icp-verify {k + 1}/{len(query_frames)} ({time.time()-t0:.0f}s)",
                  end="\r", flush=True)
    print()
    verified_icp = np.where(accept_icp[:, None], cam_rank, lidar_rank)

    def recall_at(rank: np.ndarray, D: float) -> dict:
        positive = (pd <= D) & far
        queries = np.where(positive.any(axis=1))[0]
        rec = {f"recall@{k}": 0 for k in KS}
        for q in queries:
            mr = rank[q, positive[q]].min()
            for k in KS:
                if mr < k:
                    rec[f"recall@{k}"] += 1
        nq = len(queries)
        return {"n_queries": int(nq), **{k: round(v / nq, 4) for k, v in rec.items()}}

    res = {"sequence": seq, "frames": int(n),
           "accept_frac_sc": round(float(accept_sc[query_frames].mean()), 3),
           "accept_frac_icp": round(float(accept_icp[query_frames].mean()), 3),
           "by_distance": {}}
    for D in DISTANCES:
        entry = {
            "lidar": recall_at(lidar_rank, D),
            "camera_eigenplaces": recall_at(cam_rank, D),
            "verified_sc_proxy": recall_at(verified_sc, D),
        }
        if D == 5.0:  # ICP run only on the 5 m query set
            entry["verified_full_icp"] = recall_at(verified_icp, D)
        res["by_distance"][f"{D:.0f}m"] = entry
    b = res["by_distance"]["5m"]
    print(f"  R@1@5m  LiDAR={b['lidar']['recall@1']:.3f}  "
          f"Camera={b['camera_eigenplaces']['recall@1']:.3f}  "
          f"verified[SC]={b['verified_sc_proxy']['recall@1']:.3f}  "
          f"verified[ICP]={b['verified_full_icp']['recall@1']:.3f}")
    print(f"  (camera accepted: SC-proxy {res['accept_frac_sc']*100:.0f}%, "
          f"full-ICP {res['accept_frac_icp']*100:.0f}%)")
    return res


def main() -> None:
    seqs = [a for a in sys.argv[1:] if not a.startswith("-")] or ["00", "05", "06", "07", "08"]
    results = [r for r in (evaluate(s) for s in seqs) if r]
    if results:
        print("\n=== R@1 @ 5 m: verification by SC-proxy vs full SE2-ICP ===")
        print(f"  {'seq':>4}  {'LiDAR':>7}  {'Camera':>7}  {'ver[SC]':>8}  {'ver[ICP]':>9}")
        agg = {"lidar": [], "camera_eigenplaces": [], "verified_sc_proxy": [], "verified_full_icp": []}
        for r in results:
            b = r["by_distance"]["5m"]
            for kk in agg:
                agg[kk].append(b[kk]["recall@1"])
            print(f"  {r['sequence']:>4}  {b['lidar']['recall@1']:>7.3f}  "
                  f"{b['camera_eigenplaces']['recall@1']:>7.3f}  "
                  f"{b['verified_sc_proxy']['recall@1']:>8.3f}  {b['verified_full_icp']['recall@1']:>9.3f}")
        print(f"  {'mean':>4}  {np.mean(agg['lidar']):>7.3f}  {np.mean(agg['camera_eigenplaces']):>7.3f}  "
              f"{np.mean(agg['verified_sc_proxy']):>8.3f}  {np.mean(agg['verified_full_icp']):>9.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
