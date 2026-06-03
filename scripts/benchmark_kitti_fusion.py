"""Late-fusion retrieval: LiDAR + camera under one roof, on KITTI (Recall@K).

The experiment findings.md asks for: if each sensor fails differently, does a
retriever that uses *both* recover the cases where either one is blind? We fuse
BEVMatch's two single-modality rankings — LiDAR Scan-Context and the learned
EigenPlaces camera descriptor — with **Reciprocal Rank Fusion** (RRF, Cormack et
al. 2009), a standard, parameter-light, scale-free rank combiner:

    score(j) = 1/(C + rank_lidar(j)) + 1/(C + rank_camera(j)),  C = 60

We report two fusion strategies and contrast them honestly:
  * **naive RRF** (equal weight) — and we show it *fails* on the blind-modality
    case: when one sensor is noise, equal weight drags the working sensor down.
  * **confidence-gated** — per query, trust the more self-confident sensor, where
    confidence is the top-1-vs-top-2 score margin (Lowe's ratio-test intuition),
    normalised per modality. No ground truth is used to gate.

Same protocol as the single-modality benchmarks (positive = pose <= D m AND
|dt| > 30 s; temporal window excluded; Recall@K over revisits), so the fusion
columns are directly comparable to the LiDAR-only and camera-only columns.

Runs entirely on cached descriptors (LiDAR ring-key + Scan-Context grids, camera
EigenPlaces embeddings) with numpy + BEVMatch's own `sc_alignment_distance` — no
network, no external model code. Rank matrices are vectorised; only the per-query
Scan-Context rerank of the top-RERANK ring-key survivors stays in Python.

    python scripts/benchmark_kitti_fusion.py            # all sequences
    python scripts/benchmark_kitti_fusion.py 00 08      # a subset

Prereq caches (produced by the other benchmarks):
  /tmp/kitti_seq{SEQ}_sc_cache.npz                       (benchmark_kitti_lidar)
  /tmp/kitti_vpr_emb/seq{SEQ}_eigenplaces_ResNet50_2048.npy (benchmark_kitti_vpr_learned)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.benchmark_kitti_lidar import (  # noqa: E402  (identical protocol + paths)
    DISTANCES, KS, RERANK, TIME_EXCLUDE, _paths, load_poses,
)
from bevmatch.retrieval.descriptor import sc_alignment_distance  # noqa: E402

RRF_C = 60.0
LIDAR_CACHE = "/tmp/kitti_seq{seq}_sc_cache.npz"
CAM_CACHE = "/tmp/kitti_vpr_emb/seq{seq}_eigenplaces_ResNet50_2048.npy"
OUT = ROOT / "docs" / "assets" / "kitti_fusion_results.json"


def _rank_matrix(score_desc: np.ndarray) -> np.ndarray:
    """rank[q, j] = position of j when row q is sorted by *descending* score."""
    return np.argsort(np.argsort(-score_desc, axis=1), axis=1).astype(np.float32)


def evaluate(seq: str) -> dict | None:
    lc = Path(LIDAR_CACHE.format(seq=seq))
    cc = Path(CAM_CACHE.format(seq=seq))
    if not lc.exists() or not cc.exists():
        print(f"seq{seq}: missing cache (lidar={lc.exists()}, cam={cc.exists()}), skip")
        return None
    d = np.load(lc)
    rks, scs = d["rk"], d["sc"]
    emb = np.load(cc)
    p = _paths(seq)
    pos = load_poses(p["poses"])
    times = np.loadtxt(p["times"])
    n = min(len(rks), len(emb), len(pos), len(times))
    rks, scs, emb, pos, times = rks[:n], scs[:n], emb[:n], pos[:n], times[:n]
    print(f"seq{seq} fusion: {n} frames")

    dt = np.abs(times[:, None] - times[None, :])
    far = dt > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    # --- camera: cosine similarity -> rank matrix (excluded candidates sink) ---
    cnrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    csim = cnrm @ cnrm.T
    csim[~far] = -np.inf
    cam_rank = _rank_matrix(csim)
    # per-query self-confidence: top-1 minus top-2 similarity (Lowe-style margin).
    # A blind modality has no standout candidate -> small gap -> low confidence.
    cs = np.sort(csim, axis=1)[:, ::-1]
    cam_gap = cs[:, 0] - cs[:, 1]

    # --- lidar: ring-key prefilter, then Scan-Context rerank of top-RERANK ---
    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T)
    np.maximum(rk_d2, 0, out=rk_d2)
    rk_d2[~far] = np.inf
    rk_order = np.argsort(rk_d2, axis=1)
    # baseline lidar rank = ring-key rank pushed back by RERANK; survivors overwrite 0..RERANK-1
    lidar_rank = _rank_matrix(-rk_d2) + RERANK
    lidar_gap = np.zeros(n, dtype=float)  # SC distance margin: dist@2 - dist@1
    for q in range(n):
        survivors = rk_order[q, :RERANK]
        survivors = survivors[far[q, survivors]]
        dists = np.array([sc_alignment_distance(scs[q], scs[j])[0] for j in survivors])
        sd = np.argsort(dists)
        for i, si in enumerate(sd):
            lidar_rank[q, survivors[si]] = i
        if len(sd) >= 2:
            lidar_gap[q] = dists[sd[1]] - dists[sd[0]]
        if q % 500 == 0 or q == n - 1:
            print(f"  rerank {q + 1}/{n}", end="\r", flush=True)
    print()

    # --- naive equal-weight RRF -> rank matrix ---
    rrf = 1.0 / (RRF_C + lidar_rank) + 1.0 / (RRF_C + cam_rank)
    rrf[~far] = -np.inf
    fused_rank = _rank_matrix(rrf)

    # --- confidence-gated fusion: per query, trust the more self-confident sensor.
    # Gaps live on different scales (cosine vs SC distance), so normalise each by
    # its own sequence-median before comparing. No ground truth is used. ---
    cam_n = cam_gap / (np.median(cam_gap[np.isfinite(cam_gap)]) + 1e-9)
    lid_n = lidar_gap / (np.median(lidar_gap[lidar_gap > 0]) + 1e-9)
    pick_cam = cam_n >= lid_n
    gated_rank = np.where(pick_cam[:, None], cam_rank, lidar_rank)
    res_pick = {"camera_frac": round(float(pick_cam.mean()), 3)}

    def recall_at(rank: np.ndarray, D: float) -> dict:
        positive = (pd <= D) & far
        queries = np.where(positive.any(axis=1))[0]
        rec = {f"recall@{k}": 0 for k in KS}
        for q in queries:
            minrank = rank[q, positive[q]].min()  # best-placed true positive
            for k in KS:
                if minrank < k:
                    rec[f"recall@{k}"] += 1
        nq = len(queries)
        return {"n_queries": int(nq), **{k: round(v / nq, 4) for k, v in rec.items()}}

    res = {"sequence": seq, "frames": int(n), "rrf_c": RRF_C,
           "camera_picked_frac": res_pick["camera_frac"], "by_distance": {}}
    for D in DISTANCES:
        key = f"{D:.0f}m"
        res["by_distance"][key] = {
            "lidar": recall_at(lidar_rank, D),
            "camera_eigenplaces": recall_at(cam_rank, D),
            "fusion_rrf": recall_at(fused_rank, D),
            "fusion_gated": recall_at(gated_rank, D),
        }
        b = res["by_distance"][key]
        print(f"  D={D:>4.0f}m  q={b['fusion_rrf']['n_queries']:5d}  R@1  "
              f"LiDAR={b['lidar']['recall@1']:.3f}  "
              f"Camera={b['camera_eigenplaces']['recall@1']:.3f}  "
              f"RRF={b['fusion_rrf']['recall@1']:.3f}  "
              f"Gated={b['fusion_gated']['recall@1']:.3f}")
    print(f"  (camera picked on {res_pick['camera_frac']*100:.0f}% of queries)")
    return res


def main() -> None:
    seqs = [a for a in sys.argv[1:] if not a.startswith("-")] or ["00", "05", "06", "07", "08"]
    results = [r for r in (evaluate(s) for s in seqs) if r]
    if results:
        print("\n=== R@1 @ 5 m: LiDAR vs Camera vs naive-RRF vs confidence-Gated ===")
        print(f"  {'seq':>4}  {'LiDAR':>7}  {'Camera':>7}  {'RRF':>7}  {'Gated':>7}")
        agg = {"lidar": [], "camera_eigenplaces": [], "fusion_rrf": [], "fusion_gated": []}
        for r in results:
            b = r["by_distance"]["5m"]
            for kk in agg:
                agg[kk].append(b[kk]["recall@1"])
            print(f"  {r['sequence']:>4}  {b['lidar']['recall@1']:>7.3f}  "
                  f"{b['camera_eigenplaces']['recall@1']:>7.3f}  {b['fusion_rrf']['recall@1']:>7.3f}  "
                  f"{b['fusion_gated']['recall@1']:>7.3f}")
        print(f"  {'mean':>4}  {np.mean(agg['lidar']):>7.3f}  "
              f"{np.mean(agg['camera_eigenplaces']):>7.3f}  {np.mean(agg['fusion_rrf']):>7.3f}  "
              f"{np.mean(agg['fusion_gated']):>7.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
