"""Does tuning the Scan-Context descriptor plugin recover reverse-loop recall?

seq 08's revisits are reverse-direction. Scan-Context is rotation-invariant
(column shift handles the 180 degrees), so in principle it should cope — yet the
default config (20 rings x 60 sectors, 30 m range) only reaches R@1 = 0.34. The
hypothesis: a reverse revisit is on the *opposite lane*, so the overlapping
structure sits further out; a wider range / finer polar grid should help the
hard reverse case more than the easy forward case.

This A/Bs two ScanContextConfig plugins on a forward sequence (00) and the
reverse one (08) — same retrieval code, only the descriptor config swapped.

    python scripts/experiment_scancontext_config.py
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
KS = [1, 5, 20]
TIME_EXCLUDE = 30.0
RERANK = 25
VOXEL = 0.5
OUT = ROOT / "docs" / "assets" / "kitti_scancontext_config.json"

CONFIGS = {
    "default (20x60, 30m)": ScanContextConfig(n_rings=20, n_sectors=60, max_range_m=30.0),
    "wide (40x120, 80m)":   ScanContextConfig(n_rings=40, n_sectors=120, max_range_m=80.0),
}
SEQUENCES = ["00", "08"]


def paths(seq):
    if seq == "00":
        return (f"{DATASETS}/kitti_seq00_velodyne", f"{DATASETS}/kitti_seq00_subset/poses_00.txt",
                f"{DATASETS}/kitti_seq00_full/times.txt")
    return (f"{DATASETS}/kitti_seq{seq}_velodyne", f"{SUBSETS}/seq{seq}/poses_{seq}.txt",
            f"{SUBSETS}/seq{seq}/times.txt")


def encode(seq, cfg, tag, n, velo):
    cache = Path(f"/tmp/kitti_seq{seq}_sc_{tag}.npz")
    if cache.exists():
        d = np.load(cache)
        if len(d["rk"]) == n:
            return d["rk"], d["sc"]
    rks = np.zeros((n, cfg.n_rings), dtype=np.float32)
    scs = np.zeros((n, cfg.n_rings, cfg.n_sectors), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        pts = load_kitti_bin(Path(velo) / f"{i:06d}.bin")[:, :3]
        sc = scan_context(scene_from_points(pts, f"f{i}", voxel=VOXEL, drop_ground=True).primary().xy(), cfg)
        scs[i] = sc
        rks[i] = ring_key(sc)
        if i % 400 == 0 or i == n - 1:
            print(f"    encode {seq}/{tag} {i + 1}/{n} ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print()
    np.savez(cache, rk=rks, sc=scs)
    return rks, scs


def eval_seq(seq, cfg, tag):
    velo, posep, timesp = paths(seq)
    bins = sorted(Path(velo).glob("*.bin"))
    pos = np.loadtxt(posep)[:, [3, 7, 11]]
    times = np.loadtxt(timesp)
    n = min(len(bins), len(pos), len(times))
    pos, times = pos[:n], times[:n]
    rks, scs = encode(seq, cfg, tag, n, velo)
    rks, scs = rks[:n], scs[:n]

    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T)
    np.maximum(rk_d2, 0, out=rk_d2)
    rk_order = np.argsort(rk_d2, axis=1)
    far = np.abs(times[:, None] - times[None, :]) > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    final = [None] * n
    for q in range(n):
        cand = [j for j in rk_order[q] if far[q, j]][:RERANK]
        final[q] = sorted(cand, key=lambda j: sc_alignment_distance(scs[q], scs[j])[0])

    positive = (pd <= 5.0) & far
    queries = np.where(positive.any(axis=1))[0]
    rec = {k: 0 for k in KS}
    for q in queries:
        hit = next((r for r, j in enumerate(final[q][:max(KS)]) if positive[q][j]), None)
        for k in KS:
            if hit is not None and hit < k:
                rec[k] += 1
    nq = len(queries)
    return {f"recall@{k}": round(rec[k] / nq, 4) for k in KS} | {"n_queries": int(nq)}


def main():
    results = {}
    for seq in SEQUENCES:
        results[seq] = {}
        for name, cfg in CONFIGS.items():
            tag = "def" if name.startswith("default") else "wide"
            print(f"seq{seq} :: {name}")
            r = eval_seq(seq, cfg, tag)
            results[seq][name] = r
            print(f"    R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} R@20={r['recall@20']:.3f} (q={r['n_queries']})")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    print("\n=== summary (R@1 @5m) ===")
    for seq in SEQUENCES:
        line = f"seq{seq}: " + "   ".join(f"{name} = {results[seq][name]['recall@1']:.3f}" for name in CONFIGS)
        print(line)


if __name__ == "__main__":
    main()
