"""Cross-dataset LiDAR place-recognition benchmark on NCLT (Recall@K).

Validates that BEVMatch's Scan-Context LiDAR retrieval — tuned and reported on
KITTI — generalises to a *different* dataset: the University of Michigan North
Campus Long-Term (NCLT) dataset (Carlevaris-Bianco et al., 2016). Different city,
different robot (Segway), different sensor (Velodyne HDL-32E, 32 beams vs KITTI's
64), repeated traversals of the campus within a session → real revisits.

Same protocol and same code path as scripts/benchmark_kitti_lidar.py (ring-key
KNN prefilter + Scan-Context column-shift rerank; positive = pose ≤ D m AND
|dt| > 30 s; temporal window excluded; Recall@K over revisit queries). Only the
loader differs — NCLT ships a packed `velodyne_hits.bin` hit-stream and a
high-rate ground-truth CSV, parsed here.

    python scripts/benchmark_nclt_lidar.py            # default session, keep every 15th scan

Data (login-free S3): velodyne_data/<date>_vel.tar.gz, ground_truth/groundtruth_<date>.csv
from s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bevmatch.retrieval.descriptor import (  # noqa: E402
    ScanContextConfig, ring_key, scan_context, sc_alignment_distance,
)
from bevmatch.datasets.loaders import voxel_downsample  # noqa: E402

NCLT = Path("$HOME/datasets/nclt")
SESSION = "2012-01-08"
HITS = NCLT / SESSION / "velodyne_hits.bin"
GT = NCLT / f"groundtruth_{SESSION}.csv"
OUT = ROOT / "docs" / "assets" / "nclt_lidar_results.json"

KEEP_EVERY = 15        # subsample the ~10 Hz stream (-> ~1 scan / 1.5 s)
VOXEL = 0.5
MAX_RANGE = 80.0       # parse-time range cap; >= the widest descriptor config
DISTANCES = [5.0, 10.0, 25.0]
KS = [1, 5, 10, 20]
TIME_EXCLUDE = 30.0
RERANK = 25
CONFIGS = {"default": ScanContextConfig(),                       # 20x60, 30 m (KITTI headline)
           "wide": ScanContextConfig(40, 120, 80.0)}             # 40x120, 80 m (KITTI seq08 fix)
_MAGIC = 0xAD9C


SCAN_PERIOD_US = 100_000  # 10 Hz: accumulate ~0.1 s of packets into one revolution


def parse_velodyne_hits(path: Path, keep_every: int):
    """Stream NCLT velodyne_hits.bin into full revolutions, keep every Nth.

    The file is a stream of partial packets (magic(8) num_hits(u32) utime(u64)
    pad(4), then num_hits x (x,y,z:u16, i,l:u8); x/y/z -> m via *0.005 - 100).
    ~180 packets of ~160 hits make one 10 Hz revolution, so we group packets by a
    ~0.1 s utime window into scans. Packets of *skipped* scans are seek()'d past.
    """
    hit_dt = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])
    utimes, scans = [], []
    cur_xy, cur_start, scan_idx = [], None, 0
    n_pkt = 0
    t0 = time.time()

    def finalize():
        if cur_start is None or scan_idx % keep_every != 0 or not cur_xy:
            return
        xy = np.concatenate(cur_xy, axis=0)
        r = np.hypot(xy[:, 0], xy[:, 1])
        xy = xy[(r > 2.0) & (r < MAX_RANGE)]
        if len(xy):
            scans.append(voxel_downsample(xy, VOXEL))
            utimes.append(cur_start)

    with open(path, "rb") as f:
        while True:
            magic = f.read(8)
            if len(magic) < 8:
                break
            if struct.unpack("<HHHH", magic) != (_MAGIC, _MAGIC, _MAGIC, _MAGIC):
                break  # desync — stop cleanly
            num_hits = struct.unpack("<I", f.read(4))[0]
            utime = struct.unpack("<Q", f.read(8))[0]
            f.read(4)  # padding
            nbytes = num_hits * 8

            if cur_start is None:
                cur_start = utime
            elif utime - cur_start >= SCAN_PERIOD_US:  # revolution boundary
                finalize()
                cur_xy, cur_start, scan_idx = [], utime, scan_idx + 1

            if scan_idx % keep_every == 0:  # decode only kept revolutions
                raw = f.read(nbytes)
                if len(raw) < nbytes:
                    break  # truncated tail
                h = np.frombuffer(raw, dtype=hit_dt, count=num_hits)
                x = h["x"].astype(np.float32) * 0.005 - 100.0
                y = h["y"].astype(np.float32) * 0.005 - 100.0
                cur_xy.append(np.stack([x, y], axis=1))
            else:
                f.seek(nbytes, 1)  # next magic read (<8 bytes) breaks at EOF
            n_pkt += 1
            if n_pkt % 100000 == 0:
                print(f"  parse: {n_pkt} packets, {scan_idx} revolutions, "
                      f"{len(scans)} kept ({time.time()-t0:.0f}s)", end="\r", flush=True)
    finalize()
    print()
    print(f"  parsed {n_pkt} packets -> {scan_idx + 1} revolutions, kept {len(scans)}")
    return np.array(utimes, dtype=np.int64), scans


def load_gt(path: Path):
    """NCLT ground truth: utime, x(north), y(east), z, r, p, h. Drop NaN rows."""
    a = np.loadtxt(path, delimiter=",")
    ok = np.isfinite(a[:, 1]) & np.isfinite(a[:, 2])
    a = a[ok]
    return a[:, 0].astype(np.int64), a[:, 1:3]  # utime, (x, y)


def sync_poses(scan_utimes: np.ndarray, gt_ut: np.ndarray, gt_xy: np.ndarray):
    """Nearest-time GT pose for each kept scan."""
    idx = np.searchsorted(gt_ut, scan_utimes)
    idx = np.clip(idx, 1, len(gt_ut) - 1)
    left, right = gt_ut[idx - 1], gt_ut[idx]
    pick = np.where(np.abs(scan_utimes - left) <= np.abs(scan_utimes - right), idx - 1, idx)
    return gt_xy[pick]


def load_or_parse(keep: int):
    """Parse the hit-stream once and cache the kept scans (parsing dominates)."""
    cache = Path(f"/tmp/nclt_{SESSION}_scans_k{keep}.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        off = d["offsets"]
        concat = d["concat"]
        scans = [concat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        print(f"  (loaded scan cache: {len(scans)} scans)")
        return d["utimes"], scans
    if not HITS.exists():
        print(f"missing {HITS} — extract <session>_vel.tar.gz first")
        sys.exit(1)
    scan_ut, scans = parse_velodyne_hits(HITS, keep)
    concat = np.vstack(scans).astype(np.float32)
    offsets = np.cumsum([0] + [len(s) for s in scans]).astype(np.int64)
    np.savez(cache, concat=concat, offsets=offsets, utimes=scan_ut)
    return scan_ut, scans


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    keep = int(args[0]) if args else KEEP_EVERY
    cfg_name = "wide" if "--wide" in sys.argv else "default"
    cfg = CONFIGS[cfg_name]
    print(f"NCLT {SESSION} LiDAR (HDL-32E), keep every {keep}th, config={cfg_name} "
          f"({cfg.n_rings}x{cfg.n_sectors}, {cfg.max_range_m:.0f} m)")
    scan_ut, scans = load_or_parse(keep)
    gt_ut, gt_xy = load_gt(GT)
    pos = sync_poses(scan_ut, gt_ut, gt_xy)
    times = (scan_ut - scan_ut[0]) / 1e6  # seconds from start
    n = len(scans)
    print(f"  {n} scans, {times[-1]:.0f}s, GT extent "
          f"{np.ptp(pos[:, 0]):.0f}x{np.ptp(pos[:, 1]):.0f} m")

    rks = np.zeros((n, cfg.n_rings), dtype=np.float32)
    scs = np.zeros((n, cfg.n_rings, cfg.n_sectors), dtype=np.float32)
    for i, xy in enumerate(scans):
        sc = scan_context(xy, cfg)
        scs[i] = sc
        rks[i] = ring_key(sc)
    print("  encoded Scan-Context")

    rk_norm = rks / (rks.sum(axis=1, keepdims=True) + 1e-9)
    sq = (rk_norm * rk_norm).sum(axis=1)
    rk_d2 = sq[:, None] + sq[None, :] - 2.0 * (rk_norm @ rk_norm.T)
    np.maximum(rk_d2, 0, out=rk_d2)
    rk_order = np.argsort(rk_d2, axis=1)

    dt = np.abs(times[:, None] - times[None, :])
    far = dt > TIME_EXCLUDE
    pd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    final_order = [None] * n
    t0 = time.time()
    for q in range(n):
        cand = [j for j in rk_order[q] if far[q, j]][:RERANK]
        scq = scs[q]
        final_order[q] = sorted(cand, key=lambda j: sc_alignment_distance(scq, scs[j])[0])
        if q % 500 == 0 or q == n - 1:
            print(f"  rerank {q + 1}/{n} ({time.time()-t0:.0f}s)", end="\r", flush=True)
    print()

    res = {"dataset": "NCLT", "session": SESSION, "scans": int(n),
           "sensor": "Velodyne HDL-32E", "keep_every": keep, "config_name": cfg_name,
           "descriptor": "ScanContextDescriptor (ring-key prefilter + SC rerank)",
           "config": {"n_rings": cfg.n_rings, "n_sectors": cfg.n_sectors,
                      "max_range_m": cfg.max_range_m, "voxel_m": VOXEL}, "by_distance": {}}
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
            **{k: round(v / max(nq, 1), 4) for k, v in recall.items()}}
        r = res["by_distance"][f"{D:.0f}m"]
        print(f"  D={D:>4.0f}m  queries={nq:5d}  R@1={r['recall@1']:.3f}  "
              f"R@5={r['recall@5']:.3f}  R@10={r['recall@10']:.3f}  R@20={r['recall@20']:.3f}")
    out = OUT.with_name(f"nclt_lidar_results_{cfg_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
