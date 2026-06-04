"""Cross-SESSION LiDAR place recognition on NCLT — the long-term test (Recall@K).

Finding 4 showed BEVMatch's Scan-Context retrieval *generalises* to NCLT within a
single session. This script asks the harder, real question the "Long-Term" in
NCLT's name is for: build the reference map from one day and localise against it
*months later*, after a full change of season.

    reference map : 2012-01-08  (winter)
    query traverse: 2012-08-04  (summer, ~7 months / 209 days later)

NCLT ground truth is geo-referenced to one campus frame shared across sessions, so
poses are directly comparable between days — a query frame is a true revisit of a
reference frame iff their GT poses are within D metres. There is no temporal
exclusion in the cross-session case (the two traverses are entirely separate days).

To isolate long-term degradation from anything loader-specific, the SAME sync
loader, SAME descriptor and SAME config also run a within-session baseline
(2012-01-08 query vs 2012-01-08 map, |dt| > 30 s excluded). The gap between the two
is the cost of seven months.

Loader note: the velodyne_data/<date>_vel.tar.gz ships `velodyne_sync/` — one file
per synchronised revolution, each a packed point stream (x,y,z:u16 *0.005-100,
i,l:u8 = 8 bytes/point). That is far simpler than the raw hit-stream parsed by
benchmark_nclt_lidar.py, and identical across sessions, so we use it for both.

    python scripts/benchmark_nclt_cross_session.py            # default config
    python scripts/benchmark_nclt_cross_session.py --wide     # 40x120, 80 m
"""

from __future__ import annotations

import json
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
REF_SESSION = "2012-01-08"   # winter — the stored map
QRY_SESSION = "2012-08-04"   # summer — queried ~7 months later

KEEP_EVERY = 15        # subsample synchronised revolutions
VOXEL = 0.5
MAX_RANGE = 80.0
DISTANCES = [5.0, 10.0, 25.0]
KS = [1, 5, 10, 20]
TIME_EXCLUDE = 30.0    # within-session baseline only
RERANK = 25
CONFIGS = {"default": ScanContextConfig(),            # 20x60, 30 m
           "wide": ScanContextConfig(40, 120, 80.0)}  # 40x120, 80 m (NCLT within-session winner)
OUT = ROOT / "docs" / "assets"

_PT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])


def _decode(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // 8
    h = np.frombuffer(raw, dtype=_PT, count=n)
    x = h["x"].astype(np.float32) * 0.005 - 100.0
    y = h["y"].astype(np.float32) * 0.005 - 100.0
    xy = np.stack([x, y], axis=1)
    r = np.hypot(xy[:, 0], xy[:, 1])
    return xy[(r > 2.0) & (r < MAX_RANGE)]


def load_sync_session(session: str, keep: int):
    """Read every keep-th velodyne_sync revolution; cache the kept scans."""
    cache = Path(f"/tmp/nclt_sync_{session}_k{keep}.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        off, concat = d["offsets"], d["concat"]  # read concat once (NpzFile decompresses per access)
        scans = [concat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        print(f"  ({session}: loaded {len(scans)} scans from cache)")
        return d["utimes"], scans
    sync = NCLT / session / "velodyne_sync"
    if not sync.is_dir():
        print(f"missing {sync} — extract {session}_vel.tar.gz first")
        sys.exit(1)
    files = sorted(sync.glob("*.bin"))[::keep]
    utimes, scans = [], []
    t0 = time.time()
    for k, f in enumerate(files):
        xy = _decode(f)
        if len(xy):
            scans.append(voxel_downsample(xy, VOXEL))
            utimes.append(int(f.stem))
        if k % 500 == 0:
            print(f"  {session}: read {k}/{len(files)} ({time.time()-t0:.0f}s)",
                  end="\r", flush=True)
    print()
    ut = np.array(utimes, dtype=np.int64)
    concat = np.vstack(scans).astype(np.float32)
    offsets = np.cumsum([0] + [len(s) for s in scans]).astype(np.int64)
    np.savez(cache, concat=concat, offsets=offsets, utimes=ut)
    print(f"  {session}: parsed {len(scans)} scans")
    return ut, scans


def load_gt(session: str):
    a = np.loadtxt(NCLT / f"groundtruth_{session}.csv", delimiter=",")
    ok = np.isfinite(a[:, 1]) & np.isfinite(a[:, 2])
    a = a[ok]
    return a[:, 0].astype(np.int64), a[:, 1:3]


def sync_poses(scan_ut: np.ndarray, gt_ut: np.ndarray, gt_xy: np.ndarray):
    idx = np.clip(np.searchsorted(gt_ut, scan_ut), 1, len(gt_ut) - 1)
    pick = np.where(np.abs(scan_ut - gt_ut[idx - 1]) <= np.abs(scan_ut - gt_ut[idx]),
                    idx - 1, idx)
    return gt_xy[pick]


def encode(scans, cfg):
    rks = np.zeros((len(scans), cfg.n_rings), dtype=np.float32)
    scs = np.zeros((len(scans), cfg.n_rings, cfg.n_sectors), dtype=np.float32)
    for i, xy in enumerate(scans):
        sc = scan_context(xy, cfg)
        scs[i] = sc
        rks[i] = ring_key(sc)
    return rks, scs


def _ringkey_order(rks_q: np.ndarray, rks_r: np.ndarray) -> np.ndarray:
    """Rows = queries, sorted ref indices by normalised ring-key L2."""
    qn = rks_q / (rks_q.sum(axis=1, keepdims=True) + 1e-9)
    rn = rks_r / (rks_r.sum(axis=1, keepdims=True) + 1e-9)
    d2 = (qn * qn).sum(1)[:, None] + (rn * rn).sum(1)[None, :] - 2.0 * (qn @ rn.T)
    np.maximum(d2, 0, out=d2)
    return np.argsort(d2, axis=1)


def retrieve(rks_q, scs_q, rks_r, scs_r, allow_mask):
    """SC-reranked retrieval. allow_mask[q,j]=True if ref j may answer query q."""
    order = _ringkey_order(rks_q, rks_r)
    final = [None] * len(rks_q)
    t0 = time.time()
    for q in range(len(rks_q)):
        cand = [j for j in order[q] if allow_mask[q, j]][:RERANK]
        scq = scs_q[q]
        final[q] = sorted(cand, key=lambda j: sc_alignment_distance(scq, scs_r[j])[0])
        if q % 500 == 0 or q == len(rks_q) - 1:
            print(f"  rerank {q + 1}/{len(rks_q)} ({time.time()-t0:.0f}s)",
                  end="\r", flush=True)
    print()
    return final


def recall(final_order, positive):
    """positive[q,j]=True if ref j is a true revisit of query q."""
    queries = np.where(positive.any(axis=1))[0]
    rec = {f"recall@{k}": 0 for k in KS}
    for q in queries:
        posset = positive[q]
        hit = next((r for r, j in enumerate(final_order[q][:max(KS)]) if posset[j]), None)
        for k in KS:
            if hit is not None and hit < k:
                rec[f"recall@{k}"] += 1
    nq = len(queries)
    out = {"n_queries": int(nq), **{k: round(v / max(nq, 1), 4) for k, v in rec.items()}}
    return out


def main() -> None:
    keep = next((int(a) for a in sys.argv[1:] if not a.startswith("-")), KEEP_EVERY)
    cfg_name = "wide" if "--wide" in sys.argv else "default"
    cfg = CONFIGS[cfg_name]
    print(f"NCLT cross-session — map {REF_SESSION} (winter) vs query {QRY_SESSION} "
          f"(summer), keep {keep}, config={cfg_name} "
          f"({cfg.n_rings}x{cfg.n_sectors}, {cfg.max_range_m:.0f} m)\n")

    ref_ut, ref_scans = load_sync_session(REF_SESSION, keep)
    qry_ut, qry_scans = load_sync_session(QRY_SESSION, keep)
    ref_pos = sync_poses(ref_ut, *load_gt(REF_SESSION))
    qry_pos = sync_poses(qry_ut, *load_gt(QRY_SESSION))
    ref_t = (ref_ut - ref_ut[0]) / 1e6
    print(f"  ref {len(ref_scans)} scans, extent "
          f"{np.ptp(ref_pos[:,0]):.0f}x{np.ptp(ref_pos[:,1]):.0f} m")
    print(f"  qry {len(qry_scans)} scans, extent "
          f"{np.ptp(qry_pos[:,0]):.0f}x{np.ptp(qry_pos[:,1]):.0f} m\n")

    ref_rk, ref_sc = encode(ref_scans, cfg)
    qry_rk, qry_sc = encode(qry_scans, cfg)
    print("  encoded Scan-Context (both sessions)\n")

    result = {"dataset": "NCLT", "experiment": "cross-session long-term",
              "reference_session": REF_SESSION, "query_session": QRY_SESSION,
              "days_apart": 209, "sensor": "Velodyne HDL-32E",
              "keep_every": keep, "config_name": cfg_name,
              "config": {"n_rings": cfg.n_rings, "n_sectors": cfg.n_sectors,
                         "max_range_m": cfg.max_range_m, "voxel_m": VOXEL},
              "n_ref": len(ref_scans), "n_query": len(qry_scans),
              "within_session": {}, "cross_session": {}}

    # within-session baseline: 2012-01-08 against itself, |dt|>30 s excluded
    print("[within-session baseline: 2012-01-08 vs 2012-01-08]")
    far = np.abs(ref_t[:, None] - ref_t[None, :]) > TIME_EXCLUDE
    pd_ref = np.linalg.norm(ref_pos[:, None, :] - ref_pos[None, :, :], axis=2)
    order_w = retrieve(ref_rk, ref_sc, ref_rk, ref_sc, far)
    for D in DISTANCES:
        r = recall(order_w, (pd_ref <= D) & far)
        result["within_session"][f"{D:.0f}m"] = r
        print(f"  D={D:>4.0f}m  q={r['n_queries']:5d}  R@1={r['recall@1']:.3f} "
              f"R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} R@20={r['recall@20']:.3f}")

    # cross-session: summer query against winter map, no temporal exclusion
    print("\n[cross-session: 2012-08-04 query vs 2012-01-08 map (209 days)]")
    pd_cross = np.linalg.norm(qry_pos[:, None, :] - ref_pos[None, :, :], axis=2)
    allow = np.ones((len(qry_scans), len(ref_scans)), dtype=bool)
    order_c = retrieve(qry_rk, qry_sc, ref_rk, ref_sc, allow)
    for D in DISTANCES:
        r = recall(order_c, pd_cross <= D)
        result["cross_session"][f"{D:.0f}m"] = r
        print(f"  D={D:>4.0f}m  q={r['n_queries']:5d}  R@1={r['recall@1']:.3f} "
              f"R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} R@20={r['recall@20']:.3f}")

    out = OUT / f"nclt_cross_session_{cfg_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
