# BEVMatch benchmarks — real public-data results

These are **real** numbers on **public datasets**, produced by BEVMatch's own
retrieval pipeline (not a detached reimplementation). Every table is reproducible
from a script in `scripts/` against data anyone can download.

> Two kinds of numbers live in this repo. Treat them differently:
>
> * **Real-data benchmarks (this file)** — KITTI odometry, standard place-recognition
>   protocol. These are the numbers to judge the method by.
> * **Synthetic sanity checks** (the perfect-score tables in the demos / README) —
>   tiny generated scenes that verify the *plumbing* end-to-end. A perfect score there
>   means "the pipeline runs and is wired correctly", **not** "the method is solved".

## Protocol

Standard place-recognition / loop-closure evaluation (as used by Scan-Context,
OverlapNet, NetVLAD, Patch-NetVLAD, …):

- **Database** = every frame of the sequence.
- **Positive** for query *q* = a frame whose ground-truth pose is within *D*
  metres of *q* **and** more than *T* = 30 s away in time (so trivial same-pass
  neighbours never count as a "revisit").
- **Query set** = every frame that has ≥ 1 positive (i.e. an actual revisit).
- At retrieval time the *T*-second temporal window is excluded from candidates;
  a query is **hit@K** if a positive appears in the top *K* of what remains.
- Ground-truth poses and timestamps come from the KITTI odometry ground truth.

Data: **KITTI odometry** (Geiger et al., *Are we ready for autonomous driving?*,
CVPR 2012). Sequence 00 is the classic loop-closure-rich drive.

## Camera — visual place recognition

`CameraEmbeddingDescriptor` over off-the-shelf **ResNet-18 (ImageNet)** global
features, cosine distance. No VPR-specific training or fine-tuning — this is a
baseline that shows the framework working on real images.

Reproduce: `python scripts/benchmark_kitti_vpr.py`

### KITTI seq 00 (4541 frames, 471 s)

| positive radius | queries | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|---|
| 5 m  | 1706 | **0.923** | 0.942 | 0.948 | 0.954 |
| 10 m | 1838 | 0.868 | 0.888 | 0.896 | 0.905 |
| 25 m | 2089 | 0.789 | 0.825 | 0.843 | 0.869 |

We assert in-script that BEVMatch's `SceneDatabase` reproduces this ranking
(framework check: `SceneDatabase ranking == evaluated cosine`).

*Context:* off-the-shelf ImageNet features give a strong-but-not-saturated
baseline; VPR-specialised learned descriptors (NetVLAD / Patch-NetVLAD) report
higher recall on the same data. BEVMatch treats the descriptor as a plugin, so
swapping in a learned backbone is a drop-in change — these numbers are the
floor, not the ceiling.

## LiDAR — Scan-Context place recognition

`ScanContextDescriptor`, BEVMatch's two-stage retrieval exactly as it ships:
a rotation-invariant **ring-key** KNN prefilter, then a full **Scan-Context
column-shift** distance rerank (the same design as Kim & Kim, *Scan Context*,
IROS 2018). Velodyne scans, voxel-downsampled (0.5 m) with a ground filter;
polar grid 20 rings × 60 sectors, 30 m range.

Reproduce: `python scripts/benchmark_kitti_lidar.py` (all sequences) or
`python scripts/benchmark_kitti_lidar.py 00 05` (a subset). Needs the velodyne
scans — `scripts/_fetch_kitti_velodyne_seq00.py` (seq 00) /
`scripts/_extract_velodyne_blocks.py` (seq 05–08).

### Across the KITTI loop sequences (positive radius 5 m)

| sequence | revisit queries | Recall@1 | Recall@5 | Recall@20 |
|---|---|---|---|---|
| 00 (forward loops)   | 1706 | **0.913** | 0.920 | 0.928 |
| 05                   |  963 | 0.783 | 0.797 | 0.804 |
| 06                   |  565 | 0.887 | 0.899 | 0.901 |
| 07 (few revisits)    |   94 | 0.596 | 0.628 | 0.638 |
| 08 (reverse loops)   |  616 | 0.339 | 0.385 | 0.433 |
| **mean**             |   —  | **0.704** | 0.726 | 0.741 |

Full table (5/10/25 m radii, all K) is in
[`docs/assets/kitti_lidar_results.json`](assets/kitti_lidar_results.json).

*Reading these honestly:*
- seq 00/06 (forward revisits) are where Scan-Context shines — R@1 ≈ 0.89–0.91,
  in line with the published Scan-Context baseline.
- seq 08 is dominated by **reverse-direction** revisits (the car drives back
  the opposite way). This is a documented hard case for appearance/structure
  descriptors and pulls R@1 down to 0.34 — we report it rather than hide it.
- seq 07 has very few genuine revisits (94 queries), so its number is noisy.
- The mean across sequences (R@1 = 0.70 @ 5 m) is the honest single figure.

This is a **faithful classical baseline, no learning**, with a deliberately
modest config (20×60 polar grid, 30 m range, default `ScanContextConfig`). The
descriptor is a plugin: a learned LiDAR descriptor (OverlapTransformer,
LoGG3D-Net, …) or a wider range / finer grid can be dropped in without touching
the rest of the pipeline — these numbers are the floor, not the ceiling.

## Cross-modal — same place, same protocol, two sensors

Both descriptors evaluated on **KITTI seq 00**, identical revisit protocol
(positive radius 5 m, T = 30 s). This is the point of Principle 2 — *modality
is not representation*: one retrieval framework, two sensors, comparable recall.

| modality | descriptor | Recall@1 | Recall@5 | Recall@20 |
|---|---|---|---|---|
| LiDAR  | Scan-Context (ring-key + SC rerank) | 0.913 | 0.920 | 0.928 |
| Camera | ResNet-18 embedding (ImageNet)      | 0.923 | 0.942 | 0.954 |

(seq 00; the LiDAR descriptor is also evaluated across 5 loop sequences above.)

## Notes on honesty

- The synthetic demo tables (`examples/run_*_eval.py`, some README snippets)
  exist to test wiring and stay green in CI. They are **not** evidence of
  method quality; this file is.
- We report the descriptors BEVMatch actually ships (Scan-Context, BEV grid,
  ResNet-18 embeddings). They are honest baselines, deliberately swappable.
