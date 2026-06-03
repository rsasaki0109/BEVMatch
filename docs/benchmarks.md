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

Reproduce: `python scripts/benchmark_kitti_lidar.py`
(needs the seq 00 velodyne scans — `scripts/_extract_velodyne_from_head.py`).

### KITTI seq 00 (4541 scans, 471 s)

| positive radius | queries | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|---|
| 5 m  | 1706 | **0.913** | 0.920 | 0.925 | 0.928 |
| 10 m | 1838 | 0.850 | 0.861 | 0.872 | 0.878 |
| 25 m | 2089 | 0.763 | 0.779 | 0.793 | 0.807 |

*Context:* R@1 ≈ 0.91 at 5 m is in line with the Scan-Context baseline reported
in the literature on KITTI — a faithful classical descriptor, no learning. As a
plugin it can be swapped for a learned LiDAR descriptor (OverlapTransformer,
LoGG3D-Net, …) without touching the rest of the pipeline.

## Cross-modal — same place, same protocol, two sensors

Both descriptors evaluated on **KITTI seq 00**, identical revisit protocol
(positive radius 5 m, T = 30 s). This is the point of Principle 2 — *modality
is not representation*: one retrieval framework, two sensors, comparable recall.

| modality | descriptor | Recall@1 | Recall@5 | Recall@20 |
|---|---|---|---|---|
| LiDAR  | Scan-Context (ring-key + SC rerank) | 0.913 | 0.920 | 0.928 |
| Camera | ResNet-18 embedding (ImageNet)      | 0.923 | 0.942 | 0.954 |

## Notes on honesty

- The synthetic demo tables (`examples/run_*_eval.py`, some README snippets)
  exist to test wiring and stay green in CI. They are **not** evidence of
  method quality; this file is.
- We report the descriptors BEVMatch actually ships (Scan-Context, BEV grid,
  ResNet-18 embeddings). They are honest baselines, deliberately swappable.
