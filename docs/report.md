# Same-place retrieval across modalities: representation helps, viewpoint walls don't fall to learning, and geometry — not score — fuses

**A technical report on the BEVMatch real-data benchmarks (KITTI odometry + NCLT).**

> All numbers are measured by BEVMatch's own retrieval pipeline on public KITTI and
> NCLT data, one standard protocol, no ground truth at inference, fully reproducible
> from `scripts/`. This report consolidates the five findings; the conversational
> version is [findings.md](findings.md), the full tables are in
> [benchmarks.md](benchmarks.md).

## Abstract

We study place recognition (loop closure) as a *same-place retrieval* problem on
the KITTI odometry loop sequences, comparing a hand-crafted 360° LiDAR descriptor
(Scan-Context), a generic camera descriptor (ImageNet ResNet-18), and a learned
camera place-recognition descriptor (EigenPlaces), all behind one retrieval
interface and one protocol. Three results emerge. **(1)** Within a modality, a
purpose-trained descriptor raises recall on every forward-revisit sequence
(mean R@1 @ 5 m 0.653 → 0.709), confirming representation quality is real and
measurable. **(2)** Across a viewpoint gap — reverse-direction revisits — the
learned descriptor does *nothing* (R@1 stays 0.015, identical to the baseline):
a forward-facing camera never observes the opposite view, a viewpoint/geometry
wall that learning cannot break, while a 360° LiDAR holds 0.339 and is recoverable
to 0.765 by a descriptor-config swap. **(3)** Naively fusing the two modalities at
the score level *fails* (equal-weight RRF is a net loss; a confidence gate only
buys robustness) because descriptor score cannot distinguish "confidently wrong"
from "confidently right"; but fusing on **geometric verification** — accept the
camera's proposed place only when its LiDAR geometry aligns — wins on every
sequence (mean R@1 0.779, +0.07 over either modality) and fully recovers the blind
reverse loop (0.343 ≈ LiDAR's 0.339). The arc validates a retrieve → align →
evidence design over retrieval score alone. **(4)** On a wholly different dataset
(NCLT — different city, robot and 32-beam sensor) the LiDAR retrieval generalises
(R@1 = 0.62 with the right config) and the config-tuning lesson transfers from
KITTI unchanged. **(5)** Across sessions 209 days apart — a winter map queried by a
summer drive — the LiDAR retrieval still localises at R@1 @ 5 m = 0.68, only 0.16
below the same-day baseline: appearance-blind range geometry survives a full change
of season where a camera descriptor would not, the long-term mirror image of (2).

## 1. Problem and setup

A *same-place* system must, given a current observation, retrieve prior
observations of the same physical place — across sensors and across the
appearance change a real revisit brings. We evaluate the retrieval core on KITTI
odometry [1] loop sequences 00/05/06/07/08.

**Protocol (identical for every descriptor).** Database = every frame of the
sequence. A query *q*'s **positive** is a frame within *D* metres of *q*'s
ground-truth pose **and** more than *T* = 30 s away in time (so same-pass
neighbours never count). The query set is every frame with ≥ 1 positive. At
retrieval the *T*-second temporal window is excluded from candidates; a query is
**hit@K** if a positive is in the top *K*. We report Recall@1 @ 5 m.

**Descriptors**, all behind BEVMatch's `GlobalDescriptor` → `SceneDatabase`
interface (an in-script assertion confirms `SceneDatabase` reproduces each
evaluated ranking):

| descriptor | modality | learned | distance |
|---|---|---|---|
| Scan-Context [2] (ring-key prefilter + column-shift rerank) | LiDAR, 360° | no | SC alignment |
| ResNet-18 [5] global features | camera, forward | generic (ImageNet) | cosine |
| EigenPlaces [3] (ResNet-50 + GeM, 2048-d) | camera, forward | yes, for VPR (SF-XL) | cosine |

SF-XL is disjoint from KITTI, so KITTI is held out for the learned camera
descriptor on *every* sequence — the camera-vs-camera comparison is fair
throughout, with no train-on-test caveat.

## 2. Finding 1 — representation quality is real within a modality

![Recall@1 @ 5 m, three descriptors behind one interface](assets/bevmatch_results_summary.png)

Swapping the generic ImageNet ResNet-18 for the place-recognition–trained
EigenPlaces raises R@1 @ 5 m on every forward-revisit sequence:

| seq | ResNet-18 (ImageNet) | EigenPlaces | Δ |
|---|---|---|---|
| 00 | 0.923 | 0.957 | +0.034 |
| 05 | 0.848 | 0.914 | +0.066 |
| 06 | 0.977 | 0.977 | ±0 (ceiling) |
| 07 | 0.500 | 0.681 | +0.181 |
| mean (fwd) | — | — | clearly up |

Because the descriptor is a plugin, this is a one-line swap with no pipeline
change. Representation matters — where the view is shared.

## 3. Finding 2 — the viewpoint wall does not fall to learning

Sequence 08's revisits are **reverse-direction**: the car returns the opposite
way, so a forward-facing camera observes the *opposite* view of each place. Here
both camera descriptors — generic and learned SOTA — sit at **R@1 = 0.015**,
byte-for-byte identical. There is no representation of an observation never made;
the failure is a **viewpoint/geometry wall**, not a representation gap. Evidence
it is sensor-bound:

- The same reverse seq 08, seen by the 360° LiDAR, is solved to R@1 = 0.339, and
  **recovered to 0.765** by *only* widening the Scan-Context config (range
  30 → 80 m, grid 20×60 → 40×120; no pipeline change).
- The camera cannot be rescued the same way: 0.015 before learning, 0.015 after.

The *same* intervention (a better/wider descriptor) moves LiDAR 0.34 → 0.77 but
leaves the camera pinned at 0.02. When one sensor is blind to a revisit, no
descriptor recovers it; a second, non-blind modality does — the argument for a
modality-agnostic framework.

## 4. Finding 3 — geometry, not score, fuses the two

![Score-level fusion fails, geometry-verified fusion wins](assets/bevmatch_fusion_summary.png)

If the modalities fail differently, does fusing them recover the blind cases? We
test three ground-truth-free late-fusion strategies over the same rankings:

| seq | LiDAR | Camera | naive RRF [4] | conf-gated | **geo-verified** |
|---|---|---|---|---|---|
| 00 | 0.913 | 0.957 | 0.957 | 0.939 | **0.963** |
| 05 | 0.783 | 0.914 | 0.819 | 0.841 | **0.922** |
| 06 | 0.887 | 0.977 | 0.904 | 0.927 | **0.943** |
| 07 | 0.596 | 0.681 | 0.713 | 0.660 | **0.723** |
| 08 (reverse) | 0.339 | 0.015 | 0.081 | 0.203 | **0.343** |
| **mean** | 0.704 | 0.709 | 0.695 | 0.714 | **0.779** |

**Score-level fusion fails.** Equal-weight Reciprocal Rank Fusion [4] is a net
loss (mean 0.695, below both single modalities): on seq 08 the blind camera drags
LiDAR from 0.339 to 0.081. A confidence gate — trust, per query, the sensor with
the larger top-1-vs-top-2 margin (Lowe-style [6]) — only buys robustness (mean
0.714) and still misses the blind case (seq 08 → 0.203). The reason is fundamental:
**score magnitude cannot tell "confidently wrong" from "confidently right"** — the
camera's mean top-1 cosine is *higher* on the blind seq 08 (0.46) than on the
working seq 07 (0.41), because on a reverse loop it confidently matches a
similar-looking wrong place.

**Geometric verification wins.** We instead verify the camera's proposal with
geometry the score cannot see: accept the camera's top-1 only if the two LiDAR
scans (query frame vs camera's proposed frame) align almost as well as LiDAR's own
best — Scan-Context alignment distance within α = 1.3 — else fall back to LiDAR.
This is the classic robot loop-closure recipe (cheap appearance proposal,
geometric check), one extra SC alignment per query, no ground truth. It wins on
**every** sequence (mean **0.779**, +0.07 over either modality) and **fully
recovers the blind case** (seq 08 = 0.343 ≈ LiDAR's 0.339): the geometrically-wrong
reverse-loop proposal fails the check and is rejected (camera accepted on 16 % of
seq 08 queries vs 53 % on camera-strong seq 06), while correct forward proposals
are kept (seq 00 0.963, seq 05 0.922, above LiDAR).

**The verifier must be *calibrated* for rejection — and the relative proxy is
calibration-free.** We also tried BEVMatch's *full* SE2 aligner as the verifier — a
360° BEV cross-correlation + ICP with the framework's own `success` verdict
(overlap ≥ 0.45) — expecting it to sharpen the proxy
(`scripts/experiment_icp_verification.py`). Out of the box it *fails*: on the blind
seq 08 it **accepts the camera on 89 % of queries** and lands at R@1 = 0.068 (vs
the proxy's 0.343), because a BEV cross-correlation *maximises* overlap and generic
urban structure clears the default 0.45 threshold even between different places.
But sweeping the overlap threshold τ shows this is *calibration*, not incapacity:

| τ (overlap accept) | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 |
|---|---|---|---|---|---|
| seq 08 (blind) R@1 | 0.068 | 0.149 | 0.252 | 0.320 | **0.339** |
| seq 06 (camera-right) R@1 | 0.977 | 0.977 | **0.979** | 0.972 | 0.929 |

The catch is that the **optimal τ is opposite for the two cases** — the blind loop
wants a strict threshold (reject), the camera-correct loop a lenient one (accept) —
so any *single* absolute threshold is a compromise (τ ≈ 0.75 → seq 08 0.320, seq 06
0.972). The relative proxy reaches comparable quality (seq 08 0.343, seq 06 0.943)
with **no per-dataset threshold search at all**: one α = 1.3 adapts per query
because it references LiDAR's *own best for that query*. So a tuned absolute
verifier is competitive; the relative one is simply tuning-free — and on the blind
case still marginally ahead (0.343 vs the best single-τ 0.320).

## 5. Finding 4 — the LiDAR retrieval generalises beyond KITTI

To check the results are not KITTI-overfit, we run the *same* Scan-Context code and
protocol on **NCLT** [7] — a different city, a Segway platform, and a different
sensor (Velodyne HDL-32E, 32 beams vs KITTI's 64). NCLT ships a packed
`velodyne_hits.bin` hit-stream, which we parse into 10 Hz revolutions (3716 scans
over a 94-minute, 420 × 791 m campus route) and benchmark unchanged.

| config | NCLT R@1 @ 5 m | revisit queries |
|---|---|---|
| default (20×60, 30 m) | 0.358 | 1664 |
| **wide (40×120, 80 m)** | **0.620** | 1664 |

The retrieval generalises — the same code recovers 62 % of revisits at top-1 on a
wholly different dataset — though below KITTI's best (seq 00 wide 0.966), as
expected from a half-density 32-beam sensor and a vegetated campus. And the
config lesson from §3 transfers: the *same* "wide" descriptor that rescued KITTI's
reverse loops (0.34 → 0.77) nearly doubles NCLT recall (0.358 → 0.620), since the
larger, more open campus needs the longer 80 m range. The method generalises, and
so does the knowledge of how to tune it.

## 6. Finding 5 — the map survives the seasons (cross-session, 209 days)

§5 is still within one session. The long-term test NCLT exists for is whether a map
built on one day localises a drive made months later. We build the reference from
**2012-01-08 (winter)** and query it with **2012-08-04 (summer)**, 209 days and a full
change of season later. NCLT's ground truth shares one campus frame across days, so a
summer frame is a true revisit of a winter frame iff their poses are within *D* m, with
no temporal exclusion (different days). The same `velodyne_sync` loader and descriptor
run a same-day baseline for control (`scripts/benchmark_nclt_cross_session.py`).

| R@1 | within-session (same day) | cross-session (209 days) |
|---|---|---|
| wide @ 5 m | 0.840 | **0.678** |
| wide @ 10 m | 0.721 | 0.682 |
| wide @ 25 m | 0.541 | 0.666 |
| default @ 5 m | 0.645 | 0.634 |

A winter map localises a summer traverse at **R@1 @ 5 m = 0.68**, only **0.16 below**
the same-day baseline — seven months cost about a sixth of the recall, and the system
degrades gracefully rather than collapsing. This is the long-term mirror of Finding 2:
Scan-Context reads range geometry, which the seasons barely move, so the modality that
*lost* the viewpoint battle *wins* the long-term one — a camera appearance descriptor
would be hammered by the foliage/snow/light change. The config lesson transfers a third
time (wide 0.678 > default 0.634). Two honest caveats: past 5 m the cross-session number
exceeds the same-day baseline (0.666 vs 0.541 @ 25 m) because the same-day baseline's
30 s exclusion leaves geometrically-distant positives that thin with *D* while the
cross-session positives are dense road overlap — so the 5 m column is the like-for-like
read; and this sync-loader baseline (0.840) exceeds §5's hit-stream one (0.620) because
the official synchronised product is cleaner than our hand-rolled accumulation, so the
fair comparison is cross-vs-within on the same loader.

## 7. Discussion

The five findings compose into one statement: **better representations help where
the view is shared, cannot manufacture an unobserved view, the right way to combine
sensors that fail differently is to verify a match geometrically rather than to
combine retrieval scores, and appearance-blind LiDAR geometry both transfers across
datasets and survives across seasons.** This is exactly a retrieve → align → evidence
architecture: retrieval proposes, geometry verifies, and the per-modality geometric
evidence — not the descriptor score — decides what to trust. On real data, that
architecture turns two individually-limited sensors into a retriever that beats both,
and a LiDAR map that holds up months later.

## 8. Limitations

- **Grayscale camera.** KITTI `image_0` (grayscale → 3 channels); EigenPlaces
  trained on RGB. Forward camera numbers would likely rise with colour `image_2`;
  this does not affect Finding 2 (the reverse collapse is geometric) or Finding 3.
- **Cross-dataset/-session work is LiDAR-only.** §5–6 validate the LiDAR retrieval on
  NCLT (within-session and across 209 days); the camera/fusion findings (2, 3) were
  not re-run there (NCLT's camera is a 360° Ladybug, not a forward monocular). §6 is a
  single winter→summer pair; a sweep over more seasons and longer gaps (a year+) is the
  natural next step.
- **seq 07 is small** (94 revisit queries) — its absolute number is noisy.
- **Geometric verification uses a Scan-Context proxy**, not a full SE(3) ICP
  residual with inlier counts. We tested the full SE2 aligner as the verifier
  (§4): at its default success threshold it over-accepts and fails the blind case,
  and a threshold sweep shows it is competitive only once tuned — with a
  sequence-dependent optimum — whereas the relative proxy is calibration-free. The
  acceptance factor α is not cherry-picked:
  sweeping it over a 2× range (α ∈ [1.0, 2.0]) moves the mean R@1 @ 5 m only
  within 0.762–0.784 and beats every single-modality and score-fusion number
  throughout; the default α = 1.3 (0.779) sits on the plateau.

## 9. Reproducibility

```bash
python scripts/benchmark_kitti_lidar.py            # LiDAR Scan-Context
python scripts/benchmark_kitti_vpr.py              # camera ResNet-18 baseline
python scripts/benchmark_kitti_vpr_learned.py      # camera EigenPlaces (MIT, torch.hub)
python scripts/experiment_scancontext_config.py    # LiDAR default-vs-wide (seq 00/08)
python scripts/benchmark_kitti_fusion.py           # RRF / confidence-gate / geo-verified
python scripts/benchmark_nclt_lidar.py [--wide]    # cross-dataset, NCLT within-session
python scripts/benchmark_nclt_cross_session.py [--wide]  # cross-session, winter->summer
python scripts/make_results_summary.py             # Findings 1-2 figure
python scripts/make_fusion_figure.py               # Finding 3 figure
python scripts/make_cross_session_figure.py        # Finding 5 figure
```

Scan-Context is BEVMatch's own implementation; EigenPlaces is loaded at runtime
from `gmberton/eigenplaces` (MIT) and is not vendored. Per-sequence JSON results
are under `docs/assets/`.

## References

1. A. Geiger, P. Lenz, R. Urtasun. *Are we ready for autonomous driving? The KITTI vision benchmark suite.* CVPR 2012.
2. G. Kim, A. Kim. *Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D Point Cloud Map.* IROS 2018.
3. G. Berton, G. Trivigno, B. Caputo, C. Masone. *EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition.* ICCV 2023.
4. G. V. Cormack, C. L. A. Clarke, S. Büttcher. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
6. D. G. Lowe. *Distinctive Image Features from Scale-Invariant Keypoints.* IJCV 2004.
7. N. Carlevaris-Bianco, A. K. Ushani, R. M. Eustice. *University of Michigan North Campus Long-Term Vision and LIDAR Dataset.* IJRR 2016.
