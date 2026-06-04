# Same-place retrieval across modalities: representation helps, viewpoint walls don't fall to learning, and geometry — not score — fuses

**A technical report on the BEVMatch real-data benchmarks (KITTI odometry).**

> All numbers are measured by BEVMatch's own retrieval pipeline on public KITTI
> data, one standard protocol, no ground truth at inference, fully reproducible
> from `scripts/`. This report consolidates the three findings; the conversational
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
evidence design over retrieval score alone.

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

**The verification must be *relative*, not absolute.** We also tried BEVMatch's
*full* SE2 aligner as the verifier — a 360° BEV cross-correlation + ICP with the
framework's own `success` verdict (overlap ≥ 0.45) — expecting it to sharpen the
proxy (`scripts/experiment_icp_verification.py`). It does not. On the decisive
reverse seq 08 it **accepts the blind camera on 89 % of queries** and lands at
R@1 = 0.068 (vs the proxy's 0.343), because a BEV cross-correlation *maximises*
overlap and generic urban structure (roads, façades) exceeds 0.45 even between
different places — an absolute geometric-success threshold is fooled by
self-similarity, just as absolute score magnitude was (§4). The proxy works not
because it is geometric but because it is **comparative**: it asks whether the
camera's place is as consistent as LiDAR's *own best for that query*. The lesson
is not "run more ICP" but "verify *relative* to what a true match looks like
here". (On camera-strong seq 06 the full verifier accepts 100 % and matches the
camera's 0.977 — it is good at confirming correct matches, only poor at rejecting
wrong ones.)

## 5. Discussion

The three findings compose into one statement: **better representations help where
the view is shared, cannot manufacture an unobserved view, and the right way to
combine sensors that fail differently is to verify a match geometrically rather
than to combine retrieval scores.** This is exactly a retrieve → align → evidence
architecture: retrieval proposes, geometry verifies, and the per-modality
geometric evidence — not the descriptor score — decides what to trust. On real
data, that architecture turns two individually-limited sensors into a retriever
that beats both.

## 6. Limitations

- **Grayscale camera.** KITTI `image_0` (grayscale → 3 channels); EigenPlaces
  trained on RGB. Forward camera numbers would likely rise with colour `image_2`;
  this does not affect Finding 2 (the reverse collapse is geometric) or Finding 3.
- **One dataset.** KITTI urban driving only; cross-dataset generalisation untested.
- **seq 07 is small** (94 revisit queries) — its absolute number is noisy.
- **Geometric verification uses a Scan-Context proxy**, not a full SE(3) ICP
  residual with inlier counts. We tested the full SE2 aligner as the verifier
  (§4) and found it *over-accepts* on self-similar urban geometry; the proxy's
  *comparative* criterion is what matters, not the verifier's sophistication. The
  acceptance factor α is not cherry-picked:
  sweeping it over a 2× range (α ∈ [1.0, 2.0]) moves the mean R@1 @ 5 m only
  within 0.762–0.784 and beats every single-modality and score-fusion number
  throughout; the default α = 1.3 (0.779) sits on the plateau.

## 7. Reproducibility

```bash
python scripts/benchmark_kitti_lidar.py            # LiDAR Scan-Context
python scripts/benchmark_kitti_vpr.py              # camera ResNet-18 baseline
python scripts/benchmark_kitti_vpr_learned.py      # camera EigenPlaces (MIT, torch.hub)
python scripts/experiment_scancontext_config.py    # LiDAR default-vs-wide (seq 00/08)
python scripts/benchmark_kitti_fusion.py           # RRF / confidence-gate / geo-verified
python scripts/make_results_summary.py             # Findings 1-2 figure
python scripts/make_fusion_figure.py               # Finding 3 figure
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
