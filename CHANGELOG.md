# Changelog

All notable changes to BEVMatch. Versions follow the roadmap in
[docs/architecture.md §21](docs/architecture.md).

## 1.13.0 — Cross-dataset generalisation on NCLT (Finding 4)

- `scripts/benchmark_nclt_lidar.py`: runs the *same* Scan-Context LiDAR retrieval
  and protocol on **NCLT** (Michigan North Campus Long-Term) — a different city,
  Segway platform and **different sensor (Velodyne HDL-32E, 32 beams vs KITTI's
  64)**. Includes a from-scratch parser for NCLT's packed `velodyne_hits.bin`
  hit-stream (accumulates partial packets into 10 Hz revolutions) + ground-truth
  time-sync, a scan cache, and default/wide config selection.
- **Finding 4:** the retrieval **generalises** — same code, R@1 @ 5 m = 0.620 on
  3716 scans / 1664 revisit queries of a 94-minute campus route — below KITTI's
  best (seq 00 wide 0.966), consistent with a half-density 32-beam sensor and a
  vegetated campus. And the **config lesson transfers**: the same "wide"
  descriptor that rescued KITTI seq 08 (0.34 → 0.77) nearly doubles NCLT recall
  (default 0.358 → wide 0.620) — the larger campus needs the longer 80 m range.
- docs/findings.md (now "Four findings"), docs/report.md (§5 + NCLT reference),
  docs/benchmarks.md (Cross-dataset section), README: NCLT table and takeaway.

## 1.12.1 — Overlap-threshold sweep refines the verifier claim (a second correction)

- `scripts/experiment_icp_verification.py`: sweeps the full-ICP verifier's overlap
  acceptance threshold τ ∈ {0.45..0.85} on seq 08 (blind) and seq 06 (camera-right)
  and reports verified R@1 @ 5 m + accept fraction per τ.
- **Refines v1.12.0's "relative beats absolute".** That framing was too strong. The
  sweep shows the full verifier's failure is *calibration*, not incapacity: a
  stricter τ recovers most of seq 08 (0.068 → 0.339 as τ 0.45 → 0.85). The real
  point is the optimal τ is *opposite* per sequence (blind wants strict, camera-
  right wants lenient — seq 06 0.977 → 0.929 as τ tightens), so any single absolute
  threshold is a compromise (τ ≈ 0.75 → seq 08 0.320, seq 06 0.972). The relative
  proxy matches that with no per-dataset tuning (one α = 1.3 → 0.343 / 0.943) and is
  marginally ahead on the blind case. So a tuned absolute verifier is competitive;
  the relative one is simply calibration-free.
- docs/report.md + docs/findings.md: τ-sweep table added; the absolute-vs-relative
  wording softened to calibration-free-vs-tuned.

## 1.12.0 — Full SE2-ICP verifier probe: relative beats absolute (corrects a claim)

- `scripts/experiment_icp_verification.py`: replaces the Scan-Context proxy in the
  geometric-verification fusion with BEVMatch's **real** SE2 aligner (BEV
  cross-correlation + ICP, `AlignmentHypothesis.success` = overlap ≥ 0.45) and
  compares the two verifiers on the decisive sequences (seq 08 blind, seq 06
  camera-strong).
- **Finding (and an honest correction).** v1.10/1.11 speculated a full aligner
  "should only sharpen" the proxy. Measured, it does the opposite: the absolute
  overlap-success criterion **over-accepts** — on blind seq 08 it accepts the
  camera on 89 % of queries (vs the proxy's 14 %) and collapses to R@1 = 0.068
  (vs 0.343), because a BEV cross-correlation maximises overlap and generic urban
  structure clears 0.45 even between different places. On camera-strong seq 06 it
  accepts 100 % and matches the camera (0.977). The proxy wins not by being
  geometric but by being **comparative** (camera's place vs LiDAR's own best for
  that query) — the same relative-vs-absolute lesson as the score diagnostic.
- docs/report.md + docs/findings.md: the probe result added and the earlier
  "should only sharpen" speculation corrected. Uses a coarse verifier config
  (voxel 1.0 m, yaw step 6°) for runtime; the over-acceptance is mechanistic, not
  a config artifact (confirmed by seq 06's 100 % accept of correct matches).

## 1.11.1 — Geo-verification robustness: ALPHA sweep

- `scripts/benchmark_kitti_fusion.py`: sweeps the geometric-verification
  acceptance factor ALPHA ∈ {1.0,1.1,1.2,1.3,1.5,2.0} (free once the rank matrices
  exist) and reports the mean verified R@1 @ 5 m per ALPHA.
- **Result (substantiates the report's robustness claim):** the mean barely moves
  over a 2× ALPHA range (0.762–0.784) and stays above every single-modality and
  score-fusion number throughout — looser ALPHA helps camera-strong forward loops
  and hurts the blind seq 08 (0.347 → 0.287), so the mean plateaus around 1.3–1.5.
  The default 1.3 (0.779) is not cherry-picked.
- docs/findings.md + docs/report.md: ALPHA-sweep table / sentence.

## 1.11.0 — Technical report

- `docs/report.md`: an arXiv-style, citable consolidation of the three real-data
  findings (representation helps within a modality; the reverse-view wall does not
  fall to learning; geometry — not score — fuses the two), with abstract, numbered
  sections, both result figures, honest limitations, a reproducibility block, and
  references (KITTI, Scan-Context, EigenPlaces, RRF, ResNet, Lowe).
- docs/findings.md retitled "Three findings" and linked to the report; README
  links the report.

## 1.10.1 — Fusion result figure

- `scripts/make_fusion_figure.py` + `docs/assets/bevmatch_fusion_summary.png`:
  a grouped-bar figure (read from kitti_fusion_results.json, no recompute) of
  Recall@1 @ 5 m for the two single modalities and the three fusion strategies,
  visualising Finding 3's resolution — score-level fusion (RRF, gate) collapses on
  the reverse loop while geometry-verified fusion wins on every sequence and fully
  recovers seq 08.
- README + docs/benchmarks.md: fusion figure embedded.

## 1.10.0 — Geometric-verification fusion beats both modalities (Finding 3 resolved)

- `scripts/benchmark_kitti_fusion.py` adds a third strategy, **geometric
  verification**: per query, trust the camera's proposed place only if its two
  LiDAR scans (query frame vs camera's top-1 frame) align almost as well as
  LiDAR's own best — Scan-Context alignment distance within ALPHA = 1.3 — else
  fall back to LiDAR. One extra SC alignment per query; no ground truth.
- **The resolution of Finding 3.** Where the score-level fusions failed (naive
  RRF 0.695, confidence-gate 0.714, both unable to recover the blind reverse
  loop), geometric verification **wins on every sequence** — mean R@1 @ 5 m =
  **0.779**, +0.07 over either modality alone — and **fully recovers seq 08**
  (0.343 ≈ LiDAR's 0.339): the camera's geometrically-wrong reverse-loop proposal
  fails the alignment check and is rejected (camera accepted on 16% of seq 08
  queries vs 53% on camera-strong seq 06), while its correct forward-loop
  proposals are kept (seq 00 0.963, seq 05 0.922, both above LiDAR). The
  retrieve → align → evidence design, validated on real data: fusing on geometry
  (not score) turns two individually-limited sensors into a retriever beating both.
- docs/findings.md (Finding 3 rewritten with the resolution) + docs/benchmarks.md
  + README: geo-verified column and the takeaway.

## 1.9.1 — Finding 3, deepened: "confidently wrong" is unflaggable by score

- docs/findings.md: a diagnostic measuring the camera's mean top-1 cosine per
  sequence shows the blind seq 08 (0.46) scores *higher* than seq 07 (0.41) where
  the camera works (R@1 0.68 vs 0.015). On a reverse loop the camera confidently
  matches a similar-looking *wrong* place, so "confidently wrong" is
  indistinguishable from "confidently right" by score alone — no
  descriptor-confidence gate (relative or absolute) can separate them. The real
  arbiter is **geometric verification** (BEVMatch's alignment/evidence stage),
  which reframes the "next experiment" from a better score gate to verification.

## 1.9.0 — Late-fusion benchmark (Finding 3: fusion is not a free lunch)

- `scripts/benchmark_kitti_fusion.py`: fuses BEVMatch's LiDAR Scan-Context and
  EigenPlaces camera rankings two ground-truth-free ways — equal-weight Reciprocal
  Rank Fusion and a confidence gate (per-query top-1-vs-top-2 margin, Lowe-style).
  Runs purely on cached descriptors with numpy + BEVMatch's `sc_alignment_distance`
  (no network, no external model code); rank matrices vectorised, only the
  per-query Scan-Context rerank stays in Python.
- **Finding 3 (honest negative + nuance):** naive equal-weight RRF is a *net loss*
  (mean R@1 @ 5 m = 0.695, below both single modalities) — on reverse seq 08 the
  blind camera drags LiDAR from 0.339 down to 0.081. A confidence gate is the best
  on average (0.714) and never catastrophic, **but still does not recover the
  blind case** (seq 08 → 0.203, short of LiDAR's 0.339; it picks the blind camera
  ~49% of queries because a within-sequence self-normalised margin cannot encode
  global blindness). Recovering the blind case needs absolute cross-modal
  confidence calibration, not naive score combination — which motivates
  learned/calibrated fusion as the next step.
- docs/findings.md (Finding 3) + docs/benchmarks.md (Fusion section) + README:
  fusion table and the honest takeaway.

## 1.8.1 — Technical note: two findings

- `docs/findings.md`: a short, honest technical note that states what the
  benchmarks *mean* — (1) within a modality, representation quality is real and
  measurable (a learned descriptor lifts every forward case); (2) across a
  viewpoint gap (reverse-direction seq 08) representation quality is *not* enough
  — both camera descriptors stay pinned at 0.015 while the same intervention
  recovers LiDAR from 0.34 to 0.77 — the measured argument for modality-agnostic
  same-place comparison. Includes an explicit limitations section (grayscale
  camera, single dataset, small seq 07, no fused retriever yet).
- `docs/benchmarks.md`: summary figure embedded at the top; links to findings.
- README: link to the findings note.

## 1.8.0 — One-figure results summary

- `scripts/make_results_summary.py` + `docs/assets/bevmatch_results_summary.png`:
  a single grouped-bar figure (read from the benchmark JSONs, no recompute) of
  Recall@1 @ 5 m across the KITTI loop sequences for the three descriptors that
  all run behind BEVMatch's one retrieval interface — hand-crafted LiDAR
  Scan-Context, the generic ResNet-18 camera baseline, and the learned EigenPlaces
  camera SOTA — plus a ghost bar for the LiDAR wide-config recovery on seq 08.
  Visualises the whole research story: learning lifts the forward cases, both
  camera descriptors hit the reverse-direction viewpoint wall (~0.02), and only
  the 360° LiDAR holds (0.34) and recovers (0.76) there.
- README: summary figure at the top of the real-data benchmark section.

## 1.7.0 — Learned SOTA descriptor comparison (EigenPlaces)

- `scripts/benchmark_kitti_vpr_learned.py`: swaps the generic ResNet-18 (ImageNet)
  camera descriptor for **EigenPlaces** (Berton et al., ICCV 2023, MIT — loaded
  at runtime via `torch.hub`, not vendored), a place-recognition network trained
  on San Francisco eXtra Large. SF-XL is disjoint from KITTI, so KITTI is held out
  for *every* sequence — a fair comparison everywhere, no train-on-test caveat.
  Same protocol as the ResNet-18 benchmark; the framework check confirms
  `SceneDatabase` reproduces the learned ranking too.
- **Finding:** the learned SOTA descriptor lifts every *forward* revisit case
  (seq 07 R@1 0.500 → 0.681, seq 00 0.923 → 0.957, seq 05 0.848 → 0.914; mean
  0.653 → 0.709) — representation quality clearly helps. **But on the reverse-
  direction loops (seq 08) it is pinned at 0.015, identical to the baseline.** A
  forward-facing camera never observes the opposite-direction view, so *no*
  appearance descriptor — however well trained — can match it. This is a
  viewpoint/geometry wall, not a representation gap, and it is exactly where the
  360° LiDAR descriptor is instead *recoverable* (0.339 → 0.765 by config). The
  sharpest evidence yet for Principle 2 (modality ≠ representation).
- docs/benchmarks.md + README: learned-vs-baseline camera table.
- Note: an OverlapTransformer (GPL-3.0) LiDAR comparison was scoped but dropped
  to keep the project free of GPL dependencies; EigenPlaces (MIT) is used instead.

## 1.6.0 — Cross-modal failure-mode GIF

- `scripts/make_crossmodal_gif.py` + `docs/assets/bevmatch_crossmodal.gif`:
  real KITTI seq 08 reverse-direction revisits where the forward-facing camera
  retrieves the wrong place (often hundreds of metres away) while the 360°
  LiDAR Scan-Context retrieves the genuine revisit (metres away). Frames are
  selected from real retrievals (camera top-1 > 25 m GT, LiDAR top-1 < 5 m GT).
- README: cross-modal GIF added to the hero section.

## 1.5.0 — Reverse-loop recovered by descriptor tuning

- `scripts/experiment_scancontext_config.py`: A/B of the Scan-Context descriptor
  config (default 20×60/30 m vs wide 40×120/80 m) on a forward (00) and a
  reverse (08) sequence.
- **Finding:** widening the descriptor — a pure plugin-config swap, no pipeline
  change — more than doubles reverse-loop recall (seq 08 R@1 0.339 → 0.765)
  while only nudging the forward case (seq 00 0.913 → 0.966). seq 08's low
  default number is a config limitation, not a fundamental one. The camera's
  seq 08 collapse, by contrast, is intrinsic and not tunable — the asymmetry
  that motivates a multi-modal framework.
- docs/benchmarks.md + README: reverse-loop tuning table.

## 1.4.0 — Cross-modal multi-sequence benchmark

- Camera VPR extended across the KITTI loop sequences 00/05/06/07/08
  (`scripts/benchmark_kitti_vpr.py [seqs...]`), matching the LiDAR coverage.
- **Cross-modal failure-mode finding** (Principle 2): on reverse-direction
  revisits (seq 08), the forward-facing camera collapses to R@1 = 0.015 while
  the 360° rotation-invariant LiDAR descriptor holds R@1 = 0.339 — the two
  modalities fail differently, the argument for a modality-agnostic framework.
- `scripts/_extract_gray_blocks.py`: per-sequence range download + extraction
  of KITTI gray images from the public archive.
- docs/benchmarks.md + README: combined per-sequence cross-modal tables.

## 1.3.0 — Multi-sequence LiDAR benchmark

- LiDAR Scan-Context Recall@K extended across the **KITTI loop sequences**
  00/05/06/07/08 (`scripts/benchmark_kitti_lidar.py [seqs...]`), with a
  per-sequence table + mean (R@1 = 0.704 @ 5 m across 5 sequences).
- Honest reporting of hard cases: seq 08 (reverse-direction revisits, R@1 = 0.34)
  and seq 07 (only 94 revisits) are reported, not hidden.
- `scripts/_extract_velodyne_blocks.py`: per-sequence contiguous range download
  + local-header extraction of velodyne from the public KITTI archive.

## 1.2.0 — Real public-data benchmarks

- **KITTI odometry place-recognition benchmarks** with real Recall@K, standard
  protocol (positive = pose ≤ D & |dt| > 30 s; temporal window excluded):
  - Camera VPR (`scripts/benchmark_kitti_vpr.py`): ResNet-18 embeddings,
    seq 00 R@1 = 0.923 @ 5 m.
  - LiDAR Scan-Context (`scripts/benchmark_kitti_lidar.py`): ring-key prefilter
    + SC column-shift rerank, seq 00 R@1 = 0.913 @ 5 m.
  - Cross-modal comparison on the same sequence/protocol (Principle 2).
- `docs/benchmarks.md`: real-data results + an explicit separation of
  benchmark numbers vs synthetic sanity-check tables.
- README: real-data benchmark section up top; synthetic tables labelled as
  sanity checks.

## 1.1.0 — Real data + scalable alignment

- Real-data loaders (`bevmatch.datasets`): `load_pcd` / `load_las_tile` /
  `load_kitti_bin`, `voxel_downsample`, `remove_ground`, `scene_from_points`.
- Scalable nearest-neighbour search (`bevmatch.spatial`): KD-tree / chunked
  brute force — fixes the ICP OOM on dense real LiDAR.
- GitHub Actions CI (pytest on push/PR, Python 3.11/3.12).
- README hero GIFs from **real** data: LiDAR-map localization (109M-point survey)
  and KITTI loop-closure visual place recognition (ResNet-18 camera embeddings).
- Optional extras: `perf` (scipy), `data` (laspy, open3d).

## 1.0.0 — Stable Same-Place Comparison Platform

- Stable **artifact schema** (`bevmatch.schema`, `ARTIFACT_SCHEMA_VERSION=1.0`)
  with validation, compatibility checks, and self-describing envelopes.
- Stable **plugin manifest** (`bevmatch.plugins`, §7.3) with built-in manifests.
- Stable **benchmark protocol** (`BENCHMARK_PROTOCOL_VERSION=1.0`).
- Documentation: CONTRIBUTING, GOVERNANCE, API compatibility, plugin authoring.
- Reproducible demo suite (`examples/run_all_demos.py`).

## 0.9.0 — Multi-Modal Expansion

- Modality-agnostic retrieval (LiDAR / radar / camera); `CameraEmbeddingDescriptor`,
  radar→BEV adapter, `SemanticBEV`.
- Object-level change (added/removed/moved/class-changed), scene graph,
  natural-language summaries.

## 0.8.0 — Benchmark Suite

- Dataset cards, reproducible split manifests (sha256 fingerprints), pipeline-level
  benchmark suite, leaderboard with external submission merging.

## 0.7.0 — Autoware / Nav2 Adapters

- Initial-pose candidates with covariance, localization health, point cloud map
  freshness, Lanelet2 consistency; Nav2 OccupancyGrid staleness, relocalization,
  changed-area annotation.

## 0.6.0 — ROS2 Integration

- ROS2-independent bridge (TF tree, diagnostics, markers, lifecycle bag replay)
  and an rclpy `LifecycleNode`.

## 0.5.0 — Map Validation

- Point cloud / occupancy / vector map validators, severity schema, stale-region
  report, human-review export, issue precision/recall metrics.

## 0.4.0 — Change Detection

- Occlusion-aware comparable region, temporal persistence (dynamic filtering),
  change metrics, before/after viewer.

## 0.3.0 — Alignment Framework

- SE2/SE3 aligner plugins, richer alignment evidence, failure classification,
  alignment metrics, residual visualization.

## 0.2.0 — Retrieval Framework

- Descriptor / index-backend plugins, Scan-Context and BEV-grid baselines,
  Recall@K/MRR metrics, evaluation recipe.

## 0.1.0 — Concept-Proof Architecture

- Core data model, Comparison Evidence Bundle, LiDAR retrieval + SE2 alignment +
  BEV occupancy diff, synthetic benchmark, end-to-end demo.
