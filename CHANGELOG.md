# Changelog

All notable changes to BEVMatch. Versions follow the roadmap in
[docs/architecture.md §21](docs/architecture.md).

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
