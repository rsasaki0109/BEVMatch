# BEVMatch

<p align="center">
  <a href="https://github.com/rsasaki0109/BEVMatch/actions/workflows/ci.yml"><img src="https://github.com/rsasaki0109/BEVMatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/version-1.14.0-informational" alt="v1.14.0">
</p>

<p align="center">
  <img src="docs/assets/bevmatch_camera_bev.gif" alt="BEVMatch recognizing a place in a camera-derived bird's-eye view: stereo depth turns the camera into a BEV, then the current BEV is retrieved and aligned onto the matched map BEV" width="100%">
</p>
<p align="center"><sub><b>Real data, actual pipeline output.</b> A revisit on KITTI seq 00: stereo depth turns the <b>camera</b> into a <b>bird's-eye view</b>, BEVMatch <b>retrieves</b> the matching first-visit place (score 0.98) and <b>aligns</b> the two camera BEVs (gold = matched map, cyan = current) — the same place, recognized in a camera BEV.</sub></p>

<p align="center">
  <img src="docs/assets/bevmatch_hero.gif" alt="BEVMatch localizing a real LiDAR observation against a real 109M-point survey map: it retrieves the matching map tile and recovers the SE2 pose with covariance" width="100%">
</p>
<p align="center"><sub>Same pipeline, <b>LiDAR</b> input: a real observation localized against a real 109M-point survey map — retrieve the place tile (score 0.98), align it, and recover the <b>SE2 pose + covariance</b> as an initial pose for Autoware / Nav2.</sub></p>

> **BEVMatch is not another place-recognition method.**
> It is an OSS pipeline for **finding the same place, aligning it, comparing it, and turning the differences into map-validation evidence** — across LiDAR, camera, and radar.

```
Query Scene → Retrieve → Align → Compare → Change Evidence → Map Validation
```

Core depends only on `numpy`. Optional: `scipy` / `matplotlib` / `faiss-cpu` / `rclpy` / `laspy` / `open3d`. 79 tests, CI-green, Apache-2.0.

---

## Five findings (real public data: KITTI + NCLT)

All numbers are **measured** with one place-recognition protocol (positive = GT pose ≤ D m; within-session also requires |Δt| > 30 s and excludes the temporal window). Synthetic tables elsewhere in the repo are wiring sanity-checks, not performance. Full story: **[docs/findings.md](docs/findings.md)** · **[technical report](docs/report.md)**.

<p align="center">
  <img src="docs/assets/bevmatch_results_summary.png" alt="Recall@1 across KITTI loop sequences for three descriptors behind one interface" width="88%">
</p>

**1 — Representation quality matters.** A learned SOTA camera descriptor (EigenPlaces, ICCV 2023) lifts forward revisits over a generic ResNet-18 (mean R@1 0.653 → 0.709).

**2 — The viewpoint wall is not learnable.** On reverse-direction KITTI seq 08 the forward camera collapses to R@1 = 0.015 — *even with the SOTA descriptor* — because it never observed the opposite view. The 360° LiDAR holds (0.339) and a config swap recovers it to **0.765**. Learning can't fix a viewpoint problem; a config tweak fixes a geometry problem.

| seq 08 (reverse) | ResNet-18 | EigenPlaces | LiDAR default | LiDAR wide |
|---|---|---|---|---|
| R@1 @ 5 m | 0.015 | 0.015 | 0.339 | **0.765** |

<p align="center">
  <img src="docs/assets/bevmatch_crossmodal.gif" alt="On reverse-direction revisits the camera retrieves the wrong place while LiDAR retrieves the right one" width="88%">
</p>
<p align="center"><sub>Same query, same database, opposite outcomes: the forward camera retrieves a place hundreds of metres away (red ✗); the 360° LiDAR (BEV) retrieves the true revisit (green ✓).</sub></p>

**3 — Score fusion loses; geometric verification wins.** Naive late fusion is a net loss (the blind modality drags the good one down). Verifying the camera's pick with LiDAR geometry beats *both* single modalities (mean R@1 **0.779**) and fully recovers seq 08.

| | LiDAR | Camera | naive RRF | **geo-verified** |
|---|---|---|---|---|
| mean R@1 @ 5 m | 0.704 | 0.709 | 0.695 (net loss) | **0.779** |

**4 — LiDAR retrieval generalizes across datasets.** Same code, new city / robot / sensor (NCLT, HDL-32E 32-beam): the `wide` config that rescued KITTI seq 08 nearly doubles NCLT recall (0.358 → **0.620**).

**5 — The map survives the seasons.** A winter NCLT map localizes a summer drive **209 days later** at R@1 @ 5 m = **0.678** vs 0.840 same-day — graceful, not collapse. The modality that *lost* the viewpoint battle (camera, Finding 2) *wins* the long-term battle: LiDAR range geometry barely moves with foliage, snow, or light.

```bash
python scripts/benchmark_kitti_lidar.py          # LiDAR Scan-Context, all loops
python scripts/benchmark_kitti_vpr_learned.py    # camera VPR (EigenPlaces SOTA)
python scripts/benchmark_kitti_fusion.py         # LiDAR + camera fusion
python scripts/benchmark_nclt_lidar.py --wide    # cross-dataset (NCLT)
python scripts/benchmark_nclt_cross_session.py --wide   # cross-season (209 days)
```

---

## Quickstart

```bash
pip install -e ".[viz,dev]"   # core is numpy-only; this adds matplotlib + pytest
python examples/run_demo.py   # run the pipeline on synthetic data
pytest
```

```python
from bevmatch import SamePlaceComparisonPipeline, SceneDatabase
from bevmatch.datasets import make_synthetic_same_place

data = make_synthetic_same_place()
db = SceneDatabase(); db.add_all(data.historical)
bundle = SamePlaceComparisonPipeline(database=db).run(data.query)
print(bundle.summary())
```

**Real point clouds** (PCD / LAS / KITTI `.bin`) load the same way — voxel-downsample dense clouds; alignment uses KD-tree NN so it won't OOM on large maps:

```python
from bevmatch.datasets import load_pcd, scene_from_points
scene = scene_from_points(load_pcd("scan.pcd"), "scan0", voxel=0.7, drop_ground=True)
db.add(scene)
```

The pipeline runs through retrieval → alignment → change → map validation → ROS2 → Autoware/Nav2 → benchmark → multi-modal. See `examples/` for one runnable demo per layer.

---

## Documentation

- **[Five findings](docs/findings.md)** — the readable technical note behind the results above.
- **[Technical report](docs/report.md)** — citable, arXiv-style writeup (setup → results → discussion → limits → reproducibility).
- **[Real-data benchmarks](docs/benchmarks.md)** — full Recall@K tables, protocol, and the synthetic vs real distinction.
- **[Architecture](docs/architecture.md)** — data model, plugin / pipeline design, ROS2 / Autoware / Nav2 integration, roadmap.
- **[Contributing](CONTRIBUTING.md)** · **[Governance](GOVERNANCE.md)** · **[Plugin authoring](docs/plugin_authoring.md)** · **[Changelog](CHANGELOG.md)**

## License

Apache-2.0 — see [LICENSE](LICENSE).
