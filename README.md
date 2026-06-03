# BEVMatch

> BEVMatch is not another place recognition method.
> It is an OSS pipeline for finding the same place, aligning it, comparing it, and turning differences into map validation evidence.

BEVMatch は、場所を探すだけで終わらない。
**同じ場所を見つけ、整列し、変化を説明し、地図の信頼性を検証する**「Same-Place Comparison OSS」です。

```
Query Scene
  ↓ Place Retrieval
  ↓ Geometric / Semantic Alignment
  ↓ Scene-to-Scene Comparison
  ↓ Change Evidence
  ↓ Map Validation / Map Maintenance Evidence
```

## Status

v0.9 — multi-modal expansion（camera / radar / LiDAR の modality-agnostic retrieval、semantic BEV、object-level change（added/removed/moved/class-changed）、scene graph、自然言語レポート）。BEV/LiDAR-only ではないことを実証（Principle 2）。
v0.8 benchmark suite、v0.7 Autoware/Nav2 adapters、v0.6 ROS2 integration、v0.5 map validation、v0.4 change detection、v0.3 alignment、v0.2 retrieval、v0.1 MVP も合成データでエンドツーエンド動作。コア依存は `numpy` のみ（任意で `matplotlib` / `faiss-cpu` / `rclpy`）。

## Quickstart

```bash
pip install -e .            # core (numpy)
pip install -e ".[viz,dev]" # + matplotlib + pytest

python examples/run_demo.py # 合成データでパイプラインを実行
pytest                      # テスト
```

`run_demo.py` は `out/evidence_bundle.json`（Comparison Evidence Bundle）と、
matplotlib があれば `out/same_place_comparison.png`（4 面比較ビュー: 現在 / 過去 /
整列オーバーレイ / 変化）を出力します。

```text
Best match: hist_2 (place=place_2, scan-context, score=0.87)
Alignment: x=-1.04 m, y=+1.21 m, yaw=-1.1 deg, overlap=75%, inliers=90%
Changes: 2 added, 2 removed
```

### Retrieval benchmark (v0.2)

```bash
python examples/run_retrieval_eval.py
```

同一ルート上で descriptor を差し替えて Recall@K / MRR を比較します。
rotation-invariant な Scan-Context が、revisit yaw 下で BEV-grid を上回ることを示します。

```text
descriptor      index         R@1  R@5  MRR
----------------------------------------
scan-context    brute-force   0.833  0.958  0.875
bev-grid        brute-force   0.417  0.583  0.481
```

descriptor（`GlobalDescriptor`）と index backend（`IndexBackend`）はプラグインで、
`SceneDatabase(descriptor=..., index=...)` で差し替えられます（FAISS は `make_index("faiss")`）。

### Alignment benchmark (v0.3)

```bash
python examples/run_alignment_eval.py
```

SE2/SE3 aligner を GT 相対姿勢に対して評価し、failure classification と
SE3 の縮退（平面シーンで z/roll/pitch が観測不能）を表示します。
`out/alignment_residual.png` に整列オーバーレイ + 残差マップを出力します。

```text
aligner       succ  wtol  t_err  r_err  ovlp
se2-bev-xcorr  1.000  1.000  0.083  0.13  0.772
se3-icp        1.000  1.000  0.085  0.14  0.773

Wrong-place alignment: success=False, class=overlap_insufficient, overlap=0.30
SE3 degeneracy (planar scene): unobservable=['z','roll','pitch']
```

aligner（`Aligner`）もプラグインで、`SamePlaceComparisonPipeline(aligner=...)` や
`evaluate_alignment(...)` で差し替えられます。

### Change benchmark (v0.4)

```bash
python examples/run_change_eval.py
```

§11 の「observed difference ≠ actionable change」を2点で実証します。

```text
=== Persistence (dynamic filtering) ===          # 複数フレームで移動物体を除外
actionable: added P/R=1.00/1.00 removed P/R=1.00/1.00  dynamic filtered=7
false actionable changes=0

=== Occlusion vs removal ===                     # 遮蔽を「削除」と誤らない
use_occlusion=False: removed=3 (occluded mis-reported=2)
use_occlusion=True:  removed=1 (occluded mis-reported=0)  occluded=0.33
```

- **comparable region**：両シーンが観測した領域のみ比較（polar ray-cast による遮蔽推定）。
- **temporal persistence**：複数フレームで持続する変化のみ actionable とし、移動物体は `dynamic` として除外。
- `out/change_evidence.png` に before/after + 変化エビデンスの4面ビューを出力。

### Map validation benchmark (v0.5)

```bash
python examples/run_map_validation.py
```

「この地図は現在の世界とまだ一致しているか？」を検証し（ファイル構文検証ではなく、§12.5）、
change evidence を運用判断可能な **Map Validation Issue** に変換します。

```text
| ID            | Severity | Type                     | Location     | Action                    |
| map_a-ISSUE-0 | high     | new_static_obstacle      | (+10.7,+11.2)| Inspect / add to map      |
| map_a-ISSUE-2 | medium   | missing_static_structure | (+19.1,+18.5)| Verify removal; update    |
| map_a-ISSUE-4 | medium   | map_element_unobserved   | (+19.2,+18.5)| Confirm vector element    |

issue P/R/F1 = 1.00/1.00/1.00     fresh map -> 0 issues
```

- point cloud / occupancy / vector(Lanelet2 風) の3 validator（`MapValidator` プラグイン）。
- severity schema（INFO→CRITICAL）+ recommended action + stale region 抽出。
- `out/map_validation_review.md`（人手レビュー用）と `out/map_validation_report.json` を出力。
- 既存の Lanelet2 構文 validator を置き換えず、**observation-to-map consistency** を担う（§12.5）。

### ROS2 integration (v0.6)

Core は ROS2 非依存。bag replay は ROS2 なしで動きます（offline-first, §16.1）。

```bash
python examples/run_ros_replay.py        # ROS2 不要: lifecycle bag replay
```

```text
lifecycle: unconfigured -> inactive -> active
[t=10.0] query_0  match=place_4  changes=3  markers=3  diag={retrieval:OK, alignment:OK, change:WARN}
...
lifecycle: -> finalized
```

ROS2 環境がある場合は rclpy LifecycleNode を起動できます（`MarkerArray` /
`DiagnosticArray` / `PoseWithCovarianceStamped` を publish）。

```bash
python examples/ros2_lifecycle_node.py   # 要 ROS2 (rclpy): configure -> activate -> publish
ros2 topic echo /bevmatch/markers
```

- `bevmatch.ros`：TF tree（`map→odom→base_link→sensor`）、diagnostics、markers、
  lifecycle `BagReplayPipeline`（すべて純 Python・テスト可能）。
- `bevmatch.ros.node`：rclpy `LifecycleNode`（ROS2 がある時のみ import）。

### Autoware / Nav2 adapters (v0.7)

```bash
python examples/run_autoware_nav2.py
```

BEVMatch は Autoware/Nav2 の localization を置き換えず、その周辺を支援します（§17.1, §18.1）。

```text
Autoware initial pose:  place=place_4 pose=(-3.12,-1.21,-29.0deg)  best vs GT 0.17m/0.41deg
                        cov(x,y,yaw)=0.240,0.240,0.0008  (z/roll/pitch -> unobservable)
Autoware loc-health:    reported≈truth -> OK ;  drifted +8m -> ERROR
Autoware map freshness: stale regions = [missing_static_structure]
Nav2 occupancy stale:   [map_stale_region, new_static_obstacle]  blocked areas=1
Nav2 relocalization:    AMCL initial pose = (-3.12,-1.21,-29.0deg)
```

- `AutowareAdapter`：initial pose（NDT Monte-Carlo 前段）、localization health、PCD map freshness、Lanelet2 consistency（§17.2 A–D）。
- `Nav2Adapter`：relocalization assist、`OccupancyGrid` staleness、changed-area annotation（§18.2 A–C）。
- initial pose は alignment から **covariance**（z/roll/pitch を観測不能としてマーク）を付与。

### Benchmark suite (v0.8)

```bash
python examples/run_benchmark_suite.py
```

4 タスクを同一プロトコルで評価し、leaderboard を出力します。descriptor/aligner を
追加して再実行すれば、比較可能なエントリが得られます（§20.5）。

```text
### retrieval (ranked by recall@1)
| rank | method       | recall@1 | recall@5 | mrr   |
| 1    | scan-context | 0.833    | 0.958    | 0.875 |
| 2    | bev-grid     | 0.417    | 0.583    | 0.481 |
...（alignment / change / map_validation も同様）

retrieval board with external submission:
  1. my-paper-descriptor  R@1=0.900   ← SubmissionEntry で外部結果をマージ
  2. scan-context         R@1=0.833
```

- **dataset cards**（`bevmatch.benchmarks.CARDS`）：各ベンチマークの内容・条件・ライセンスを記述。
- **再現可能な split manifest**：seed から決定的に生成し、ground-truth の **fingerprint(sha256)** で再現性を検証。
- **leaderboard**：task ごとに primary metric でランク。`SubmissionEntry` で外部 plugin の結果を同一 board に統合。

### Multi-modal expansion (v0.9)

```bash
python examples/run_multimodal.py
```

BEVMatch は BEV/LiDAR 専用ではありません（§1.3, Principle 2）。

```text
Modality-agnostic retrieval:
  LiDAR  (Scan-Context BEV): place_2 [OK]
  Radar  (-> BEV occupancy): place_2 [OK]
  Camera (image embedding) : place_2 [OK]

Object-level change:  class_changed(pole->building), moved(vehicle 2.0m), added(vehicle), removed(vehicle)
NL summary: "An object 18 m to the east changed from pole to building. A vehicle moved 2.0 m ..."
```

- **modality**（camera/radar/LiDAR）と **representation**（BEV / image embedding）を分離。
  `CameraEmbeddingDescriptor`、radar→BEV、`SemanticBEV` を提供。
- **object-level change**（`detect_object_changes`）：added / removed / moved / class-changed。
- **scene graph**（`build_scene_graph`）と **自然言語サマリ**（`bevmatch.nl`、VLM/LLM に差し替え可能）。

### Library 利用

```python
from bevmatch import SamePlaceComparisonPipeline, SceneDatabase
from bevmatch.datasets import make_synthetic_same_place

data = make_synthetic_same_place()
db = SceneDatabase(); db.add_all(data.historical)
bundle = SamePlaceComparisonPipeline(database=db).run(data.query)
print(bundle.summary())
```

## v0.1 MVP Pipeline

```
Query LiDAR scene
  ↓ Retrieve Top-K historical scenes   bevmatch.retrieval  (Scan-Context descriptor)
  ↓ Align best candidate               bevmatch.alignment  (SE2 BEV xcorr + ICP)
  ↓ Compare in BEV                     bevmatch.representations
  ↓ Added/removed occupancy diff       bevmatch.change
  ↓ Export comparison evidence         bevmatch.io  (ComparisonEvidenceBundle → JSON)
```

| Module | 責務 (architecture.md) |
| --- | --- |
| `bevmatch.core` | データモデル・evidence schema・plugin registry・pipeline (§5, §6, §7, §8) |
| `bevmatch.representations` | BEV occupancy 表現 (§5.4) |
| `bevmatch.retrieval` | descriptor / index プラグイン + Top-K retriever (§9, §7.2) |
| `bevmatch.alignment` | SE2/SE3 aligner プラグイン（BEV相互相関 + ICP）+ failure 分類 (§10, §7.2) |
| `bevmatch.change` | occlusion-aware diff + comparable region + persistence (§11) |
| `bevmatch.maps` | map 検証 validator + issue severity + review report (§12) |
| `bevmatch.ros` | ROS2 統合: TF / diagnostics / markers / lifecycle replay / node (§16) |
| `bevmatch.integrations` | Autoware / Nav2 アダプタ（initial pose / health / staleness）(§17, §18) |
| `bevmatch.eval` | retrieval / alignment / change / map メトリクス (§13) |
| `bevmatch.benchmarks` | dataset cards / 再現 split / suite / leaderboard (§0.8, §13) |
| `bevmatch.sensors` | camera / radar アダプタ（modality-agnostic）(§1.3, Principle 2) |
| `bevmatch.scene_graph` / `bevmatch.nl` | object scene graph / 自然言語サマリ (§5.4, §0.9) |
| `bevmatch.viz` | 整列オーバーレイ・残差可視化（matplotlib 任意）(§15) |
| `bevmatch.datasets` | 合成 same-place / route ベンチマーク (§14) |
| `bevmatch.io` | evidence エクスポート (§16.4) |

## Documentation

- [Master Architecture Design Document](docs/architecture.md) — 全体設計、データモデル、plugin / pipeline 設計、評価、ROS2 / Autoware / Nav2 連携、ロードマップ。

詳細な MVP スコープは [§22 Recommended MVP Scope](docs/architecture.md#22-recommended-mvp-scope) を参照。

## License

Apache-2.0 (`bevmatch` core)。詳細は [LICENSE](LICENSE)。
