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

v0.3 — alignment framework（SE2/SE3 aligner プラグイン、richer alignment evidence、overlap 推定、failure classification、translation/rotation error 評価、残差可視化）。
v0.2 retrieval framework（descriptor / index プラグイン、Recall@K/MRR 評価）、v0.1 MVP パイプライン（retrieval → alignment → change diff → evidence）も合成データでエンドツーエンド動作。依存は `numpy` のみ（可視化は任意で `matplotlib`、大規模 index は任意で `faiss-cpu`）。

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
| `bevmatch.change` | geometry-level BEV occupancy diff (§11) |
| `bevmatch.eval` | retrieval / alignment メトリクスと評価レシピ (§13) |
| `bevmatch.viz` | 整列オーバーレイ・残差可視化（matplotlib 任意）(§15) |
| `bevmatch.datasets` | 合成 same-place / route ベンチマーク (§14) |
| `bevmatch.io` | evidence エクスポート (§16.4) |

## Documentation

- [Master Architecture Design Document](docs/architecture.md) — 全体設計、データモデル、plugin / pipeline 設計、評価、ROS2 / Autoware / Nav2 連携、ロードマップ。

詳細な MVP スコープは [§22 Recommended MVP Scope](docs/architecture.md#22-recommended-mvp-scope) を参照。

## License

Apache-2.0 (`bevmatch` core)。詳細は [LICENSE](LICENSE)。
