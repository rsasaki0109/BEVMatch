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

v0.1 MVP — 合成データでエンドツーエンドに動作（retrieval → alignment → change diff → evidence）。依存は `numpy` のみ（可視化は任意で `matplotlib`）。

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
| `bevmatch.retrieval` | Scan-Context descriptor + Top-K retriever (§9) |
| `bevmatch.alignment` | SE2 alignment（BEV相互相関 + ICP refine）(§10) |
| `bevmatch.change` | geometry-level BEV occupancy diff (§11) |
| `bevmatch.datasets` | 合成 same-place ベンチマーク (§14) |
| `bevmatch.io` | evidence エクスポート (§16.4) |

## Documentation

- [Master Architecture Design Document](docs/architecture.md) — 全体設計、データモデル、plugin / pipeline 設計、評価、ROS2 / Autoware / Nav2 連携、ロードマップ。

詳細な MVP スコープは [§22 Recommended MVP Scope](docs/architecture.md#22-recommended-mvp-scope) を参照。

## License

Apache-2.0 (`bevmatch` core)。詳細は [LICENSE](LICENSE)。
