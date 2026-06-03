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

Draft v0.1 — アーキテクチャ設計フェーズ。

## Documentation

- [Master Architecture Design Document](docs/architecture.md) — 全体設計、データモデル、plugin / pipeline 設計、評価、ROS2 / Autoware / Nav2 連携、ロードマップ。

## v0.1 MVP Scope

```
Query LiDAR scene
  ↓ Retrieve Top-K historical scenes
  ↓ Align best candidate
  ↓ Generate BEV overlay
  ↓ Generate simple added/removed occupancy diff
  ↓ Export comparison evidence
```

詳細は [§22 Recommended MVP Scope](docs/architecture.md#22-recommended-mvp-scope) を参照。
