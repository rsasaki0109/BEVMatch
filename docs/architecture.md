# BEVMatch — Master Architecture Design Document

> Draft v0.1 / 2026-06-03

## 0. Executive Summary

BEVMatch は「Place Recognition OSS」ではなく、「Same-Place Comparison OSS」である。

Place Recognition は「ここは以前見た場所か？」に答える。
BEVMatch は、その先まで含めて答える。

同じ場所を見つけ、整列し、比較し、変化を説明し、地図の信頼性に変換する。

つまり BEVMatch の本質は、単一の特徴量・単一の論文手法・単一のセンサではない。
本質は次の一連の判断を、再利用可能なOSSパイプラインとして標準化することにある。

```
Query Scene
  ↓
Place Retrieval
  ↓
Geometric / Semantic Alignment
  ↓
Scene-to-Scene Comparison
  ↓
Change Evidence
  ↓
Map Validation / Map Maintenance Evidence
```

Scan Context、OverlapNet、BEVPlace++ のような手法は、BEVMatch の中では重要な Retrieval / Alignment plugin になり得る。Scan Context はLiDAR点群のglobal descriptorとしてplace retrievalやlong-term localizationに使われ、LiDAR odometryやradarへの統合例も示されている。OverlapNet はLiDAR range imageのペアからoverlapとrelative yawを推定する。BEVPlace++ はLiDAR点群をBEV画像へ投影し、place recognitionとpose estimationを順に行うglobal localization手法である。

しかし BEVMatch はそれらの置き換えではない。
BEVMatch は、それらを **実運用の比較ワークフローへ接続する器** である。

## 1. Why BEVMatch Should Exist

### 1.1 問題設定

Robotics / Autonomous Driving / Mapping の現場では、単に「似た場所を検索する」だけでは不十分である。

実際に必要なのは次の問いである。

- この現在シーンは、過去のどのシーンと比較すべきか？
- 比較対象は本当に同じ場所か？
- どの座標変換で比較すればよいか？
- 差分は実際の変化か、視点差・遮蔽・動的物体・センサ差か？
- その変化は地図更新・運用停止・人手レビューに値するか？

多くの研究・OSSは、このうち 1 つまたは 2 つだけを扱う。

BEVMatch が扱うべき単位は method ではなく **operational evidence chain** である。

### 1.2 BEVMatch の一文定義

> BEVMatch is an open-source framework for retrieving, aligning, comparing, and validating same-place observations across time, sensors, maps, and robotic platforms.

日本語では次の定義がよい。

> BEVMatch は、ロボットが観測した現在の場所を、過去の観測や地図と照合し、整列し、変化を抽出し、地図や運用判断に使える証拠へ変換するOSSである。

### 1.3 なぜ「BEV」という名前を残すのか

BEV は必須入力ではない。
BEV は **比較しやすい共通内部表現の一つ** である。

BEV の価値は、次の点にある。

- 地面基準で幾何と意味を揃えやすい
- LiDAR、Camera、Radar、Occupancy、HD Map を共通平面へ投影しやすい
- 道路・構造物・占有・空き領域・車線・工事領域を比較しやすい
- 可視化しやすく、OSSデモとして伝わりやすい

ただし BEVMatch は BEV-only であってはならない。

将来的には、BEV、range image、voxel、point cloud、image embedding、topological graph、HD map feature、occupancy grid をすべて同じパイプライン内の representation として扱う。

## 2. BEVMatch はなぜ単なる Place Recognition OSS ではないのか

ここが最重要である。

### 2.1 Place Recognition OSS との境界

典型的な Place Recognition OSS は、主に次を扱う。

```
Input: Query sensor data
Output: Similar database entries / loop candidates / global descriptor score
```

BEVMatch は次を扱う。

```
Input:
  Query scene
  Historical observations
  Map artifacts
  Calibration / pose / uncertainty / metadata

Output:
  Retrieved places
  Relative poses
  Alignment confidence
  Scene difference evidence
  Added / removed / modified entities
  Potential map errors
  Stale map regions
  Human-review-ready evidence package
```

つまり BEVMatch の主語は descriptor ではなく **comparison** である。

### 2.2 BEVMatch の中心成果物: Comparison Evidence Bundle

BEVMatch の中核概念は、単なる Top-K result ではない。

中核成果物は **Comparison Evidence Bundle** である。

これは以下をまとめた一つの証拠パッケージである。

| Evidence | 内容 |
| --- | --- |
| Retrieval evidence | どの過去シーンが候補になったか、なぜ候補になったか |
| Alignment evidence | どの相対姿勢で整列したか、どれほど信頼できるか |
| Overlap evidence | 観測がどの範囲で重なっているか |
| Occlusion evidence | 比較不能な領域はどこか |
| Change evidence | 追加・削除・移動・形状変化・意味変化 |
| Map validation evidence | 地図エラー候補、古い地図領域、更新優先度 |
| Provenance | 使用データ、時刻、センサ、plugin、パラメータ、モデル版 |
| Uncertainty | pose covariance、score calibration、false-positive risk |

Place Recognition が candidate を返すのに対し、BEVMatch は decision evidence を返す。

### 2.3 BEVMatch の独自ポジション

BEVMatch の差別化は次である。

| 観点 | 従来のPlace Recognition | BEVMatch |
| --- | --- | --- |
| 目的 | 似た場所を探す | 同じ場所を比較して判断する |
| 出力 | Top-K / loop candidate | retrieval + pose + change + map issue |
| 主な評価 | Recall@K, F1 | retrieval, alignment, change, validation, system latency |
| 単位 | descriptor / model | scene comparison pipeline |
| 時間軸 | 現在 vs database | current vs historical vs map version |
| 実運用 | SLAM loop closure補助 | map maintenance, relocalization, QA, fleet memory |
| センサ | 手法依存 | LiDAR, camera, radar, map, occupancyを拡張可能 |
| OSS価値 | 手法再現 | 実運用パイプライン、可視化、評価、統合 |

結論:
BEVMatch は Place Recognition の上位概念ではなく、Place Recognition を一部品として使う **Place Comparison Infrastructure** である。

## 3. Existing OSS Landscape

### 3.1 研究手法系 OSS

| 領域 | 代表例 | 強み | BEVMatch から見たギャップ |
| --- | --- | --- | --- |
| LiDAR global descriptor | Scan Context | LiDAR point cloud descriptor。LiDAR odometryやradarへの統合例がある。 | retrieval中心。alignment以後のchange/map validationは対象外 |
| LiDAR overlap / yaw | OverlapNet | range imageペアからoverlapとrelative yawを推定。 | loop closure寄り。change evidenceやmap issue化は対象外 |
| BEV LiDAR localization | BEVPlace++ | BEV画像を使い、place recognitionとpose estimationを順に行う。 | global localization手法であり、比較・差分・地図検証基盤ではない |
| Visual localization | hloc | image retrievalとfeature matchingを組み合わせた6-DoF visual localization toolbox。 | camera localizationに強いが、マルチモーダル変化検出基盤ではない |
| Visual BoW | DBoW2 | imageをbag-of-words表現へ変換し、image databaseで高速検索する。 | descriptor/index layerであり、比較判断までは持たない |

### 3.2 SLAM / Localization OSS

| OSS | 強み | BEVMatch との違い |
| --- | --- | --- |
| RTAB-Map | RGB-D、Stereo、LiDARのgraph-based SLAM。incremental appearance-based loop closure detectorを持つ。 | SLAMとmap optimizationが主目的。BEVMatchはSLAM後・地図運用時のsame-place comparisonを主目的にする |
| OpenVSLAM / ORB-SLAM系 | visual SLAM / relocalizationに強い | 地図差分・HD map validation・センサ横断比較は主目的ではない |
| Autoware localization | LiDAR、camera、GNSS、IMU、map data、tf等を入力に、pose/covarianceやdiagnosticsを出す設計。 | 自動運転stack内のlocalization機能。BEVMatchはlocalization候補生成・map freshness検証・比較証拠生成を担う |
| Nav2 | AMCLはstatic map内でのlocalization server。Map Serverはgrid mapのloading/saving/publishingを担う。 | 2D navigation基盤。BEVMatchは地図が古いか、現在観測と過去観測がどう違うかを補助する |

### 3.3 基盤ライブラリ

BEVMatch は、既存の強力な基盤を再発明すべきではない。

| 領域 | 候補 |
| --- | --- |
| Point cloud processing | PCL は2D/3D image and point cloud processingの大規模OSS。 |
| 3D processing / registration / visualization | Open3D は3D dataを扱うOSSで、3D processing、scene reconstruction、surface alignment、visualization等を提供する。 |
| Similarity search | Faiss はdense vectorのefficient similarity search and clustering library。 |
| Graph optimization / uncertainty | GTSAM はfactor graphを用いたrobotics / computer vision向けsensor fusion library。 |

BEVMatch のOSS戦略は、これらの基盤を抱え込むことではなく、**same-place comparison の標準的な接続面を作ること** である。

## 4. Gap Analysis

### 4.1 現在のギャップ

現状のOSS landscapeには、次の空白がある。

**Gap A: Retrieval と Change Detection が分断されている**

Place Retrieval は「似た場所」を返す。
Change Detection は「2つの入力の差分」を返す。
しかし実運用では、その間に alignment、overlap、occlusion、uncertainty、temporal persistence が必要である。

BEVMatch はこの空白を埋める。

**Gap B: Map Validation が観測比較と接続されていない**

Lanelet2 map validator のようなツールは、Lanelet2 .osm map がAutowareで正しく機能するかを検証する。入力はLanelet2 mapとrequirement setで、出力はvalidation resultである。

これは重要だが、BEVMatch が狙うのは別の層である。

- 既存validator: map file itself is structurally valid?
- BEVMatch: map still matches the current world?

この違いが非常に重要である。

**Gap C: Dataset evaluation が単一タスクに閉じている**

KITTI、KITTI-360、Oxford RobotCar、MulRan、nuScenes、SemanticKITTI、Argoverse 2 などは強力なデータ資産である。Oxford RobotCar はOxfordの同一路線を1年以上にわたり100回以上走行し、天候・交通・歩行者・工事・道路工事などの長期変化を含む。MulRan はRadar/LiDAR、複数都市、月単位の時間差、逆方向再訪などを含むplace recognition向けデータセットである。

しかし、OSSとして次の統一評価はまだ弱い。

```
Retrieval success
  + Alignment success
  + Change correctness
  + Map validation usefulness
  + Review burden
```

BEVMatch は、タスク単体のSOTA比較ではなく、**pipeline-level benchmark** を作るべきである。

**Gap D: Robotics integration が弱い**

多くの研究コードは、論文再現には使えるが、ROS2、Autoware、Nav2、bag replay、diagnostics、visualization、dataset adapter、plugin lifecycleまで含めたOSS productにはなっていない。

BEVMatch は research code collection ではなく、**robotics-facing product** として設計する。

## 5. Core Concepts

### 5.1 Place

Place は、世界内の意味的・幾何的に再訪可能な領域である。

Place は単一時刻の観測ではない。
Place は複数の Scene を持つ。

```
Place
  ├── Scene at time t1
  ├── Scene at time t2
  ├── Scene at time t3
  └── Map representation version m1/m2/...
```

Place は以下を持つ。

- rough global pose
- local coordinate frame
- spatial extent
- semantic tags
- map association
- observation history
- change history

### 5.2 Scene

Scene は、ある時刻・あるロボット・あるセンサ構成から見た観測単位である。

Scene は以下を含む。

- timestamp
- robot pose estimate
- sensor observations
- calibration
- ego-motion compensation状態
- weather / route / mission metadata
- raw and derived representations

Scene は「比較の最小単位」である。

### 5.3 Observation

Observation はセンサごとの生データまたは準生データである。

例:

- LiDAR scan / aggregated submap
- camera image / multi-camera rig
- radar scan
- occupancy grid
- HD map excerpt
- semantic segmentation
- object tracks
- road surface estimate

Observation は Scene に属するが、同じ Scene 内に複数 modality が存在してよい。

### 5.4 Representation

Representation は、retrieval / alignment / comparison のための内部表現である。

例:

- BEV occupancy
- BEV intensity
- semantic BEV
- range image
- voxel grid
- point cloud submap
- image embedding
- map feature graph
- lane topology graph
- object-level scene graph

重要なのは、Representation は **入力形式ではない** ということ。
BEVMatch の入力は Scene であり、BEV は Scene から派生する比較表現である。

### 5.5 Candidate

Candidate は、Query Scene と比較すべき過去 Scene / Place / Map region である。

Candidate は単なる距離スコアではなく、以下を持つ。

- retrieval score
- descriptor type
- expected pose hypothesis
- expected overlap
- temporal distance
- modality compatibility
- known uncertainty
- reason for selection

### 5.6 Alignment Hypothesis

Alignment Hypothesis は、Query Scene と Candidate Scene を比較可能にする座標変換である。

必要な情報は以下。

- relative pose
- covariance / uncertainty
- inlier evidence
- overlap region
- residual distribution
- degeneracy / observability
- failure reason

BEVMatch では、alignment が不確かな場合に change detection を強く主張してはいけない。

### 5.7 Change Hypothesis

Change Hypothesis は、整列後の比較から導かれる変化候補である。

例:

- added object
- removed object
- moved object
- road boundary changed
- lane marking changed
- construction area
- vegetation growth
- temporary occlusion
- sensor artifact
- map stale region

重要なのは、すべての差分を「変化」と呼ばないこと。
BEVMatch は observed difference と actionable change を区別する。

### 5.8 Map Validation Issue

Map Validation Issue は、Change Hypothesis が map artifact と矛盾した結果として生成される運用上のissueである。

例:

- lanelet geometry mismatch
- traffic sign missing / moved
- road surface changed
- construction zone not in map
- static obstacle appears in drivable area
- point cloud map stale
- localization map feature disappeared
- HD map element no longer observable

Map Validation Issue は、地図更新担当者や運用システムが扱える粒度にする。

## 6. Architecture Overview

### 6.1 全体アーキテクチャ

```
Data Sources
  ↓
Scene Normalization
  ↓
Representation Layer
  ↓
Retrieval Layer
  ↓
Candidate Verification
  ↓
Alignment Layer
  ↓
Comparison Layer
  ↓
Change Reasoning Layer
  ↓
Map Validation Layer
  ↓
Evidence Export / Visualization / ROS2 Integration
```

### 6.2 Layer Responsibilities

| Layer | 責務 |
| --- | --- |
| Data Sources | dataset、ROS bag、live ROS topic、map file、databaseから入力を取り込む |
| Scene Normalization | timestamp、frame、calibration、motion compensation、ground handling、ego maskを整理 |
| Representation Layer | BEV、range image、voxel、image embedding、map graphなどを生成 |
| Retrieval Layer | descriptor生成、index検索、Top-K candidate取得 |
| Candidate Verification | semantic/geometric sanity check、reranking、false candidate除去 |
| Alignment Layer | relative pose、overlap、inlier、covarianceを推定 |
| Comparison Layer | aligned scenesを比較し、差分候補を生成 |
| Change Reasoning Layer | dynamic object、occlusion、viewpoint差、seasonal差を考慮してchange hypothesis化 |
| Map Validation Layer | changeをHD map / point cloud map / occupancy mapのissueに変換 |
| Evidence Export | reports、visualization、ROS2 messages、JSON/Parquet/MCAPなどへ出力 |

### 6.3 Repository Product Boundary

BEVMatch は、最初から巨大なmonorepoにすべきではない。
ただし、最初から責務分離を明確にする。

推奨する論理構成:

| Area | 内容 |
| --- | --- |
| BEVMatch Core | データモデル、pipeline orchestration、plugin registry、artifact schema |
| BEVMatch Algorithms | baseline retrieval / alignment / change detection plugins |
| BEVMatch Datasets | dataset adapters、split definitions、benchmark recipes |
| BEVMatch Visualization | RViz/Foxglove/web viewer向けexport |
| BEVMatch ROS | ROS2 integration、bag replay、topic bridge |
| BEVMatch Autoware | Autoware-specific adapters、map validation workflows |
| BEVMatch Nav2 | 2D navigation / occupancy map workflows |
| BEVMatch Benchmarks | evaluation definitions、leaderboard tooling |
| BEVMatch Labs | experimental plugins、research integration |

### 6.4 Design Principles

**Principle 1: Evidence-first**

すべてのstageは、単なるscoreではなく **根拠付きartifact** を出す。

悪い例:

```
candidate_id = 42, score = 0.87
```

良い考え方:

```
candidate 42 was selected by LiDAR-BEV descriptor,
verified by overlap estimate,
aligned with SE2 pose,
valid over 62% of visible area,
low confidence near occluded construction zone.
```

**Principle 2: Modality-agnostic, Representation-plural**

BEVMatch は LiDAR-first で始めてよい。
しかし architecture は LiDAR-only にしてはいけない。

入力 modality と内部 representation を分離する。

```
LiDAR      → point cloud / range image / BEV
Camera     → image embedding / depth / semantic BEV
Radar      → radar polar image / BEV occupancy
HD Map     → lane graph / vector map / rasterized BEV
Occupancy  → 2D/3D occupancy grid
```

**Principle 3: Alignment-gated change detection**

Change detection は alignment confidence によってgateされるべきである。

Poseが不確かなまま差分を出すと、地図更新候補がノイズになる。

BEVMatch は次を明示する。

- comparable region
- non-comparable region
- aligned region
- uncertain region
- dynamic-object region
- map-relevant region

**Principle 4: Map-aware but not map-dependent**

BEVMatch は map validation に強いOSSを目指す。
ただし、map がなくても動くべきである。

動作モード:

- Observation-to-observation comparison
- Observation-to-map comparison
- Map-to-map version comparison
- Fleet observation aggregation
- Live relocalization assistance

**Principle 5: Offline-first, live-ready**

最初のOSS成長は offline dataset / ROS bag replay から始めるべきである。
実運用・live robotics はその後でよい。

理由:

- 再現性が高い
- GitHub demoが作りやすい
- CI評価しやすい
- 研究者と実務者の両方に刺さる
- Autoware/Nav2連携へ自然に進める

## 7. Plugin System Design

### 7.1 Plugin の目的

BEVMatch の plugin system は、アルゴリズム差し替えのためだけではない。

目的は次である。

- 新しい研究手法を pipeline 内へ安全に取り込む
- modalityやrepresentationを追加できる
- retrieval / alignment / change detection を独立評価できる
- dataset / map / ROS integration を増やせる
- 研究コードをOSS productへ昇格しやすくする

### 7.2 Plugin Categories

| Plugin category | 役割 |
| --- | --- |
| Data Source Plugin | dataset、bag、live topic、map fileをSceneへ変換 |
| Scene Normalizer Plugin | de-skew、frame transform、ground handling、ego filtering |
| Representation Plugin | BEV、range image、voxel、image embedding、map graph生成 |
| Descriptor Plugin | retrieval用descriptor生成 |
| Index Backend Plugin | dense vector / sparse / spatial / hybrid index |
| Retriever Plugin | Top-K candidate取得 |
| Reranker Plugin | candidate再順位付け |
| Alignment Plugin | SE2/SE3/pose graph/feature-based alignment |
| Overlap Plugin | 比較可能領域・overlap推定 |
| Change Detector Plugin | geometric/semantic/object/map-level change抽出 |
| Dynamic Filter Plugin | moving object、temporary obstacle、occlusion除去 |
| Map Validator Plugin | change evidenceをmap issueへ変換 |
| Evaluation Plugin | task-specific metrics |
| Visualization Exporter Plugin | RViz/Foxglove/web/report向け出力 |
| Confidence Plugin | score calibration、uncertainty estimation、failure classification |

### 7.3 Plugin Manifest

各pluginは、実装の詳細ではなく capability を宣言する。

Manifest に含めるべき情報:

| 項目 | 例 |
| --- | --- |
| Input modality | LiDAR, camera, radar, occupancy, Lanelet2, PCD |
| Required representation | point cloud, BEV occupancy, range image |
| Output artifact | descriptor, candidate list, alignment, change hypothesis |
| Pose assumption | no pose, rough GNSS, odom prior, map prior |
| Invariance | rotation, viewpoint, season, illumination, sensor height |
| Runtime profile | offline, realtime, GPU required, CPU-only |
| Scale profile | single scene, route-scale, city-scale |
| Uncertainty support | none, score only, covariance, calibrated probability |
| License / model weights | OSS compatible, research-only, commercial-safe |
| Dataset compatibility | KITTI, MulRan, Oxford RobotCar, custom |
| Failure modes | repetitive structures, vegetation, tunnels, sparse scenes |

この manifest により、BEVMatch は「pluginが動く」だけでなく、「どの条件で信用できるか」を扱える。

### 7.4 Plugin Design Rule

BEVMatch の plugin は、pipeline の共通データモデルを破ってはいけない。

特に以下を禁止する。

- plugin固有の隠れた座標系
- scoreの意味が不明な出力
- uncertaintyなしの強いchange assertion
- dataset固有IDへの依存
- visualization不能なblack-box result
- provenanceを残さない処理

## 8. Data Model Design

### 8.1 Data Model の中心思想

BEVMatch の data model は、ML dataset schema でも、ROS message schema でも、SLAM map schema でもない。

中心は **same-place comparison evidence schema** である。

### 8.2 Core Entities

```
Dataset
  └── Mission / Log / Traverse
        └── Scene
              ├── Observation
              ├── Representation
              ├── Descriptor
              └── Metadata

Place
  ├── Scene references
  ├── Map region references
  └── Change history

Comparison Evidence Bundle
  ├── Query Scene
  ├── Retrieved Candidates
  ├── Alignment Hypotheses
  ├── Overlap / Occlusion Evidence
  ├── Change Hypotheses
  ├── Map Validation Issues
  └── Provenance / Uncertainty
```

### 8.3 Required Coordinate Concepts

BEVMatch は ROS2 / Autoware / Nav2 との親和性を持つため、frame設計を曖昧にしてはいけない。

必要な座標概念:

| Frame | 役割 |
| --- | --- |
| sensor frame | 個別センサの観測座標 |
| base frame | ロボット/車両基準 |
| odom frame | 連続的な局所推定 |
| map frame | 地図基準 |
| local scene frame | 比較のための局所座標 |
| place frame | place単位で安定した比較座標 |
| earth/global frame | GNSS/地理座標系との接続 |

Nav2の概念では、少なくとも `map -> odom -> base_link -> sensor frames` のTF treeが必要で、global positioning側が `map -> odom` を提供し、odometry側が `odom -> base_link` を提供する、という整理がある。

BEVMatch はこのTF文化に合わせるべきである。

### 8.4 Scene Metadata

Scene は sensor data だけでは不十分である。
比較品質を左右する metadata を保持する。

必須に近い metadata:

- timestamp
- vehicle/robot ID
- mission/log ID
- approximate pose
- pose uncertainty
- calibration version
- sensor configuration
- weather / lighting if available
- map version
- route ID
- speed / motion state
- localization status
- semantic labels if available

### 8.5 Difference Between Place ID and Scene ID

これは設計上重要である。

- Scene ID: ある時刻の観測
- Place ID: 複数Sceneが属する空間的場所
- Map Region ID: map artifact内の領域
- Change ID: 時間をまたいだ変化仮説
- Issue ID: map validation / human review対象

Place Recognition系OSSは Scene retrieval で止まりやすい。
BEVMatch は Place、Scene、Change、Issue を分離する。

## 9. Retrieval Pipeline Design

### 9.1 Retrieval の役割

Retrieval は BEVMatch の入口である。
ただし最終目的ではない。

Retrieval の目的は、Query Scene に対して **比較すべき候補** を効率よく絞ることである。

### 9.2 Retrieval Pipeline

```
Query Scene
  ↓
Scene Normalization
  ↓
Representation Generation
  ↓
Descriptor Extraction
  ↓
Index Search
  ↓
Top-K Candidate Generation
  ↓
Spatial / Temporal / Semantic Filtering
  ↓
Candidate Reranking
  ↓
Retrieval Evidence
```

### 9.3 Retrieval Modes

| Mode | 用途 |
| --- | --- |
| Loop closure mode | SLAM中の再訪検出 |
| Global localization mode | 初期位置推定・kidnapped robot回復 |
| Historical comparison mode | 過去観測との比較 |
| Map validation mode | 地図領域との対応候補探索 |
| Fleet memory mode | 複数ロボットの観測履歴検索 |
| Cross-modal mode | LiDAR query vs camera/map/radar history |

### 9.4 Retrieval Output

Retrieval は Top-K のリストだけでは不十分である。

出力すべき情報:

- candidate scene / place / map region
- retrieval score
- score calibration
- descriptor type
- expected relative pose if available
- retrieval uncertainty
- temporal gap
- condition gap
- modality compatibility
- expected overlap
- reason for candidate selection

### 9.5 Baseline Strategy

v0.1〜v0.3 では、派手なdeep modelより **強いbaseline** を重視する。

初期baseline候補:

- Scan Context系 LiDAR descriptor
- BEV occupancy descriptor
- BEV intensity descriptor
- simple voxel/global histogram
- image embedding retrieval
- map-element spatial retrieval
- FAISS-based dense vector index

FAISSはdense vectorの効率的なsimilarity searchを扱えるため、descriptor index backendの初期候補として自然である。

## 10. Alignment Pipeline Design

### 10.1 Alignment の役割

Retrieval は「同じ場所かもしれない」を返す。
Alignment は「この座標変換なら比較できる」を返す。

Change Detection の品質は Alignment に強く依存する。

### 10.2 Alignment Pipeline

```
Query Scene + Candidate Scene
  ↓
Initial Pose Hypothesis
  ↓
Coarse Alignment
  ↓
Fine Alignment
  ↓
Overlap Estimation
  ↓
Residual / Inlier Analysis
  ↓
Uncertainty Estimation
  ↓
Alignment Evidence
```

### 10.3 Alignment Levels

| Level | 変換 | 用途 |
| --- | --- | --- |
| Yaw-only | relative yaw | Scan Context / OverlapNet的用途 |
| SE2 | x, y, yaw | ground robot / road scene / BEV comparison |
| SE3 | x, y, z, roll, pitch, yaw | 3D LiDAR / visual localization / uneven terrain |
| Map-element alignment | lane, curb, pole, sign, wall | HD map validation |
| Object-level alignment | static object graph | indoor / warehouse / semantic map |
| Deformable/local alignment | local warping | long-term construction / map deformation analysis |

### 10.4 Alignment Evidence

Alignment の出力は pose だけではいけない。

必要な evidence:

- relative pose
- covariance
- inlier ratio
- overlap mask
- residual map
- degeneracy flags
- local minima risk
- comparable area ratio
- excluded dynamic regions
- failure reason

Autoware の NDT scan matcher は、pointcloud mapとraw sensor pointcloud等を入力に、estimated pose、pose with covariance、diagnostics、aligned pointcloud、transform probability、execution timeなどを出す。BEVMatch のalignment layerも、単なるposeではなくdiagnosticsとdebuggable evidenceを出す思想に合わせるべきである。

### 10.5 Alignment Failure Is a First-Class Result

BEVMatch では、alignment failure も重要な成果物である。

失敗理由の例:

- overlap insufficient
- repetitive structure
- dynamic scene dominates
- map too stale
- sensor mismatch
- initial pose too far
- viewpoint gap too large
- vegetation / seasonal change
- calibration inconsistent
- insufficient geometric constraints

失敗を明示できるOSSは、実運用で信用される。

## 11. Change Detection Pipeline Design

### 11.1 Change Detection の基本思想

BEVMatch の change detection は、単純な差分ではない。

```
Raw Difference ≠ Real Change
Real Change ≠ Map-Relevant Change
Map-Relevant Change ≠ Immediate Update
```

BEVMatch はこの4段階を分ける。

1. Observed difference — センサ上の差分
2. Explained difference — 遮蔽・動的物体・視点差・センサ差で説明できる差分
3. Persistent change hypothesis — 時間的に継続する可能性のある変化
4. Map validation issue — 地図や運用に影響する可能性のある変化

### 11.2 Change Detection Pipeline

```
Aligned Scene Pair
  ↓
Comparable Region Estimation
  ↓
Dynamic / Occlusion Filtering
  ↓
Geometric Difference
  ↓
Semantic Difference
  ↓
Object-level Difference
  ↓
Temporal Persistence Reasoning
  ↓
Change Hypothesis Generation
  ↓
Confidence / Severity Estimation
```

### 11.3 Change Categories

| Category | 説明 | 例 |
| --- | --- | --- |
| Added object | 過去になく現在ある | 工事バリケード、新しいポール、駐車車両 |
| Removed object | 過去にあり現在ない | 標識撤去、建物解体、ガードレール撤去 |
| Moved object | 位置が変わった | 標識移設、信号機移設 |
| Road geometry change | 路面・縁石・境界の変化 | 車線拡張、道路工事 |
| Lane marking change | 車線標示の変化 | 白線引き直し、横断歩道追加 |
| Construction area | 広範囲・一時的・構造的変化 | 工事区間 |
| Vegetation change | 草木の成長・伐採 | 長期運用ロボットで重要 |
| Temporary obstacle | 地図更新不要な一時物 | 歩行者、車両、荷物 |
| Sensor artifact | 雨・雪・反射・欠損 | false change除去対象 |
| Unknown change | 分類不能だが差分はある | human review対象 |

### 11.4 Change Evidence Output

Change Hypothesis は以下を持つ。

- change category
- geometry
- semantic label
- confidence
- spatial extent
- affected map elements
- supporting observations
- conflicting observations
- temporal persistence
- alignment dependency
- recommended action

### 11.5 Object-level vs Geometry-level

BEVMatch は最初から高度なobject detectionに依存してはいけない。

初期設計は二層にする。

1. Geometry-level change
   - occupancy difference
   - height difference
   - surface difference
   - curb / wall / pole-like feature difference
2. Semantic/object-level change
   - object class added/removed
   - sign/lane/traffic light relation
   - drivable area affected
   - construction semantics

v0.1〜v0.3 は geometry-first。
v0.4以降に semantic/object-level を拡張する。

## 12. Map Validation Pipeline Design

### 12.1 Map Validation の目的

BEVMatch の Map Validation は、地図ファイルの構文検証ではない。

目的は次である。

> Current observation と map artifact の不一致を、運用判断可能な issue に変換する。

### 12.2 Supported Map Types

| Map type | Validation target |
| --- | --- |
| Point cloud map | localization features, static structures, map freshness |
| Lanelet2 / vector map | lane geometry, stop line, crosswalk, traffic sign relation |
| Occupancy map | free/occupied mismatch, stale obstacles, blocked routes |
| Semantic map | object-level semantic consistency |
| HD map tiles | tile-level stale region, update priority |
| Learned map / feature map | descriptor drift, localization feature disappearance |

Autoware はlocalization architectureでpoint cloud map、lanelet2 map、feature mapなどをmap data例として挙げ、センサ構成やuse caseに応じてmap formatを選ぶ設計になっている。
BEVMatch はこの多様なmap dataを「現在観測と比較する対象」として扱う。

### 12.3 Map Validation Pipeline

```
Current Scene
  ↓
Map Region Retrieval
  ↓
Observation-to-Map Alignment
  ↓
Map Element Projection
  ↓
Evidence Comparison
  ↓
Issue Hypothesis
  ↓
Severity / Confidence Assignment
  ↓
Human Review / Map Update Candidate
```

### 12.4 Map Validation Issues

| Issue | 説明 |
| --- | --- |
| Stale point cloud region | localization mapが現在環境と合わない |
| Missing static structure | 地図上にあるべき構造が観測されない |
| New static obstacle | 地図にない静的障害物が現れた |
| Lane geometry mismatch | lane boundary / centerlineが観測と合わない |
| Road closure / construction | drivable mapと現地が矛盾 |
| Traffic sign mismatch | 標識・信号・停止線の存在や位置が矛盾 |
| Localization risk region | map featureが薄い、または変化でlocalization品質が落ちる |
| Review-required region | 自動判断には不十分だが、人手確認が必要 |

### 12.5 BEVMatch と既存 Map Validator の関係

既存の Lanelet2 validator は必要である。
BEVMatch はそれを置き換えない。

役割分担:

| Tool | Question |
| --- | --- |
| Lanelet2 validator | このLanelet2 mapはAutowareの要求に対して構造的に妥当か？ |
| BEVMatch | このmapは現在の世界とまだ一致しているか？ |
| Map editor | issueを受けて地図をどう修正するか？ |
| Autoware | 修正済みmapで安全に走れるか？ |

BEVMatch は **map validation evidence generator** として位置づける。

## 13. Evaluation Framework Design

### 13.1 評価の基本方針

BEVMatch は単一タスクのSOTA争いを主目的にしない。
pipeline-level の実用評価を行う。

評価単位:

```
Retrieval quality
+ Alignment quality
+ Change quality
+ Map validation usefulness
+ Runtime / reproducibility
```

### 13.2 Retrieval Metrics

| Metric | 意味 |
| --- | --- |
| Recall@K | Top-K内に正解placeがあるか |
| MRR | 正解候補の順位 |
| Precision@K | 候補のfalse positive率 |
| Pose-constrained Recall | 一定距離・角度以内の候補を正解とする |
| Condition-stratified Recall | night/rain/snow/season/reverse route別 |
| Cross-modal Recall | LiDAR→camera、radar→LiDARなど |
| Long-term Recall | 時間差ごとの性能 |

### 13.3 Alignment Metrics

| Metric | 意味 |
| --- | --- |
| Translation error | 相対位置誤差 |
| Rotation error | yaw / full rotation誤差 |
| Success rate | 許容誤差内のalignment成功率 |
| Overlap accuracy | overlap推定の正確さ |
| Inlier calibration | inlier scoreと成功確率の一致 |
| Covariance calibration | covarianceが実誤差を説明できるか |
| Failure classification accuracy | 失敗理由が正しいか |

### 13.4 Change Detection Metrics

| Metric | 意味 |
| --- | --- |
| Pixel/BEV IoU | raster change mapの正確さ |
| Point-level IoU | point cloud change labelの正確さ |
| Instance precision/recall | added/removed object単位 |
| False changes per km | 運用上のノイズ量 |
| Persistent change precision | 一時物を除いた変化精度 |
| Occlusion-aware precision | 比較不能領域で誤検出しないか |
| Severity calibration | 重大度scoreが人手判断と合うか |

### 13.5 Map Validation Metrics

| Metric | 意味 |
| --- | --- |
| Issue precision | 報告issueのうち本当にmap問題か |
| Issue recall | 実際のmap問題を拾えたか |
| Human review burden | 1kmあたりレビュー件数 |
| Time-to-detect | 変化発生から検出まで |
| Map update usefulness | 地図更新作業に使えた割合 |
| Localization risk prediction | localization failure領域を予測できたか |

### 13.6 System Metrics

| Metric | 意味 |
| --- | --- |
| Query latency | 1 queryあたり処理時間 |
| Index build time | map/history index構築時間 |
| Index update time | 新規scene追加時間 |
| Memory footprint | route/city scaleでのメモリ |
| Throughput | bag replay / batch処理性能 |
| Determinism | 同じ入力で同じ結果になるか |
| Reproducibility | dataset split / config / artifactの再現性 |

## 14. Dataset Strategy

### 14.1 Dataset Selection Principles

BEVMatch の dataset strategy は、単に有名datasetをサポートすることではない。

選定基準:

- repeated traversal がある
- pose / trajectory ground truth がある
- long-term changes がある
- LiDAR / camera / radar / map の少なくとも一部がある
- commercial-friendlyか、少なくともOSS demoに使いやすい
- small subsetを作れる
- ROS2 replayしやすい
- evaluation splitを定義しやすい

### 14.2 初期Dataset候補

| Dataset | BEVMatchでの用途 |
| --- | --- |
| KITTI / KITTI Odometry | LiDAR retrieval / alignment baseline |
| KITTI-360 | 3D/2D annotation、semantic mapping、localization。KITTI-360は73.7km、32万超の画像、10万 laser scans、3D/2D annotationを持つ。 |
| SemanticKITTI | LiDAR semantic / dynamic object handling。SemanticKITTIはKITTI Odometry全sequenceへsemantic annotationを提供し、10Hz point cloud sequenceやmoving/non-moving traffic participant annotationを含む。 |
| Oxford RobotCar | long-term appearance / construction / roadworks / weather comparison。 |
| MulRan | LiDAR + Radar place recognition、複数都市、月単位ギャップ、逆方向再訪。 |
| Boreas | multi-season、adverse weather、LiDAR/Radar/Camera、metric localization。Boreasは1年間の繰り返し走行、350km超、128-channel LiDAR、360° radar、camera、cm-level poseを持つ。 |
| nuScenes | multi-camera + radar + LiDAR + map文脈。nuScenesは6 cameras、5 radars、1 lidarを持つ1000 scenesの自動運転datasetである。 |
| Argoverse 2 | HD map付き自動運転dataset。Argoverse 2は6都市のopen-source autonomous driving dataとHD mapsを含む。 |

### 14.3 BEVMatch Mini Benchmark

OSS初期成長には、巨大datasetより **小さく動くbenchmark** が重要である。

v0.1で作るべきもの:

- 5〜10分で動く小規模sample
- query → retrieval → alignment → visual overlay まで
- 可能なら change-like example を含む
- dataset downloadを強制しない
- CIで少なくともschema validationが動く
- GitHub READMEにGIFを貼れる

名前例:

- BEVMatch Mini Route Benchmark
- BEVMatch Same-Place Demo Pack
- BEVMatch Long-Term Toy Benchmark

### 14.4 Annotation Strategy

最初から大規模ラベルを作らない。

段階的に進める。

| Stage | Annotation |
| --- | --- |
| v0.1 | pose-based positive/negative pairs |
| v0.2 | alignment ground truth pairs |
| v0.3 | synthetic added/removed object labels |
| v0.4 | manual small change labels |
| v0.5 | map issue labels |
| v1.0 | community benchmark submission format |

## 15. Visualization Strategy

### 15.1 Visualization の役割

BEVMatch は可視化が非常に重要である。

なぜなら、BEVMatchの価値は「似ているscore」ではなく、**比較して何が起きたかを説明すること** だからである。

### 15.2 Primary Visualization: Same-Place Comparison Viewer

基本UIは4画面構成がよい。

```
[Current Scene]   [Historical Scene]

[Aligned Overlay] [Change Evidence]
```

表示すべきもの:

- query scene
- retrieved historical scene
- aligned overlay
- overlap region
- occluded / non-comparable region
- added objects
- removed objects
- road changes
- construction area
- map issue severity
- confidence
- time gap
- candidate ranking
- alignment residuals

### 15.3 ROS Visualization

ROS2ユーザ向けには以下が必要。

- RViz overlay
- MarkerArray style visualization
- aligned point cloud / BEV layers
- change region markers
- map issue markers
- diagnostics
- TF sanity visualization

### 15.4 Web / Foxglove Strategy

GitHub growthには、Web viewerやFoxglove向けexportが強い。

Nav2やAutowareのユーザはbagやMCAPを扱うことが多い。
BEVMatch は、offline review workflowとして以下を提供すべきである。

- scene pair review
- candidate list browser
- map tile stale heatmap
- change timeline
- issue export
- reviewer decision log

## 16. ROS2 Integration Strategy

### 16.1 ROS2 Integration の方針

BEVMatch Core は ROS2 に依存しない。
ただし ROS2 integration は first-class citizen にする。

理由:

- robotics OSSとしての採用に必須
- Autoware / Nav2 integrationの土台
- bag replayによる再現性
- RViz / Foxglove可視化
- TF / diagnostics / lifecycleとの親和性

### 16.2 ROS2 Node Philosophy

ROS2側は一枚岩の巨大nodeにしない。

推奨する構成思想:

- lifecycle-managed components
- composable pipeline
- offline bag replay profile
- live query profile
- map validation batch profile
- diagnostics-first
- TF-aware
- parameters are reproducible artifacts

ROS2 lifecycle nodeは、known interfaceとknown lifecycle state machineを提供し、管理ツールがcompliant nodeを扱えるようにするという思想を持つ。
BEVMatch ROS integrationも、この思想に合わせるべきである。

### 16.3 ROS2 Use Cases

| Use case | 入力 | 出力 |
| --- | --- | --- |
| Bag benchmark | ROS bag / MCAP | evidence report, metrics |
| Live place retrieval | sensor topics | Top-K places, pose candidates |
| Live map validation | sensor topics + map | stale regions, issue markers |
| Offline map QA | map + logs | validation report |
| Nav2 relocalization assist | 2D lidar / occupancy | coarse pose candidates |
| Autoware map freshness | LiDAR + PCD/Lanelet2 | map issue evidence |

### 16.4 Message Strategy

最初から複雑なcustom messagesに寄せすぎない。
初期は次を分ける。

- robotics runtime向け軽量message
- offline evidence向けrich artifact
- visualization向けmarker/export
- benchmark向けstructured report

runtime message は小さく、offline artifact は豊かにする。

## 17. Autoware Integration Strategy

### 17.1 Autoware に対する立ち位置

BEVMatch は Autoware の localization を置き換えない。

BEVMatch は Autoware に対して以下を提供する。

- Global localization candidate provider
- Initial pose hypothesis provider
- Localization health / map freshness monitor
- Point cloud map stale region detector
- Lanelet2 observation consistency checker
- Offline map QA pipeline
- Human-review evidence generator

Autoware localization architectureは、sensor message、map data、tf/static_tfを入力とし、pose with covariance、twist、accel、diagnostics、tfなどを出力する設計である。
BEVMatch はこの入出力思想と衝突せず、周辺支援コンポーネントとして入るべきである。

### 17.2 Autoware Integration Patterns

**Pattern A: Initial Pose Assistance**

```
Current LiDAR / camera scene
  ↓
BEVMatch retrieval
  ↓
Top-K map regions / historical scenes
  ↓
Alignment
  ↓
Initial pose candidates
  ↓
Autoware localization initialization
```

NDT scan matcherはMonte Carlo methodによるinitial position estimation serviceを持つ。
BEVMatch は、その前段でより良い候補領域を提供する役割を担える。

**Pattern B: Localization Health Monitoring**

```
Autoware localization pose
  +
Current observation
  +
Map / historical scene
  ↓
BEVMatch alignment residual / retrieval agreement
  ↓
Localization confidence evidence
```

NDTのscoreやcovarianceだけでなく、過去観測とのsame-place consistencyを見る。

**Pattern C: Point Cloud Map Freshness**

```
Current LiDAR submap
  ↓
Retrieve corresponding PCD map region
  ↓
Align
  ↓
Compare geometry
  ↓
Stale point cloud region report
```

これは localization map maintenance に直結する。

**Pattern D: Lanelet2 Observation Validation**

```
Current observation
  +
Lanelet2 map region
  ↓
Project map elements into observation/BEV
  ↓
Compare visible evidence
  ↓
Potential lane/sign/crosswalk issue
```

既存のLanelet2 map validatorがmap file validationを担い、BEVMatchがobservation-to-map consistencyを担う、という分担がよい。

## 18. Nav2 Integration Strategy

### 18.1 Nav2 に対する立ち位置

Nav2 では、AMCLがstatic map内でrobotをlocalizeするserverを実装し、Map Serverはgrid mapのloading/saving/publishingなどを担う。

BEVMatch は Nav2 に対して以下を提供する。

- kidnapped robot recovery の粗い候補生成
- long-term map staleness detection
- occupancy map difference detection
- changed area warning
- costmap review evidence
- warehouse / indoor route memory

### 18.2 Nav2 Use Cases

**Use Case A: Relocalization Assistance**

2D lidar / depth / RGB-D から過去placeをretrievalし、AMCL initial pose候補を出す。

**Use Case B: Static Map Staleness**

現在のlaser scanやlocal costmapと、保存済みoccupancy mapを比較し、古い領域を検出する。

**Use Case C: Changed Area Annotation**

通れなくなった領域や恒久障害物を検出し、operator reviewに回す。

**Use Case D: Topological Place Memory**

Nav2の単一mapだけでなく、建物内のplace memoryとして利用する。

## 19. Future Physical AI Extensions

### 19.1 Physical AI時代にBEVMatchが残る理由

Physical AI / embodied AI の時代には、ロボットは大量のセンサ履歴、世界モデル、行動履歴、地図、タスク文脈を持つ。

そこで必要になるのは、単なる地図でも、単なる特徴量でもない。

必要なのは:

> grounded spatial memory that can retrieve and compare the same place across time.

BEVMatch はまさにそこに位置する。

OSRAはPhysical AI SIGを設け、ROS等のOSRA projects向けにinterfaces、data pipelines、reference platforms、embodied AIのopen standardsを進める方向を示している。
また、NVIDIA CosmosはPhysical AI向けのworld foundation models、data processing、training、evaluation frameworksを掲げている。

この潮流の中で、BEVMatch は **ロボットの空間記憶・地図記憶・変化記憶を扱うOSS** として価値を持つ。

### 19.2 Physical AI Extensions

| Extension | 内容 |
| --- | --- |
| Scene Memory Retrieval | VLM/robot policyが過去の同じ場所を検索 |
| Change-aware World Model | world model predictionと実観測を比較 |
| Natural Language Map QA | 「先週から工事が始まった場所は？」に答える |
| Fleet Spatial Memory | 複数ロボットが共有するplace/change memory |
| Task-conditioned Retrieval | 「配送ロボットに影響する変化だけ」検索 |
| Embodied Agent Grounding | LLM/VLMの空間推論を実観測履歴に接地 |
| Simulation-to-Real Validation | simulated map/worldとreal observationの差分 |
| Continual Map Learning | 地図更新の候補と履歴を学習データ化 |

### 19.3 BEVMatch as Retrieval-Augmented Spatial Memory

将来の構図:

```
Robot Foundation Model / VLM / Policy
  ↓ asks
BEVMatch Spatial Memory
  ↓ returns
Same-place evidence, aligned history, changes, map issues
  ↓ grounds
Planning / reasoning / human explanation
```

BEVMatch はAI model本体ではない。
しかし、AI model が物理世界で信頼できる判断をするための **grounded memory infrastructure** になれる。

## 20. GitHub Growth Strategy

### 20.1 Positioning

README の最初に置くべきメッセージ:

> BEVMatch is not another place recognition method.
> It is an OSS pipeline for finding the same place, aligning it, comparing it, and turning differences into map validation evidence.

日本語なら:

> BEVMatch は、場所を探すだけで終わらない。
> 同じ場所を見つけ、整列し、変化を説明し、地図の信頼性を検証するOSSです。

### 20.2 Initial Target Users

| User | 欲しい価値 |
| --- | --- |
| Localization engineer | global localization / initial pose / failure analysis |
| Mapping engineer | map freshness / stale region / update evidence |
| SLAM researcher | retrieval + alignment pipeline benchmark |
| Autonomous driving engineer | HD map validation / construction detection |
| Nav2 user | long-term map staleness / relocalization assist |
| Robotics researcher | multi-session comparison framework |
| Physical AI researcher | spatial memory / scene comparison artifacts |

### 20.3 Must-Have GitHub Demos

v0.1〜v0.3で必要なデモ:

- LiDAR query → Top-K same places
- Query vs retrieved scene aligned overlay
- Before/after BEV difference
- Map stale region heatmap
- ROS bag replay to evidence report
- RViz/Foxglove visualization
- Dataset benchmark table
- Plugin example without heavy model dependency

最初のstar獲得には、論文風の数値よりも **一目で価値が分かるGIF** が重要。

### 20.4 OSS Differentiation Message

競合と比較した一言:

| Compared to | Message |
| --- | --- |
| Scan Context | BEVMatch can use Scan Context, but continues to alignment, change, and map validation |
| OverlapNet | BEVMatch uses overlap as evidence, not the final product |
| BEVPlace++ | BEVMatch treats BEV localization as one plugin in a broader comparison pipeline |
| RTAB-Map | BEVMatch is not a SLAM system; it is post-/around-SLAM same-place comparison and map validation |
| Autoware | BEVMatch complements localization and map workflows |
| Nav2 | BEVMatch adds long-term place memory and map freshness checking |

### 20.5 Community Strategy

重要なのは、研究者にも実務者にも参加理由を作ること。

**For researchers**

- pluginを追加するとbenchmarkで比較できる
- dataset adapterを追加すると評価対象が広がる
- paper methodを実運用pipelineへ接続できる

**For practitioners**

- ROS bagで試せる
- map QAに使える
- visualizationで説明できる
- Autoware/Nav2とつながる
- model依存なしbaselineから始められる

### 20.6 Governance

推奨:

- Core: permissive license
- Plugin: license明示必須
- Model weights: separate license
- Dataset: download scripts and adapters only, redistributionに注意
- Benchmark: reproducible split definition
- Contribution: plugin manifest required
- Documentation: architecture-first
- Stability: artifact schema versioning

## 21. Roadmap: v0.1 → v1.0

### v0.1: Concept-Proof Architecture

目的: BEVMatch が何者かをGitHubで一瞬で理解できる状態

Deliverables:

- core concept document
- data model draft
- comparison evidence schema draft
- single LiDAR dataset adapter
- simple retrieval baseline
- simple alignment baseline
- aligned visualization
- small sample dataset / instructions
- no deep dependency required

成功条件:

- READMEを見て「Place Recognitionだけではない」と伝わる
- query→retrieval→alignment overlay が動く
- contributionの方向性が明確

### v0.2: Retrieval Framework

目的: 複数retrieval手法を同じ土俵で比較できる状態

Deliverables:

- descriptor plugin system
- index backend abstraction
- Top-K evidence artifact
- Scan Context-style baseline
- BEV descriptor baseline
- retrieval metrics
- KITTI / MulRan-style evaluation recipe

成功条件:

- Recall@K評価が再現できる
- plugin追加の価値が見える

### v0.3: Alignment Framework

目的: retrieval candidateを比較可能なscene pairへ変換する

Deliverables:

- alignment evidence artifact
- SE2 / SE3 baseline
- overlap estimation
- residual visualization
- alignment metrics
- failure classification

成功条件:

- Top-K候補がpose付きcomparison pairになる
- alignment失敗も明示される

### v0.4: Change Detection MVP

目的: BEVMatchがPlace Recognition OSSでないことを決定的に示す

Deliverables:

- BEV occupancy diff
- point/voxel-level diff
- comparable / non-comparable mask
- dynamic region filtering baseline
- added / removed region output
- before/after viewer

成功条件:

- aligned historical/current sceneから差分が説明可能に見える
- false changeの主要原因を表示できる

### v0.5: Map Validation MVP

目的: map maintenance OSSとしての価値を出す

Deliverables:

- point cloud map comparison
- occupancy map comparison
- Lanelet2 region association
- stale region report
- issue severity schema
- human review report export

成功条件:

- 「地図が古い可能性のある領域」を出せる
- mapping engineerがレビューに使える

### v0.6: ROS2 Integration

目的: robotics OSSとして使える状態

Deliverables:

- ROS bag replay workflow
- ROS2 visualization output
- diagnostics
- TF-aware scene handling
- lifecycle-friendly runtime profile
- Foxglove/RViz demo

成功条件:

- ROS2ユーザが自分のbagで試せる

### v0.7: Autoware / Nav2 Adapters

目的: 既存robotics ecosystemに接続する

Deliverables:

- Autoware map QA workflow
- initial pose candidate workflow
- NDT localization support report
- Nav2 occupancy map staleness workflow
- simple relocalization assist demo

成功条件:

- Autoware / Nav2 usersに明確な導入理由が生まれる

### v0.8: Benchmark Suite

目的: 研究者が参加したくなる評価基盤にする

Deliverables:

- retrieval benchmark
- alignment benchmark
- change benchmark
- map validation benchmark
- dataset cards
- reproducible splits
- leaderboard-ready output

成功条件:

- paper methodをBEVMatch pluginとして追加する動機ができる

### v0.9: Multi-Modal Expansion

目的: BEV-only誤解を消し、Physical AI時代へ拡張する

Deliverables:

- camera retrieval adapter
- radar representation adapter
- semantic BEV representation
- object-level change hypothesis
- scene graph / map graph prototype
- natural-language report summary

成功条件:

- LiDAR以外でもBEVMatchの価値が伝わる

### v1.0: Stable Same-Place Comparison Platform

目的: production-like OSSとして安定版を出す

Deliverables:

- stable artifact schema
- stable plugin manifest
- stable benchmark protocol
- stable ROS2 integration
- documentation complete
- reproducible demo suite
- contribution guide
- governance and release policy
- API compatibility policy at artifact level

成功条件:

- GitHubユーザが「自分のretrieval/alignment/change手法をBEVMatchに載せたい」と思う
- mapping/localization engineerが「自分のbagとmapで試したい」と思う
- Autoware/Nav2ユーザが「map freshness確認に使える」と思う

## 22. Recommended MVP Scope

最初のMVPは広げすぎない。

**v0.1 MVP**

Input:

- LiDAR point cloud sequence
- approximate poses
- historical scenes
- optional map

Internal representations:

- point cloud
- BEV occupancy
- simple descriptor

Pipeline:

```
Query LiDAR scene
  ↓
Retrieve Top-K historical scenes
  ↓
Align best candidate
  ↓
Generate BEV overlay
  ↓
Generate simple added/removed occupancy diff
  ↓
Export comparison evidence
```

Output:

- Top-K places
- relative pose
- aligned overlay
- simple change regions
- evidence report

これだけで、BEVMatch の独自性は十分に伝わる。

## 23. Anti-Patterns to Avoid

BEVMatch が失敗する典型パターンは次である。

- Scan Context cloneになる
- BEVPlace再実装になる
- deep model zooになる
- ROS wrapperだけの薄いrepoになる
- change detectionだけに寄りすぎる
- map validationを構文検証と混同する
- pluginが出力scoreだけ返して根拠を持たない
- alignment confidenceなしでchangeを主張する
- dataset downloadが重すぎて誰も試せない
- visualizationが弱く、価値が伝わらない
- Autoware/Nav2に寄せすぎて一般roboticsで使いにくい
- Physical AIを意識しすぎて初期MVPが曖昧になる

## 24. Final Architecture Thesis

BEVMatch の勝ち筋は、最先端モデルを最初に実装することではない。

勝ち筋は次である。

> Place Recognition、Alignment、Change Detection、Map Validation を、同じデータモデル・同じ証拠形式・同じ可視化・同じ評価で接続すること。

この接続こそがOSSとして価値を持つ。

BEVMatch は、次のように定義するのが最も強い。

> BEVMatch is the missing comparison layer between robot observations and long-term spatial memory.

日本語では:

> BEVMatch は、ロボットの現在観測と長期空間記憶を接続する、欠けていた比較レイヤである。

そして、最も重要なメッセージはこれである。

> BEVMatch は「同じ場所を探すOSS」ではない。
> 「同じ場所を見つけて、比較可能にし、変化を説明し、地図の信頼性へ変換するOSS」である。
