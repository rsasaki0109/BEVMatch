# Two findings from the BEVMatch benchmarks

A short, honest technical note. Everything here is measured on **public KITTI
odometry** data by BEVMatch's own retrieval pipeline, on one standard
place-recognition protocol, and is reproducible from `scripts/`. Numbers and the
exact protocol live in [benchmarks.md](benchmarks.md); this note is about what
they *mean*.

## Setup in one paragraph

We evaluate place recognition (loop closure) on the KITTI loop sequences
00/05/06/07/08. A query's **positive** is a frame within *D* metres of its
ground-truth pose and more than 30 s away in time; the 30 s temporal window is
excluded from retrieval candidates so trivial same-pass neighbours never count.
We report **Recall@1 @ 5 m**. Three descriptors run behind BEVMatch's single
retrieval interface (`GlobalDescriptor` → `SceneDatabase`):

| descriptor | modality | learned? | training data |
|---|---|---|---|
| Scan-Context (ring-key + column-shift) | LiDAR, 360° | no | — |
| ResNet-18 (ImageNet) global features | camera, forward | pretrained, generic | ImageNet |
| EigenPlaces (ResNet-50 + GeM, 2048-d) | camera, forward | yes, for VPR | SF-XL street-view |

EigenPlaces never sees KITTI in training (SF-XL ⟂ KITTI), so every KITTI sequence
is a genuine held-out domain — the camera-vs-camera comparison is fair everywhere.

| seq | LiDAR Scan-Context | Camera ResNet-18 | Camera EigenPlaces |
|---|---|---|---|
| 00 | 0.913 | 0.923 | **0.957** |
| 05 | 0.783 | 0.848 | **0.914** |
| 06 | 0.887 | 0.977 | 0.977 |
| 07 | 0.596 | 0.500 | **0.681** |
| 08 (reverse) | **0.339** | 0.015 | 0.015 |
| mean | 0.704 | 0.653 | **0.709** |

## Finding 1 — within a modality, representation quality is real and measurable

Swapping the generic ImageNet ResNet-18 for a place-recognition–trained
descriptor (EigenPlaces) improves recall on **every forward-revisit sequence**:
seq 07 jumps +0.18 (0.500 → 0.681), seq 05 +0.07, seq 00 +0.03; seq 06 is already
at the 0.977 ceiling. The mean over the camera-favourable cases moves clearly up.
This is the expected, encouraging result: a better-learned representation of the
*same* observation retrieves better, and because the descriptor is a plugin, the
upgrade is a one-line swap with no pipeline change. **Representation matters.**

## Finding 2 — across a viewpoint gap, representation quality is not enough

Sequence 08's revisits are **reverse-direction**: the car drives back down the
same road the opposite way. A forward-facing camera then observes the *opposite
view* of each place. Here both camera descriptors — the generic baseline **and**
the learned SOTA — sit at **R@1 = 0.015**, byte-for-byte identical. Throwing a
stronger, purpose-trained network at the problem changes *nothing*, because the
sensor never records the view that would need to be matched. There is no
representation of an observation that was never made.

This is the distinction the framework is built around (Principle 2:
**modality ≠ representation**). The reverse-loop failure is not a representation
gap that more learning closes; it is a **viewpoint / geometry wall** set by the
sensor. The evidence that it is *sensor*-bound, not *descriptor*-bound:

- The same reverse seq 08, seen by a **360° LiDAR**, is solved to **R@1 = 0.339**
  with the hand-crafted Scan-Context — and **recovered to 0.765** by *only*
  widening the descriptor config (range 30→80 m, grid 20×60→40×120; no pipeline
  change). A rotation-invariant sensor *does* observe the place on the opposite
  pass, so a better descriptor of that observation helps.
- The camera, observing the opposite view, cannot be rescued the same way: 0.015
  before learning, 0.015 after.

So the *same* intervention — a better/wider descriptor — moves LiDAR from 0.34 to
0.77 but leaves the camera pinned at 0.02. The asymmetry is the point: **when one
sensor is blind to a revisit, no descriptor recovers it; a second modality that
is not blind does.** That is the concrete, measured argument for a
modality-agnostic same-place framework rather than a single-sensor pipeline.

## Honest limitations

- **Grayscale camera.** We use KITTI `image_0` (grayscale, replicated to 3
  channels); EigenPlaces trained on RGB. The forward-case camera numbers would
  likely be somewhat higher with color `image_2`. This does **not** affect
  Finding 2 — the reverse-loop collapse is geometric, not photometric, and color
  cannot manufacture an unobserved view.
- **One dataset.** KITTI only, urban driving. The findings are consistent across
  its five loop sequences, but cross-dataset generalisation is untested here.
- **seq 07 is small.** 94 revisit queries; treat its absolute number as noisy
  (the *direction* of the EigenPlaces improvement is still clear).
- **Two modalities, not a fusion.** We compare LiDAR and camera under one
  interface and show their failure modes are complementary; we do not yet report
  a *fused* retriever that always picks the modality that can see the revisit.
  That is the natural next experiment this note motivates.

## Reproduce

```bash
python scripts/benchmark_kitti_lidar.py            # LiDAR Scan-Context, all sequences
python scripts/benchmark_kitti_vpr.py              # Camera ResNet-18 baseline
python scripts/benchmark_kitti_vpr_learned.py      # Camera EigenPlaces (MIT, torch.hub)
python scripts/experiment_scancontext_config.py    # LiDAR default-vs-wide on seq 00/08
python scripts/make_results_summary.py             # the summary figure
```

Descriptor sources: Scan-Context is BEVMatch's own implementation; EigenPlaces is
loaded at runtime from `gmberton/eigenplaces` (MIT) and is **not** vendored.
