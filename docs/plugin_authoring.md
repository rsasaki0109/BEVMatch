# Authoring a BEVMatch plugin

This guide shows how to add a retrieval descriptor and an aligner. Other plugin
categories (change detector, map validator, dataset adapter, index backend)
follow the same pattern: implement the interface, ship a manifest, add tests.

## A descriptor plugin

```python
import numpy as np
from bevmatch.retrieval.base import GlobalDescriptor, DescriptorCode
from bevmatch.plugins import PluginManifest, register_manifest

class MyDescriptor(GlobalDescriptor):
    name = "my-descriptor"

    def extract(self, scene) -> DescriptorCode:
        pts = scene.primary().xy()
        vector = my_embedding(pts)            # rotation-invariant prefilter vector
        return DescriptorCode(vector=vector, payload=pts)

    def distance(self, query, ref) -> tuple[float, float | None]:
        d = float(np.linalg.norm(query.vector - ref.vector))
        return d, None                        # (distance, optional yaw estimate)

register_manifest(PluginManifest(
    name="my-descriptor", category="descriptor", output_artifact="descriptor",
    input_modality=("lidar",), required_representation=("point_cloud",),
    invariance=("rotation",), uncertainty_support="score_only",
    failure_modes=("repetitive_structure",),
))
```

Use it anywhere a descriptor is accepted:

```python
from bevmatch.retrieval import SceneDatabase
db = SceneDatabase(descriptor=MyDescriptor())
```

## An aligner plugin

```python
from bevmatch.alignment.base import Aligner
from bevmatch.core.datamodel import AlignmentHypothesis, Pose2D

class MyAligner(Aligner):
    name = "my-aligner"

    def align(self, query, reference) -> AlignmentHypothesis:
        pose, overlap, inliers, ok = my_register(query, reference)
        return AlignmentHypothesis(
            relative_pose=pose, overlap_ratio=overlap, inlier_ratio=inliers,
            success=ok, failure_class=None if ok else "overlap_insufficient",
        )
```

## Benchmarking your plugin

```python
from bevmatch.benchmarks import run_retrieval_benchmark, format_leaderboard
results = run_retrieval_benchmark(descriptors=[MyDescriptor()])
print(format_leaderboard(results, "retrieval"))
```

To publish an external result, emit a `SubmissionEntry` with the dataset
fingerprint (`bevmatch.benchmarks.make_manifest(card).fingerprint`) and the
`protocol_version`.

## Rules (architecture.md §7.4)

- one coordinate convention (`p_hist = relative_pose.transform(p_query)`),
- meaningful, documented scores,
- no change assertion without alignment confidence,
- provenance on every artifact,
- a manifest is required.
