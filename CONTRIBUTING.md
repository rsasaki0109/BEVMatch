# Contributing to BEVMatch

BEVMatch grows by adding **plugins** (descriptors, aligners, change detectors,
map validators, dataset adapters, index backends) on a shared data model, evidence
schema, visualization and evaluation. Contributions are welcome from both
researchers (new methods, benchmarks) and practitioners (datasets, integrations).

## Quickstart

```bash
pip install -e ".[viz,dev]"
pytest                         # all tests should pass
python examples/run_all_demos.py   # reproducible demo suite
```

## Adding a plugin

1. **Implement the interface** for your plugin category:
   - descriptor → `bevmatch.retrieval.base.GlobalDescriptor`
   - aligner → `bevmatch.alignment.base.Aligner`
   - change detector / map validator → see `bevmatch.change` / `bevmatch.maps`
   - dataset adapter → produce `Scene` objects (see `bevmatch.datasets.synthetic`)
2. **Ship a manifest** (`bevmatch.plugins.PluginManifest`) declaring capability:
   input modality, required representation, output artifact, pose assumption,
   invariances, runtime/scale profile, uncertainty support, license, dataset
   compatibility, and failure modes. Register it with `register_manifest(...)`.
   A manifest is **required** for a plugin contribution (§7.3, §20.6).
3. **Respect the data model** (Plugin Design Rule, §7.4). Do **not**:
   - introduce a hidden coordinate frame,
   - return scores with undefined meaning,
   - assert change without alignment confidence,
   - depend on dataset-specific IDs,
   - emit black-box results with no provenance.
4. **Add a benchmark entry**: run `examples/run_benchmark_suite.py` (or
   `bevmatch.benchmarks`) and, for external results, submit a `SubmissionEntry`
   carrying the dataset fingerprint and `protocol_version`.
5. **Add tests** under `tests/` and keep them deterministic (seeded).

## Design principles (architecture.md §6.4)

- **Evidence-first** — every stage emits a justified artifact, not just a score.
- **Modality-agnostic, representation-plural** — separate input sensor from
  internal representation.
- **Alignment-gated change detection** — never assert change on an uncertain pose.
- **Map-aware but not map-dependent** — must run without a map.
- **Offline-first, live-ready** — keep the core ROS2-independent.

## Stability

The public contract is the **artifact schema** (`bevmatch.schema`,
`ARTIFACT_SCHEMA_VERSION`). See [docs/api_compatibility.md](docs/api_compatibility.md).
Changes that break an artifact's shape require a major-version bump.

## Code style

- Python ≥ 3.10, `numpy` for the core (no heavy deps in `bevmatch` core).
- Match the surrounding code's naming, comment density, and idioms.
- New optional dependencies go behind a `pyproject.toml` extra and a lazy import.
