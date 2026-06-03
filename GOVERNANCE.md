# BEVMatch Governance

This document describes how BEVMatch is licensed, versioned, and maintained
(architecture.md §20.6).

## Licensing

- **Core** (`bevmatch`): Apache-2.0 (permissive).
- **Plugins**: must declare their license in the `PluginManifest`. Research-only
  or non-commercial plugins are allowed but must say so.
- **Model weights**: distributed under a separate license from the code; never
  vendored into the core.
- **Datasets**: BEVMatch ships download scripts / adapters only. Respect each
  dataset's redistribution terms; do not commit dataset contents.

## Versioning and stability

- **Artifact schema** (`bevmatch.schema.ARTIFACT_SCHEMA_VERSION`) is the public
  contract. Semantic: same major = compatible (additive), major bump = breaking.
- **Plugin manifest** (`MANIFEST_VERSION`) and **benchmark protocol**
  (`BENCHMARK_PROTOCOL_VERSION`) are versioned independently.
- The package version follows the artifact schema's major version.

## Release policy

1. All tests pass (`pytest`) and the demo suite runs (`examples/run_all_demos.py`).
2. Update `CHANGELOG.md`.
3. Breaking artifact-schema changes bump the major version and are documented in
   [docs/api_compatibility.md](docs/api_compatibility.md).
4. Tag the release; benchmark fingerprints in the leaderboard must be regenerated
   if any dataset card changed.

## Contributions

- Plugins require a manifest (§7.3) and tests.
- Documentation is architecture-first: see [docs/architecture.md](docs/architecture.md).
- See [CONTRIBUTING.md](CONTRIBUTING.md).

## Maintainers

- Project lead: repository owner. Additional maintainers are added by consensus
  of existing maintainers.
