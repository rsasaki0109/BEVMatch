# API Compatibility Policy (artifact level)

BEVMatch's stability guarantee is at the **artifact level**: the JSON shape of the
Comparison Evidence Bundle and the other artifacts a consumer reads. Python
function signatures may evolve; the artifact schema is the contract.

## Versioning

`bevmatch.schema.ARTIFACT_SCHEMA_VERSION` (currently **1.0**) uses semantic
versioning on the schema:

| Change | Version bump | Example |
| --- | --- | --- |
| Add an optional key | minor (1.0 → 1.1) | new evidence field |
| Add a new artifact type | minor | a new validator's issue |
| Remove/rename a key, change a key's meaning/type | **major** (1.x → 2.0) | rename `relative_pose` |

Within the same **major** version, consumers written against an older minor keep
working (only additive changes).

## Checking compatibility

```python
from bevmatch.schema import is_compatible, validate_artifact, require_compatible

bundle = ...  # loaded JSON dict
require_compatible(bundle["schema_version"])          # raises if major mismatch
problems = validate_artifact("comparison_evidence_bundle", bundle)
assert not problems, problems
```

## Self-describing envelopes

For storage/transport, wrap a payload so the artifact type and version travel
with it:

```python
from bevmatch.schema import envelope
record = envelope("change_hypothesis", change.to_dict())
# {"artifact": "change_hypothesis", "schema_version": "1.0", "payload": {...}}
```

## Stable artifacts (v1.0)

- `comparison_evidence_bundle`
- `alignment_hypothesis`
- `change_hypothesis`
- `map_validation_issue`
- `map_validation_report`
- `initial_pose_candidate`

Required keys per artifact are enforced by `validate_artifact`. Adding keys is
backward compatible; removing or renaming them is not.
