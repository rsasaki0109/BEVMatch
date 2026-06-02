"""Serialise a Comparison Evidence Bundle to JSON (offline evidence, §16.4)."""

from __future__ import annotations

import json
from pathlib import Path

from bevmatch.core.evidence import ComparisonEvidenceBundle


def bundle_to_json(bundle: ComparisonEvidenceBundle, indent: int = 2) -> str:
    return json.dumps(bundle.to_dict(), indent=indent, ensure_ascii=False)


def save_bundle(bundle: ComparisonEvidenceBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle_to_json(bundle), encoding="utf-8")
    return path
