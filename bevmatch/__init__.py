"""BEVMatch — Same-Place Comparison OSS.

v0.1 MVP: LiDAR query -> Top-K retrieval -> SE2 alignment -> BEV overlay/diff
-> Comparison Evidence Bundle.

See docs/architecture.md for the full design.
"""

from bevmatch.core.datamodel import (
    AlignmentHypothesis,
    Candidate,
    ChangeHypothesis,
    Observation,
    Pose2D,
    Scene,
)
from bevmatch.core.evidence import ComparisonEvidenceBundle
from bevmatch.core.pipeline import SamePlaceComparisonPipeline
from bevmatch.retrieval.retriever import SceneDatabase

__version__ = "1.11.0"

__all__ = [
    "Pose2D",
    "Observation",
    "Scene",
    "Candidate",
    "AlignmentHypothesis",
    "ChangeHypothesis",
    "ComparisonEvidenceBundle",
    "SceneDatabase",
    "SamePlaceComparisonPipeline",
    "__version__",
]
