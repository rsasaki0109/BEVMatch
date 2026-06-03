"""Alignment plugin interface (§7.2 Alignment Plugin, §10).

An ``Aligner`` consumes a query and reference Scene and returns an
``AlignmentHypothesis`` (pose + evidence). Different DOF levels (SE2, SE3,
map-element) implement the same interface so the pipeline and benchmark can swap
them freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bevmatch.core.datamodel import AlignmentHypothesis, Scene


class Aligner(ABC):
    name: str = "aligner"

    @abstractmethod
    def align(self, query: Scene, reference: Scene) -> AlignmentHypothesis:
        """Estimate the transform mapping the query into the reference frame."""
