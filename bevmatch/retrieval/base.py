"""Descriptor plugin interface (§7.2 Descriptor Plugin, §9).

A ``GlobalDescriptor`` turns a Scene into a ``DescriptorCode``: a rotation-
invariant ``vector`` used by an index backend for cheap Top-N prefiltering, plus
an opaque ``payload`` the descriptor's own ``distance`` uses for fine,
rotation-robust scoring (and an optional yaw estimate). This two-part design
lets a dense-vector index (FAISS) and a structured matcher (Scan-Context column
shift) coexist behind one interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from bevmatch.core.datamodel import Scene


@dataclass
class DescriptorCode:
    vector: np.ndarray  # rotation-invariant prefilter vector
    payload: Any = None  # descriptor-specific data for fine distance


class GlobalDescriptor(ABC):
    """Base class for place-level descriptors."""

    #: stable identifier used in evidence/provenance (Candidate.descriptor_type)
    name: str = "global-descriptor"

    @abstractmethod
    def extract(self, scene: Scene) -> DescriptorCode:
        """Compute the descriptor code for a scene."""

    @abstractmethod
    def distance(self, query: DescriptorCode, ref: DescriptorCode) -> tuple[float, float | None]:
        """Return ``(distance, yaw_estimate)``; ``yaw_estimate`` may be ``None``.

        Smaller distance = more similar.
        """
