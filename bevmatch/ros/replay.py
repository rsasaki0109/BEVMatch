"""Lifecycle-managed bag replay pipeline (§16.2, §16.3 Bag benchmark).

A ROS2-independent runtime that mirrors the managed-lifecycle pattern
(unconfigured -> inactive -> active -> finalized). It replays a stream of
timestamped scenes (a "bag"), running retrieval -> alignment -> change per
message and emitting evidence, diagnostics and markers. The rclpy LifecycleNode
in ``bevmatch.ros.node`` delegates to this class, so the same logic runs offline
and live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.se2 import SE2Aligner
from bevmatch.change.bev_diff import ChangeConfig
from bevmatch.core.datamodel import Scene
from bevmatch.core.evidence import ComparisonEvidenceBundle
from bevmatch.core.pipeline import SamePlaceComparisonPipeline
from bevmatch.retrieval.retriever import SceneDatabase
from bevmatch.ros.diagnostics import DiagnosticStatus, diagnostics_from_bundle
from bevmatch.ros.markers import change_markers


class LifecycleState(Enum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"


@dataclass
class ReplayOutput:
    timestamp: float | None
    scene_id: str
    bundle: ComparisonEvidenceBundle
    diagnostics: list[DiagnosticStatus]
    markers: list[dict]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "scene_id": self.scene_id,
            "summary": self.bundle.summary(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "n_markers": len(self.markers),
        }


@dataclass
class BagReplayPipeline:
    """A managed-lifecycle wrapper around the comparison pipeline."""

    database: SceneDatabase
    aligner: Aligner = field(default_factory=SE2Aligner)
    top_k: int = 5
    change_config: ChangeConfig = field(default_factory=ChangeConfig)
    map_frame: str = "map"

    state: LifecycleState = LifecycleState.UNCONFIGURED
    _pipeline: SamePlaceComparisonPipeline | None = None

    # --- lifecycle transitions (mirror ROS2 managed nodes, §16.2) ---
    def configure(self) -> LifecycleState:
        self._pipeline = SamePlaceComparisonPipeline(
            database=self.database, top_k=self.top_k,
            aligner=self.aligner, change_config=self.change_config,
        )
        self.state = LifecycleState.INACTIVE
        return self.state

    def activate(self) -> LifecycleState:
        if self.state != LifecycleState.INACTIVE:
            raise RuntimeError(f"cannot activate from {self.state.value}")
        self.state = LifecycleState.ACTIVE
        return self.state

    def deactivate(self) -> LifecycleState:
        if self.state != LifecycleState.ACTIVE:
            raise RuntimeError(f"cannot deactivate from {self.state.value}")
        self.state = LifecycleState.INACTIVE
        return self.state

    def cleanup(self) -> LifecycleState:
        self._pipeline = None
        self.state = LifecycleState.UNCONFIGURED
        return self.state

    def shutdown(self) -> LifecycleState:
        self._pipeline = None
        self.state = LifecycleState.FINALIZED
        return self.state

    # --- processing ---
    def process(self, scene: Scene) -> ReplayOutput:
        """Process one incoming message (scene). Only valid when ACTIVE."""
        if self.state != LifecycleState.ACTIVE or self._pipeline is None:
            raise RuntimeError(f"pipeline not ACTIVE (state={self.state.value})")
        bundle = self._pipeline.run(scene)
        diagnostics = diagnostics_from_bundle(bundle)
        markers = change_markers(bundle.changes, frame_id=self.map_frame)
        return ReplayOutput(
            timestamp=scene.timestamp, scene_id=scene.scene_id,
            bundle=bundle, diagnostics=diagnostics, markers=markers,
        )

    def replay(self, scenes: list[Scene]) -> list[ReplayOutput]:
        """Replay a full bag (auto configure/activate if needed)."""
        if self.state == LifecycleState.UNCONFIGURED:
            self.configure()
        if self.state == LifecycleState.INACTIVE:
            self.activate()
        return [self.process(s) for s in scenes]
