"""Same-place comparison pipeline (§6.1, §22 MVP).

Query Scene -> retrieve Top-K -> align best candidate -> change diff ->
Comparison Evidence Bundle. Change detection is alignment-gated (Principle 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bevmatch.alignment.se2 import SE2AlignConfig, align_se2
from bevmatch.change.bev_diff import ChangeConfig, detect_changes
from bevmatch.core.datamodel import Scene
from bevmatch.core.evidence import ComparisonEvidenceBundle
from bevmatch.retrieval.retriever import SceneDatabase


@dataclass
class SamePlaceComparisonPipeline:
    """Orchestrates the v0.1 retrieval -> alignment -> change pipeline."""

    database: SceneDatabase
    top_k: int = 5
    align_config: SE2AlignConfig = field(default_factory=SE2AlignConfig)
    change_config: ChangeConfig = field(default_factory=ChangeConfig)

    def run(self, query_scene: Scene) -> ComparisonEvidenceBundle:
        bundle = ComparisonEvidenceBundle(query_scene_id=query_scene.scene_id)
        bundle.provenance = {
            "descriptor": "scan-context-ring-key",
            "aligner": "se2-bev-xcorr",
            "change_detector": "bev-occupancy-diff",
            "top_k": self.top_k,
            "bev": {
                "range_m": self.align_config.bev.range_m,
                "resolution_m": self.align_config.bev.resolution_m,
            },
        }

        candidates = self.database.query(query_scene, top_k=self.top_k)
        bundle.candidates = candidates
        if not candidates:
            bundle.uncertainty = {"note": "no candidates retrieved"}
            return bundle

        best = candidates[0]
        bundle.best_candidate = best
        ref_scene = self.database.get_scene(best.scene_id)

        alignment = align_se2(
            query_scene.primary().xy(),
            ref_scene.primary().xy(),
            self.align_config,
        )
        bundle.alignment = alignment
        bundle.uncertainty = {
            "retrieval_score": best.score,
            "alignment_overlap": alignment.overlap_ratio,
            "alignment_inliers": alignment.inlier_ratio,
            "alignment_success": alignment.success,
        }

        # Principle 3: only assert changes when alignment is trustworthy.
        if alignment.success:
            bundle.changes = detect_changes(
                query_scene.primary().xy(),
                ref_scene.primary().xy(),
                alignment.relative_pose,
                self.change_config,
                align_overlap=alignment.overlap_ratio,
            )
        else:
            bundle.uncertainty["note"] = (
                "alignment failed; change detection suppressed to avoid false changes"
            )
        return bundle
