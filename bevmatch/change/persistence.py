"""Temporal persistence reasoning (§11.1 stage 3, §11.2, §11.3).

A single before/after diff cannot tell a real change from a passing pedestrian.
Given the same place observed over several frames, a genuine added/removed
object recurs at the same location across frames, while a dynamic/temporary
object appears once or jumps around. This consolidator tracks detections across
frames and keeps only the persistent ones as actionable; the rest are relabelled
``dynamic`` so they can be shown but excluded from map-update evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.core.datamodel import ChangeHypothesis


@dataclass(frozen=True)
class PersistenceConfig:
    match_dist_m: float = 2.0  # detections within this distance are the same object
    persistence_min: float = 0.6  # fraction of frames required to be actionable


def _track(per_frame: list[list[ChangeHypothesis]], category: str, cfg: PersistenceConfig):
    """Greedily group same-category detections across frames into tracks."""
    tracks: list[dict] = []  # {"centroids": [...], "frames": set, "members": [ch...]}
    for f, frame in enumerate(per_frame):
        for ch in frame:
            if ch.category != category:
                continue
            c = np.array(ch.centroid_xy)
            best, best_d = None, cfg.match_dist_m
            for tr in tracks:
                d = float(np.linalg.norm(np.array(tr["centroid"]) - c))
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                tracks.append({"centroid": ch.centroid_xy, "frames": {f}, "members": [ch]})
            else:
                best["frames"].add(f)
                best["members"].append(ch)
                pts = np.array([m.centroid_xy for m in best["members"]])
                best["centroid"] = tuple(pts.mean(axis=0))
    return tracks


def consolidate_changes(
    per_frame: list[list[ChangeHypothesis]],
    config: PersistenceConfig | None = None,
) -> list[ChangeHypothesis]:
    """Consolidate per-frame change lists into persistence-scored hypotheses."""
    cfg = config or PersistenceConfig()
    n_frames = max(1, len(per_frame))
    consolidated: list[ChangeHypothesis] = []

    for category in ("added", "removed"):
        for tr in _track(per_frame, category, cfg):
            members = tr["members"]
            persistence = len(tr["frames"]) / n_frames
            rep = max(members, key=lambda m: m.num_cells)  # most-supported detection
            actionable = persistence >= cfg.persistence_min
            out_category = category if actionable else "dynamic"
            consolidated.append(
                ChangeHypothesis(
                    category=out_category,
                    centroid_xy=tuple(np.array([m.centroid_xy for m in members]).mean(axis=0)),
                    area_m2=float(np.mean([m.area_m2 for m in members])),
                    num_cells=int(rep.num_cells),
                    confidence=float(np.clip(persistence * rep.confidence, 0.0, 1.0)),
                    bbox_xy=rep.bbox_xy,
                    persistence=persistence,
                    evidence={
                        "frames_support": len(tr["frames"]),
                        "n_frames": n_frames,
                        "original_category": category,
                    },
                )
            )

    consolidated.sort(key=lambda c: (c.actionable, c.area_m2), reverse=True)
    return consolidated
