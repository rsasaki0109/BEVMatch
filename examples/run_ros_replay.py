"""Offline bag-replay workflow (§16.3 Bag benchmark) — no ROS2 required.

    python examples/run_ros_replay.py

Replays a stream of timestamped scenes through the lifecycle-managed pipeline,
printing managed-lifecycle transitions and per-message diagnostics, and writing
a Foxglove/web-importable markers JSON plus an evidence report. The same
``BagReplayPipeline`` is what the rclpy LifecycleNode wraps.
"""

from __future__ import annotations

import json
from pathlib import Path

from bevmatch.datasets import make_synthetic_route
from bevmatch.retrieval import SceneDatabase
from bevmatch.ros import BagReplayPipeline

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    route = make_synthetic_route(seed=0, n_places=10, n_queries=8)
    db = SceneDatabase()
    db.add_all(route.historical)

    pipeline = BagReplayPipeline(database=db)
    print(f"lifecycle: {pipeline.state.value}")
    pipeline.configure(); print(f"lifecycle: -> {pipeline.state.value}")
    pipeline.activate(); print(f"lifecycle: -> {pipeline.state.value}\n")

    outputs = []
    for q in route.queries:
        out = pipeline.process(q.scene)
        outputs.append(out)
        diag = {d.name.split("/")[-1]: d.level.name for d in out.diagnostics}
        n = len(out.bundle.changes)
        print(f"[t={out.timestamp:>5}] {out.scene_id:<9} match={out.bundle.best_candidate.place_id:<9} "
              f"changes={n:<2} markers={len(out.markers):<2} diag={diag}")

    pipeline.deactivate(); pipeline.shutdown()
    print(f"\nlifecycle: -> {pipeline.state.value}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = [m for o in outputs for m in o.markers]
    (OUT_DIR / "ros_markers.json").write_text(json.dumps(markers, indent=2), encoding="utf-8")
    (OUT_DIR / "ros_replay_report.json").write_text(
        json.dumps([o.to_dict() for o in outputs], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Markers (Foxglove/web) -> {OUT_DIR / 'ros_markers.json'}")
    print(f"Replay report          -> {OUT_DIR / 'ros_replay_report.json'}")


if __name__ == "__main__":
    main()
