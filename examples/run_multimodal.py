"""Multi-modal expansion demo (§0.9, §1.3, Principle 2).

    python examples/run_multimodal.py

Shows BEVMatch is not LiDAR/BEV-only: the same retrieval framework works on
LiDAR, radar and camera; object-level change reasoning yields added/removed/
moved/class-changed hypotheses; and a natural-language summary grounds the
evidence in plain sentences.
"""

from __future__ import annotations

from bevmatch.alignment import SE2Aligner
from bevmatch.change import detect_object_changes
from bevmatch.datasets import make_multimodal_places, make_object_change_case
from bevmatch.nl import summarize_object_changes
from bevmatch.retrieval import ScanContextDescriptor, SceneDatabase
from bevmatch.scene_graph import build_scene_graph
from bevmatch.sensors import CameraEmbeddingDescriptor


def main() -> None:
    print("=== Modality-agnostic place retrieval (Principle 2) ===")
    mm = make_multimodal_places(seed=0)
    rows = [
        ("LiDAR  (Scan-Context BEV)", mm.lidar_hist, mm.lidar_query, ScanContextDescriptor()),
        ("Radar  (-> BEV occupancy)", mm.radar_hist, mm.radar_query, ScanContextDescriptor()),
        ("Camera (image embedding) ", mm.camera_hist, mm.camera_query, CameraEmbeddingDescriptor()),
    ]
    for name, hist, query, desc in rows:
        db = SceneDatabase(descriptor=desc)
        db.add_all(hist)
        top = db.query(query, top_k=1)[0]
        ok = "OK" if top.place_id == mm.gt_place_id else "MISS"
        print(f"  {name}: retrieved {top.place_id} (gt {mm.gt_place_id})  [{ok}]")

    print("\n=== Object-level change (§11.5) ===")
    case = make_object_change_case(seed=4)
    alignment = SE2Aligner().align(case.query_lidar, case.ref_lidar)
    changes = detect_object_changes(case.query_objects, case.ref_objects, alignment.relative_pose)
    for c in changes:
        print(f"  {c.category:<14} {c.object_class}"
              + (f"  ({c.from_class}->{c.to_class})" if c.category == 'class_changed' else "")
              + (f"  moved {c.displacement_m:.1f} m" if c.category == 'moved' else ""))

    print("\n=== Natural-language summary (§0.9, §19.2) ===")
    print("  " + summarize_object_changes(changes))

    print("\n=== Object scene graph (§5.4 prototype) ===")
    graph = build_scene_graph(case.ref_objects)
    print(f"  {len(graph.objects)} objects, {len(graph.edges)} 'near' edges, "
          f"classes: {graph.class_counts()}")


if __name__ == "__main__":
    main()
