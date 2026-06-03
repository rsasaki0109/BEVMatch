"""rclpy LifecycleNode demo (§16.2) — requires a sourced ROS2 environment.

    python examples/ros2_lifecycle_node.py

Brings up the BEVMatch managed-lifecycle node, drives it through
configure -> activate, replays a few synthetic scenes (publishing MarkerArray /
DiagnosticArray / PoseWithCovarianceStamped), then deactivates and shuts down.
Inspect topics with, e.g.:  ros2 topic echo /bevmatch/markers
"""

from __future__ import annotations

import rclpy

from bevmatch.datasets import make_synthetic_route
from bevmatch.retrieval import SceneDatabase
from bevmatch.ros.node import BevmatchLifecycleNode
from bevmatch.ros.replay import BagReplayPipeline


def main() -> None:
    rclpy.init()
    route = make_synthetic_route(seed=0, n_places=8, n_queries=5)
    db = SceneDatabase()
    db.add_all(route.historical)

    node = BevmatchLifecycleNode(BagReplayPipeline(database=db))
    try:
        node.trigger_configure()
        node.trigger_activate()
        for q in route.queries:
            node.process_scene(q.scene)
            rclpy.spin_once(node, timeout_sec=0.05)
            node.get_logger().info(f"published evidence for {q.scene.scene_id}")
        node.trigger_deactivate()
    finally:
        node.trigger_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
