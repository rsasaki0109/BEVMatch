"""BEVMatch rclpy LifecycleNode (§16.2 lifecycle-managed, composable).

Requires a ROS2 environment (rclpy). This module is imported only where ROS2 is
available — never from ``bevmatch.ros.__init__`` — so the core stays usable
without ROS2. The node is a thin wrapper: all logic lives in the ROS2-
independent ``BagReplayPipeline``; the node converts its evidence / diagnostics /
markers into ROS messages and publishes them.

Run the demo:  python examples/ros2_lifecycle_node.py
"""

from __future__ import annotations

import math

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from visualization_msgs.msg import Marker, MarkerArray

from bevmatch.core.datamodel import Scene
from bevmatch.ros.replay import BagReplayPipeline

_MARKER_TYPE = {"SPHERE": Marker.SPHERE, "CUBE": Marker.CUBE, "CYLINDER": Marker.CYLINDER}


class BevmatchLifecycleNode(LifecycleNode):
    """Publishes BEVMatch evidence as MarkerArray / DiagnosticArray / pose."""

    def __init__(self, pipeline: BagReplayPipeline, **kwargs) -> None:
        super().__init__("bevmatch", **kwargs)
        self._pipeline = pipeline
        self._markers_pub = None
        self._diag_pub = None
        self._pose_pub = None

    # --- managed lifecycle callbacks ---
    def on_configure(self, state) -> TransitionCallbackReturn:
        self._markers_pub = self.create_lifecycle_publisher(MarkerArray, "~/markers", 10)
        self._diag_pub = self.create_lifecycle_publisher(DiagnosticArray, "/diagnostics", 10)
        self._pose_pub = self.create_lifecycle_publisher(PoseWithCovarianceStamped, "~/pose", 10)
        self._pipeline.configure()
        self.get_logger().info("bevmatch configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._pipeline.activate()
        self.get_logger().info("bevmatch active")
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        self._pipeline.deactivate()
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._pipeline.cleanup()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self._pipeline.shutdown()
        return TransitionCallbackReturn.SUCCESS

    # --- processing one incoming scene ("message") ---
    def process_scene(self, scene: Scene) -> None:
        output = self._pipeline.process(scene)
        stamp = self.get_clock().now().to_msg()

        self._markers_pub.publish(self._to_marker_array(output.markers, stamp))
        self._diag_pub.publish(self._to_diag_array(output.diagnostics, stamp))
        if output.bundle.alignment and output.bundle.alignment.success:
            self._pose_pub.publish(self._to_pose(output.bundle.alignment.relative_pose, stamp))

    # --- conversions (pure-Python artifacts -> ROS messages) ---
    def _to_marker_array(self, markers, stamp) -> MarkerArray:
        arr = MarkerArray()
        for m in markers:
            msg = Marker()
            msg.header.stamp = stamp
            msg.header.frame_id = m["frame_id"]
            msg.ns = m["ns"]
            msg.id = m["id"]
            msg.type = _MARKER_TYPE.get(m["type"], Marker.SPHERE)
            msg.action = Marker.ADD
            msg.pose.position.x = m["pose"]["position"]["x"]
            msg.pose.position.y = m["pose"]["position"]["y"]
            msg.pose.position.z = m["pose"]["position"]["z"]
            msg.pose.orientation.w = 1.0
            msg.scale.x, msg.scale.y, msg.scale.z = m["scale"]["x"], m["scale"]["y"], m["scale"]["z"]
            msg.color.r, msg.color.g = m["color"]["r"], m["color"]["g"]
            msg.color.b, msg.color.a = m["color"]["b"], m["color"]["a"]
            msg.text = m["text"]
            arr.markers.append(msg)
        return arr

    def _to_diag_array(self, statuses, stamp) -> DiagnosticArray:
        arr = DiagnosticArray()
        arr.header.stamp = stamp
        for s in statuses:
            ds = DiagnosticStatus()
            ds.level = bytes([int(s.level)])
            ds.name = s.name
            ds.message = s.message
            ds.values = [KeyValue(key=str(k), value=str(v)) for k, v in s.values.items()]
            arr.status.append(ds)
        return arr

    def _to_pose(self, pose2d, stamp) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._pipeline.map_frame
        msg.pose.pose.position.x = pose2d.x
        msg.pose.pose.position.y = pose2d.y
        msg.pose.pose.orientation.z = math.sin(pose2d.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(pose2d.yaw / 2.0)
        return msg
