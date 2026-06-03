"""ROS2 integration (§16).

The BEVMatch core is ROS2-independent (§16.1); this package provides a
ROS2-facing layer. Everything exported here is pure Python (no rclpy) so it is
testable and usable offline (Principle 5: offline-first, live-ready). The rclpy
LifecycleNode lives in ``bevmatch.ros.node`` and is imported separately, only
where a ROS2 environment is available.
"""

from bevmatch.ros.diagnostics import (
    DiagnosticLevel,
    DiagnosticStatus,
    diagnostics_from_bundle,
    diagnostics_from_map_report,
)
from bevmatch.ros.markers import change_markers, issue_markers
from bevmatch.ros.replay import (
    BagReplayPipeline,
    LifecycleState,
    ReplayOutput,
)
from bevmatch.ros.tf import TFTree

__all__ = [
    "TFTree",
    "DiagnosticLevel",
    "DiagnosticStatus",
    "diagnostics_from_bundle",
    "diagnostics_from_map_report",
    "change_markers",
    "issue_markers",
    "BagReplayPipeline",
    "LifecycleState",
    "ReplayOutput",
]
