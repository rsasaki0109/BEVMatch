"""Sensor adapters (§1.3, Principle 2): bring non-LiDAR modalities into the pipeline.

The internal representation (BEV, embedding) is decoupled from the input sensor.
Camera observations carry an image embedding; radar scans are projected to the
same BEV occupancy used by LiDAR, so retrieval/alignment work unchanged.
"""

from bevmatch.sensors.camera import CameraEmbeddingDescriptor, camera_scene
from bevmatch.sensors.radar import radar_scene, radar_to_points

__all__ = [
    "CameraEmbeddingDescriptor",
    "camera_scene",
    "radar_to_points",
    "radar_scene",
]
