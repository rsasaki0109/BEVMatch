"""Vector / Lanelet2 map validator (§12.2, §12.5, §17.2 Pattern D).

For each vector-map element (lane boundary, stop line, ...), checks whether the
current observation supports it: sample points along the element's polyline and
test for nearby returns. An element with little support becomes a
``map_element_unobserved`` issue — the map claims geometry the world no longer
shows. This complements (does not replace) a Lanelet2 file-syntax validator.
"""

from __future__ import annotations

import numpy as np

from bevmatch.core.datamodel import Pose2D, Scene
from bevmatch.maps.datamodel import MapValidationIssue, VectorMap
from bevmatch.maps.severity import assess_severity


def _resample(polyline: np.ndarray, step: float) -> np.ndarray:
    poly = np.asarray(polyline, dtype=float)
    if len(poly) < 2:
        return poly
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        n = max(1, int(length / step))
        for i in range(1, n + 1):
            out.append(a + seg * (i / n))
    return np.array(out)


class VectorMapValidator:
    name = "vector-map-validator"

    def __init__(self, support_radius_m: float = 1.5, min_support_frac: float = 0.3, sample_step_m: float = 1.0) -> None:
        self.support_radius_m = support_radius_m
        self.min_support_frac = min_support_frac
        self.sample_step_m = sample_step_m

    def validate(
        self,
        current: Scene,
        vmap: VectorMap,
        pose: Pose2D | None = None,
    ) -> list[MapValidationIssue]:
        pose = pose or Pose2D()
        obs = pose.transform(current.primary().xy())
        issues: list[MapValidationIssue] = []
        for elem in vmap.elements:
            samples = _resample(elem.polyline, self.sample_step_m)
            if len(samples) == 0:
                continue
            d = np.sqrt(((samples[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            support = float(np.mean(d <= self.support_radius_m))
            if support < self.min_support_frac:
                centroid = samples.mean(axis=0)
                issues.append(
                    MapValidationIssue(
                        issue_type="map_element_unobserved",
                        severity=assess_severity("map_element_unobserved", confidence=1.0 - support),
                        centroid_xy=(float(centroid[0]), float(centroid[1])),
                        confidence=float(1.0 - support),
                        evidence={
                            "source": "vector-map-projection",
                            "element_id": elem.element_id,
                            "element_type": elem.element_type,
                            "support_fraction": round(support, 3),
                        },
                    )
                )
        return issues
