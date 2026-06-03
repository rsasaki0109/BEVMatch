"""Scalable NN search and real-data loader helpers."""

from __future__ import annotations

import numpy as np

from bevmatch.alignment import SE2Aligner
from bevmatch.core.datamodel import Observation, Pose2D, Scene
from bevmatch.datasets import remove_ground, scene_from_points, voxel_downsample
from bevmatch.spatial import nearest_neighbors


def test_nearest_neighbors_matches_bruteforce():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2))
    query = rng.normal(size=(300, 2))
    dist, idx = nearest_neighbors(query, ref)
    # brute-force reference
    d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    bidx = d2.argmin(axis=1)
    assert np.array_equal(idx, bidx)
    assert np.allclose(dist, np.sqrt(d2[np.arange(len(query)), bidx]))


def test_nearest_neighbors_empty():
    d, i = nearest_neighbors(np.zeros((0, 2)), np.ones((5, 2)))
    assert len(d) == 0 and len(i) == 0


def test_dense_alignment_does_not_oom():
    # ~20k points would build a 20k x 20k matrix under the old brute force.
    rng = np.random.default_rng(1)
    ref = rng.uniform(-25, 25, size=(20000, 2))
    gt = Pose2D(2.0, -1.5, np.deg2rad(15))
    query = gt.inverse().transform(ref)
    a = SE2Aligner().align(Scene("q", observations={"l": Observation("l", query)}),
                           Scene("r", observations={"l": Observation("l", ref)}))
    assert a.success
    assert abs(a.relative_pose.x - gt.x) < 0.6
    assert abs(np.rad2deg(a.relative_pose.yaw - gt.yaw)) < 3.0


def test_voxel_downsample():
    pts = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.05, 4.95]])
    out = voxel_downsample(pts, voxel=1.0)
    assert len(out) == 2  # two clusters collapse to two points


def test_remove_ground():
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-10, 10, 200), rng.uniform(-10, 10, 200), np.zeros(200)])
    poles = np.column_stack([rng.uniform(-10, 10, 50), rng.uniform(-10, 10, 50), rng.uniform(2, 4, 50)])
    out = remove_ground(np.vstack([ground, poles]), margin_m=0.6)
    assert len(out) == 50  # only above-ground structure survives


def test_scene_from_points():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-20, 20, 5000), rng.uniform(-20, 20, 5000), rng.uniform(0, 5, 5000)])
    scene = scene_from_points(pts, "s0", voxel=1.0, drop_ground=True)
    xy = scene.primary().xy()
    assert xy.shape[1] == 2
    assert len(xy) < 5000  # downsampled
