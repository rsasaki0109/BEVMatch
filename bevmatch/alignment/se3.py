"""SE3 alignment via point-to-point ICP (§10.3).

A full 6-DoF baseline for 3D LiDAR / uneven terrain. It is seeded by the SE2
BEV cross-correlation, then refined by 3D ICP. On planar scenes the out-of-plane
DOF (z, roll, pitch) are unobservable; rather than reporting a confident-but-
meaningless value, the aligner flags the degeneracy (§5.6, §10.4 observability).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bevmatch.alignment.base import Aligner
from bevmatch.alignment.failure import classify_alignment_failure
from bevmatch.alignment.se2 import SE2AlignConfig, align_se2, bev_overlap
from bevmatch.core.datamodel import AlignmentHypothesis, PoseSE3, Scene
from bevmatch.representations.bev import BEVConfig
from bevmatch.spatial import nearest_neighbors


@dataclass(frozen=True)
class SE3AlignConfig:
    bev: BEVConfig = field(default_factory=BEVConfig)
    icp_iters: int = 30
    icp_max_dist_m: float = 1.5
    min_overlap_ratio: float = 0.45  # genuine revisits ≳0.74; unrelated places ≲0.33
    min_points: int = 20
    rmse_fail_m: float = 1.0
    planar_eig_ratio: float = 1e-3  # below this the scene is treated as planar


def _umeyama_se3(src: np.ndarray, dst: np.ndarray) -> PoseSE3:
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return PoseSE3.from_matrix(R, t)


def _icp_se3(query: np.ndarray, ref: np.ndarray, init: PoseSE3, cfg: SE3AlignConfig):
    pose = init
    inlier_ratio, rmse, n_corr = 0.0, float("inf"), 0
    nn_keep = None
    for i in range(cfg.icp_iters):
        moved = pose.transform(query)
        dist, nn = nearest_neighbors(moved, ref)
        gate = max(cfg.icp_max_dist_m * (1.0 - 0.6 * i / max(1, cfg.icp_iters)), 0.4)
        keep = dist < gate
        inlier_ratio = float(keep.mean())
        n_corr = int(keep.sum())
        rmse = float(np.sqrt(np.mean(dist[keep] ** 2))) if n_corr else float("inf")
        if n_corr < 3:
            break
        nn_keep = nn[keep]
        new_pose = _umeyama_se3(query[keep], ref[nn[keep]])
        if np.allclose(new_pose.transform(query), moved, atol=1e-4):
            pose = new_pose
            break
        pose = new_pose
    return pose, inlier_ratio, rmse, n_corr, nn_keep


def _degeneracy(ref_inliers: np.ndarray, cfg: SE3AlignConfig) -> dict:
    """Flag unobservable DOF from the spread of the matched reference geometry."""
    if ref_inliers is None or len(ref_inliers) < 3:
        return {"planar_scene": False, "eig_ratio": None, "unobservable": []}
    cov = np.cov(ref_inliers.T)
    eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))
    ratio = float(eig[0] / (eig[-1] + 1e-12))
    planar = ratio < cfg.planar_eig_ratio
    return {
        "planar_scene": planar,
        "eig_ratio": round(ratio, 6),
        # in a planar scene z translation and out-of-plane rotation are unconstrained
        "unobservable": ["z", "roll", "pitch"] if planar else [],
    }


def align_se3(
    query_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    config: SE3AlignConfig | None = None,
) -> AlignmentHypothesis:
    """Estimate the SE3 transform mapping ``query_xyz`` into the ``ref_xyz`` frame."""
    cfg = config or SE3AlignConfig()
    q = np.asarray(query_xyz, dtype=float)[:, :3]
    r = np.asarray(ref_xyz, dtype=float)[:, :3]

    if len(q) < cfg.min_points or len(r) < cfg.min_points:
        return AlignmentHypothesis(
            relative_pose=PoseSE3().project_se2(),
            overlap_ratio=0.0,
            inlier_ratio=0.0,
            success=False,
            failure_reason="insufficient geometric constraints",
            failure_class="insufficient_constraints",
            pose_se3=PoseSE3().to_dict(),
        )

    # Seed with the SE2 BEV cross-correlation, then refine in full 6-DoF.
    coarse2d = align_se2(q[:, :2], r[:, :2], SE2AlignConfig(bev=cfg.bev)).relative_pose
    pose, inlier_ratio, rmse, n_corr, nn_keep = _icp_se3(q, r, PoseSE3.from_se2(coarse2d), cfg)

    q_in_ref = pose.transform(q)
    overlap_ratio, grid_inlier = bev_overlap(q_in_ref[:, :2], r[:, :2], cfg.bev)
    inlier_ratio = max(inlier_ratio, grid_inlier)
    degeneracy = _degeneracy(r[nn_keep] if nn_keep is not None else None, cfg)

    success = overlap_ratio >= cfg.min_overlap_ratio
    failure_class = classify_alignment_failure(
        overlap_ratio=overlap_ratio,
        inlier_ratio=inlier_ratio,
        rmse_m=rmse,
        num_correspondences=n_corr,
        success=success,
        min_overlap_ratio=cfg.min_overlap_ratio,
        min_correspondences=cfg.min_points,
        rmse_fail_m=cfg.rmse_fail_m,
    )
    return AlignmentHypothesis(
        relative_pose=pose.project_se2(),
        pose_se3=pose.to_dict(),
        overlap_ratio=overlap_ratio,
        inlier_ratio=inlier_ratio,
        comparable_area_ratio=overlap_ratio,
        num_correspondences=n_corr,
        rmse_m=rmse,
        success=success,
        score=inlier_ratio,
        failure_reason=None if success else "overlap insufficient",
        failure_class=failure_class,
        degeneracy=degeneracy,
    )


class SE3Aligner(Aligner):
    """SE3 (6-DoF) ICP baseline aligner (§10.3)."""

    name = "se3-icp"

    def __init__(self, config: SE3AlignConfig | None = None) -> None:
        self.config = config or SE3AlignConfig()

    def align(self, query: Scene, reference: Scene) -> AlignmentHypothesis:
        return align_se3(query.primary().xyz(), reference.primary().xyz(), self.config)
