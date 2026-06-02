"""SE2 alignment via BEV cross-correlation (§10.2, §10.3).

Yaw is searched by brute force over the circle; for each yaw the query BEV is
rotated and FFT cross-correlated against the reference BEV to recover (x, y).
The convention is ``p_hist = relative_pose.transform(p_query)``.

Alignment failure is a first-class result (§10.5): insufficient overlap yields
``success=False`` with a reason rather than a silently wrong pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bevmatch.core.datamodel import AlignmentHypothesis, Pose2D, _wrap_angle
from bevmatch.representations.bev import BEVConfig, points_to_bev


@dataclass(frozen=True)
class SE2AlignConfig:
    bev: BEVConfig = BEVConfig()
    yaw_step_deg: float = 2.0
    occ_threshold: float = 0.5
    min_overlap_ratio: float = 0.30
    min_points: int = 20
    icp_iters: int = 25
    icp_max_dist_m: float = 1.5  # correspondence gate (shrinks over iterations)


def _rotate(points_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return points_xy @ R.T


def _umeyama_se2(src: np.ndarray, dst: np.ndarray) -> Pose2D:
    """Least-squares rigid SE2 mapping ``src`` onto ``dst`` (Umeyama, no scale)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection guard
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return Pose2D(float(t[0]), float(t[1]), float(np.arctan2(R[1, 0], R[0, 0])))


def _icp_se2(query: np.ndarray, ref: np.ndarray, init: Pose2D, cfg: SE2AlignConfig) -> tuple[Pose2D, float]:
    """Refine ``init`` (query->ref) by point-to-point ICP. Returns (pose, inliers)."""
    pose = init
    inlier_ratio = 0.0
    for i in range(cfg.icp_iters):
        moved = pose.transform(query)
        # nearest ref neighbour per moved query point (brute force; small N)
        d2 = ((moved[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        nn = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(moved)), nn])
        gate = max(cfg.icp_max_dist_m * (1.0 - 0.6 * i / max(1, cfg.icp_iters)), 0.4)
        keep = dist < gate
        inlier_ratio = float(keep.mean())
        if keep.sum() < 3:
            break
        new_pose = _umeyama_se2(query[keep], ref[nn[keep]])
        moved_new = new_pose.transform(query)
        if np.allclose(moved_new, moved, atol=1e-4):
            pose = new_pose
            break
        pose = new_pose
    return pose, inlier_ratio


def _dilate(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation without scipy."""
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
    return out


def align_se2(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    config: SE2AlignConfig | None = None,
) -> AlignmentHypothesis:
    """Estimate the SE2 transform mapping ``query_xy`` into the ``ref_xy`` frame."""
    cfg = config or SE2AlignConfig()
    q = np.asarray(query_xy, dtype=float)[:, :2]
    r = np.asarray(ref_xy, dtype=float)[:, :2]

    if len(q) < cfg.min_points or len(r) < cfg.min_points:
        return AlignmentHypothesis(
            relative_pose=Pose2D(),
            overlap_ratio=0.0,
            inlier_ratio=0.0,
            success=False,
            failure_reason="insufficient geometric constraints",
        )

    size = cfg.bev.size
    res = cfg.bev.resolution_m
    ref_grid = points_to_bev(r, cfg.bev).occupied(cfg.occ_threshold).astype(float)
    f_ref = np.fft.rfft2(ref_grid)
    ref_norm = np.sqrt(ref_grid.sum()) + 1e-9

    best = {"score": -np.inf, "yaw": 0.0, "drow": 0, "dcol": 0}
    for deg in np.arange(0.0, 360.0, cfg.yaw_step_deg):
        yaw = np.deg2rad(deg)
        q_grid = points_to_bev(_rotate(q, yaw), cfg.bev).occupied(cfg.occ_threshold).astype(float)
        denom = (np.sqrt(q_grid.sum()) + 1e-9) * ref_norm
        # correlation[s] = sum_x ref[x] * q[x - s]  ->  IFFT(F_ref * conj(F_q))
        corr = np.fft.irfft2(f_ref * np.conj(np.fft.rfft2(q_grid)), s=ref_grid.shape)
        idx = int(np.argmax(corr))
        drow, dcol = np.unravel_index(idx, corr.shape)
        score = corr[drow, dcol] / denom
        if score > best["score"]:
            drow = drow - size if drow > size // 2 else drow
            dcol = dcol - size if dcol > size // 2 else dcol
            best = {"score": float(score), "yaw": float(yaw), "drow": int(drow), "dcol": int(dcol)}

    coarse = Pose2D(
        x=best["dcol"] * res,
        y=best["drow"] * res,
        yaw=_wrap_angle(best["yaw"]),
    )

    # Refine the cell/degree-quantised coarse estimate to sub-cell accuracy so
    # that matched structure cancels and only genuine changes survive the diff.
    rel, icp_inliers = _icp_se2(q, r, coarse, cfg)

    # Quantify overlap/inliers with the chosen transform.
    q_in_ref = rel.transform(q)
    q_grid = _dilate(points_to_bev(q_in_ref, cfg.bev).occupied(cfg.occ_threshold))
    ref_dil = _dilate(ref_grid.astype(bool))
    inter = int(np.count_nonzero(q_grid & ref_dil))
    n_q = int(np.count_nonzero(q_grid))
    n_ref = int(np.count_nonzero(ref_dil))
    overlap_ratio = inter / max(1, min(n_q, n_ref))
    inlier_ratio = inter / max(1, n_q)

    success = overlap_ratio >= cfg.min_overlap_ratio
    return AlignmentHypothesis(
        relative_pose=rel,
        overlap_ratio=overlap_ratio,
        inlier_ratio=max(inlier_ratio, icp_inliers),
        success=success,
        score=best["score"],
        failure_reason=None if success else "overlap insufficient",
    )
