"""Evaluation metrics and consistency checks (guide Step 9, Section 7).

No trajectory alignment is needed for ATE: the 4D map defines the world frame and the
ground-truth poses live in that same frame (Section 7.1).
"""
import numpy as np
import torch
from scipy.stats import chi2

from se3 import se3_inv, se3_log, rot_err_deg


def pose_errors(T_est, T_gt):
    te = (T_est[..., :3, 3] - T_gt[..., :3, 3]).norm(dim=-1)
    re = rot_err_deg(T_est[..., :3, :3], T_gt[..., :3, :3])
    return te, re


def tangent_error(T_est, T_gt):
    """e = Log(T_est^-1 T_gt) in R^6 (rho, phi)."""
    return se3_log(se3_inv(T_est) @ T_gt)


def ate_rmse(te):  # no alignment needed: the map fixes the world frame
    te = torch.as_tensor(te)
    return torch.sqrt((te ** 2).mean()).item()


def nees(T_est, P, T_gt, reg=1e-12):
    e = tangent_error(T_est, T_gt).double()
    P = P.double() + reg * torch.eye(6, dtype=torch.float64, device=P.device)
    return (e @ torch.linalg.solve(P, e)).item()


def nees_bounds(n_runs, dof=6, conf=0.95):
    a = (1 - conf) / 2
    return chi2.ppf([a, 1 - a], dof * n_runs) / n_runs   # band for the average NEES


def success(te, re, t_thr=0.05, r_thr=5.0):
    te, re = torch.as_tensor(te), torch.as_tensor(re)
    return ((te < t_thr) & (re < r_thr)).float().mean().item()


def covered(T_est, P, T_gt, k=2.0):
    """Distribution-free consistency check (Section 7.2): is the true position within k weighted
    standard deviations of the mean on every axis?  (Axes of the estimate's camera frame.)"""
    e = tangent_error(T_est, T_gt)
    sd = torch.sqrt(torch.diagonal(P).clamp(min=1e-12))
    return bool((e[:3].abs() <= k * sd[:3]).all().item())


def convergence_step(te, re, t_thr=0.05, r_thr=5.0):
    """First step after which the error STAYS under the success threshold (None if never)."""
    ok = (np.asarray(te) < t_thr) & (np.asarray(re) < r_thr)
    if not ok[-1]:
        return None
    bad = np.where(~ok)[0]
    return 0 if bad.size == 0 else int(bad[-1] + 1)


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return med, q1, q3
