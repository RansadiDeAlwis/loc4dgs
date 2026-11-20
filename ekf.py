"""EKF and iterated EKF baselines on SE(3) (guide Step 8, Section 3.9).

State: pose T with a right-multiplied error, T_true = T Exp(delta), delta ~ N(0, P) (6x6).
Measurement: blurred, normalized grayscale image (M values); Jacobian H (M x 6) by central
finite differences (12 extra renders, batched).  Update in INFORMATION form, so only 6x6
matrices are inverted (the M x M innovation covariance is never formed):

    P+ = (P-^-1 + H^T H / s_z^2)^-1,   delta = P+ H^T (z - h(T-)) / s_z^2,   T+ = T- Exp(delta)

iters=1 is the EKF; iters=3..5 is the iterated EKF (Gauss-Newton on the posterior, close to
what iNeRF / iComMa do with gradients).
"""
import torch

from se3 import se3_exp, se3_inv, se3_log, adjoint
from render import render_batch
from observations import gaussian_blur


def gray(x):  # (...,H,W,3) -> (...,H*W), normalized
    g = (x * torch.tensor([0.299, 0.587, 0.114], device=x.device)).sum(-1).flatten(-2)
    return (g - g.mean(-1, keepdim=True)) / (g.std(-1, keepdim=True) + 1e-6)


class PoseEKF:
    def __init__(self, T0, P0, Q, sigma_z, eps=1e-3, iters=1, blur_sigma=0.0):
        self.T, self.P, self.Q = T0.clone().float(), P0.clone().float(), Q.float()
        self.s2, self.eps, self.iters, self.blur = sigma_z ** 2, eps, iters, blur_sigma
        self.n_renders = 0

    def predict(self, U=None):
        """T- = T U~,   P- = Ad_{U~^-1} P Ad_{U~^-1}^T + Q   (U=None: no motion, P- = P + Q)."""
        if U is not None:
            self.T = self.T @ U
            Ad = adjoint(se3_inv(U))
            self.P = Ad @ self.P @ Ad.T + self.Q
        else:
            self.P = self.P + self.Q

    def _h(self, fr, poses, K, W, H):
        return gray(gaussian_blur(render_batch(fr, se3_inv(poses), K, W, H), self.blur))

    def update(self, fr, obs, K, W, H):
        z = gray(gaussian_blur(obs, self.blur))
        T_prior, P_prior_inv = self.T.clone(), torch.linalg.inv(self.P)
        T = T_prior.clone()
        E = torch.eye(6, device=T.device) * self.eps
        for _ in range(self.iters):                               # iters=1: EKF, >1: iterated EKF
            poses = torch.cat([T[None], T[None] @ se3_exp(torch.cat([E, -E]))])
            h = self._h(fr, poses, K, W, H)                       # 13, M
            self.n_renders += 13
            Hj = ((h[1:7] - h[7:13]) / (2 * self.eps)).T          # M, 6
            r = z - h[0]
            d = se3_log(se3_inv(T_prior) @ T)                     # current offset from prior
            Pinv = P_prior_inv + Hj.T @ Hj / self.s2
            delta = torch.linalg.solve(Pinv, Hj.T @ r / self.s2 - P_prior_inv @ d)
            T = T @ se3_exp(delta)
        self.T, self.P = T, torch.linalg.inv(Pinv)
        self.P = 0.5 * (self.P + self.P.T)                        # keep it symmetric
        self.err = (r ** 2).mean()
        return self.T

    def estimate(self):
        return self.T, self.P
