"""Particle filter on SE(3) with a render-and-compare likelihood (guide Step 6, Sections 2.4, 3.3-3.8).

One SIR step:  predict (motion model + right-multiplied noise)  ->  update (render every
particle from G(t), tempered likelihood, log-space weights)  ->  resample if N_eff < N/2.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from se3 import se3_exp, se3_inv, se3_log
from render import render_batch
from observations import photometric_error


def systematic_resample(w, n_out):
    """u ~ U[0,1/N), points u + k/N, pick the particle whose cumulative-weight interval contains
    each point.  O(N), low variance, keeps every particle whose weight exceeds 1/N."""
    u = (torch.rand(1, device=w.device) + torch.arange(n_out, device=w.device)) / n_out
    c = torch.cumsum(w, 0); c[-1] = 1.0
    return torch.searchsorted(c, u).clamp(max=w.numel() - 1)


class ParticleFilter:
    def __init__(self, X0, sig_p, sig_r, lam, trim=0.0, likelihood="tempered", loss="mse",
                 chunk=None):
        """likelihood: 'tempered' -> log p = -lam * e        (Section 3.4)
                       'locnerf'  -> w = (M / sum r^2)^4, i.e. log p = -4 log e  (Loc-NeRF heuristic)
           loss: 'mse' | 'huber';  trim: fraction of worst pixels ignored (robust variant)."""
        self.X = X0.clone()                                       # N,4,4 particle poses (c2w)
        self.N = X0.shape[0]
        self.logw = torch.full((self.N,), -np.log(self.N), device=X0.device)
        self.sig_p, self.sig_r, self.lam, self.trim = sig_p, sig_r, lam, trim
        self.likelihood, self.loss, self.chunk = likelihood, loss, chunk
        self.err = None
        self.n_renders = 0
        self.bad_steps = 0                                        # consecutive steps with a poor best error

    @property
    def w(self):
        return torch.softmax(self.logw, 0)                        # stable normalization

    def ess(self):
        return 1.0 / (self.w ** 2).sum().item()

    def predict(self, U=None, scale=1.0):
        """x_t^i = x_{t-1}^i U~_t Exp(delta^i), delta^i ~ N(0, Q).  U=None: random walk only."""
        if U is not None:
            self.X = self.X @ U                                   # odometry in the body frame
        d = self.X.device
        xi = torch.cat([torch.randn(self.N, 3, device=d) * self.sig_p * scale,
                        torch.randn(self.N, 3, device=d) * self.sig_r * scale], -1)
        self.X = self.X @ se3_exp(xi)

    def loglik(self, err):
        if self.likelihood == "locnerf":
            return -4.0 * torch.log(err.clamp(min=1e-8))
        return -self.lam * err

    def update(self, frame, obs, K, W, H):
        imgs = render_batch(frame, se3_inv(self.X), K, W, H, chunk=self.chunk)
        self.n_renders += self.N
        self.err = photometric_error(imgs, obs, self.trim, self.loss)    # N
        # l^i = log w_{t-1}^i - lam e^i ;  normalized with the max-subtraction trick (log_softmax)
        self.logw = torch.log_softmax(self.logw + self.loglik(self.err), 0)
        return imgs

    def resample(self, n_out=None):
        n_out = n_out or self.N
        idx = systematic_resample(self.w, n_out)
        self.X, self.N = self.X[idx], n_out
        if self.err is not None:
            self.err = self.err[idx]
        self.logw = torch.full((n_out,), -np.log(n_out), device=self.X.device)

    def inject(self, sampler, frac=0.1):                          # recovery from wrong convergence
        k = int(frac * self.N)
        idx = torch.randperm(self.N, device=self.X.device)[:k]
        self.X[idx] = sampler(k)
        self.logw = torch.full((self.N,), -np.log(self.N), device=self.X.device)

    def maybe_inject(self, sampler, err_thr, patience, frac=0.1):
        """Random particle injection (Section 3.6): if the best particle's error stays above
        err_thr for `patience` consecutive steps, replace a fraction of the particles."""
        if self.err is None:
            return False
        self.bad_steps = self.bad_steps + 1 if self.err.min().item() > err_thr else 0
        if self.bad_steps >= patience:
            self.inject(sampler, frac)
            self.bad_steps = 0
            return True
        return False

    def estimate(self):
        """Weighted-mean position, Markley quaternion average, tangent-space covariance (3.8)."""
        w = self.w
        p = (w[:, None] * self.X[:, :3, 3]).sum(0)
        # SciPy's Rotation.mean(weights) = Markley's eigenvector method (sign-safe)
        R = Rot.from_matrix(self.X[:, :3, :3].double().cpu().numpy()).mean(weights=w.double().cpu().numpy())
        T = torch.eye(4, device=self.X.device)
        T[:3, :3] = torch.tensor(R.as_matrix(), device=self.X.device, dtype=torch.float32)
        T[:3, 3] = p
        eps = se3_log(se3_inv(T)[None] @ self.X)                  # N,6 in the tangent space
        P = (w[:, None, None] * eps[:, :, None] @ eps[:, None, :]).sum(0)
        return T, P

    def spread(self):                                             # used by annealing
        w = self.w
        p = self.X[:, :3, 3]
        mu = (w[:, None] * p).sum(0)
        return torch.sqrt((w * ((p - mu) ** 2).sum(-1)).sum()).item()

    def best(self):
        """Highest-weight particle (useful for visualizing renders)."""
        return self.X[torch.argmax(self.logw)]
