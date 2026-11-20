"""Tiny synthetic 4D world: the fallback of guide Section 10.4.

A "sports hall" of a few hundred coloured 3D Gaussians (floor, walls, a few static props)
plus three moving "people" and a juggled ball on scripted paths over T = 150 frames, seen
by 31 cameras on a dome-like ring looking inward (same camera indexing as Panoptic, so test
cameras 0, 10, 15, 30 are held out).  It has exactly the same interface as scene.Scene4D,
so every script, experiment and plot runs unchanged with `--scene synthetic`.
World frame: z up.  Cameras: OpenCV convention (x right, y down, z forward).
"""
import numpy as np
import torch

import config
from scene import Scene4D
from se3 import look_at, rotmat_to_quat
from scipy.spatial.transform import Rotation as Rot


def _make_cameras(n=31, radius=2.8, center=(0, 0, 0.9), seed=0):
    rng = np.random.default_rng(seed)
    az = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.05, 0.05, n)
    h = 1.0 + 1.2 * (np.arange(n) % 3) / 2 + rng.uniform(-0.1, 0.1, n)
    r = radius + rng.uniform(-0.15, 0.15, n)
    pos = torch.tensor(np.stack([r * np.cos(az), r * np.sin(az), h], -1), dtype=torch.float64)
    tgt = torch.tensor(center, dtype=torch.float64).expand_as(pos)
    up = torch.tensor([0, 0, 1.0], dtype=torch.float64).expand_as(pos)
    return look_at(pos, tgt, up).numpy()                                    # n,4,4 c2w


class SyntheticScene4D(Scene4D):
    def __init__(self, device="cuda", T=150, seed=0):
        rng = np.random.default_rng(seed)
        self.dev = device
        means, scales, cols, rots, fg = [], [], [], [], []

        def add(mu, sc, col, q=None, is_fg=False):
            mu = np.atleast_2d(mu); n = mu.shape[-2]
            means.append(np.broadcast_to(mu, (T, n, 3)) if mu.ndim == 2 else mu)
            scales.append(np.broadcast_to(sc, (n, 3)))
            cols.append(np.broadcast_to(col, (n, 3)))
            q = np.tile([1, 0, 0, 0.0], (n, 1)) if q is None else q
            rots.append(np.broadcast_to(q, (T, n, 4)))
            fg.append(np.full(n, is_fg))

        # floor: checkerboard disk (flat Gaussians)
        g = np.arange(-3.0, 3.01, 0.3)
        X, Y = np.meshgrid(g, g)
        m = X ** 2 + Y ** 2 < 3.2 ** 2
        fl = np.stack([X[m], Y[m], np.zeros(m.sum())], -1)
        chk = ((np.floor(X[m] / 0.6) + np.floor(Y[m] / 0.6)) % 2)[:, None]
        add(fl, [0.17, 0.17, 0.02], chk * [0.8, 0.75, 0.6] + (1 - chk) * [0.35, 0.25, 0.2])
        # walls: cylinder radius 3.6 with coloured vertical bands (gives azimuth information)
        az = np.arange(0, 2 * np.pi, 0.12)
        zz = np.arange(0.15, 2.6, 0.3)
        A, Z = np.meshgrid(az, zz)
        wall = np.stack([3.6 * np.cos(A.ravel()), 3.6 * np.sin(A.ravel()), Z.ravel()], -1)
        hue = (A.ravel() / (2 * np.pi))
        wc = np.stack([0.5 + 0.45 * np.cos(2 * np.pi * hue), 0.5 + 0.45 * np.cos(2 * np.pi * (hue - 1 / 3)),
                       0.5 + 0.45 * np.cos(2 * np.pi * (hue - 2 / 3))], -1)
        wc = wc * (0.75 + 0.25 * (Z.ravel()[:, None] > 1.3))
        qw = rotmat_to_quat(torch.tensor(Rot.from_euler("z", A.ravel()).as_matrix())).numpy()
        add(wall, [0.03, 0.2, 0.17], wc, q=qw)
        # static props: three boxes
        for c, col in [((1.6, 0.4, 0.3), [0.9, 0.2, 0.2]), ((-1.2, 1.5, 0.25), [0.2, 0.3, 0.9]),
                       ((-0.6, -1.7, 0.35), [0.95, 0.85, 0.1])]:
            pts = np.array(c) + rng.uniform(-0.25, 0.25, (20, 3)) * [1, 1, 0.8]
            add(pts, [0.12, 0.12, 0.12], col)
        # moving people: vertical clusters on circular paths; plus a juggled ball
        t = np.arange(T) / T
        for k, (rad, ph, spd, col) in enumerate([(0.6, 0.0, 1.0, [0.95, 0.5, 0.1]),
                                                 (1.1, 2.1, -0.7, [0.1, 0.8, 0.3]),
                                                 (0.9, 4.2, 1.3, [0.8, 0.1, 0.8])]):
            body = np.stack([rng.normal(0, 0.12, 30), rng.normal(0, 0.12, 30), rng.uniform(0.1, 1.7, 30)], -1)
            ang = ph + 2 * np.pi * spd * t
            ctr = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.zeros(T)], -1)
            add(ctr[:, None] + body[None], [0.12, 0.12, 0.2], col * (0.8 + 0.2 * rng.random((30, 1))), is_fg=True)
        ball = np.stack([0.6 * np.cos(2 * np.pi * t) + 0.3, 0.6 * np.sin(2 * np.pi * t),
                         1.6 + 0.5 * np.abs(np.sin(6 * np.pi * t))], -1)
        add(ball[:, None] + rng.normal(0, 0.03, (1, 5, 3)), [0.08, 0.08, 0.08], [1.0, 1.0, 1.0], is_fg=True)

        self.means = np.concatenate(means, 1).astype(np.float32)
        self.rots = np.concatenate(rots, 1).astype(np.float32)
        col = np.concatenate(cols, 0).astype(np.float32)
        self.cols = np.broadcast_to(col, self.means.shape)
        N = self.means.shape[1]
        self.opac = torch.full((N,), 0.95, device=device)
        self.scales = torch.tensor(np.concatenate(scales, 0), dtype=torch.float32, device=device)
        self.is_fg = torch.tensor(np.concatenate(fg), device=device)
        self.cam_m = self.cam_c = None
        self.T, self.N = T, N
        self._gpu, self._cache = None, {}

        # camera metadata in the Dynamic3DGaussians format
        c2w = _make_cameras(seed=seed)
        fx = 0.75 * config.FULL_W
        K = [[fx, 0, config.FULL_W / 2], [0, fx, config.FULL_H / 2], [0, 0, 1]]
        test = config.TEST_CAMS
        train = [c for c in range(len(c2w)) if c not in test]

        def meta(ids):
            w2c = [np.linalg.inv(c2w[c]).tolist() for c in ids]
            return dict(w=config.FULL_W, h=config.FULL_H, k=[[K] * len(ids)] * T, w2c=[w2c] * T,
                        fn=[[None] * len(ids)] * T, cam_id=[list(ids)] * T)
        self.train_meta, self.test_meta = meta(train), meta(test)
