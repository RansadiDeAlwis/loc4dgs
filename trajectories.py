"""Synthetic camera trajectories, odometry noise and the pose priors (guide Step 5, Sections 3.3, 3.7).

The Panoptic cameras do not move, so moving cameras are created by smoothly interpolating
between neighbouring real cameras (SLERP for rotation, linear for position).  The scene
looks good from these in-between poses because they stay inside the trained region.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot, Slerp

from se3 import se3_exp, look_at


# --------------------------------------------------------------------------- dome geometry
def cam_centers_and_c2w(md, t=0):
    c2ws = [np.linalg.inv(np.array(w2c)) for w2c in md["w2c"][t]]
    return np.stack(c2ws)                                                # C,4,4


def dome_frame(c2ws):
    """World up, ring centre and two horizontal axes (a, b) of the camera dome."""
    centers = c2ws[:, :3, 3]
    up = -c2ws[:, :3, 1].mean(0); up /= np.linalg.norm(up)              # OpenCV: camera y points down
    mid = centers.mean(0)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(up, ref); a /= np.linalg.norm(a); b = np.cross(up, a)
    return up, mid, a, b


def scene_center(c2ws):
    """Point closest (least squares) to all camera optical axes = where the dome looks."""
    A, rhs = np.zeros((3, 3)), np.zeros(3)
    for T in c2ws:
        d = T[:3, 2] / np.linalg.norm(T[:3, 2]); P = np.eye(3) - np.outer(d, d)
        A += P; rhs += P @ T[:3, 3]
    return np.linalg.solve(A, rhs)


def sorted_by_azimuth(c2ws):
    up, mid, a, b = dome_frame(c2ws)
    centers = c2ws[:, :3, 3]
    az = np.arctan2((centers - mid) @ b, (centers - mid) @ a)
    return np.argsort(az)                                                # cameras sorted around the dome


def angular_gaps(c2ws, order):
    """Angle (deg) between viewing directions of consecutive cameras in `order`."""
    z = c2ws[order][:, :3, 2]
    return np.rad2deg(np.arccos(np.clip((z[:-1] * z[1:]).sum(-1), -1, 1)))


def best_starts(c2ws, n_cams=5, k=3, max_gap_deg=None):
    """Start indices (into the azimuth order) whose n_cams consecutive cameras have the smallest
    maximum angular spacing (guide: 'choose consecutive cameras whose angular spacing is small')."""
    order = sorted_by_azimuth(c2ws)
    gaps = angular_gaps(c2ws, order)
    scores = [gaps[s:s + n_cams - 1].max() for s in range(len(order) - n_cams + 1)]
    idx = np.argsort(scores)
    chosen = []
    for s in idx:                                                        # prefer non-overlapping segments
        if all(abs(s - c) >= n_cams for c in chosen):
            if max_gap_deg is None or scores[s] <= max_gap_deg:
                chosen.append(int(s))
        if len(chosen) == k:
            break
    return chosen, [float(scores[s]) for s in chosen]


def make_trajectory(c2ws, n_cams=5, start=0, n_frames=150, wobble=0.0, seed=0):
    rng = np.random.default_rng(seed)
    order = sorted_by_azimuth(c2ws)
    keys = c2ws[order[start:start + n_cams]]
    s_key = np.linspace(0, 1, n_cams)
    s = np.linspace(0, 1, n_frames)
    slerp = Slerp(s_key, Rot.from_matrix(keys[:, :3, :3]))
    T = np.tile(np.eye(4), (n_frames, 1, 1))
    T[:, :3, :3] = slerp(s).as_matrix()
    for j in range(3):
        T[:, j, 3] = np.interp(s, s_key, keys[:, j, 3])
    if wobble > 0:                                                       # handheld-like smooth shake
        T[:, :3, 3] += wobble * np.cumsum(rng.normal(size=(n_frames, 3)), 0) / np.sqrt(n_frames)
    return T                                                             # ground-truth c2w per frame


def relative_motions(T_gt):
    """True relative motions U_t = T_{t-1}^-1 T_t (U_0 = I)."""
    U = [np.eye(4)]
    for t in range(1, len(T_gt)):
        U.append(np.linalg.inv(T_gt[t - 1]) @ T_gt[t])
    return np.stack(U)


def odometry(T_gt, sig_p, sig_r, rng, U_true=None):
    """Noisy relative motions U~_t = (T_{t-1}^-1 T_t) Exp(noise).
    U_true can override the true relative motions (used for the kidnapping, E7)."""
    U_true = relative_motions(T_gt) if U_true is None else U_true
    U = [np.eye(4)]
    for t in range(1, len(T_gt)):
        n = np.concatenate([rng.normal(0, sig_p, 3), rng.normal(0, sig_r, 3)])
        U.append(U_true[t] @ se3_exp(torch.tensor(n)).numpy())
    return np.stack(U)


def kidnap_trajectory(T_a, T_b, frame):
    """E7: the true camera follows T_a until `frame`, then jumps to T_b (another segment).
    Returns the ground truth and the relative motions the odometry 'thinks' happened (the
    jump itself is hidden: at `frame` the odometry reports T_a's own motion)."""
    T = np.concatenate([T_a[:frame], T_b[frame:]])
    U = relative_motions(T)
    U[frame] = np.linalg.inv(T_a[frame - 1]) @ T_a[frame]
    return T, U


# --------------------------------------------------------------------------- priors (Section 3.7)
def local_prior(T_init, n, sig_p, sig_r, gen=None):
    """x_0^i = T_init Exp(delta^i), delta^i ~ N(0, diag(sig_p^2 I, sig_r^2 I))."""
    d = T_init.device
    xi = torch.cat([torch.randn(n, 3, device=d, generator=gen) * sig_p,
                    torch.randn(n, 3, device=d, generator=gen) * sig_r], -1)
    return T_init[None] @ se3_exp(xi)


class GlobalPrior:
    """Physically informed global prior: positions inside the region spanned by the training
    cameras (random azimuth, height and radius within their ranges, around the dome axis),
    orientations looking at the scene centre perturbed by uniform yaw/pitch/roll noise."""

    def __init__(self, train_c2ws, rot_noise_deg=20.0, device="cuda", seed=0):
        self.up, _, self.a, self.b = dome_frame(train_c2ws)
        self.center = scene_center(train_c2ws)
        rel = train_c2ws[:, :3, 3] - self.center
        h = rel @ self.up
        horiz = rel - h[:, None] * self.up
        self.r_rng = (np.linalg.norm(horiz, axis=-1).min(), np.linalg.norm(horiz, axis=-1).max())
        self.h_rng = (h.min(), h.max())
        az = np.sort(np.arctan2(horiz @ self.b, horiz @ self.a))
        gaps = np.diff(np.concatenate([az, az[:1] + 2 * np.pi]))
        g = np.argmax(gaps)                                               # largest empty sector
        self.az0, self.az_span = az[(g + 1) % len(az)], 2 * np.pi - gaps[g]
        self.rot_noise = np.deg2rad(rot_noise_deg)
        self.dev = device
        self.rng = np.random.default_rng(seed)

    def __call__(self, n):
        rng = self.rng
        az = self.az0 + rng.uniform(0, self.az_span, n)
        r = rng.uniform(*self.r_rng, n)
        h = rng.uniform(*self.h_rng, n)
        pos = self.center + r[:, None] * (np.cos(az)[:, None] * self.a + np.sin(az)[:, None] * self.b) \
            + h[:, None] * self.up
        T = look_at(torch.tensor(pos), torch.tensor(self.center).expand(n, 3),
                    torch.tensor(self.up).expand(n, 3))
        noise = Rot.from_euler("yxz", rng.uniform(-self.rot_noise, self.rot_noise, (n, 3))).as_matrix()
        T[:, :3, :3] = T[:, :3, :3] @ torch.tensor(noise)                 # yaw/pitch/roll in camera frame
        return T.float().to(self.dev)

    def describe(self):
        return (f"GlobalPrior: centre={np.round(self.center, 3)}, up={np.round(self.up, 3)}, "
                f"radius in [{self.r_rng[0]:.2f},{self.r_rng[1]:.2f}], "
                f"height in [{self.h_rng[0]:.2f},{self.h_rng[1]:.2f}], "
                f"azimuth span {np.rad2deg(self.az_span):.0f} deg, "
                f"rotation noise +-{np.rad2deg(self.rot_noise):.0f} deg (uniform yaw/pitch/roll)")
