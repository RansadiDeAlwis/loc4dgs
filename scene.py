"""Load a pretrained Dynamic 3D Gaussians scene and build the map G(t) (guide Step 2, Section 2.2).

G(t) = { mu_k(t), q_k(t), s_k, alpha_k, c_k }:  only centres and rotations move with t.
The full arrays (~2 GB) stay on the CPU; one frame at a time is uploaded to the GPU
(set preload=True on a big GPU to keep everything on the device for speed).
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

import config


@dataclass
class Frame:                        # the map G(t), ready for rendering (activations applied)
    means: torch.Tensor             # (N,3)
    quats: torch.Tensor             # (N,4) wxyz, normalized
    scales: torch.Tensor            # (N,3) = exp(log_scales)
    opac: torch.Tensor              # (N,)  = sigmoid(logit_opacities)
    colors: torch.Tensor            # (N,3) used as stored (no sigmoid)
    is_fg: torch.Tensor = None      # (N,) bool, foreground (people / objects)


class Scene4D:
    def __init__(self, params_path, device="cuda", preload=False):
        p = dict(np.load(params_path))
        self.dev = device
        self.means = p["means3D"].astype(np.float32)                          # T,N,3 (CPU)
        self.rots = p["unnorm_rotations"].astype(np.float32)                  # T,N,4 (wxyz)
        cols = p["rgb_colors"].astype(np.float32)                             # T,N,3 (or N,3)
        self.cols = cols if cols.ndim == 3 else np.broadcast_to(cols, self.means.shape)
        self.opac = torch.sigmoid(torch.tensor(p["logit_opacities"])).reshape(-1).float().to(device)
        self.scales = torch.exp(torch.tensor(p["log_scales"])).float().to(device)
        self.is_fg = torch.tensor(p["seg_colors"][:, 0] > 0.5).to(device)
        self.cam_m = p.get("cam_m")      # per-camera colour correction (TRAINING cameras only)
        self.cam_c = p.get("cam_c")
        self.T = self.means.shape[0]
        self.N = self.means.shape[1]
        self._gpu = None
        if preload:
            self._gpu = (torch.from_numpy(self.means).to(device),
                         torch.from_numpy(self.rots).to(device),
                         torch.from_numpy(np.ascontiguousarray(self.cols)).to(device))
        self._cache = {}

    def _arrays(self, t):
        if self._gpu is not None:
            return self._gpu[0][t], self._gpu[1][t], self._gpu[2][t]
        return (torch.from_numpy(self.means[t]).to(self.dev),
                torch.from_numpy(self.rots[t]).to(self.dev),
                torch.from_numpy(np.array(self.cols[t])).to(self.dev))

    def frame(self, t, mode="full", offset=0):
        """mode: full | background | frozen0.  offset shifts time for the ablation (E3)."""
        if mode == "frozen0":
            t = 0
        t = int(np.clip(t + offset, 0, self.T - 1))
        key = (t, mode)
        if key in self._cache:
            return self._cache[key]
        m, q, c = self._arrays(t)
        q = F.normalize(q, dim=-1)
        keep = ~self.is_fg if mode == "background" else torch.ones_like(self.is_fg)
        fr = Frame(m[keep], q[keep], self.scales[keep], self.opac[keep], c[keep], self.is_fg[keep])
        self._cache = {key: fr}             # keep only the latest frame (GPU memory)
        return fr

    def fg_frame(self, t):
        """Same geometry as frame(t) but coloured white = foreground, black = background.
        Rendering it gives the fraction of pixels covered by people (E3 analysis)."""
        fr = self.frame(t, "full")
        col = fr.is_fg.float()[:, None].expand(-1, 3).contiguous()
        return Frame(fr.means, fr.quats, fr.scales, fr.opac, col, fr.is_fg)

    def background_points(self, t=0, n=5000, seed=0):
        """Subsampled background Gaussian centres for plots (F3)."""
        idx = np.where(~self.is_fg.cpu().numpy())[0]
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=min(n, idx.size), replace=False)
        return self.means[t][idx], self.cols[t][idx]


def load_cameras(meta_path):
    md = json.load(open(meta_path))
    return md   # use md["k"][t][c], md["w2c"][t][c], md["fn"][t][c], md["w"], md["h"], md["cam_id"][t][c]


def camera(md, t, c, device="cuda"):
    """-> K (3,3) at md['w'] x md['h'], c2w (4,4), filename, cam_id."""
    K = torch.tensor(md["k"][t][c], dtype=torch.float32, device=device)
    w2c = torch.tensor(md["w2c"][t][c], dtype=torch.float32, device=device)
    c2w = torch.linalg.inv(w2c)
    return K, c2w, md["fn"][t][c], md["cam_id"][t][c]


def camera_centers(md, t=0):
    """Camera centres c = -R^T t from each w2c (guide Section 4.5, gotcha 3)."""
    out = []
    for w2c in md["w2c"][t]:
        w2c = np.array(w2c)
        out.append(-w2c[:3, :3].T @ w2c[:3, 3])
    return np.stack(out)


def unit_report(md, t=0):
    """Print camera-to-camera distances so you can decide whether scene units are metres."""
    c = camera_centers(md, t)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    mid = c.mean(0)
    r = np.linalg.norm(c - mid, axis=-1)
    msg = (f"{len(c)} cameras | max pairwise distance {d.max():.3f} | "
           f"mean radius from centroid {r.mean():.3f} | "
           f"nearest-neighbour distance (median) {np.median(np.sort(d, 1)[:, 1]):.3f}")
    print(msg)
    print("The Panoptic dome is several metres across: if the max distance is ~5-6, units are metres; "
          "otherwise report errors in scene units and state the conversion.")
    return dict(max_dist=d.max(), mean_radius=r.mean())


def load_scene(name, device="cuda", preload=False):
    """Factory: real Panoptic scene, or the synthetic fallback world (Section 10.4).
    Returns (scene, train_meta, test_meta)."""
    if name == config.SYNTHETIC:
        from synthetic import SyntheticScene4D
        s = SyntheticScene4D(device=device)
        return s, s.train_meta, s.test_meta
    scene = Scene4D(config.params_path(name), device=device, preload=preload)
    return scene, load_cameras(config.meta_path(name, "train")), load_cameras(config.meta_path(name, "test"))
