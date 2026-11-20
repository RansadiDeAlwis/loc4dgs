"""Image preprocessing, the photometric error and synthetic observation corruption (guide Step 4, 3.4).

Every image is normalized per colour channel (zero mean, unit std) before comparison. This
removes brightness / contrast differences between the map and the camera (the held-out test
cameras have no learned colour correction, Section 4.5 gotcha 1).  With normalization the
error of an empty render is ~1 and the error is 2(1 - rho) per channel (rho = correlation).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from render import render_batch
from se3 import se3_inv


def load_obs(path, W, H, device="cuda"):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return torch.tensor(img, device=device).float() / 255.0          # H,W,3


def load_mask(path, W, H, device="cuda"):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    return torch.tensor(m, device=device).float() / 255.0            # H,W


def normalize(img, eps=1e-6):  # per image, per channel: zero mean, unit std
    dims = (-3, -2)
    return (img - img.mean(dims, keepdim=True)) / (img.std(dims, keepdim=True) + eps)


def photometric_error(renders, obs, trim=0.0, loss="mse", huber_delta=1.0):
    """renders: (B,H,W,3), obs: (H,W,3) -> (B,) mean (robust) squared error after normalization.

    trim > 0  : drop the worst `trim` fraction of pixels per particle (trimmed mean).
    loss=huber: Huber penalty instead of the squared error (Section 3.4, robust variant).
    """
    r = normalize(renders) - normalize(obs)[None]
    if loss == "huber":
        a = r.abs()
        se = torch.where(a <= huber_delta, 0.5 * r ** 2, huber_delta * (a - 0.5 * huber_delta)) * 2
    else:
        se = r ** 2
    se = se.mean(-1).flatten(1)                                         # B,HW
    if trim > 0:                                                        # robust: drop the worst pixels
        k = int(se.shape[1] * (1 - trim))
        se, _ = se.topk(k, dim=1, largest=False)
    return se.mean(1)


def downsample(img, W, H):
    """Area interpolation of (H0,W0,3) or (B,H0,W0,3) torch images to (…,H,W,3)."""
    x = img if img.dim() == 4 else img[None]
    x = F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode="area").permute(0, 2, 3, 1)
    return x if img.dim() == 4 else x[0]


def gaussian_blur(img, sigma):
    """Separable Gaussian blur of (…,H,W,C) torch images (used by the EKF)."""
    if sigma <= 0:
        return img
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=img.dtype) / sigma) ** 2)
    k = k / k.sum()
    shp = img.shape
    x = img.reshape(-1, *shp[-3:]).permute(0, 3, 1, 2)                # B,C,H,W
    C = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)
    return x.permute(0, 2, 3, 1).reshape(shp)


def corrupt(img, noise=0.02, blur=0, occluders=0, rng=np.random, coverage=0.0):
    """Make a synthetic observation look less like a perfect render.
    occluders: number of random grey rectangles; coverage: add rectangles until at least this
    fraction of the image is covered (E6)."""
    x = img.clone()
    if blur > 0:
        k = 2 * blur + 1
        x = torch.tensor(cv2.GaussianBlur(x.cpu().numpy(), (k, k), 0), device=img.device)
    x = x * (0.85 + 0.3 * rng.rand()) + 0.05 * (rng.rand() - 0.5)       # exposure change
    x = x + noise * torch.randn_like(x)
    H, W = x.shape[:2]
    for _ in range(occluders):                                          # random grey rectangles
        h, w = rng.randint(H // 8, H // 3), rng.randint(W // 8, W // 3)
        y0, x0 = rng.randint(0, H - h), rng.randint(0, W - w)
        x[y0:y0 + h, x0:x0 + w] = rng.rand()
    if coverage > 0:
        covered = torch.zeros(H, W, dtype=torch.bool, device=x.device)
        while covered.float().mean() < coverage:
            h, w = rng.randint(H // 8, H // 2), rng.randint(W // 8, W // 2)
            y0, x0 = rng.randint(0, H - h), rng.randint(0, W - w)
            x[y0:y0 + h, x0:x0 + w] = rng.rand()
            covered[y0:y0 + h, x0:x0 + w] = True
    return x.clamp(0, 1)


def synth_observation(fr, c2w, K_full, W0, H0, W, H, rng, obs_wh=(320, 180), **corrupt_kw):
    """Synthetic observation (Step 4): render the ground-truth pose at a HIGHER resolution,
    area-downsample to the filter resolution, then corrupt it.  This reduces the inverse crime."""
    from render import scale_K
    Wo, Ho = obs_wh
    Ko = scale_K(K_full, W0, H0, Wo, Ho)
    img = render_batch(fr, se3_inv(c2w)[None], Ko, Wo, Ho)[0]
    return corrupt(downsample(img, W, H), rng=rng, **corrupt_kw)
