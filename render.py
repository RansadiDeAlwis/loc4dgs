"""Batched renderer = the measurement function h(x) (guide Step 3, Section 2.1).

Backends
--------
* "gsplat": the CUDA rasterizer (default whenever gsplat + CUDA are available).
* "torch" : a small pure-PyTorch EWA splatter for the synthetic fallback world (Section 10.4).
            Brute force (every pixel x every Gaussian), so only use it for a few hundred
            Gaussians at ~64 x 36. It implements exactly the equations of Section 2.1.
Both take world-to-camera matrices (OpenCV: x right, y down, z forward) and return images
(B,H,W,3) in [0,1] on a black background.
"""
import os
import torch

import config

try:
    import gsplat
    HAVE_GSPLAT = True
except Exception:                                   # pragma: no cover
    gsplat = None
    HAVE_GSPLAT = False

_BACKEND = os.environ.get("LOC4DGS_BACKEND") or ("gsplat" if (HAVE_GSPLAT and torch.cuda.is_available()) else "torch")


# Dynamic3DGaussians' rasterizer culls every Gaussian closer than 0.2 to the camera, and the
# pretrained scenes contain floaters there (often at camera positions).  With a smaller near
# plane those floaters cover the whole view and renders come out black, so use the same 0.2.
NEAR_PLANE = 0.2


def set_backend(name):
    global _BACKEND
    assert name in ("gsplat", "torch")
    if name == "gsplat" and not HAVE_GSPLAT:
        raise RuntimeError("gsplat is not installed")
    _BACKEND = name


def get_backend():
    return _BACKEND


def scale_K(K, W0, H0, W, H):                       # K for the original W0xH0 -> K for WxH
    K = K.clone()
    K[0, :] *= W / W0
    K[1, :] *= H / H0
    return K


@torch.no_grad()
def render_batch(fr, w2c, K, W, H, chunk=None, backend=None):
    """fr: Frame, w2c: (B,4,4), K: (3,3) already scaled -> images (B,H,W,3) in [0,1]"""
    backend = backend or _BACKEND
    # the torch splatter's memory grows with chunk x pixels x Gaussians, so keep it small there
    chunk = chunk or (config.RENDER_CHUNK if backend == "gsplat" else 64)
    w2c = w2c.float()
    K = K.float()
    out = []
    for i in range(0, w2c.shape[0], chunk):
        V = w2c[i:i + chunk]
        Ks = K[None].expand(V.shape[0], 3, 3)
        if backend == "gsplat":
            img, _, _ = gsplat.rasterization(fr.means, fr.quats, fr.scales, fr.opac,
                                             fr.colors, V, Ks, W, H,
                                             near_plane=NEAR_PLANE, far_plane=100.0)
        else:
            img = _torch_splat(fr, V, Ks, W, H, near=NEAR_PLANE)
        out.append(img.clamp(0, 1))
    return torch.cat(out)


# --------------------------------------------------------------------------- pure PyTorch fallback
def _torch_splat(fr, V, Ks, W, H, near=0.01, blur=0.3, pix_chunk=4096):
    """EWA splatting + front-to-back alpha blending, Section 2.1 equations, no tiling."""
    from se3 import quat_to_rotmat
    B, dev = V.shape[0], V.device
    Rk = quat_to_rotmat(fr.quats)                                   # N,3,3
    S = torch.diag_embed(fr.scales)
    Sig = Rk @ S @ S.transpose(-1, -2) @ Rk.transpose(-1, -2)       # Sigma_k = R S S^T R^T

    Rcw, tcw = V[:, :3, :3], V[:, :3, 3]
    Xc = torch.einsum("bij,nj->bni", Rcw, fr.means) + tcw[:, None]  # B,N,3 camera coords
    z = Xc[..., 2].clamp(min=near)
    fx, fy, cx, cy = Ks[:, 0, 0, None], Ks[:, 1, 1, None], Ks[:, 0, 2, None], Ks[:, 1, 2, None]
    u = fx * Xc[..., 0] / z + cx
    v = fy * Xc[..., 1] / z + cy
    # Jacobian evaluated with x/z, y/z clamped to 1.3x the field of view (as 3DGS / gsplat do),
    # so Gaussians far outside the view do not get exploding footprints
    tan_x, tan_y = 0.5 * W / fx, 0.5 * H / fy
    tx = z * torch.minimum(torch.maximum(Xc[..., 0] / z, -(cx / fx + 0.3 * tan_x)), (W - cx) / fx + 0.3 * tan_x)
    ty = z * torch.minimum(torch.maximum(Xc[..., 1] / z, -(cy / fy + 0.3 * tan_y)), (H - cy) / fy + 0.3 * tan_y)
    J = torch.zeros(B, Xc.shape[1], 2, 3, device=dev, dtype=Xc.dtype)  # projection Jacobian
    J[..., 0, 0] = fx / z
    J[..., 0, 2] = -fx * tx / z ** 2
    J[..., 1, 1] = fy / z
    J[..., 1, 2] = -fy * ty / z ** 2
    JR = J @ Rcw[:, None]                                           # B,N,2,3
    Sig2 = JR @ Sig[None] @ JR.transpose(-1, -2)                    # Sigma' = J R Sigma R^T J^T
    Sig2 = Sig2 + blur * torch.eye(2, device=dev)                   # low-pass, as in 3DGS
    det = Sig2[..., 0, 0] * Sig2[..., 1, 1] - Sig2[..., 0, 1] ** 2
    inv = torch.stack([Sig2[..., 1, 1], -Sig2[..., 0, 1], Sig2[..., 0, 0]], -1) / det[..., None]
    valid = (Xc[..., 2] > near) & (det > 0)

    order = torch.argsort(torch.where(valid, Xc[..., 2], torch.full_like(z, 1e9)), dim=1)
    g = lambda x: torch.gather(x, 1, order.view(B, -1, *([1] * (x.dim() - 2))).expand_as(x))
    u, v, inv, valid = g(u), g(v), g(inv), g(valid)
    col = fr.colors[order]                                          # B,N,3
    op = fr.opac[order] * valid                                     # B,N

    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5,
                            indexing="ij")
    px, py = xs.reshape(-1), ys.reshape(-1)
    imgs = []
    for p0 in range(0, px.numel(), pix_chunk):
        dx = px[None, p0:p0 + pix_chunk, None] - u[:, None]         # B,P,N
        dy = py[None, p0:p0 + pix_chunk, None] - v[:, None]
        q = inv[:, None, :, 0] * dx * dx + 2 * inv[:, None, :, 1] * dx * dy + inv[:, None, :, 2] * dy * dy
        a = (op[:, None] * torch.exp(-0.5 * q)).clamp(max=0.99)
        a = torch.where(a < 1 / 255, torch.zeros_like(a), a)
        Tr = torch.cumprod(torch.cat([torch.ones_like(a[..., :1]), 1 - a[..., :-1]], -1), -1)
        imgs.append(torch.einsum("bpn,bnc->bpc", a * Tr, col))
    return torch.cat(imgs, 1).view(B, H, W, 3)
