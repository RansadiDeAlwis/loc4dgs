"""SE(3) / SO(3) utilities (guide Section 2.5 and Step 1).

Conventions
-----------
* A pose T = [R t; 0 1] is camera-to-world (c2w) unless a name says w2c.
* A tangent vector xi = (rho, phi) in R^6: translation part first, rotation part second.
* Perturbations are applied on the RIGHT: T' = T @ Exp(xi), i.e. in the camera's own frame.
* Quaternions are w-x-y-z (gsplat and Dynamic3DGaussians order).
All functions are batched over leading dimensions and work for float32 and float64.
"""
import numpy as np
import torch


def hat(v):                                   # v: (...,3) -> (...,3,3) skew-symmetric matrix
    x, y, z = v.unbind(-1)
    O = torch.zeros_like(x)
    return torch.stack([torch.stack([O, -z, y], -1),
                        torch.stack([z, O, -x], -1),
                        torch.stack([-y, x, O], -1)], -2)


def so3_exp(phi):                             # (...,3) -> (...,3,3), Rodrigues' formula
    th = phi.norm(dim=-1)[..., None, None]
    small = th < 1e-4
    ths = torch.where(small, torch.ones_like(th), th)
    A = torch.where(small, torch.ones_like(th), torch.sin(ths) / ths)
    B = torch.where(small, torch.full_like(th, 0.5), (1 - torch.cos(ths)) / ths ** 2)
    Phi = hat(phi)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(Phi.shape)
    return I + A * Phi + B * (Phi @ Phi)


def se3_exp(xi):                              # xi: (...,6) = (rho, phi) -> (...,4,4)
    rho, phi = xi[..., :3], xi[..., 3:]
    th = phi.norm(dim=-1)[..., None, None]
    small = th < 1e-4
    ths = torch.where(small, torch.ones_like(th), th)
    A = torch.where(small, torch.ones_like(th), torch.sin(ths) / ths)
    B = torch.where(small, torch.full_like(th, 0.5), (1 - torch.cos(ths)) / ths ** 2)
    C = torch.where(small, torch.full_like(th, 1 / 6), (ths - torch.sin(ths)) / ths ** 3)
    Phi = hat(phi)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(Phi.shape)
    R = I + A * Phi + B * (Phi @ Phi)
    V = I + B * Phi + C * (Phi @ Phi)
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).repeat(*xi.shape[:-1], 1, 1)
    T[..., :3, :3] = R
    T[..., :3, 3] = (V @ rho[..., None])[..., 0]
    return T


def se3_inv(T):                               # (...,4,4)
    R, t = T[..., :3, :3], T[..., :3, 3:]
    Ti = T.clone()
    Ti[..., :3, :3] = R.transpose(-1, -2)
    Ti[..., :3, 3:] = -R.transpose(-1, -2) @ t
    return Ti


def so3_log(R):                               # (...,3,3) -> (...,3); fine away from 180 degrees
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(cos)[..., None]
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)
    k = torch.where(th < 1e-4, torch.full_like(th, 0.5), th / (2 * torch.sin(th)))
    return k * w


def se3_log(T):                               # (...,4,4) -> (...,6)
    phi = so3_log(T[..., :3, :3])
    th = phi.norm(dim=-1)[..., None, None]
    small = th < 1e-4
    ths = torch.where(small, torch.ones_like(th), th)
    D = torch.where(small, torch.full_like(th, 1 / 12),
                    1 / ths ** 2 - (1 + torch.cos(ths)) / (2 * ths * torch.sin(ths)))
    Phi = hat(phi)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(Phi.shape)
    Vinv = I - 0.5 * Phi + D * (Phi @ Phi)
    rho = (Vinv @ T[..., :3, 3:])[..., 0]
    return torch.cat([rho, phi], -1)


def adjoint(T):                               # (...,4,4) -> (...,6,6), for xi ordered (rho, phi)
    R, t = T[..., :3, :3], T[..., :3, 3]
    Ad = torch.zeros(*T.shape[:-2], 6, 6, device=T.device, dtype=T.dtype)
    Ad[..., :3, :3] = R
    Ad[..., 3:, 3:] = R
    Ad[..., :3, 3:] = hat(t) @ R
    return Ad


def rot_err_deg(Ra, Rb):                      # angle between two rotations
    c = (((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


# --------------------------------------------------------------------------- quaternion helpers
def quat_to_rotmat(q):                        # (...,4) wxyz (any norm) -> (...,3,3)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotmat_to_quat(R):                        # (...,3,3) -> (...,4) wxyz, w >= 0
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    qw = 0.5 * torch.sqrt((1 + m00 + m11 + m22).clamp(min=0))
    qx = 0.5 * torch.sqrt((1 + m00 - m11 - m22).clamp(min=0))
    qy = 0.5 * torch.sqrt((1 - m00 + m11 - m22).clamp(min=0))
    qz = 0.5 * torch.sqrt((1 - m00 - m11 + m22).clamp(min=0))
    qx = torch.copysign(qx, R[..., 2, 1] - R[..., 1, 2])
    qy = torch.copysign(qy, R[..., 0, 2] - R[..., 2, 0])
    qz = torch.copysign(qz, R[..., 1, 0] - R[..., 0, 1])
    q = torch.stack([qw, qx, qy, qz], -1)
    return q / q.norm(dim=-1, keepdim=True)


def quat_average(q, w=None):
    """Markley's weighted quaternion average (guide Section 3.8).

    M = sum_i w_i q_i q_i^T (4x4); the mean is the eigenvector of the largest eigenvalue.
    Sign-safe because q and -q give the same outer product.  q: (N,4), w: (N,) -> (4,)
    """
    if w is None:
        w = torch.full((q.shape[0],), 1.0 / q.shape[0], device=q.device, dtype=q.dtype)
    M = (w[:, None, None] * q[:, :, None] * q[:, None, :]).sum(0)
    evals, evecs = torch.linalg.eigh(M.double())
    qm = evecs[:, -1].to(q.dtype)
    return qm if qm[0] >= 0 else -qm


def look_at(position, target, up):
    """OpenCV camera-to-world pose (x right, y down, z forward) at `position` looking at `target`.
    position/target/up: (...,3) torch tensors. `up` is the world up direction."""
    z = target - position
    z = z / z.norm(dim=-1, keepdim=True)
    y = -up - (-up * z).sum(-1, keepdim=True) * z           # camera y points DOWN
    y = y / y.norm(dim=-1, keepdim=True)
    x = torch.cross(y, z, dim=-1)
    T = torch.eye(4, dtype=position.dtype, device=position.device).repeat(*position.shape[:-1], 1, 1)
    T[..., :3, 0], T[..., :3, 1], T[..., :3, 2], T[..., :3, 3] = x, y, z, position
    return T


def random_offset(trans, rot_deg, rng, n=None, dtype=torch.float32):
    """Tangent vector(s) with EXACT translation norm `trans` and rotation angle `rot_deg`
    in uniformly random directions (used to build T_init = T_gt Exp(offset), Section 3.7)."""
    shape = (3,) if n is None else (n, 3)
    a = rng.normal(size=shape); a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b = rng.normal(size=shape); b /= np.linalg.norm(b, axis=-1, keepdims=True)
    xi = np.concatenate([trans * a, np.deg2rad(rot_deg) * b], -1)
    return torch.tensor(xi, dtype=dtype)

