"""Step 1 sanity checks for se3.py.  Run:  python -m pytest tests -q   (from loc4dgs/)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from se3 import (se3_exp, se3_log, se3_inv, so3_exp, adjoint, rot_err_deg, quat_to_rotmat,
                 rotmat_to_quat, quat_average, look_at)


def rand_xi(n, max_rot=2.5, dtype=torch.float64):
    rho = torch.randn(n, 3, dtype=dtype)
    ax = torch.nn.functional.normalize(torch.randn(n, 3, dtype=dtype), dim=-1)
    return torch.cat([rho, ax * torch.rand(n, 1, dtype=dtype) * max_rot], -1)


def test_log_exp_roundtrip():
    xi = rand_xi(1000)
    assert torch.allclose(se3_log(se3_exp(xi)), xi, atol=1e-8)
    xi32 = rand_xi(1000, dtype=torch.float32)
    assert torch.allclose(se3_log(se3_exp(xi32)), xi32, atol=2e-3)


def test_small_angles():
    xi = rand_xi(100) * 1e-6
    assert torch.allclose(se3_log(se3_exp(xi)), xi, atol=1e-12)


def test_inverse():
    T = se3_exp(rand_xi(100))
    I = torch.eye(4, dtype=T.dtype).expand_as(T)
    assert torch.allclose(T @ se3_inv(T), I, atol=1e-10)


def test_rotation_valid():
    R = se3_exp(rand_xi(100))[:, :3, :3]
    assert torch.allclose(torch.linalg.det(R), torch.ones(100, dtype=R.dtype), atol=1e-10)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=R.dtype).expand_as(R), atol=1e-10)


def test_so3_matches_scipy():
    phi = rand_xi(50)[:, 3:]
    assert np.allclose(so3_exp(phi).numpy(), Rot.from_rotvec(phi.numpy()).as_matrix(), atol=1e-10)


def test_adjoint():
    # T Exp(xi) T^-1 = Exp(Ad_T xi)
    T = se3_exp(rand_xi(20))
    xi = rand_xi(20, 0.5)
    lhs = T @ se3_exp(xi) @ se3_inv(T)
    rhs = se3_exp((adjoint(T) @ xi[..., None])[..., 0])
    assert torch.allclose(lhs, rhs, atol=1e-9)


def test_rot_err():
    phi = torch.tensor([[0, 0, np.deg2rad(30.0)]], dtype=torch.float64)
    assert abs(rot_err_deg(torch.eye(3, dtype=torch.float64)[None], so3_exp(phi)).item() - 30) < 1e-6


def test_quaternions():
    R = se3_exp(rand_xi(100))[:, :3, :3]
    q = rotmat_to_quat(R)
    assert torch.allclose(quat_to_rotmat(q), R, atol=1e-8)
    qs = Rot.from_matrix(R.numpy()).as_quat()            # xyzw
    qs = np.concatenate([qs[:, 3:], qs[:, :3]], 1) * np.sign(qs[:, 3:])
    assert np.allclose(q.numpy(), qs, atol=1e-8)


def test_quat_average_sign_safe():
    q0 = torch.nn.functional.normalize(torch.randn(4, dtype=torch.float64), dim=0)
    noise = torch.randn(200, 4, dtype=torch.float64) * 0.01
    q = torch.nn.functional.normalize(q0 + noise, dim=-1)
    q[::2] *= -1                                           # flip half the signs
    qm = quat_average(q)
    assert min((qm - q0).norm(), (qm + q0).norm()) < 0.01


def test_look_at():
    p = torch.tensor([[3.0, 0, 1]], dtype=torch.float64)
    T = look_at(p, torch.zeros(1, 3, dtype=torch.float64), torch.tensor([[0, 0, 1.0]], dtype=torch.float64))
    z = T[0, :3, 2]
    assert torch.allclose(z, torch.nn.functional.normalize(-p[0], dim=0))
    assert T[0, 2, 1] < 0                                   # camera y points down (world -z component)
    assert abs(torch.linalg.det(T[0, :3, :3]).item() - 1) < 1e-10
