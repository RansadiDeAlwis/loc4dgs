"""Debugging checklist of guide Section 10.2, run on the synthetic world (no dataset needed).

  * render from a real camera is not empty and changes with pose (w2c vs c2w convention)
  * with noise off and particles at the true pose, the estimate stays at the truth
  * a single particle at the true pose among random ones wins the weights (tests the likelihood)
  * weights stay finite (log space); resampling indices in range; weights uniform afterwards
  * same seed -> same result (reproducibility)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

import config
from common import set_seed
from scene import load_scene, camera
from render import scale_K, render_batch, get_backend
from se3 import se3_exp, se3_inv
from observations import synth_observation
from trajectories import cam_centers_and_c2w, GlobalPrior, local_prior
from pf import ParticleFilter, systematic_resample
from metrics import pose_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, H = (config.PF_W, config.PF_H) if get_backend() == "gsplat" else (config.SYNTH_W, config.SYNTH_H)


@pytest.fixture(scope="module")
def setup():
    scene, train_md, test_md = load_scene("synthetic", DEV)
    K, T_gt, _, _ = camera(test_md, 40, 1, DEV)
    Ks = scale_K(K, test_md["w"], test_md["h"], W, H)
    obs = synth_observation(scene.frame(40), T_gt, K, test_md["w"], test_md["h"], W, H,
                            np.random.RandomState(0), obs_wh=(2 * W, 2 * H), noise=0.01, blur=0)
    return scene, train_md, Ks, T_gt, obs


def test_render_not_empty_and_pose_sensitive(setup):
    scene, _, K, T_gt, _ = setup
    a = render_batch(scene.frame(40), se3_inv(T_gt)[None], K, W, H)[0]
    assert a.mean() > 0.05, "render is (almost) black: check the w2c / c2w convention"
    away = T_gt @ se3_exp(torch.tensor([0, 0, 0, 0, np.pi, 0], device=DEV, dtype=torch.float32))
    b = render_batch(scene.frame(40), se3_inv(away)[None], K, W, H)[0]
    assert (a - b).abs().mean() > 0.05


def test_systematic_resample():
    w = torch.softmax(torch.randn(500), 0)
    idx = systematic_resample(w, 500)
    assert idx.min() >= 0 and idx.max() < 500
    big = torch.where(w > 1 / 500)[0]
    assert set(big.tolist()) <= set(idx.tolist())          # keeps every particle with w > 1/N


def test_true_particle_wins(setup):
    scene, train_md, K, T_gt, obs = setup
    set_seed(0)
    prior = GlobalPrior(cam_centers_and_c2w(train_md), device=DEV)
    X = torch.cat([T_gt[None], prior(199)])
    pf = ParticleFilter(X, 0, 0, lam=20.0)
    pf.update(scene.frame(40), obs, K, W, H)
    assert torch.isfinite(pf.logw).all()
    assert torch.argmax(pf.w).item() == 0


def test_stays_at_truth_without_noise(setup):
    scene, _, K, T_gt, obs = setup
    pf = ParticleFilter(T_gt[None].repeat(50, 1, 1), 0.0, 0.0, lam=5.0)
    for _ in range(5):
        pf.predict(None)
        pf.update(scene.frame(40), obs, K, W, H)
        pf.resample()
    T, P = pf.estimate()
    te, re = pose_errors(T, T_gt)
    assert te.item() < 1e-4 and re.item() < 1e-2
    assert torch.allclose(pf.w, torch.full_like(pf.w, 1 / 50))


def test_local_relocalization_converges(setup):
    """Median over 3 seeds: a single run is not reproducible bit-for-bit on the GPU (the
    rasterizer is not deterministic), and one unlucky run must not fail the test."""
    scene, _, K, T_gt, obs = setup
    T_init = T_gt @ se3_exp(torch.tensor([0.1, -0.05, 0.05, 0.03, -0.05, 0.02], device=DEV))
    te0, _ = pose_errors(T_init, T_gt)
    finals = []
    for seed in (1, 2, 3):
        set_seed(seed)
        pf = ParticleFilter(local_prior(T_init, 300, 0.1, np.deg2rad(5)), 0.05, np.deg2rad(2.5),
                            lam=config.RELOC["lam"], trim=0.1)          # the frozen lambda_0
        for k in range(30):
            pf.predict(None)
            pf.update(scene.frame(40), obs, K, W, H)
            pf.resample()
            if k % 5 == 4:
                pf.sig_p *= 0.6; pf.sig_r *= 0.6
        finals.append(pose_errors(pf.estimate()[0], T_gt)[0].item())
    assert np.median(finals) < 0.5 * te0.item(), (finals, te0.item())


def test_reproducible(setup):
    scene, _, K, T_gt, obs = setup
    outs = []
    for _ in range(2):
        set_seed(123)
        pf = ParticleFilter(local_prior(T_gt, 100, 0.05, 0.05), 0.02, 0.02, lam=5.0)
        for _ in range(3):
            pf.predict(None); pf.update(scene.frame(40), obs, K, W, H); pf.resample()
        outs.append(pf.estimate()[0])
    assert torch.allclose(outs[0], outs[1])
