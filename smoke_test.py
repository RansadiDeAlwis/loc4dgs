"""Stage 0 smoke tests on a REAL scene (run before any experiment; any failure means a bug).

    python smoke_test.py --scene juggle                 # all four checks
    python smoke_test.py --scene juggle --only converge timing --cam 1 --t 60

1. truth    : particles start exactly at the true pose, no noise -> estimate stays at the truth.
2. winner   : one particle at the true pose, the rest from the global prior -> it gets most weight.
3. converge : particles start 0.2 units / 10 deg away -> the error falls within ~20-30 iterations
              (uses the real relocalization loop of run_reloc.py with the settings in config.py).
4. timing   : ms per update for the PF (N = 300 tracking, 600 / 200 relocalization) and the EKF /
              iterated EKF, and a rough run-time budget for the plan (E1, E2, E3 on 3 scenes).

The observation is a real held-out test photo (a corrupted render for --scene synthetic).
"""
import argparse
import types
import numpy as np
import torch

import config
from common import Timer, device, set_seed
from scene import load_scene
from render import set_backend, get_backend
from se3 import se3_exp, random_offset
from trajectories import cam_centers_and_c2w, GlobalPrior, local_prior
from pf import ParticleFilter
from ekf import PoseEKF
from metrics import pose_errors, convergence_step
import run_reloc

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class _Rows:                                       # stands in for CSVLog: keeps rows in memory
    def __init__(self):
        self.rows = []

    def add(self, **r):
        self.rows.append(r)


def check_truth(obs, fr, W, H):
    pf = ParticleFilter(obs.T_gt[None].repeat(100, 1, 1), 0.0, 0.0, lam=config.RELOC["lam"],
                        trim=config.RELOC["trim"])
    for _ in range(5):
        pf.predict(None)
        pf.update(fr, obs.obs, obs.K, W, H)
        pf.resample()
    te, re = pose_errors(pf.estimate()[0], obs.T_gt)
    ok = te.item() < 1e-4 and re.item() < 1e-2
    print(f"[1 truth]    final error {te.item():.2e} units, {re.item():.2e} deg  -> {PASS if ok else FAIL}")
    return ok


def check_winner(obs, fr, prior, W, H, n=300):
    X = torch.cat([obs.T_gt[None], prior(n - 1)])
    pf = ParticleFilter(X, 0.0, 0.0, lam=config.RELOC["lam"], trim=config.RELOC["trim"])
    pf.update(fr, obs.obs, obs.K, W, H)
    w0, rank = pf.w[0].item(), int((pf.err < pf.err[0]).sum().item())
    others = pf.err[1:]
    ok = rank == 0 and torch.isfinite(pf.logw).all().item()
    print(f"[2 winner]   true particle: error {pf.err[0].item():.3f} (others: min {others.min().item():.3f}, "
          f"median {others.median().item():.3f}), rank {rank}, weight {w0:.1%} (uniform = {1 / n:.2%}) "
          f"-> {PASS if ok else FAIL}")
    if ok and w0 < 0.5:
        print(f"             (rank 0 but weight < 50%: lambda={config.RELOC['lam']} is mild on purpose; "
              f"tempering raises it during relocalization)")
    return ok


def check_converge(obs, fr, prior, iters, seed=0):
    init_t, init_r = 0.2, 10.0
    rng = np.random.default_rng(seed)
    set_seed(seed)
    T_init = obs.T_gt @ se3_exp(random_offset(init_t, init_r, rng).to(obs.T_gt.device))
    R = config.RELOC
    X0 = local_prior(T_init, R["N"], init_t, np.deg2rad(init_r))
    args = types.SimpleNamespace(iters=iters, inject=0, no_temper=False, N_small=R["N_small"],
                                 likelihood="tempered")
    log = _Rows()
    base = dict(level="0.2u_10deg", init_t=init_t, init_r=init_r)
    run_reloc.run_pf(obs, fr, X0, args, R["N"], R["lam"], 1, "trimmed", prior, log, base)
    te = np.array([r["te"] for r in log.rows]); re = np.array([r["re"] for r in log.rows])
    show = [k for k in (0, 5, 10, 20, 30, 40, 60, iters - 1) if k < iters]
    print("[3 converge] step : " + "  ".join(f"{k:>11d}" for k in show))
    print("             error: " + "  ".join(f"{te[k]:.3f}/{re[k]:4.1f}deg" for k in show))
    c = convergence_step(te, re, config.SUCCESS_T, config.SUCCESS_R)
    if c is not None and c <= 30:
        verdict = PASS
    elif te[-1] < 0.5 * te[0] or c is not None:
        verdict = WARN
    else:
        verdict = FAIL
    print(f"             start {te[0]:.3f} units / {re[0]:.1f} deg; inside the success threshold "
          f"({config.SUCCESS_T} units, {config.SUCCESS_R} deg) from step {c} -> {verdict}")
    if verdict != PASS:
        print("             (WARN/FAIL here is what Stage 1 tuning is for: try other --lam / --noise_frac)")
    return verdict != FAIL


def _time(fn, reps):
    for _ in range(2):                              # warm-up (CUDA kernels, allocator)
        fn()
    ts = []
    for _ in range(reps):
        with Timer() as tm:
            fn()
        ts.append(tm.ms)
    return float(np.median(ts))


def check_timing(obs, fr, W, H, n_images, reps=10):
    ms = {}
    for N in (300, 600, 200):
        pf = ParticleFilter(local_prior(obs.T_gt, N, 0.02, np.deg2rad(1.0)), 0.01, np.deg2rad(0.5),
                            lam=config.TRACK["lam"], trim=config.TRACK["trim"])

        def step(pf=pf):                            # one tracking update, as in run_track.py
            pf.predict(None)
            pf.update(fr, obs.obs, obs.K, W, H)
            if pf.ess() < pf.N / 2:
                pf.resample()
            pf.estimate()
        ms[f"pf{N}"] = _time(step, reps)
    E = config.EKF
    Q = torch.diag(torch.tensor([E["q_p"]] * 3 + [E["q_r"]] * 3, device=obs.T_gt.device) ** 2).float()
    P0 = torch.diag(torch.tensor([0.02] * 3 + [np.deg2rad(1.0)] * 3, device=obs.T_gt.device) ** 2).float()
    for name, iters in (("ekf", 1), ("iekf", E["iters_iekf"])):
        ekf = PoseEKF(obs.T_gt.clone(), P0.clone(), Q, E["sigma_z"], eps=E["eps"], iters=iters,
                      blur_sigma=E["blur_sigma"])

        def step(ekf=ekf):
            ekf.predict(None)
            ekf.update(fr, obs.obs_ekf, obs.K_ekf, config.EKF_W, config.EKF_H)
        ms[name] = _time(step, reps)
    print(f"[4 timing]   ms per update: PF N=300 {ms['pf300']:.1f} | N=600 {ms['pf600']:.1f} | "
          f"N=200 {ms['pf200']:.1f} | EKF {ms['ekf']:.1f} | IEKF {ms['iekf']:.1f}   "
          f"(backend {get_backend()}, {W}x{H}, RENDER_CHUNK={config.RENDER_CHUNK})")

    # rough budget for the plan: E1, E2, E3 on 3 scenes (the sweeps on juggle come on top)
    R, it = config.RELOC, config.RELOC["iters"]
    k600 = R["anneal_patience"]                     # at most this many iterations at N before the first anneal
    pf_reloc = k600 * ms["pf600"] + (it - k600) * ms["pf200"]
    e1 = n_images * 3 * 3 * (pf_reloc + E["reloc_steps"] * (ms["ekf"] + ms["iekf"]))
    e2 = 2 * 5 * config.N_FRAMES * (ms["pf300"] + ms["ekf"])
    e3 = n_images * 3 * 5 * pf_reloc + 2 * 5 * 5 * config.N_FRAMES * ms["pf300"]
    h = lambda x: x / 3.6e6
    print(f"             rough budget per scene: E1 {h(e1):.1f} h | E2 {h(e2):.1f} h | E3 {h(e3):.1f} h  "
          f"-> 3 scenes ~ {3 * h(e1 + e2 + e3):.1f} h")
    print(f"             (E1: {n_images} images x 3 levels x 3 seeds, PF+EKF+IEKF; E2: 2 traj x 5 seeds, PF+EKF; "
          f"E3: 5 maps, reloc 3 seeds + track 2 traj x 5 seeds; excludes observation rendering and I/O)")
    return True


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--scene", default=config.TUNE_SCENE)
    a.add_argument("--cam", type=int, default=0, help="index into the test cameras")
    a.add_argument("--t", type=int, default=0)
    a.add_argument("--iters", type=int, default=config.RELOC["iters"])
    a.add_argument("--only", nargs="*", default=["truth", "winner", "converge", "timing"])
    a.add_argument("--backend", default=None, choices=["gsplat", "torch"])
    args = a.parse_args()
    dev = device()
    if args.backend:
        set_backend(args.backend)
    small = get_backend() == "torch"
    W, H = (config.SYNTH_W, config.SYNTH_H) if small else (config.PF_W, config.PF_H)
    scene, train_md, test_md = load_scene(args.scene, dev)
    t = min(args.t, scene.T - 1)
    obs = run_reloc.Observation(args.scene, scene, test_md, t, args.cam, W, H, dev, seed=0)
    fr = scene.frame(t)
    prior = GlobalPrior(cam_centers_and_c2w(train_md), config.GLOBAL_PRIOR["rot_noise_deg"], dev)
    print(f"scene {args.scene}, test cam {obs.cam_id}, t={t}, {'real photo' if obs.real else 'synthetic obs'}, "
          f"{W}x{H}, backend {get_backend()}")
    set_seed(0)
    ok = True
    if "truth" in args.only: ok &= check_truth(obs, fr, W, H)
    if "winner" in args.only: ok &= check_winner(obs, fr, prior, W, H)
    if "converge" in args.only: ok &= check_converge(obs, fr, prior, args.iters)
    if "timing" in args.only:
        n_images = len(test_md["fn"][0]) * 5
        ok &= check_timing(obs, fr, W, H, n_images)
    print("all checks passed" if ok else "SOME CHECKS FAILED: fix before running experiments")


if __name__ == "__main__":
    main()
