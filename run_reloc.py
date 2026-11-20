"""Relocalization experiments (guide Step 7, Sections 3.1, 3.6, 3.7; experiments E1, E3, E4, E5).

One REAL image from a held-out test camera (0, 10, 15, 30) at a fixed time t.  The filter
iterates repeatedly on the same image (no motion: U = I, the noise is a random-walk search
that shrinks through annealing).  PF, EKF and iterated EKF all start from the SAME perturbed
pose T_init = T_gt Exp(offset) with the same seeds (fairness rules, Section 6.2).

Examples
--------
E1 : python run_reloc.py --scene juggle --exp E1 --methods pf ekf iekf --levels 0.1:5 0.3:20 1.0:45 global
E3 : python run_reloc.py --scene juggle --exp E3 --methods pf --levels 0.3:20 \
         --maps full:0 background:0 full:5 full:-5 full:15 full:-15 frozen0:0
E4 : python run_reloc.py --scene juggle --exp E4 --methods pf --levels 0.3:20 --N 50 100 300 600 1000 --anneal 1 0
E5 : python run_reloc.py --scene juggle --exp E5 --methods pf --levels 0.3:20 --lam 0.5 1 2 5 10 20 50 \
         --no_temper --loss_types mse trimmed
Use --scene synthetic to run everything without the dataset (CPU / no-gsplat fallback).
"""
import argparse
import zlib
import numpy as np
import torch
from tqdm import tqdm

import config
from common import (CSVLog, Timer, device, set_seed, parse_maps, parse_levels, level_name,
                    result_path, vis_path)
from scene import load_scene, camera
from render import scale_K, render_batch, set_backend, get_backend
from se3 import se3_exp, se3_inv, random_offset
from observations import load_obs, synth_observation, downsample
from trajectories import cam_centers_and_c2w, GlobalPrior, local_prior
from pf import ParticleFilter
from ekf import PoseEKF
from metrics import pose_errors


def get_args():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--scene", default=config.TUNE_SCENE)
    a.add_argument("--exp", default="E1", help="tag; rows go to results/<exp>_reloc_<scene>.csv")
    a.add_argument("--methods", nargs="+", default=["pf", "ekf", "iekf"])
    a.add_argument("--levels", nargs="+", default=["0.1:5", "0.3:20", "1.0:45", "global"])
    a.add_argument("--maps", nargs="+", default=["full:0"])
    a.add_argument("--seeds", type=int, default=len(config.SEEDS))
    a.add_argument("--n_times", type=int, default=10, help="timesteps per test camera (evenly spaced)")
    a.add_argument("--times", type=int, nargs="+", default=None,
                   help="explicit timesteps, e.g. 0 30 60 90 120 (overrides --n_times)")
    a.add_argument("--noise_frac", type=float, nargs="+", default=[config.RELOC["local_noise_frac"]],
                   help="local prior: random-walk sigma = frac x initial error (tuning, Stage 1)")
    a.add_argument("--max_images", type=int, default=None)
    a.add_argument("--N", type=int, nargs="+", default=[config.RELOC["N"]])
    a.add_argument("--N_small", type=int, default=config.RELOC["N_small"])
    a.add_argument("--iters", type=int, default=config.RELOC["iters"])
    a.add_argument("--lam", type=float, nargs="+", default=[config.RELOC["lam"]])
    a.add_argument("--no_temper", action="store_true", help="keep lambda fixed (E5)")
    a.add_argument("--lam_max", type=float, nargs="+", default=None,
                   help="cap of the tempered lambda (default RELOC['lam_max']); several values = tuning")
    a.add_argument("--anneal", type=int, nargs="+", default=[1])
    # EKF tuning (Stage 1 for the baseline): several values = grid; omitted = config.EKF
    a.add_argument("--ekf_blur", type=float, nargs="+", default=None, help="Gaussian blur sigma [px at EKF res]")
    a.add_argument("--ekf_sigma_z", type=float, nargs="+", default=None, help="pixel noise std (trust in the image)")
    a.add_argument("--ekf_eps", type=float, nargs="+", default=None, help="finite-difference step")
    a.add_argument("--inject", type=int, default=0)
    a.add_argument("--likelihood", default="tempered", choices=["tempered", "locnerf"])
    a.add_argument("--loss_types", nargs="+", default=["trimmed"], choices=["mse", "trimmed", "huber"])
    a.add_argument("--W", type=int, default=None)
    a.add_argument("--H", type=int, default=None)
    a.add_argument("--backend", default=None, choices=["gsplat", "torch"])
    a.add_argument("--save_vis", type=int, default=0, help="save visualization npz for the first k runs")
    a.add_argument("--preload", action="store_true", help="keep the whole 4D scene on the GPU")
    return a.parse_args()


def loss_cfg(name):
    return dict(mse=(0.0, "mse"), trimmed=(config.RELOC["trim"], "mse"), huber=(0.0, "huber"))[name]


class Observation:
    """One test image: ground truth, filter-resolution observation and EKF-resolution observation."""

    def __init__(self, scene_name, scene, md, t, c, W, H, dev, seed):
        K, self.T_gt, fn, self.cam_id = camera(md, t, c, dev)
        W0, H0 = md["w"], md["h"]
        self.t, self.c = t, c
        self.K = scale_K(K, W0, H0, W, H)
        self.K_ekf = scale_K(K, W0, H0, config.EKF_W, config.EKF_H)
        if fn is not None:                                       # real held-out photo
            path = config.image_path(scene_name, fn)
            self.obs = load_obs(path, W, H, dev)
            self.obs_ekf = load_obs(path, config.EKF_W, config.EKF_H, dev)
            self.real = True
        else:                                                    # synthetic world: corrupted render
            rng = np.random.RandomState(seed)
            self.obs = synth_observation(scene.frame(t), self.T_gt, K, W0, H0, W, H, rng,
                                         obs_wh=(4 * W, 4 * H), **config.OBS_CORRUPT)
            self.obs_ekf = downsample(self.obs, config.EKF_W, config.EKF_H)
            self.real = False
        fgimg = render_batch(scene.fg_frame(t), se3_inv(self.T_gt)[None], self.K, W, H)[0]
        self.fg_frac = fgimg[..., 0].mean().item()


def run_pf(obs, fr, X0, args, N, lam, anneal, loss, sampler, log, base, vis=None, noise_frac=None,
           lam_max=None):
    trim, lossf = loss_cfg(loss)
    local = base["level"] != "global"
    noise_frac = config.RELOC["local_noise_frac"] if noise_frac is None else noise_frac
    lam_max = config.RELOC["lam_max"] if lam_max is None else lam_max
    if local:
        sig_p = noise_frac * base["init_t"]
        sig_r = noise_frac * np.deg2rad(base["init_r"])
    else:
        sig_p, sig_r = config.RELOC["sig_p"], config.RELOC["sig_r"]
    pf = ParticleFilter(X0, sig_p, sig_r, lam, trim=trim, likelihood=args.likelihood, loss=lossf)
    spread_thr = config.RELOC["spread_frac0"] * pf.spread()
    since_anneal = 0
    W, H = obs.obs.shape[1], obs.obs.shape[0]
    T_hat = None
    elapsed = 0.0
    for k in range(args.iters):
        with Timer() as tm:
            pf.predict(None)
            injected = args.inject and pf.maybe_inject(sampler, config.RELOC["inject_err"],
                                                       config.RELOC["inject_patience"],
                                                       config.RELOC["inject_frac"])
            pf.update(fr, obs.obs, obs.K, W, H)
            ess = pf.ess()
            pf.resample()
            if anneal and not args.no_temper:                     # likelihood tempering lambda_k = lambda_0 gamma^k
                pf.lam = min(lam * config.RELOC["lam_gamma"] ** (k + 1), lam_max)
            since_anneal += 1
            if anneal and (pf.spread() < spread_thr or since_anneal >= config.RELOC["anneal_patience"]):
                pf.sig_p *= config.RELOC["anneal_factor"]; pf.sig_r *= config.RELOC["anneal_factor"]
                if pf.N > args.N_small:                           # annealing (Section 3.6)
                    pf.resample(args.N_small)
                spread_thr = min(spread_thr, pf.spread()) * config.RELOC["anneal_factor"]
                since_anneal = 0
            T_hat, P = pf.estimate()
        elapsed += tm.ms
        te, re = pose_errors(T_hat, obs.T_gt)
        log.add(**base, method="pf", N0=N, lam0=lam, anneal=anneal, loss=loss,
                noise_frac=noise_frac if local else np.nan, step=k,
                te=te.item(), re=re.item(), ess=ess, N=pf.N, lam=pf.lam, sig_p=pf.sig_p,
                sig_r=np.rad2deg(pf.sig_r), spread=pf.spread(), err_best=pf.err.min().item(),
                injected=int(bool(injected)), time_ms=tm.ms, elapsed_ms=elapsed, n_renders=pf.n_renders,
                **({"lam_max": lam_max} if getattr(args, "lam_max", None) else {}))
        if vis is not None:
            img = render_batch(fr, se3_inv(T_hat)[None], obs.K, W, H)[0]
            sub = torch.randperm(pf.N)[:200] if pf.N >= 200 else torch.randint(pf.N, (200,))
            vis["est_render"].append(img.cpu().numpy())
            vis["particles"].append(pf.X[sub, :3, 3].cpu().numpy())
            vis["particle_dirs"].append(pf.X[sub, :3, 2].cpu().numpy())
            vis["weights"].append(pf.w[sub].cpu().numpy())
            vis["est"].append(T_hat.cpu().numpy())
    return T_hat


def run_ekf(obs, fr, T0, P0, iters, args, log, base, blur=None, sigma_z=None, eps=None):
    E = config.EKF
    blur = E["blur_sigma"] if blur is None else blur
    sigma_z = E["sigma_z"] if sigma_z is None else sigma_z
    eps = E["eps"] if eps is None else eps
    Q = torch.diag(torch.tensor([E["q_p"]] * 3 + [E["q_r"]] * 3, device=T0.device) ** 2)
    ekf = PoseEKF(T0, P0, Q, sigma_z, eps=eps, iters=iters, blur_sigma=blur)
    tuning = any(getattr(args, k, None) for k in ("ekf_blur", "ekf_sigma_z", "ekf_eps"))
    extra = dict(ekf_blur=blur, ekf_sigma_z=sigma_z, ekf_eps=eps) if tuning else {}
    name = "ekf" if iters == 1 else "iekf"
    elapsed = 0.0
    for k in range(E["reloc_steps"]):
        with Timer() as tm:
            ekf.predict(None)
            ekf.update(fr, obs.obs_ekf, obs.K_ekf, config.EKF_W, config.EKF_H)
        elapsed += tm.ms
        te, re = pose_errors(ekf.T, obs.T_gt)
        if not torch.isfinite(ekf.T).all():                       # diverged numerically
            te, re = torch.tensor(np.inf), torch.tensor(180.0)
        log.add(**base, method=name, N0=1, lam0=np.nan, anneal=0, loss="gray", noise_frac=np.nan, step=k,
                te=te.item(), re=re.item(), ess=np.nan, N=1, lam=np.nan, sig_p=np.nan, sig_r=np.nan,
                spread=torch.sqrt(torch.trace(ekf.P[:3, :3])).item(), err_best=ekf.err.item(),
                injected=0, time_ms=tm.ms, elapsed_ms=elapsed, n_renders=ekf.n_renders, **extra)
        if not torch.isfinite(ekf.T).all():
            break
    return ekf.T


def main():
    args = get_args()
    dev = device()
    if args.backend:
        set_backend(args.backend)
    small = get_backend() == "torch"
    W = args.W or (config.SYNTH_W if small else config.PF_W)
    H = args.H or (config.SYNTH_H if small else config.PF_H)
    scene, train_md, test_md = load_scene(args.scene, dev, preload=args.preload)
    prior = GlobalPrior(cam_centers_and_c2w(train_md), config.GLOBAL_PRIOR["rot_noise_deg"], dev)
    print(prior.describe())
    levels, maps = parse_levels(args.levels), parse_maps(args.maps)
    log = CSVLog(result_path(args.exp, args.scene, "reloc"))

    n_cams = len(test_md["fn"][0])
    if args.times:
        times = sorted({min(t, scene.T - 1) for t in args.times})
    else:
        times = np.linspace(0, scene.T - 1, args.n_times).astype(int)
    images = [(int(t), c) for c in range(n_cams) for t in times][:args.max_images]
    vis_left = args.save_vis
    jobs = [(i, t, c, lv, s) for i, (t, c) in enumerate(images) for lv in levels for s in range(args.seeds)]
    obs_cache = {}
    for img_id, t, c, level, seed in tqdm(jobs, desc=f"{args.exp}/{args.scene}"):
        if img_id not in obs_cache:
            obs_cache = {img_id: Observation(args.scene, scene, test_md, t, c, W, H, dev, seed=1000 + img_id)}
        obs = obs_cache[img_id]
        # identical initial pose / prior for every method and map (fairness)
        key = zlib.crc32(f"{args.scene}-{t}-{c}-{level_name(level)}-{seed}".encode())
        rng = np.random.default_rng(key)
        if level == "global":
            prior.rng = np.random.default_rng(key)
            T_init = prior(1)[0]
            init_t, init_r = np.nan, np.nan
        else:
            init_t, init_r = level
            T_init = obs.T_gt @ se3_exp(random_offset(init_t, init_r, rng).to(dev))
        base = dict(scene=args.scene, mode="reloc", level=level_name(level), init_t=init_t, init_r=init_r,
                    t=t, cam=obs.cam_id, img=img_id, seed=seed, real=int(obs.real), fg_frac=obs.fg_frac,
                    backend=get_backend(), W=W, H=H)
        for map_mode, offset in maps:
            fr = scene.frame(t, map_mode, offset)
            b = dict(base, map_mode=map_mode, offset=offset)
            for m in args.methods:
                if m == "pf":
                    for N in args.N:
                        for lam in args.lam:
                            for anneal in args.anneal:
                                for loss in args.loss_types:
                                    for nf, lmax in [(a, m) for a in (args.noise_frac if level != "global" else [None])
                                                     for m in (args.lam_max or [None])]:
                                        set_seed(key % 2 ** 31)
                                        if level == "global":
                                            prior.rng = np.random.default_rng(key + 1)
                                            X0 = prior(N)
                                        else:
                                            X0 = local_prior(T_init, N, init_t, np.deg2rad(init_r))
                                        vis = None
                                        if vis_left > 0:
                                            vis = dict(est_render=[], particles=[], particle_dirs=[],
                                                       weights=[], est=[])
                                        run_pf(obs, fr, X0, args, N, lam, anneal, loss, prior, log, b, vis,
                                               noise_frac=nf, lam_max=lmax)
                                        if vis is not None:
                                            vis_left -= 1
                                            bg, bgc = scene.background_points(t, 3000)
                                            np.savez_compressed(
                                                vis_path(f"reloc_{args.exp}_{args.scene}_img{img_id}_{level_name(level)}"
                                                         f"_s{seed}_{map_mode}{offset}_N{N}_lam{lam:g}"
                                                         + (f"_nf{nf:g}" if nf is not None else "")),
                                                obs=obs.obs.cpu().numpy(), T_gt=obs.T_gt.cpu().numpy(),
                                                T_init=T_init.cpu().numpy(), bg=bg, bg_col=bgc,
                                                **{k: np.stack(v) for k, v in vis.items()})
                else:
                    set_seed(key % 2 ** 31)
                    if level == "global":
                        P0 = torch.diag(torch.tensor([1.0] * 3 + [np.deg2rad(30.0)] * 3, device=dev) ** 2)
                    else:
                        P0 = torch.diag(torch.tensor([init_t] * 3 + [np.deg2rad(init_r)] * 3, device=dev) ** 2)
                    iters = 1 if m == "ekf" else config.EKF["iters_iekf"]
                    for bl in (args.ekf_blur or [None]):
                        for sz in (args.ekf_sigma_z or [None]):
                            for ep in (args.ekf_eps or [None]):
                                set_seed(key % 2 ** 31)
                                run_ekf(obs, fr, T_init, P0.float(), iters, args, log, b, blur=bl, sigma_z=sz, eps=ep)
        log.flush()
    log.flush()
    print("wrote", log.path)


if __name__ == "__main__":
    main()
