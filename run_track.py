"""Tracking experiments (guide Step 7, Sections 3.1, 3.3; experiments E2, E3, E4, E5, E6, E7).

A camera moves along a synthetic trajectory (interpolated between neighbouring Panoptic
cameras) through the moving scene; the time index advances with each frame and the map used
at frame t is G(t).  Observations are rendered from the FULL 4D map at the correct time at a
higher resolution, area-downsampled and corrupted (noise, blur, exposure, occluders).  Every
method / map sees exactly the same observations and odometry for a given (trajectory, seed).

Examples
--------
E2 : python run_track.py --scene juggle --exp E2 --methods pf ekf --odo low medium high
E3 : python run_track.py --scene juggle --exp E3 --maps full:0 background:0 full:5 full:-5 full:15 full:-15 frozen0:0
E4 : python run_track.py --scene juggle --exp E4 --N 50 100 300 600 1000
E5 : python run_track.py --scene juggle --exp E5 --lam 0.5 1 2 5 10 20 50 --loss_types mse trimmed
E6 : python run_track.py --scene juggle --exp E6 --occ 0 0.1 0.3 0.5 --loss_types mse trimmed
E7 : python run_track.py --scene juggle --exp E7 --kidnap --inject 0 1
no odometry (constant velocity): add --motion cv
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm

import config
from common import CSVLog, Timer, device, set_seed, parse_maps, result_path, vis_path
from scene import load_scene
from render import scale_K, render_batch, set_backend, get_backend
from se3 import se3_exp, se3_inv
from observations import synth_observation, downsample
from trajectories import (cam_centers_and_c2w, sorted_by_azimuth, best_starts, make_trajectory,
                          odometry, kidnap_trajectory, relative_motions, local_prior, GlobalPrior)
from pf import ParticleFilter
from ekf import PoseEKF
from metrics import pose_errors, nees, tangent_error, covered


def get_args():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--scene", default=config.TUNE_SCENE)
    a.add_argument("--exp", default="E2")
    a.add_argument("--methods", nargs="+", default=["pf"])
    a.add_argument("--n_traj", type=int, default=3)
    a.add_argument("--n_frames", type=int, default=config.N_FRAMES)
    a.add_argument("--seeds", type=int, default=len(config.SEEDS))
    a.add_argument("--odo", nargs="+", default=["medium"], choices=list(config.ODO_LEVELS))
    a.add_argument("--motion", default="odo", choices=["odo", "cv"])
    a.add_argument("--maps", nargs="+", default=["full:0"])
    a.add_argument("--N", type=int, nargs="+", default=[config.TRACK["N"]])
    a.add_argument("--lam", type=float, nargs="+", default=[config.TRACK["lam"]])
    a.add_argument("--noise", nargs="+",
                   default=[f"{config.TRACK['sig_p']:g}:{np.rad2deg(config.TRACK['sig_r']):g}"],
                   help="process noise on top of odometry as sig_p:sig_r_deg, e.g. 0.01:0.5 0.02:1 (Stage 1)")
    a.add_argument("--likelihood", default="tempered", choices=["tempered", "locnerf"])
    a.add_argument("--loss_types", nargs="+", default=["trimmed"], choices=["mse", "trimmed", "huber"])
    a.add_argument("--occ", type=float, nargs="+", default=[0.0], help="occluder coverage fractions (E6)")
    a.add_argument("--kidnap", action="store_true", help="E7: teleport the true camera at KIDNAP_FRAME")
    a.add_argument("--inject", type=int, nargs="+", default=[0])
    a.add_argument("--W", type=int, default=None)
    a.add_argument("--H", type=int, default=None)
    a.add_argument("--backend", default=None, choices=["gsplat", "torch"])
    a.add_argument("--save_vis", type=int, default=0)
    a.add_argument("--preload", action="store_true")
    return a.parse_args()


def loss_cfg(name):
    return dict(mse=(0.0, "mse"), trimmed=(config.TRACK["trim"], "mse"), huber=(0.0, "huber"))[name]


def build_trajectories(train_md, n_traj, n_frames, kidnap, dev):
    c2ws = cam_centers_and_c2w(train_md)
    n = config.TRACK["n_cams"]
    starts, gaps = best_starts(c2ws, n, k=n_traj + (1 if kidnap else 0))
    order = sorted_by_azimuth(c2ws)
    trajs = []
    for j, s in enumerate(starts[:n_traj]):
        T = make_trajectory(c2ws, n, s, n_frames, config.TRACK["wobble"], seed=j)
        U = relative_motions(T)
        if kidnap:                                               # jump to another segment
            other = starts[(j + 1) % len(starts)]
            T_b = make_trajectory(c2ws, n, other, n_frames, config.TRACK["wobble"], seed=100 + j)
            T, U = kidnap_trajectory(T, T_b, config.KIDNAP_FRAME)
        cam = int(order[s])
        K = torch.tensor(train_md["k"][0][cam], dtype=torch.float32, device=dev)
        trajs.append(dict(id=j, start=s, max_gap=gaps[j], T=T, U=U, K=K))
        print(f"trajectory {j}: start={s} cameras={order[s:s + n].tolist()} max angular gap={gaps[j]:.1f} deg", flush=True)
    return trajs


def main():
    args = get_args()
    dev = device()
    if args.backend:
        set_backend(args.backend)
    small = get_backend() == "torch"
    W = args.W or (config.SYNTH_W if small else config.PF_W)
    H = args.H or (config.SYNTH_H if small else config.PF_H)
    scene, train_md, _ = load_scene(args.scene, dev, preload=args.preload)
    n_frames = min(args.n_frames, scene.T)
    W0, H0 = train_md["w"], train_md["h"]
    prior = GlobalPrior(cam_centers_and_c2w(train_md), config.GLOBAL_PRIOR["rot_noise_deg"], dev)
    maps = parse_maps(args.maps)
    log = CSVLog(result_path(args.exp, args.scene, "track"))
    trajs = build_trajectories(train_md, args.n_traj, n_frames, args.kidnap, dev)
    TR, E = config.TRACK, config.EKF
    vis_left = args.save_vis

    for tr in trajs:
        K = scale_K(tr["K"], W0, H0, W, H)
        K_ekf = scale_K(tr["K"], W0, H0, config.EKF_W, config.EKF_H)
        T_gt = torch.tensor(tr["T"], dtype=torch.float32, device=dev)
        # fraction of pixels covered by people at the true pose (explains when 4D matters, E3)
        fg = [render_batch(scene.fg_frame(t), se3_inv(T_gt[t])[None], K, W, H)[0, ..., 0].mean().item()
              for t in range(n_frames)]
        for seed in range(args.seeds):
            for occ in args.occ:
                rng = np.random.RandomState(10_000 * tr["id"] + 100 * seed + int(occ * 100))
                set_seed(10_000 * tr["id"] + seed)
                obs = [synth_observation(scene.frame(t), T_gt[t], tr["K"], W0, H0, W, H, rng,
                                         obs_wh=(2 * W, 2 * H), coverage=occ, **config.OBS_CORRUPT)
                       for t in range(n_frames)]
                obs_ekf = [downsample(o, config.EKF_W, config.EKF_H) for o in obs]
                for odo_name in args.odo:
                    sp, sr = config.ODO_LEVELS[odo_name]
                    U_odo = odometry(tr["T"], sp, np.deg2rad(sr), np.random.default_rng(seed), U_true=tr["U"])
                    U_odo = torch.tensor(U_odo, dtype=torch.float32, device=dev)
                    init_xi = torch.cat([torch.randn(3) * TR["init_sig_p"], torch.randn(3) * TR["init_sig_r"]])
                    T_init = T_gt[0] @ se3_exp(init_xi.to(dev))
                    for noise in args.noise:
                        jp, jr = (float(v) for v in noise.split(":"))
                        # process noise = odometry error + diversity jitter
                        q_p, q_r = np.hypot(jp, sp), np.hypot(np.deg2rad(jr), np.deg2rad(sr))
                        base = dict(scene=args.scene, mode="track", traj=tr["id"], seed=seed, odo=odo_name,
                                    motion=args.motion, occ=occ, kidnap=int(args.kidnap), noise=noise,
                                    backend=get_backend(), W=W, H=H)
                        for map_mode, offset in maps:
                            b = dict(base, map_mode=map_mode, offset=offset)
                            for m in args.methods:
                                if m == "pf":
                                    for N in args.N:
                                        for lam in args.lam:
                                            for loss in args.loss_types:
                                                for inj in args.inject:
                                                    set_seed(10_000 * tr["id"] + seed)
                                                    vis = vis_left > 0
                                                    run_pf(scene, map_mode, offset, obs, U_odo, T_gt, T_init, K, W, H,
                                                           N, lam, loss, inj, q_p, q_r, prior, fg, args, log,
                                                           dict(b, N=N, lam=lam, loss=loss, inject=inj),
                                                           vis_name=(f"track_{args.exp}_{args.scene}_traj{tr['id']}_s{seed}"
                                                                     f"_{odo_name}_occ{occ:g}_{map_mode}{offset}_N{N}"
                                                                     f"_lam{lam:g}_{loss}_inj{inj}_q{noise.replace(':', '-')}")
                                                                     if vis else None)
                                                    vis_left -= int(vis)
                                else:
                                    set_seed(10_000 * tr["id"] + seed)
                                    run_ekf(scene, map_mode, offset, obs_ekf, U_odo, T_gt, T_init, K_ekf,
                                            1 if m == "ekf" else E["iters_iekf"], q_p, q_r, fg, args, log,
                                            dict(b, N=1, lam=np.nan, loss="gray", inject=0, method=m))
                log.flush()
    log.flush()
    print("wrote", log.path, flush=True)


def _log_step(log, b, t, T_hat, P, T_gt, fg, **extra):
    te, re = pose_errors(T_hat, T_gt)
    e = tangent_error(T_hat, T_gt)
    sd = torch.sqrt(torch.diagonal(P).clamp(min=0))                    # per-axis sigma (same frame as e)
    log.add(**b, step=t, te=te.item(), re=re.item(),
            e_x=e[0].item(), e_y=e[1].item(), e_z=e[2].item(),
            sd_x=sd[0].item(), sd_y=sd[1].item(), sd_z=sd[2].item(),
            sig_t=torch.sqrt(torch.trace(P[:3, :3]).clamp(min=0)).item(),
            sig_r=np.rad2deg(torch.sqrt(torch.trace(P[3:, 3:]).clamp(min=0)).item()),
            nees=nees(T_hat, P, T_gt), covered=int(covered(T_hat, P, T_gt)), fg_frac=fg[t], **extra)


def run_pf(scene, map_mode, offset, obs, U_odo, T_gt, T_init, K, W, H, N, lam, loss, inject,
           q_p, q_r, prior, fg, args, log, b, vis_name=None):
    TR = config.TRACK
    trim, lossf = loss_cfg(loss)
    X0 = local_prior(T_init, N, TR["init_sig_p"], TR["init_sig_r"])
    pf = ParticleFilter(X0, q_p, q_r, lam, trim=trim, likelihood=args.likelihood, loss=lossf)
    b = dict(b, method="pf")
    est = []
    vis = dict(est_render=[], particles=[], weights=[], est=[]) if vis_name else None
    for t in range(len(obs)):
        fr = scene.frame(t, mode=map_mode, offset=offset)              # the 4D part
        with Timer() as tm:
            injected = False
            if t > 0:
                if args.motion == "odo":
                    pf.predict(U_odo[t])
                else:                                                   # constant velocity, larger Q
                    U = se3_inv(est[-2]) @ est[-1] if len(est) >= 2 else None
                    pf.predict(U, scale=TR["cv_scale"])
                if inject:
                    injected = pf.maybe_inject(prior, TR["inject_err"], TR["inject_patience"], TR["inject_frac"])
            pf.update(fr, obs[t], K, W, H)
            ess = pf.ess()
            if ess < pf.N / 2:
                pf.resample()
            T_hat, P = pf.estimate()
        est.append(T_hat)
        _log_step(log, b, t, T_hat, P, T_gt[t], fg, ess=ess, err_best=pf.err.min().item(),
                  injected=int(bool(injected)), time_ms=tm.ms, n_renders=pf.n_renders)
        if vis is not None:
            vis["est_render"].append(render_batch(fr, se3_inv(T_hat)[None], K, W, H)[0].cpu().numpy())
            sub = torch.randperm(pf.N)[:200] if pf.N >= 200 else torch.randint(pf.N, (200,))
            vis["particles"].append(pf.X[sub, :3, 3].cpu().numpy())
            vis["weights"].append(pf.w[sub].cpu().numpy())
            vis["est"].append(T_hat.cpu().numpy())
    _report(b, est, T_gt)
    if vis is not None:
        bg, bgc = scene.background_points(0, 3000)
        np.savez_compressed(vis_path(vis_name), obs=torch.stack(obs).cpu().numpy(),
                            T_gt=T_gt.cpu().numpy(), bg=bg, bg_col=bgc, **{k: np.stack(v) for k, v in vis.items()})


def run_ekf(scene, map_mode, offset, obs_ekf, U_odo, T_gt, T_init, K_ekf, iters, q_p, q_r, fg, args, log, b):
    TR, E = config.TRACK, config.EKF
    dev = T_init.device
    P0 = torch.diag(torch.tensor([TR["init_sig_p"]] * 3 + [TR["init_sig_r"]] * 3, device=dev) ** 2)
    Q = torch.diag(torch.tensor([q_p] * 3 + [q_r] * 3, device=dev, dtype=torch.float32) ** 2)
    ekf = PoseEKF(T_init, P0.float(), Q, E["sigma_z"], eps=E["eps"], iters=iters, blur_sigma=E["blur_sigma"])
    est = []
    for t in range(len(obs_ekf)):
        fr = scene.frame(t, mode=map_mode, offset=offset)
        with Timer() as tm:
            if t > 0:
                if args.motion == "odo":
                    ekf.predict(U_odo[t])
                else:
                    U = se3_inv(est[-2]) @ est[-1] if len(est) >= 2 else torch.eye(4, device=dev)
                    ekf.predict(U)
            ekf.update(fr, obs_ekf[t], K_ekf, config.EKF_W, config.EKF_H)
        if not torch.isfinite(ekf.T).all():
            print("EKF diverged numerically at frame", t)
            break
        est.append(ekf.T.clone())
        _log_step(log, b, t, ekf.T, ekf.P, T_gt[t], fg, ess=np.nan, err_best=ekf.err.item(),
                  injected=0, time_ms=tm.ms, n_renders=ekf.n_renders)
    _report(b, est, T_gt)


def _report(b, est, T_gt):
    """One progress line per finished run (flushed, so it shows up through pipes / tee / nohup)."""
    if not est:
        print(f"[{b['method']}] traj {b['traj']} seed {b['seed']}: diverged at frame 0", flush=True)
        return
    te, re = pose_errors(torch.stack(est), T_gt[:len(est)])
    cfg = (f" N={b['N']} lam={b['lam']:g} noise={b['noise']} loss={b['loss']}" if b["method"] == "pf"
           else f" noise={b['noise']}")
    print(f"[{b['method']}] traj {b['traj']} seed {b['seed']} odo={b['odo']} map={b['map_mode']}{b['offset']:+d}"
          f"{cfg}: ATE {torch.sqrt((te ** 2).mean()).item():.4f}, mean rot {re.mean().item():.2f} deg, "
          f"{len(est)} frames", flush=True)


if __name__ == "__main__":
    main()
