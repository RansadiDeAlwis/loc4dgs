"""Week-1 go/no-go checks (guide Step 3 "do not skip", Section 4.5, Step 5, Section 10.1).

    python sanity_check.py --scene juggle             # all checks
    python sanity_check.py --scene juggle --only render units

1. render : render test camera 0 at t=0 at full 640x360 with its real K and w2c, next to the
            real photo.  If the render is blank, flipped or shifted, the pose convention is wrong.
2. psnr   : PSNR on the TRAINING cameras (with their learned colour correction
            corrected = exp(cam_m[c]) * render + cam_c[c]); expect the high twenties.
3. units  : camera-centre distances, to decide whether scene units are metres.
4. chunk  : how many particles render per call without running out of memory, and renders/s.
5. traj   : render a preview video of each tracking trajectory; discard broken ones.
6. prior  : describe the global prior and check that its samples see the scene (non-empty renders).
"""
import argparse
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from scene import load_scene, camera, unit_report
from render import render_batch, scale_K, get_backend
from se3 import se3_inv
from observations import load_obs
from trajectories import cam_centers_and_c2w, best_starts, make_trajectory, GlobalPrior

OUT = os.path.join(config.RESULTS, "sanity")


def psnr(a, b):
    return (10 * torch.log10(1.0 / ((a - b) ** 2).mean())).item()


def check_render(name, scene, md, dev):
    K, c2w, fn, cam_id = camera(md, 0, 0, dev)
    W, H = md["w"], md["h"]
    img = render_batch(scene.frame(0), se3_inv(c2w)[None], K, W, H)[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
    ax[1].imshow(img.cpu().numpy()); ax[1].set_title("render at test camera pose, t=0")
    if fn is not None:
        real = load_obs(config.image_path(name, fn), W, H, dev)
        ax[0].imshow(real.cpu().numpy()); ax[0].set_title(f"real image (cam {cam_id})")
        ax[2].imshow((real - img).abs().mean(-1).cpu().numpy(), cmap="magma"); ax[2].set_title("|difference|")
        print(f"[render] test cam {cam_id}: PSNR vs real photo (no colour correction) = {psnr(img, real):.2f} dB")
    for a in ax:
        a.axis("off")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"render_{name}.png"); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[render] mean intensity {img.mean().item():.3f} (blank render ~0 means a wrong w2c/c2w convention) -> {p}")


def check_psnr(name, scene, md, dev, times=(0, 75, 149)):
    if md["fn"][0][0] is None:
        print("[psnr] synthetic scene: skipped"); return
    from observations import photometric_error
    raw, cor, nerr = [], [], []
    for t in times:
        t = min(t, scene.T - 1)
        for c in range(len(md["fn"][t])):
            K, c2w, fn, cam_id = camera(md, t, c, dev)
            img = render_batch(scene.frame(t), se3_inv(c2w)[None], K, md["w"], md["h"])[0]
            real = load_obs(config.image_path(name, fn), md["w"], md["h"], dev)
            raw.append(psnr(img, real))
            nerr.append(photometric_error(img[None], real).item())
            if scene.cam_m is not None:
                imc = (torch.exp(torch.tensor(scene.cam_m[cam_id], device=dev)) * img
                       + torch.tensor(scene.cam_c[cam_id], device=dev)).clamp(0, 1)
                cor.append(psnr(imc, real))
    print(f"[psnr] training cameras ({len(raw)} images): raw render mean {np.mean(raw):.2f} dB; "
          f"with learned colour correction mean {np.mean(cor):.2f} dB (expect high twenties)")
    bad = [(i, e) for i, e in enumerate(nerr) if e > 0.3]
    print(f"[psnr] filter's normalized error: median {np.median(nerr):.3f} (0 = perfect, ~1 = unrelated); "
          f"{len(bad)} image(s) above 0.3")


def check_chunk(scene, md, dev, W=config.PF_W, H=config.PF_H):
    K, c2w, _, _ = camera(md, 0, 0, dev)
    Ks = scale_K(K, md["w"], md["h"], W, H)
    poses = se3_inv(c2w)[None].repeat(1000, 1, 1)
    fr = scene.frame(0)
    for chunk in [512, 256, 128, 64, 32]:
        try:
            render_batch(fr, poses[:chunk], Ks, W, H, chunk=chunk)
            torch.cuda.synchronize() if dev == "cuda" else None
            t0 = time.perf_counter()
            render_batch(fr, poses[:600], Ks, W, H, chunk=chunk)
            torch.cuda.synchronize() if dev == "cuda" else None
            dt = time.perf_counter() - t0
            print(f"[chunk] chunk={chunk}: 600 renders at {W}x{H} in {dt * 1000:.0f} ms "
                  f"({600 / dt:.0f} renders/s) -> set RENDER_CHUNK={chunk} in config.py")
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[chunk] chunk={chunk}: out of memory")


def check_traj(name, scene, train_md, dev, n_traj=3, W=320, H=180):
    import imageio.v2 as imageio
    c2ws = cam_centers_and_c2w(train_md)
    starts, gaps = best_starts(c2ws, config.TRACK["n_cams"], k=n_traj)
    os.makedirs(OUT, exist_ok=True)
    from trajectories import sorted_by_azimuth
    order = sorted_by_azimuth(c2ws)
    for j, (s, g) in enumerate(zip(starts, gaps)):
        T = make_trajectory(c2ws, config.TRACK["n_cams"], s, scene.T)
        K = scale_K(torch.tensor(train_md["k"][0][int(order[s])], dtype=torch.float32, device=dev),
                    train_md["w"], train_md["h"], W, H)
        Tt = torch.tensor(T, dtype=torch.float32, device=dev)
        frames = [(render_batch(scene.frame(t), se3_inv(Tt[t])[None], K, W, H)[0].cpu().numpy() * 255).astype(np.uint8)
                  for t in range(0, scene.T, 2)]
        p = os.path.join(OUT, f"traj_{name}_{j}.mp4")
        imageio.mimsave(p, frames, fps=15, macro_block_size=1)
        step = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
        print(f"[traj] {j}: start={s}, max angular gap {g:.1f} deg, path length {step.sum():.2f}, "
              f"mean step {step.mean():.4f} units/frame -> {p}  (watch it; discard broken ones)")


def check_prior(scene, train_md, test_md, dev, n=64):
    prior = GlobalPrior(cam_centers_and_c2w(train_md), config.GLOBAL_PRIOR["rot_noise_deg"], dev)
    print("[prior]", prior.describe())
    X = prior(n)
    K, _, _, _ = camera(test_md, 0, 0, dev)
    Ks = scale_K(K, test_md["w"], test_md["h"], config.PF_W, config.PF_H)
    imgs = render_batch(scene.frame(0), se3_inv(X), Ks, config.PF_W, config.PF_H)
    m = imgs.mean((1, 2, 3))
    print(f"[prior] fraction of prior samples with a non-empty view: {(m > 0.02).float().mean().item():.0%}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--scene", default=config.TUNE_SCENE)
    a.add_argument("--only", nargs="*", default=["render", "psnr", "units", "chunk", "traj", "prior"])
    args = a.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("backend:", get_backend(), "| device:", dev)
    scene, train_md, test_md = load_scene(args.scene, dev)
    print(f"scene {args.scene}: {scene.N} Gaussians ({int(scene.is_fg.sum())} foreground), T={scene.T}")
    if "render" in args.only: check_render(args.scene, scene, test_md, dev)
    if "psnr" in args.only: check_psnr(args.scene, scene, train_md, dev)
    if "units" in args.only: unit_report(train_md)
    if "chunk" in args.only and dev == "cuda": check_chunk(scene, test_md, dev)
    if "traj" in args.only: check_traj(args.scene, scene, train_md, dev)
    if "prior" in args.only: check_prior(scene, train_md, test_md, dev)


if __name__ == "__main__":
    main()
