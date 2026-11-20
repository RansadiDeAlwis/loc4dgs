"""Figures, tables and animations from the result CSVs (guide Section 8).

Every figure makes ONE point (stated in its title / caption below) and is produced only from
results/*.csv and results/vis/*.npz, so re-running and fixing plots is painless.

    python plots.py all                       # every figure / table whose CSVs exist
    python plots.py f1                        # pipeline diagram
    python plots.py f2 f3 --vis results/vis/reloc_....npz
    python plots.py f4 --scene juggle         # likelihood landscape (needs the renderer)
    python plots.py gifs                      # animations for every vis file
    python plots.py site                      # GitHub Pages demo (docs/index.html)

Figures                                         point it makes
F1  pipeline diagram                            how the parts connect
F2  observed / render at estimate / |diff|      the likelihood is visual and intuitive
F3  particle cloud snapshots                    multimodality and collapse
F4  likelihood landscape (yaw x translation)    why the EKF fails: many local minima
F5  success rate vs initial error (E1)          PF beats EKF from bad starts
F6  error vs time with +-3 sigma (E2)           tracking accuracy and honest uncertainty
F6b one representative run, per axis, +-3 sigma  same point, on a single run (python plots.py f6b)
F7  ATE / error per map type (E3)               whether 4D matters (+ error vs people coverage)
F8  accuracy and runtime vs N (E4)              cost / accuracy trade-off
F9  ESS traces for several lambda (E5)          degeneracy vs over-diffusion
F10 average NEES with 95% chi^2 band (E2)       consistency
F11 error vs time during kidnapping (E7)        recovery with injection
F12 error vs occluder coverage (E6)             robustness of the trimmed error
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config
from metrics import nees_bounds, convergence_step

FIG = os.path.join(config.RESULTS, "figures")
TAB = os.path.join(config.RESULTS, "tables")
GIF = os.path.join(config.RESULTS, "gifs")
UNIT = "m"                    # sanity_check units report: camera distances up to 5.5 -> metres

RELOC_KEYS = ["scene", "method", "level", "img", "seed", "map_mode", "offset", "N0", "lam0", "anneal", "loss",
              "noise_frac"]
TRACK_KEYS = ["scene", "method", "traj", "seed", "odo", "motion", "occ", "map_mode", "offset", "N", "lam",
              "loss", "inject", "kidnap", "noise"]
LEVEL_ORDER = ["0.1u_5deg", "0.3u_20deg", "1u_45deg", "global"]


# --------------------------------------------------------------------------- helpers
def load(exp):
    files = sorted(glob.glob(os.path.join(config.RESULTS, f"{exp}_*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def keys(df):
    k = RELOC_KEYS if df["mode"].iloc[0] == "reloc" else TRACK_KEYS
    return [c for c in k if c in df.columns]


def final_rows(df):
    """Last iteration of every relocalization run."""
    return df.sort_values("step").groupby(keys(df), dropna=False).tail(1).copy()


def per_run(df):
    """One row per tracking run: ATE, mean rotation error, success, mean NEES, runtime."""
    g = df.groupby(keys(df), dropna=False)
    out = g.agg(ate=("te", lambda x: float(np.sqrt((x ** 2).mean()))), rot=("re", "mean"),
                nees=("nees", "mean"), coverage=("covered", "mean"), ms=("time_ms", "median"),
                fg=("fg_frac", "mean")).reset_index()
    return out


def add_success(f):
    f["success"] = ((f["te"] < config.SUCCESS_T) & (f["re"] < config.SUCCESS_R)).astype(float)
    return f


def map_label(r):
    if r["map_mode"] == "full" and r["offset"] == 0:
        return "full 4D"
    if r["map_mode"] == "full":
        return f"4D offset {int(r['offset']):+d}"
    return {"background": "background only", "frozen0": "frozen t=0"}.get(r["map_mode"], r["map_mode"])


REPORT_FIG = os.path.join(config.ROOT, "report", "figures")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(p + ".pdf", bbox_inches="tight")
    # report copy: drop the "F<k>. ..." headline titles (the LaTeX caption states the point)
    import re
    if fig._suptitle is not None and re.match(r"F\d+b?\.", fig._suptitle.get_text()):
        fig._suptitle.set_text("")
    for ax in fig.axes:
        if re.match(r"F\d+b?\.", ax.get_title()):
            ax.set_title("")
    # per-run figures (F2/F3) and names with extra dots are copied into the report by hand
    if os.path.isdir(REPORT_FIG) and not name.startswith(("F2_", "F3_")) and "." not in name:
        fig.savefig(os.path.join(REPORT_FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", p + ".png")


def table(df, name):
    os.makedirs(TAB, exist_ok=True)
    p = os.path.join(TAB, name)
    df.to_csv(p + ".csv", index=False)
    with open(p + ".md", "w") as f:
        try:
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
        except ImportError:                                   # tabulate not installed
            f.write(df.to_string(index=False, float_format="%.3f"))
    print("saved", p + ".md")


def summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return pd.Series(dict(mean=np.nan, median=np.nan, q1=np.nan, q3=np.nan, n=0))
    return pd.Series(dict(mean=x.mean(), median=np.median(x), q1=np.percentile(x, 25),
                          q3=np.percentile(x, 75), n=x.size))


def band(ax, x, med, lo, hi, color, label, ls="-"):
    ax.plot(x, med, color=color, label=label, ls=ls)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)


def color(method):
    return config.METHOD_COLORS.get(method, "k")


# --------------------------------------------------------------------------- F1
def f1():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    boxes = [("Observed image z_t\n(downsample 160x90,\nnormalize per channel)", 0.02, 0.62),
             ("Particles x_t^i in SE(3)\npredict: x U~ Exp(delta)", 0.02, 0.12),
             ("4D map G(t)\n(Dynamic 3D Gaussians\nat time t)", 0.25, 0.62),
             ("Batch render\n(gsplat)  I^i", 0.25, 0.12),
             ("Photometric error e^i\n(trimmed MSE)", 0.46, 0.37),
             ("Weights\nlog w += -lambda e^i\n(log-softmax)", 0.64, 0.37),
             ("Resample if\nN_eff < N/2\n(systematic)", 0.81, 0.62),
             ("Estimate\nmean p, Markley q,\ncovariance P", 0.81, 0.12)]
    for txt, x, y in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.16, 0.26, boxstyle="round,pad=0.01", fc="#e8f0fb", ec="#3b6fb6"))
        ax.text(x + 0.08, y + 0.13, txt, ha="center", va="center", fontsize=8.5)
    arrows = [((0.18, 0.25), (0.25, 0.25)), ((0.33, 0.62), (0.33, 0.38)), ((0.41, 0.25), (0.46, 0.45)),
              ((0.18, 0.75), (0.46, 0.55)), ((0.62, 0.5), (0.64, 0.5)), ((0.80, 0.55), (0.82, 0.62)),
              ((0.80, 0.45), (0.82, 0.38)), ((0.89, 0.62), (0.89, 0.38)),
              ((0.81, 0.18), (0.18, 0.18))]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(0.5, 0.03, "next time step: t <- t+1 (tracking)  /  next iteration on the same image (relocalization)",
            ha="center", fontsize=8, style="italic")
    ax.set_title("F1. Particle filter localization in a 4D Gaussian map")
    save(fig, "F1_pipeline")


# --------------------------------------------------------------------------- F2, F3 (visualization files)
def _norm_img(x):
    x = (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-6)
    return x


def f2(vis):
    d = np.load(vis)
    obs = d["obs"] if d["obs"].ndim == 3 else d["obs"][0]
    steps = [0, len(d["est_render"]) - 1]
    fig, axs = plt.subplots(2, 3, figsize=(11, 4.4))
    for r, k in enumerate(steps):
        o = d["obs"] if d["obs"].ndim == 3 else d["obs"][k]
        ren = d["est_render"][k]
        diff = np.abs(_norm_img(o) - _norm_img(ren)).mean(-1)
        axs[r, 0].imshow(o); axs[r, 1].imshow(ren)
        im = axs[r, 2].imshow(diff, cmap="magma", vmin=0, vmax=2)
        axs[r, 0].set_ylabel("start" if r == 0 else "after convergence")
    for a, t in zip(axs[0], ["observed image", "render at estimated pose", "|normalized difference|"]):
        a.set_title(t)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axs[:, 2], shrink=0.8)
    fig.suptitle("F2. The likelihood is visual: the difference vanishes as the estimate converges")
    save(fig, "F2_" + os.path.basename(vis)[:-4])


def _plane(d):
    """Two axes spanning the floor (largest PCA components of the background points)."""
    bg = d["bg"]
    mu = bg.mean(0)
    _, _, Vt = np.linalg.svd(bg - mu, full_matrices=False)
    return mu, Vt[:2]


def f3(vis, n_snap=6):
    d = np.load(vis)
    mu, A = _plane(d)
    P = d["particles"]
    ks = np.unique(np.linspace(0, len(P) - 1, n_snap).astype(int))
    fig, axs = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.4))
    bg2 = (d["bg"] - mu) @ A.T
    gt = d["T_gt"]
    gt2 = ((gt[..., :3, 3] if gt.ndim == 2 else gt[:, :3, 3]).reshape(-1, 3) - mu) @ A.T
    # zoom on the particles: the cloud is centimetres wide, the dome metres
    allp = np.concatenate([(P[k] - mu) @ A.T for k in ks] + [gt2])
    lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.25 + 1e-3
    for a, k in zip(np.atleast_1d(axs), ks):
        inside = (np.abs(bg2 - c) < half).all(1)
        a.scatter(bg2[inside, 0], bg2[inside, 1], s=1, c=np.clip(d["bg_col"][inside], 0, 1), alpha=0.3)
        p2 = (P[k] - mu) @ A.T
        w = d["weights"][k]
        a.scatter(p2[:, 0], p2[:, 1], s=6, c=w, cmap="viridis")
        if "particle_dirs" in d:                                  # little frusta = viewing direction
            v = d["particle_dirs"][k] @ A.T * 0.12 * half
            a.quiver(p2[::4, 0], p2[::4, 1], v[::4, 0], v[::4, 1], width=0.003, color="k", alpha=0.4,
                     angles="xy", scale_units="xy", scale=1)
        g = gt2[min(k, len(gt2) - 1)]
        a.plot(g[0], g[1], "r*", ms=12, label="truth")
        a.set_xlim(c[0] - half, c[0] + half); a.set_ylim(c[1] - half, c[1] + half)
        a.set_title(f"iteration / frame {k}"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a0 = np.atleast_1d(axs)[0]
    a0.plot([c[0] - half * 0.9, c[0] - half * 0.9 + 0.1], [c[1] - half * 0.9] * 2, "k-", lw=2)
    a0.text(c[0] - half * 0.9 + 0.05, c[1] - half * 0.82, f"10 cm", ha="center", fontsize=7)
    a0.legend(loc="upper left", fontsize=7)
    fig.suptitle("F3. The particle cloud keeps several hypotheses, then collapses on the true pose (top view)")
    save(fig, "F3_" + os.path.basename(vis)[:-4])


# --------------------------------------------------------------------------- F4 likelihood landscape
def f4(scene_name, t=0, cam=0, n=41, x_range=0.5, yaw_range=30.0, map_mode="full"):
    import torch
    from scene import load_scene, camera
    from render import scale_K, render_batch, get_backend
    from se3 import se3_exp, se3_inv
    from observations import load_obs, synth_observation, photometric_error
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    small = get_backend() == "torch"
    W, H = (config.SYNTH_W, config.SYNTH_H) if small else (config.PF_W, config.PF_H)
    scene, _, md = load_scene(scene_name, dev)
    K, T_gt, fn, _ = camera(md, t, cam, dev)
    Ks = scale_K(K, md["w"], md["h"], W, H)
    if fn is not None:
        obs = load_obs(config.image_path(scene_name, fn), W, H, dev)
    else:
        obs = synth_observation(scene.frame(t), T_gt, K, md["w"], md["h"], W, H, np.random.RandomState(0),
                                obs_wh=(2 * W, 2 * H), **config.OBS_CORRUPT)
    xs = np.linspace(-x_range, x_range, n)
    yaws = np.deg2rad(np.linspace(-yaw_range, yaw_range, n))
    X, Y = np.meshgrid(xs, yaws)
    xi = np.zeros((n * n, 6)); xi[:, 0] = X.ravel(); xi[:, 4] = Y.ravel()   # x (camera right), yaw (about camera y)
    poses = T_gt[None] @ se3_exp(torch.tensor(xi, dtype=torch.float32, device=dev))
    fr = scene.frame(t, map_mode)
    err = torch.cat([photometric_error(render_batch(fr, se3_inv(poses[i:i + 256]), Ks, W, H), obs)
                     for i in range(0, len(poses), 256)]).cpu().numpy().reshape(n, n)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(err, origin="lower", aspect="auto", cmap="viridis",
                   extent=[xs[0], xs[-1], np.rad2deg(yaws[0]), np.rad2deg(yaws[-1])])
    ax.plot(0, 0, "r*", ms=14, label="true pose")
    ax.contour(X, np.rad2deg(Y), err, levels=12, colors="w", linewidths=0.4, alpha=0.6)
    ax.set_xlabel(f"x offset (camera right) [{UNIT}]"); ax.set_ylabel("yaw offset [deg]")
    fig.colorbar(im, label="photometric error e")
    ax.legend(loc="upper right")
    ax.set_title(f"F4. Likelihood landscape over x and yaw ({scene_name}, t={t}, test cam {cam})", fontsize=9)
    np.savez(os.path.join(config.RESULTS, f"landscape_{scene_name}_t{t}_c{cam}.npz"), err=err, xs=xs, yaws=yaws)
    save(fig, f"F4_landscape_{scene_name}")


# --------------------------------------------------------------------------- F5 (E1)
def f5(df):
    f = add_success(final_rows(df[(df.map_mode == "full") & (df.offset == 0)]))
    conv = {k: convergence_step(g.te.values, g.re.values, config.SUCCESS_T, config.SUCCESS_R)
            for k, g in df.sort_values("step").groupby(keys(df), dropna=False)}
    f["conv_step"] = [conv.get(tuple(r)) for r in f[keys(df)].itertuples(index=False)]
    f["conv_step"] = f["conv_step"].astype(float)
    tab = f.groupby(["scene", "method", "level"]).apply(
        lambda g: pd.Series(dict(success=g.success.mean(), conv_iters_median=g.conv_step.median(),
                                 te_median=g.te.median(),
                                 te_q1=g.te.quantile(.25), te_q3=g.te.quantile(.75),
                                 re_median=g.re.median(), ms_per_update=g.time_ms.median(),
                                 total_s_median=(g.elapsed_ms.median() / 1000 if "elapsed_ms" in g
                                                 else np.nan),
                                 renders=g.n_renders.median(), runs=len(g)))).reset_index()
    table(tab, "E1_relocalization")
    levels = [l for l in LEVEL_ORDER if l in f.level.unique()] + \
             [l for l in f.level.unique() if l not in LEVEL_ORDER]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in f.groupby("method"):
        s = g.groupby("level").success.mean().reindex(levels)
        # standard error over runs, as error bars
        n = g.groupby("level").success.count().reindex(levels)
        se = np.sqrt(s * (1 - s) / n)
        ax.errorbar(range(len(levels)), s.values, yerr=se.values, marker="o", capsize=3,
                    color=color(m), label=m.upper())
    ax.set_xticks(range(len(levels)), [l.replace("u_", " u / ").replace("deg", " deg") for l in levels])
    ax.set_xlabel("initial error"); ax.set_ylabel(f"success rate (< {config.SUCCESS_T} {UNIT}, < {config.SUCCESS_R} deg)")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("F5. The PF finds the pose from starts where the EKF cannot")
    save(fig, "F5_success_vs_init")


# --------------------------------------------------------------------------- F6, F10 (E2)
def f6(df):
    d = df[(df.map_mode == "full") & (df.offset == 0)]
    for odo, g0 in d.groupby("odo"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        for m, g in g0.groupby("method"):
            s = g.groupby("step")
            band(axs[0], s.te.median().index, s.te.median(), s.te.quantile(.25), s.te.quantile(.75), color(m),
                 f"{m.upper()} error (median, IQR)")
            axs[0].plot(s.sig_t.median().index, 3 * s.sig_t.median(), color=color(m), ls="--", lw=1,
                        label=f"{m.upper()} 3 sigma")
            band(axs[1], s.re.median().index, s.re.median(), s.re.quantile(.25), s.re.quantile(.75), color(m), m.upper())
            axs[1].plot(s.sig_r.median().index, 3 * s.sig_r.median(), color=color(m), ls="--", lw=1)
        axs[0].set_ylabel(f"translation error [{UNIT}]"); axs[1].set_ylabel("rotation error [deg]")
        for a in axs:
            a.set_xlabel("frame"); a.grid(alpha=0.3); a.set_yscale("log")
        axs[0].legend(fontsize=7)
        fig.suptitle(f"F6. Tracking a moving camera in a moving scene (odometry noise: {odo}); dashed = filter 3 sigma")
        save(fig, f"F6_tracking_error_{odo}")
    r = per_run(d)
    tab = r.groupby(["scene", "method", "odo"]).apply(
        lambda g: pd.Series(dict(ATE_mean=g.ate.mean(), ATE_median=g.ate.median(), ATE_q1=g.ate.quantile(.25),
                                 ATE_q3=g.ate.quantile(.75), rot_mean=g.rot.mean(), NEES=g.nees.mean(),
                                 coverage=g.coverage.mean(), ms=g.ms.median(), runs=len(g)))).reset_index()
    table(tab, "E2_tracking_ATE")


def f6_single(df):
    """One representative run per scene: the PF run whose ATE is the median of all PF runs, and the
    EKF on the SAME trajectory / seed / odometry.  Per-axis error with its own +-3 sigma band."""
    d = df[(df.map_mode == "full") & (df.offset == 0)]
    if "sd_x" not in d or d.empty:
        print("F6 single run: CSV has no per-axis sigma (sd_x); rerun E2 with the current run_track.py")
        return
    r = per_run(d)
    for scene, rs in r.groupby("scene"):
        pf_runs = rs[rs.method == "pf"]
        if pf_runs.empty:
            continue
        rep = pf_runs.iloc[(pf_runs.ate - pf_runs.ate.median()).abs().argsort().iloc[0]]
        same = ["scene", "traj", "seed", "odo", "motion", "occ"] + (["noise"] if "noise" in d else [])
        sel = d.copy()
        for c in same:
            sel = sel[sel[c] == rep[c]]
        methods = [m for m in ["pf", "ekf", "iekf"] if m in sel.method.unique()]
        fig, axs = plt.subplots(len(methods), 3, figsize=(12, 2.6 * len(methods)), squeeze=False, sharex=True)
        for i, m in enumerate(methods):
            g = sel[sel.method == m]
            if m == "pf":                                          # the chosen PF configuration only
                for c in ["N", "lam", "loss", "inject"]:
                    if c in g and c in rep:
                        g = g[g[c] == rep[c]]
            g = g.sort_values("step")
            for j, ax_name in enumerate("xyz"):
                ax = axs[i, j]
                sd = 3 * g[f"sd_{ax_name}"]
                ax.fill_between(g.step, -sd, sd, color=color(m), alpha=0.2, lw=0, label="+-3 sigma")
                ax.plot(g.step, g[f"e_{ax_name}"], color=color(m), lw=1, label="error")
                ax.axhline(0, color="k", lw=0.5)
                ax.set_title(f"{m.upper()}  {ax_name} error [{UNIT}]", fontsize=9)
                ax.grid(alpha=0.3)
                if i == len(methods) - 1:
                    ax.set_xlabel("frame")
            inside = np.mean([(g[f"e_{a}"].abs() <= 3 * g[f"sd_{a}"]).mean() for a in "xyz"])
            axs[i, 0].set_ylabel(f"{m.upper()}: {inside:.0%} inside")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"F6b. One representative run ({scene}, trajectory {rep.traj}, seed {rep.seed}, "
                     f"{rep.odo} odometry): error stays inside the filter's own 3 sigma", fontsize=10)
        save(fig, f"F6b_single_run_{scene}")


def f10(df):
    d = df[(df.map_mode == "full") & (df.offset == 0)]
    for odo, g0 in d.groupby("odo"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        R = None
        for m, g in g0.groupby("method"):
            s = g.groupby("step").nees
            R = s.count().min()
            ax.plot(s.mean().index, s.mean(), color=color(m), label=f"{m.upper()} (coverage {g.covered.mean():.0%})")
        if R:
            lo, hi = nees_bounds(R)
            ax.axhspan(lo, hi, color="gray", alpha=0.25, label=f"95% chi^2 band (R={R} runs)")
            ax.axhline(6, color="k", lw=0.8, ls=":")
        ax.set_yscale("log"); ax.set_xlabel("frame"); ax.set_ylabel("average NEES")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(f"F10. Consistency: above the band = overconfident ({odo} odometry)\n"
                     "(PF NEES is approximate: the particle belief is not Gaussian)", fontsize=9)
        save(fig, f"F10_nees_{odo}")


# --------------------------------------------------------------------------- F7 (E3)
def f7(df, tag):
    df = df.copy()
    df["map"] = df.apply(map_label, axis=1)
    order = ["full 4D", "background only", "frozen t=0"] + sorted(m for m in df["map"].unique() if "offset" in m)
    order = [o for o in order if o in df["map"].unique()]
    if df["mode"].iloc[0] == "track":
        r = per_run(df); r["map"] = r.apply(map_label, axis=1); val, lab = "ate", f"ATE [{UNIT}]"
    else:
        r = add_success(final_rows(df)); r["map"] = r.apply(map_label, axis=1); val, lab = "te", f"final error [{UNIT}]"
    scenes = sorted(r.scene.unique())
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(scenes)), 4))
    w = 0.8 / len(order)
    cmap = plt.get_cmap("tab10")
    for i, mp in enumerate(order):
        g = r[r["map"] == mp].groupby("scene")[val]
        med = g.median().reindex(scenes)
        lo, hi = g.quantile(.25).reindex(scenes), g.quantile(.75).reindex(scenes)
        x = np.arange(len(scenes)) + (i - (len(order) - 1) / 2) * w
        ax.bar(x, med, w, yerr=[med - lo, hi - med], capsize=2, label=mp, color=cmap(i))
    ax.set_xticks(range(len(scenes)), scenes); ax.set_ylabel(lab + " (median, IQR)"); ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"F7. Does the 4D map matter? ({tag})")
    save(fig, f"F7_map_ablation_{tag}")
    agg = dict(median=(val, "median"), q1=(val, lambda x: x.quantile(.25)), q3=(val, lambda x: x.quantile(.75)),
               mean=(val, "mean"), runs=(val, "size"))
    if "success" in r:
        agg["success"] = ("success", "mean")
    table(r.groupby(["scene", "map"]).agg(**agg).reset_index(), f"E3_maps_{tag}")
    # error vs fraction of pixels covered by people: explains WHEN 4D matters
    src = df if df["mode"].iloc[0] == "track" else r
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(0.05, src.fg_frac.max()), 8)
    for i, mp in enumerate(order):
        g = src[src["map"] == mp]
        b = pd.cut(g.fg_frac, bins)
        s = g.groupby(b, observed=True).te.median()
        ax.plot([iv.mid for iv in s.index], s.values, marker="o", color=cmap(i), label=mp)
    ax.set_xlabel("fraction of image covered by people"); ax.set_ylabel(f"translation error [{UNIT}] (median)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("F7b. Static maps hurt most when people fill the view")
    save(fig, f"F7b_error_vs_people_{tag}")


# --------------------------------------------------------------------------- F8 (E4)
def f8(df, tag):
    if df["mode"].iloc[0] == "reloc":
        # relocalization: N0 is the STARTING number of particles (all runs drop to N_small after the
        # first anneal), so per-update time is meaningless; use the total time per relocalization
        r = add_success(final_rows(df)); ncol, acc, lab = "N0", "success", "success rate"
        r["ms"] = r["elapsed_ms"] / 1000 if "elapsed_ms" in r else r["time_ms"]
    else:
        r = per_run(df); ncol, acc, lab = "N", "ate", f"ATE [{UNIT}]"
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    anneal = r["anneal"] if "anneal" in r else pd.Series(1, index=r.index)
    for a_val, g in r.groupby(anneal):
        s = g.groupby(ncol)
        name = "annealing on" if a_val else "annealing off"
        axs[0].plot(s[acc].mean().index, s[acc].mean() if acc == "success" else s[acc].median(), "o-",
                    label=name, ls="-" if a_val else "--", color="tab:blue")
        axs[1].plot(s.ms.median().index, s.ms.median(), "o-", ls="-" if a_val else "--", color="tab:red", label=name)
    for a in axs:
        a.set_xscale("log"); a.set_xlabel("number of particles N"); a.grid(alpha=0.3); a.legend(fontsize=8)
    axs[0].set_ylabel(lab); axs[1].set_yscale("log")
    axs[1].set_ylabel("time per relocalization [s]" if ncol == "N0" else "time per update [ms]")
    if ncol == "N0":
        axs[0].set_xlabel("initial number of particles N (drops to N_small after the first anneal)")
    fig.suptitle(f"F8. Accuracy saturates while runtime keeps growing with N ({tag})")
    save(fig, f"F8_particles_{tag}")
    table(r.groupby([ncol] + (["anneal"] if "anneal" in r else [])).agg(
        acc=(acc, "mean"), ms=("ms", "median"), runs=(acc, "size")).reset_index(), f"E4_particles_{tag}")


# --------------------------------------------------------------------------- F9 (E5)
def f9(df, tag):
    # tuning runs (Stage 1) vary the noise level too: one figure per noise level
    for ncol in ("noise", "noise_frac"):
        if ncol in df and df[ncol].dropna().nunique() > 1:
            for v, g in df.groupby(ncol):
                f9(g.drop(columns=ncol), f"{tag}_{ncol}{str(v).replace(':', '-')}")
            return
    lamcol = "lam0" if ("lam0" in df and df["lam0"].notna().any()) else "lam"
    cmap = plt.get_cmap("viridis")
    lams = sorted(df[lamcol].dropna().unique())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for loss, ls in [("trimmed", "-"), ("mse", "--"), ("huber", ":")]:
        g0 = df[df.loss == loss]
        if g0.empty:
            continue
        for i, lam in enumerate(lams):
            s = g0[g0[lamcol] == lam].groupby("step").ess
            if len(s):
                axs[0].plot(s.median().index, s.median(), ls=ls, color=cmap(i / max(1, len(lams) - 1)),
                            label=f"lambda={lam:g} ({loss})" if loss == "trimmed" or len(lams) < 4 else None)
        if df["mode"].iloc[0] == "reloc":
            r = add_success(final_rows(g0))
            fail = 1 - r.groupby(lamcol).success.mean()
            axs[1].plot(fail.index, fail.values, "o" + ls, label=loss)
        else:                                                     # tracking: accuracy vs lambda
            r = per_run(g0)
            ate = r.groupby("lam").ate
            axs[1].errorbar(ate.median().index, ate.median(), capsize=3, fmt="o" + ls, label=loss,
                            yerr=[ate.median() - ate.quantile(.25), ate.quantile(.75) - ate.median()])
    axs[0].set_yscale("log"); axs[0].set_xlabel("iteration / frame"); axs[0].set_ylabel("ESS (median)")
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].set_xscale("log"); axs[1].set_xlabel("lambda (initial lambda_0 in relocalization)")
    axs[1].set_ylabel("failure rate (not within 5 cm / 5 deg)" if lamcol == "lam0" else f"ATE [{UNIT}] (median, IQR)")
    axs[1].legend()
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(f"F9. Larger lambda concentrates the particles (lower ESS) and improves accuracy ({tag})")
    save(fig, f"F9_lambda_{tag}")


# --------------------------------------------------------------------------- F11 (E7), F12 (E6)
def f11(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for inj, g in df.groupby("inject"):
        s = g.groupby("step").te
        band(ax, s.median().index, s.median(), s.quantile(.25), s.quantile(.75),
             "tab:green" if inj else "tab:red", "with injection" if inj else "without injection")
        for k, run in g.groupby(keys(g), dropna=False):
            run = run.sort_values("step")
            after = run[run.step >= config.KIDNAP_FRAME]
            c = convergence_step(after.te.values, after.re.values, 3 * config.SUCCESS_T, 3 * config.SUCCESS_R)
            rows.append(dict(inject=inj, recovered=c is not None, frames_to_recovery=np.nan if c is None else c))
    ax.axvline(config.KIDNAP_FRAME, color="k", ls="--", label="kidnap")
    ax.set_yscale("log"); ax.set_xlabel("frame"); ax.set_ylabel(f"translation error [{UNIT}]")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("F11. Random particle injection lets the filter recover from a kidnapping")
    save(fig, "F11_kidnap")
    t = pd.DataFrame(rows).groupby("inject").agg(recovery_rate=("recovered", "mean"),
                                                 frames_median=("frames_to_recovery", "median"),
                                                 runs=("recovered", "size")).reset_index()
    table(t, "E7_kidnap")


def f12(df):
    r = per_run(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    for loss, g in r.groupby("loss"):
        s = g.groupby("occ").ate
        ax.errorbar(s.median().index * 100, s.median(), yerr=[s.median() - s.quantile(.25), s.quantile(.75) - s.median()],
                    marker="o", capsize=3, label=loss)
    ax.set_xlabel("occluder coverage [% of image]"); ax.set_ylabel(f"ATE [{UNIT}] (median, IQR)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("F12. The trimmed error keeps occluders from dominating the score")
    save(fig, "F12_occlusion")
    table(r.groupby(["occ", "loss"]).ate.apply(summary).unstack().reset_index(), "E6_occlusion")


# --------------------------------------------------------------------------- animations + site
def gif_from_vis(vis, fps=8):
    import imageio.v2 as imageio
    d = np.load(vis)
    os.makedirs(GIF, exist_ok=True)
    out = os.path.join(GIF, os.path.basename(vis)[:-4] + ".gif")
    mu, A = _plane(d)
    bg2 = (d["bg"] - mu) @ A.T
    gt = d["T_gt"].reshape(-1, 4, 4)
    frames = []
    fig = plt.figure(figsize=(9, 2.6), dpi=90)
    for k in range(len(d["est_render"])):
        fig.clf()
        axs = fig.subplots(1, 3)
        o = d["obs"] if d["obs"].ndim == 3 else d["obs"][k]
        axs[0].imshow(np.clip(o, 0, 1)); axs[0].set_title("observed", fontsize=8)
        axs[1].imshow(np.clip(d["est_render"][k], 0, 1)); axs[1].set_title("render at estimate", fontsize=8)
        axs[2].scatter(bg2[:, 0], bg2[:, 1], s=0.5, c=np.clip(d["bg_col"], 0, 1), alpha=0.3)
        p2 = (d["particles"][k] - mu) @ A.T
        axs[2].scatter(p2[:, 0], p2[:, 1], s=3, c=d["weights"][k], cmap="viridis")
        g2 = (gt[:, :3, 3] - mu) @ A.T
        axs[2].plot(g2[:, 0], g2[:, 1], "r-", lw=0.8)
        e2 = (d["est"][:k + 1, :3, 3] - mu) @ A.T
        axs[2].plot(e2[:, 0], e2[:, 1], "b-", lw=0.8)
        axs[2].set_title(f"particles (top view), step {k}", fontsize=8); axs[2].set_aspect("equal")
        for a in axs:
            a.set_xticks([]); a.set_yticks([])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(out, frames, duration=1000 / fps, loop=0)
    print("saved", out)
    return out


def site():
    """Simple GitHub Pages demo: one card per animation + the main tables."""
    docs = os.path.join(config.ROOT, "docs")
    os.makedirs(os.path.join(docs, "gifs"), exist_ok=True)
    import shutil
    cards = []
    for g in sorted(glob.glob(os.path.join(GIF, "*.gif"))):
        shutil.copy(g, os.path.join(docs, "gifs", os.path.basename(g)))
        cards.append(f'<div class="card"><img src="gifs/{os.path.basename(g)}"><p>{os.path.basename(g)[:-4]}</p></div>')
    tables = []
    for t in sorted(glob.glob(os.path.join(TAB, "*.csv"))):
        tables.append(f"<h3>{os.path.basename(t)[:-4]}</h3>" + pd.read_csv(t).to_html(index=False, float_format="%.3f"))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>4DGS Particle Filter Localization</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;background:#fff;color:#222}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}} .card{{border:1px solid #ddd;border-radius:8px;padding:8px;max-width:100%}}
.card img{{max-width:100%}} table{{border-collapse:collapse;font-size:12px;overflow-x:auto;display:block}}
td,th{{border:1px solid #ccc;padding:3px 6px}}</style></head><body>
<h1>Particle filter 6-DoF camera localization in dynamic 4D Gaussian Splatting scenes</h1>
<p>Observed frame | render from the estimated pose | top view of the particle cloud (red: true trajectory, blue: estimate).</p>
<div class="grid">{''.join(cards)}</div><h2>Results</h2>{''.join(tables)}</body></html>"""
    open(os.path.join(docs, "index.html"), "w").write(html)
    print("saved", os.path.join(docs, "index.html"))


# --------------------------------------------------------------------------- main
def run_all():
    f1()
    if (d := load("E1")) is not None:
        f5(d)
    if (d := load("E2")) is not None:
        f6(d); f6_single(d); f10(d)
    if (d := load("E3")) is not None:
        for mode, g in d.groupby("mode"):
            f7(g, mode)
    if (d := load("E4")) is not None:
        for mode, g in d.groupby("mode"):
            f8(g, mode)
    if (d := load("E5")) is not None:
        for mode, g in d.groupby("mode"):
            f9(g, mode)
    if (d := load("E6")) is not None:
        f12(d)
    if (d := load("E7")) is not None:
        f11(d)
    for v in sorted(glob.glob(os.path.join(config.RESULTS, "vis", "*.npz"))):
        f2(v); f3(v)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("what", nargs="+", help="all | f1 | f2 | f3 | f4 | f5..f12 | gifs | site")
    a.add_argument("--vis", nargs="*", default=None)
    a.add_argument("--exp", default=None, help="CSV tag to use instead of the default E-number")
    a.add_argument("--scene", default=config.TUNE_SCENE)
    a.add_argument("--t", type=int, default=0)
    a.add_argument("--cam", type=int, default=0)
    args = a.parse_args()
    vis = args.vis or sorted(glob.glob(os.path.join(config.RESULTS, "vis", "*.npz")))
    single = {"f5": ("E1", f5), "f6": ("E2", f6), "f6b": ("E2", f6_single), "f10": ("E2", f10), "f11": ("E7", f11), "f12": ("E6", f12)}
    for w in args.what:
        if w == "all":
            run_all()
        elif w == "f1":
            f1()
        elif w in ("f2", "f3"):
            for v in vis:
                (f2 if w == "f2" else f3)(v)
        elif w == "f4":
            f4(args.scene, args.t, args.cam)
        elif w in single:
            exp, fn = single[w]
            fn(load(args.exp or exp))
        elif w in ("f7", "f8", "f9"):
            exp = args.exp or {"f7": "E3", "f8": "E4", "f9": "E5"}[w]
            d = load(exp)
            for mode, g in d.groupby("mode"):
                {"f7": f7, "f8": f8, "f9": f9}[w](g, mode)
        elif w == "gifs":
            for v in vis:
                gif_from_vis(v)
        elif w == "site":
            site()
        else:
            raise SystemExit(f"unknown target {w}")


if __name__ == "__main__":
    main()
