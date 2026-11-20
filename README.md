# loc4dgs: particle-filter 6-DoF camera localization in dynamic 4D Gaussian Splatting scenes

A particle filter on SE(3) estimates where a camera is and which way it points. Its only input is
the image the camera sees. It compares that image with renders of a pretrained **Dynamic 3D Gaussians**
(4D) map at the correct time `t`. The repository also contains an EKF / iterated-EKF baseline,
seven experiments (E1–E7), metrics with NEES consistency checks, figures, animations and a demo page.

## Layout

| file | content |
|---|---|
| `config.py` | paths, image sizes, filter settings (tune on `TUNE_SCENE`, then freeze) |
| `se3.py` | Exp / Log / inverse / adjoint on SE(3), quaternion helpers, Markley average, look-at |
| `scene.py` | loads `params.npz` and meta files, builds `G(t)` (`full`, `background`, `frozen0`, time offset) |
| `render.py` | batched renderer `h(x)` with gsplat, plus a pure-PyTorch EWA splatter as a fallback |
| `synthetic.py` | small synthetic 4D world (fallback from Section 10.4), with the same interface as a real scene |
| `observations.py` | normalization, photometric error (MSE / trimmed / Huber), blur, corruption, occluders |
| `trajectories.py` | camera paths interpolated between Panoptic cameras, odometry noise, kidnapping, local and global priors |
| `pf.py` | SIR particle filter: predict, tempered likelihood, systematic resampling, injection, estimate and covariance |
| `ekf.py` | information-form EKF and iterated EKF with finite-difference Jacobians |
| `metrics.py` | translation/rotation error, ATE, NEES and χ² band, success rate, coverage, convergence time |
| `run_reloc.py` | relocalization from one real test image (E1, E3, E4, E5) |
| `run_track.py` | tracking along a moving camera (E2, E3, E4, E5, E6, E7, constant-velocity model) |
| `plots.py` | figures F1–F12, tables (median and IQR), GIFs, GitHub Pages demo (`docs/`) |
| `sanity_check.py` | week-1 go/no-go checks: render vs photo, PSNR, units, chunk size, trajectory videos, prior |
| `smoke_test.py` | Stage 0 on a real scene: stays at truth, true particle wins, converges from 0.2 / 10°, ms per update + run budget |
| `run_plan.sh` | the trimmed project plan: `smoke`, `tune`, `main` (E1–E3 × 3 scenes), `sweeps`, `plots` |
| `tests/` | SE(3) tests and the debugging checklist from Section 10.2 (`python -m pytest tests -q`) |
| `results/` | CSVs (`<exp>_<reloc|track>_<scene>.csv`), `figures/`, `tables/`, `gifs/`, `vis/`, `sanity/` |

## 1. Install (guide Section 4.2)

```bash
bash setup_env.sh            # conda env "loc4dgs": torch 2.4.1+cu121, gsplat prebuilt wheel, utilities
conda activate loc4dgs
python -m pytest tests -q    # 16 tests; they run on the synthetic world, so no dataset is needed
```

## 2. Data (guide Section 4.3)

```bash
bash download_data.sh        # clones Dynamic3DGaussians, downloads and unzips data.zip and output.zip
```
By default the code expects `../data/<scene>/...` and `../output/pretrained/<scene>/params.npz`.
Override the locations with `LOC4DGS_DATA`, `LOC4DGS_OUTPUT` and `LOC4DGS_EXP`.

## 3. Week-1 checks (go/no-go)

```bash
python sanity_check.py --scene juggle
```
Look at `results/sanity/render_juggle.png`: the render must match the photo. Training-camera PSNR
should be in the high twenties. The units report tells you whether errors are in metres; if they
are not, set `SUCCESS_T` in `config.py` and `UNIT` in `plots.py`. Also watch the trajectory videos.

## 4. Smoke tests, tune, then freeze (project plan)

```bash
bash run_plan.sh smoke     # Stage 0
bash run_plan.sh tune      # Stage 1: --lam 1 5 20 x --noise / --noise_frac, juggle only
bash run_plan.sh main      # E1, E2, E3 on juggle tennis basketball
bash run_plan.sh sweeps    # N sweep on juggle
bash run_plan.sh plots
```


Tune `RELOC`, `TRACK` and `EKF` in `config.py` on **one** scene and **one** trajectory only. Then
freeze them and say so in the report.

## 5. Experiments (one command each)

```bash
python run_reloc.py --scene juggle --exp E1 --methods pf ekf iekf --levels 0.1:5 0.3:20 1.0:45 global --save_vis 2
python run_track.py --scene juggle --exp E2 --methods pf ekf iekf --odo low medium high --save_vis 1
python run_reloc.py --scene juggle --exp E3 --methods pf --levels 0.3:20 --maps full:0 background:0 full:5 full:-5 full:15 full:-15 frozen0:0
python run_track.py --scene juggle --exp E3 --maps full:0 background:0 full:5 full:-5 full:15 full:-15 frozen0:0
python run_reloc.py --scene juggle --exp E4 --methods pf --levels 0.3:20 --N 50 100 300 600 1000 --anneal 1 0
python run_track.py --scene juggle --exp E5 --lam 0.5 1 2 5 10 20 50 --loss_types mse trimmed
python run_track.py --scene juggle --exp E6 --occ 0 0.1 0.3 0.5 --loss_types mse trimmed
python run_track.py --scene juggle --exp E7 --kidnap --inject 0 1 --save_vis 2
python plots.py all gifs site && python plots.py f4 --scene juggle
```
`bash run_all.sh juggle tennis basketball` runs all of them. Add `--scene synthetic` to any command
to run it without the dataset. `--backend torch` forces the CPU-capable fallback renderer.

## Design notes worth stating in the report

* **Conventions.** Particles store camera-to-world poses. The renderer takes world-to-camera poses.
  Noise is right-multiplied, `T·Exp(ξ)`, with ξ = (ρ, φ).
* **Likelihood.** `log p = −λ·e`, where `e` is the trimmed MSE between per-channel-normalized images
  (2(1−ρ) per channel). Tempering is needed because pixel errors are correlated. `--likelihood locnerf`
  selects Loc-NeRF's `(M/Σr²)⁴` instead.
* **Relocalization.** The filter uses a random-walk search on one image. Annealing halves σ when the
  particle spread drops below a threshold, or after `anneal_patience` iterations without such a drop.
  N goes from 600 to 200 after the first anneal, and λ_k = λ₀·γᵏ (capped). For the local prior,
  σ = `local_noise_frac` × the initial error level.
* **Global prior.** Positions are sampled in the azimuth, radius and height ranges of the training
  cameras, around the dome axis. Orientations look at the least-squares convergence point of the
  camera axes, with uniform ±20° yaw/pitch/roll noise.
* **EKF.** The EKF uses the information-form update on 80×45 blurred, normalized grayscale images
  with central-difference Jacobians (12 renders). In relocalization mode it repeats `predict(I)` and
  `update`. The EKF and the PF start from the same `T_init` for every seed.
* **Tracking.** Observations are rendered from the full 4D map at the correct time, at twice the
  filter resolution, then area-downsampled and corrupted. This reduces the inverse crime. Process
  noise is √(jitter² + odometry noise²).
* **No ATE alignment.** The map fixes the world frame, so estimates and ground truth are compared
  directly.
* **PF NEES is approximate.** The particle belief can be non-Gaussian, so the CSVs also log
  `covered` (truth within 2σ on every axis).
