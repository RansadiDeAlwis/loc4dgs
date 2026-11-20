"""Central configuration: paths, image sizes and filter settings (guide Section 4.6).

Every experiment script reads its defaults from here. Tune the filter parameters on ONE
scene / ONE trajectory (TUNE_SCENE), then freeze them before running the full experiments
(guide Section 6.2: "otherwise you are tuning on your test set").
"""
import os
import numpy as np

# --------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
# Unzipped data.zip from Dynamic3DGaussians:   <DATA_ROOT>/<scene>/{ims,seg,train_meta.json,test_meta.json}
DATA_ROOT = os.environ.get("LOC4DGS_DATA", os.path.join(ROOT, "..", "data"))
# Unzipped output.zip:                         <OUTPUT_ROOT>/<EXPERIMENT>/<scene>/params.npz
OUTPUT_ROOT = os.environ.get("LOC4DGS_OUTPUT", os.path.join(ROOT, "..", "output"))
EXPERIMENT = os.environ.get("LOC4DGS_EXP", "pretrained")
RESULTS = os.path.join(ROOT, "results")

SCENES_CORE = ["juggle", "tennis", "basketball"]          # start with these (Section 6)
SCENES_EXTRA = ["boxes", "softball", "football"]          # add if time allows
SYNTHETIC = "synthetic"                                    # CPU / no-gsplat fallback (Section 10.4)
TUNE_SCENE = "juggle"                                      # tune here only, then freeze

TEST_CAMS = [0, 10, 15, 30]                                # held-out Panoptic cameras
N_FRAMES = 150                                             # T in params.npz


def params_path(scene):
    return os.path.join(OUTPUT_ROOT, EXPERIMENT, scene, "params.npz")


def meta_path(scene, split="test"):
    return os.path.join(DATA_ROOT, scene, f"{split}_meta.json")


def image_path(scene, fn):
    return os.path.join(DATA_ROOT, scene, "ims", fn)


def seg_path(scene, fn):
    return os.path.join(DATA_ROOT, scene, "seg", fn.replace(".jpg", ".png"))


# --------------------------------------------------------------------------- image sizes
FULL_W, FULL_H = 640, 360        # Panoptic undistorted images
PF_W, PF_H = 160, 90             # particle-filter likelihood resolution (1/4)
EKF_W, EKF_H = 80, 45            # EKF Jacobian resolution (grayscale, blurred)
OBS_W, OBS_H = 320, 180          # synthetic observations are rendered here, then area-downsampled
SYNTH_W, SYNTH_H = 64, 36        # resolution used with the pure-PyTorch fallback splatter
RENDER_CHUNK = 512               # particles per gsplat call (sanity_check: 512 fits); lower it if you run out of memory

# --------------------------------------------------------------------------- relocalization (E1)
RELOC = dict(
    N=600, N_small=200,          # start with many particles, drop to N_small after first anneal
    iters=80,                    # filter iterations on one image
    sig_p=0.10,                  # random-walk noise (scene units) for the GLOBAL prior
    sig_r=np.deg2rad(5.0),       # random-walk noise (rad) for the GLOBAL prior
    local_noise_frac=0.3,        # LOCAL prior: random-walk sigma = frac * initial error level
    lam=20.0, lam_gamma=1.1, lam_max=200.0,  # tempering lambda_k = lam0 * gamma^k (capped); FROZEN after Stage 1 tuning (E5, E5b on juggle)
    trim=0.1,                    # robust trimmed error (drop worst 10% pixels)
    anneal_factor=0.5,           # shrink sigma_p, sigma_r by this when spread < threshold
    spread_frac0=0.5,            # first threshold = frac * initial particle spread
    anneal_patience=10,          # also anneal if the spread has not dropped for this many iterations
    inject_frac=0.1, inject_err=1.2, inject_patience=5,
)
RELOC_LEVELS = [(0.1, 5.0), (0.3, 20.0), (1.0, 45.0), "global"]   # (units, degrees)
GLOBAL_PRIOR = dict(rot_noise_deg=20.0, radius_margin=0.0, height_margin=0.0)

EKF = dict(
    sigma_z=30.0,                # inflated pixel noise (correlated pixels!); FROZEN after EKF tuning (E5ekf on juggle)
    eps=1e-2,                    # finite-difference step; FROZEN after EKF tuning
    blur_sigma=1.0,              # Gaussian blur (px at EKF_W x EKF_H); tried 1/3/6, 1 is best; FROZEN
    iters_iekf=5,                # iterated EKF = a few Gauss-Newton steps
    reloc_steps=30,              # repeated predict(I)/update cycles in relocalization mode
    q_p=1e-4, q_r=np.deg2rad(0.05),   # process noise std in relocalization mode
)

# --------------------------------------------------------------------------- tracking (E2)
TRACK = dict(
    N=300,
    lam=50.0, trim=0.1,                  # FROZEN after Stage 1 tuning (E5 on juggle)
    sig_p=0.01, sig_r=np.deg2rad(0.5),   # process noise on top of odometry
    cv_scale=3.0,                        # constant-velocity model uses cv_scale x the noise
    init_sig_p=0.02, init_sig_r=np.deg2rad(1.0),
    inject_frac=0.1, inject_err=1.0, inject_patience=3,
    n_cams=5, wobble=0.0,
)
# odometry noise per frame: (sigma_p [units], sigma_r [deg])
ODO_LEVELS = {"low": (0.002, 0.1), "medium": (0.005, 0.3), "high": (0.01, 0.8)}
OBS_CORRUPT = dict(noise=0.02, blur=1, occluders=0)
OCCLUSION_LEVELS = [0.0, 0.1, 0.3, 0.5]
KIDNAP_FRAME = 75

# --------------------------------------------------------------------------- ablations
MAP_MODES = [("full", 0), ("background", 0), ("full", 5), ("full", -5),
             ("full", 15), ("full", -15), ("frozen0", 0)]
N_SWEEP = [50, 100, 300, 600, 1000]
LAM_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# --------------------------------------------------------------------------- evaluation
SUCCESS_T = 0.05                 # 5 cm (adjust if scene units are not metres)
SUCCESS_R = 5.0                  # degrees
SEEDS = list(range(10))          # >= 10 seeds per setting
METHOD_COLORS = {"pf": "tab:blue", "ekf": "tab:orange", "iekf": "tab:green"}
MAP_COLORS = {"full": "tab:blue", "background": "tab:red", "offset": "tab:purple",
              "frozen0": "tab:gray"}
