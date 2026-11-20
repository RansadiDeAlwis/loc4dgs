"""Small helpers shared by the experiment scripts: CSV logging, seeding, timing, argument parsing."""
import os
import time
import random
import numpy as np
import pandas as pd
import torch

import config


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        sync(); self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        sync(); self.ms = 1000 * (time.perf_counter() - self.t0)


class CSVLog:
    """Buffered row logger.  Every run is saved as rows (scene, mode, map_mode, N, lambda, seed,
    step, errors, ESS, NEES, time, ...); all figures are made from these CSVs by plots.py."""

    def __init__(self, path, flush_every=2000):
        self.path, self.rows, self.flush_every = path, [], flush_every
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add(self, **row):
        self.rows.append(row)
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        header = not os.path.exists(self.path)
        if not header:                                            # keep column order of the existing file
            cols = pd.read_csv(self.path, nrows=0).columns.tolist()
            for c in df.columns:
                if c not in cols:
                    raise ValueError(f"new column {c!r} not in {self.path}; use a new --exp tag")
            df = df.reindex(columns=cols)
        df.to_csv(self.path, mode="a", header=header, index=False)
        self.rows = []


def parse_maps(specs):
    """'full:0 background:0 full:5 frozen0:0' -> [('full',0), ...]"""
    out = []
    for s in specs:
        m, _, o = s.partition(":")
        out.append((m, int(o or 0)))
    return out


def parse_levels(specs):
    """'0.1:5 0.3:20 global' -> [(0.1,5.0), (0.3,20.0), 'global']"""
    out = []
    for s in specs:
        if s == "global":
            out.append("global")
        else:
            a, b = s.split(":")
            out.append((float(a), float(b)))
    return out


def level_name(level):
    return "global" if level == "global" else f"{level[0]:g}u_{level[1]:g}deg"


def result_path(exp, scene, mode):
    """results/<exp>_<mode>_<scene>.csv  (mode = reloc | track, so both can share an E-number)."""
    return os.path.join(config.RESULTS, f"{exp}_{mode}_{scene}.csv")


def vis_path(name):
    d = os.path.join(config.RESULTS, "vis")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ".npz")
