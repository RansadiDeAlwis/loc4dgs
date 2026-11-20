#!/usr/bin/env bash
# One command per experiment (guide Sections 6 and 10.5). Usage: bash run_all.sh [scenes...]
# Tune sigma_p, sigma_r and lambda in config.py on TUNE_SCENE first, then FREEZE them.
set -e
cd "$(dirname "$0")"
SCENES=${@:-"juggle tennis basketball"}
for S in $SCENES; do
  python run_reloc.py --scene $S --exp E1 --methods pf ekf iekf --levels 0.1:5 0.3:20 1.0:45 global --save_vis 2   # E1
  python run_track.py --scene $S --exp E2 --methods pf ekf iekf --odo low medium high --save_vis 1                  # E2
  python run_reloc.py --scene $S --exp E3 --methods pf --levels 0.3:20 \
         --maps full:0 background:0 full:5 full:-5 full:15 full:-15 frozen0:0                                       # E3 (reloc)
  python run_track.py --scene $S --exp E3 --maps full:0 background:0 full:5 full:-5 full:15 full:-15 frozen0:0     # E3 (track)
  python run_reloc.py --scene $S --exp E4 --methods pf --levels 0.3:20 --N 50 100 300 600 1000 --anneal 1 0        # E4 (reloc)
  python run_track.py --scene $S --exp E4 --N 50 100 300 600 1000                                                  # E4 (track)
  python run_reloc.py --scene $S --exp E5 --methods pf --levels 0.3:20 --lam 0.5 1 2 5 10 20 50 --no_temper \
         --loss_types mse trimmed                                                                                   # E5 (reloc)
  python run_track.py --scene $S --exp E5 --lam 0.5 1 2 5 10 20 50 --loss_types mse trimmed                        # E5 (track)
  python run_track.py --scene $S --exp E6 --occ 0 0.1 0.3 0.5 --loss_types mse trimmed                             # E6
  python run_track.py --scene $S --exp E7 --kidnap --inject 0 1 --save_vis 2                                       # E7
done
python run_track.py --scene ${SCENES%% *} --exp E2cv --motion cv --odo medium        # extra: tracking without odometry
python plots.py all gifs site
python plots.py f4 --scene ${SCENES%% *}
