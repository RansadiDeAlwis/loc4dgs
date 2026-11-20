#!/usr/bin/env bash
# The project plan, stage by stage (three claims, three scenes, two sweeps on juggle).
#
#   bash run_plan.sh smoke        # Stage 0: smoke tests on juggle (half a day)
#   bash run_plan.sh tune         # Stage 1: 3 lambdas x 2 noise levels on juggle, ONE trajectory
#                                 #          -> pick the best, write it into config.py, never retune
#   bash run_plan.sh main         # Stages 2-3 + Claim 3: E1, E2, E3 on juggle tennis basketball
#   bash run_plan.sh sweeps       # discussion: N sweep on juggle (the lambda sweep = the tune runs)
#   bash run_plan.sh plots        # figures + tables
#   bash run_plan.sh all          # main + sweeps + plots (after tuning!)
#
# Override the scenes with SCENES="juggle tennis" bash run_plan.sh main
# Occlusion (E6) and kidnapping (E7) are left out on purpose; see run_all.sh if time is left over.
set -e
cd "$(dirname "$0")"
SCENES=${SCENES:-"juggle tennis basketball"}
TUNE=juggle
TIMES="0 30 60 90 120"                       # E1 / E3 relocalization timesteps
MAPS="full:0 background:0 frozen0:0 full:15 full:-15"   # Claim 3: correct time vs static vs wrongly timed

smoke() {
  python sanity_check.py --scene $TUNE --only render units
  python smoke_test.py --scene $TUNE
}

tune() {
  # tracking, one trajectory: lambda x process noise (sig_p:sig_r_deg)
  python run_track.py --scene $TUNE --exp E5 --methods pf --n_traj 1 --seeds 5 --odo medium \
         --lam 1 5 20 --noise 0.01:0.5 0.02:1.0
  # relocalization on a few images: lambda_0 x random-walk noise fraction
  python run_reloc.py --scene $TUNE --exp E5 --methods pf --levels 0.3:20 --times 0 60 120 --seeds 3 \
         --lam 1 5 20 --noise_frac 0.3 0.5
  echo "Now compare results/E5_*_$TUNE.csv (python plots.py f9), write the chosen lam / sig_p / sig_r /"
  echo "local_noise_frac into config.py (RELOC and TRACK), and do not change them again."
}

main_runs() {
  for S in $SCENES; do
    # Claim 1 (E1): PF vs EKF vs IEKF, 4 test cams x 5 timesteps, 3 start levels, 3 seeds
    python run_reloc.py --scene $S --exp E1 --methods pf ekf iekf --levels 0.1:5 0.3:20 1.0:45 \
           --times $TIMES --seeds 3 --save_vis 2
    # Claim 2 (E2): tracking, 2 trajectories x 5 seeds, PF and EKF, same start and odometry
    python run_track.py --scene $S --exp E2 --methods pf ekf --n_traj 2 --seeds 5 --odo medium --save_vis 1
    # Claim 3 (E3): the map at the correct time vs static / wrongly timed maps
    python run_reloc.py --scene $S --exp E3 --methods pf --levels 0.3:20 --times $TIMES --seeds 3 --maps $MAPS
    python run_track.py --scene $S --exp E3 --methods pf --n_traj 2 --seeds 5 --odo medium --maps $MAPS
  done
}

sweeps() {
  python run_reloc.py --scene $TUNE --exp E4 --methods pf --levels 0.3:20 --times 0 60 120 --seeds 3 \
         --N 100 300 600 1000
  python run_track.py --scene $TUNE --exp E4 --methods pf --n_traj 1 --seeds 3 --odo medium --N 50 100 300 600
}

plots() {
  python plots.py all gifs site
  python plots.py f4 --scene $TUNE
}

case "${1:-}" in
  smoke)  smoke ;;
  tune)   tune ;;
  main)   main_runs ;;
  sweeps) sweeps ;;
  plots)  plots ;;
  all)    main_runs; sweeps; plots ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac
