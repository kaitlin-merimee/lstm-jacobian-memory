#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J jtalk
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o logs/jtalk_%j.out
# Submit from the repo root with the neuralhydrology env active:  sbatch slurm/backup/nudge_lags_lyapunov.slurm
set -euo pipefail
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
python src/backup/nudge_through_melt.py
python src/backup/lag_profiles.py --seeds 1 2 3
python src/backup/lyapunov_seeds.py --seeds 1 2 3
python src/synthetic_4_figure.py --scen S2
grep -h "SLIDE NUMBERS" -A 4 logs/jtalk_${SLURM_JOB_ID}.out || true
