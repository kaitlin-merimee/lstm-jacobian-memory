#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J checks
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o logs/nb22_checks_%j.out
# Checks 1, 3 and 4. Submit from the repo root with the neuralhydrology env active:   sbatch slurm/backup/whole_model_checks.slurm
set -euo pipefail
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
python src/backup/whole_model_checks.py --check3 --check4 --check1
