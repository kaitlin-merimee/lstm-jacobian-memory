#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J readfollow
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=1-3
#SBATCH -o logs/nb28_followup_%A_%a.out
# One task per model version (the array index is the seed); each handles the five snow basins together. CPU only.
# Submit from the repo root:   mkdir -p logs && sbatch slurm/where_snow_followup.slurm
# Other date (1 April):   sbatch --export=ALL,DATE=04-01 slurm/where_snow_followup.slurm
# When done:   grep -h -E "^ANSWER|^\[" logs/nb28_followup_*.out
set -eo pipefail
# module load miniconda3   # load your cluster's conda/python module here
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuralhydrology
set -u
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
python src/where_snow_followup.py --seed "$SLURM_ARRAY_TASK_ID" --date "${DATE:-03-01}"
