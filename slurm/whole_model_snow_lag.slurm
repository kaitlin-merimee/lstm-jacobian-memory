#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J obslag
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --array=0-11
#SBATCH -o logs/nb26_obslag_%A_%a.out
# Snow basins only (freezing-day precipitation share above 0.2 in the whole_model_memory table). CPU only. Cached basins are skipped on resubmit.
# Submit from the repo root:   mkdir -p logs && sbatch slurm/whole_model_snow_lag.slurm
# When done:   grep -l FAILED logs/nb26_obslag_*.out ;  PYTHONPATH=src python src/whole_model_snow_lag.py --summarize
set -eo pipefail
# module load miniconda3   # load your cluster's conda/python module here
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuralhydrology
set -u
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
python src/whole_model_snow_lag.py --task "$SLURM_ARRAY_TASK_ID" --n-tasks 12
