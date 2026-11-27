#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J allmem
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --array=0-53
#SBATCH -o logs/nb23_all_%A_%a.out
# All 531 basins, about 10 per task. CPU only. Resubmitting is safe: cached basins are skipped.
# Submit from the repo root:   mkdir -p logs && sbatch slurm/nb23_all_basins.slurm
# When done:   grep -l FAILED logs/nb23_all_*.out ;  PYTHONPATH=src python src/nb23_all_basins_memory.py --summarize
set -eo pipefail
# module load miniconda3   # load your cluster's conda/python module here
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuralhydrology
set -u
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
python src/nb23_all_basins_memory.py --task "$SLURM_ARRAY_TASK_ID" --n-tasks 54
