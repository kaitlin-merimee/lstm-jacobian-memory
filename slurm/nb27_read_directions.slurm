#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J readdir
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --array=0-14
#SBATCH -o logs/nb27_readdir_%A_%a.out
# Five snow basins x three model versions. Task 0 is Encampment, seed 1. CPU only.
# Submit from the repo root:   mkdir -p logs && sbatch slurm/nb27_read_directions.slurm      (one run only: sbatch --array=0 ...)
# Other date (1 April):   sbatch --export=ALL,DATE=04-01 slurm/nb27_read_directions.slurm   then   python src/nb27_collect.py --date 04-01
# When done:   grep -h "^ANSWER" logs/nb27_readdir_*.out ;  python src/nb27_collect.py
set -eo pipefail
# module load miniconda3   # load your cluster's conda/python module here
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuralhydrology
set -u
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
BASINS=(06623800 12178100 13235000 09066300 09312600)
SEEDS=(1 2 3)
B=${BASINS[$((SLURM_ARRAY_TASK_ID / 3))]}
S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "basin $B seed $S"
python src/nb27_read_directions.py --basin "$B" --seed "$S" --date "${DATE:-03-01}"
