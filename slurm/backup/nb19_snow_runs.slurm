#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J snowruns
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --array=0-14
#SBATCH -o logs/nb19_snow_%A_%a.out
# Five snow basins x three seeds. The snow unit is re-selected in every run (--unit auto).
# Submit from the repo root with the neuralhydrology env active:   sbatch slurm/backup/nb19_snow_runs.slurm
# When all tasks are done:   PYTHONPATH=src python src/backup/nb22_checks.py --collect
set -euo pipefail
export OMP_NUM_THREADS=8 PYTHONPATH=src:${PYTHONPATH:-}
BASINS=(06623800 12178100 13235000 09066300 09312600)
SEEDS=(1 2 3)
B=${BASINS[$((SLURM_ARRAY_TASK_ID / 3))]}
S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "basin $B seed $S"
python src/backup/nb19_open_questions.py --basin "$B" --seed "$S" --unit auto --only ABCDJ
