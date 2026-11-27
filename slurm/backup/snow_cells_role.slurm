#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J snowcells
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --array=0-14
#SBATCH -o logs/nb30_snowcells_%A_%a.out
# Five snow basins x three model versions. Task 0 is Encampment, seed 1, with unit 77 named. CPU only.
# Section J needs the where_snow_followup files (analysis/read_directions/nb28_directions_seed*.pkl), which the earlier where_snow_followup runs wrote.
# Submit from the repo root:   mkdir -p logs && sbatch slurm/backup/snow_cells_role.slurm
# When done:   grep -h "^ANSWER" logs/nb30_snowcells_*.out ;  python src/backup/snow_cells_role_collect.py
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
UNIT=auto; if [ "$B" = "06623800" ] && [ "$S" = "1" ]; then UNIT=77; fi
echo "basin $B seed $S unit $UNIT"
python src/backup/snow_cells_role.py --basin "$B" --seed "$S" --unit "$UNIT" --date "${DATE:-03-01}"
