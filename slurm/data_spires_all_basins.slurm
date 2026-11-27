#!/bin/bash
#SBATCH --account <your-account>
#SBATCH --partition <your-partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --job-name=data_spires_all_basins
#SBATCH --array=1-14
#SBATCH --output=./logs/spires_all_%A_%a.out
# one task per water year: task 1 = WY2001 ... task 14 = WY2014 (CAMELS forcing ends in WY2014)
set -euo pipefail
YEAR=$((2000 + SLURM_ARRAY_TASK_ID))
PROJECT_DIR=.
module purge
# module load miniconda3/25.9.1   # load your cluster's conda/python module here
source activate neuralhydrology
cd "${PROJECT_DIR}"
echo "WY${YEAR} on ${SLURM_NODELIST} | $(date)"
python src/data_spires_all_basins.py \
  --h5dir <path-to>/spires_swe \
  --basins data/basins --basin-file basin_lists/camels_531_basins.txt \
  --year "${YEAR}" --out analysis/spires/all
echo "done $(date)"
