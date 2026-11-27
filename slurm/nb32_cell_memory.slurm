#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -J cellmem
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH --array=0-14
#SBATCH -o logs/nb32_cellmem_%A_%a.out
# Five snow basins x three model versions. Task 0 is Encampment, seed 1, with unit 77 named. CPU only.
# Per run: 12 start days x 6 water years, each with ~240 flow gradients and one forward Jacobian pass.
# Submit from the repo root:   mkdir -p logs && sbatch slurm/nb32_cell_memory.slurm
# When done:   grep -h -E "^ANSWER|check" logs/nb32_cellmem_*.out ;  python src/nb32_collect.py ;  python src/nb32_fig_memory.py
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
python src/nb32_cell_memory.py --basin "$B" --seed "$S" --unit "$UNIT" --horizon "${HORIZON:-240}"
