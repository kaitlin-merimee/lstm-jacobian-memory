#!/bin/bash
# nb15 all-basin reach-back / state memory. 54 tasks x ~10 basins, ~8 min per basin -> ~80 min per task.
# Submit from the project root with the neuralhydrology env active (the job inherits it):
#   mkdir -p logs && sbatch slurm/backup/nb15_all.slurm
# Re-submitting is safe: basins already in analysis/lyapunov/nb15_cache/ are skipped.
# When every task is done:  python src/backup/nb15_all_basins.py --summarize
#SBATCH -J nb15_all
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH --array=0-53%20
#SBATCH -o logs/nb15_all_%A_%a.out

# If the env is not inherited, uncomment and adjust:
# source ~/.bashrc; conda activate neuralhydrology

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd .
python src/backup/nb15_all_basins.py --task "$SLURM_ARRAY_TASK_ID" --ntasks 54
