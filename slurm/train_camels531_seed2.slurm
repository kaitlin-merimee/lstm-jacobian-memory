#!/bin/bash
#SBATCH --account <your-account>
#SBATCH --partition <your-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=camels531_daymet
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err


echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start:     $(date)"
echo "=========================================="

module purge
# module load miniconda3/25.9.1   # load your cluster's conda/python module here
# module load cuda/12.1   # load your cluster's conda/python module here

source activate neuralhydrology

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

PROJECT_DIR=.
mkdir -p ${PROJECT_DIR}/logs ${PROJECT_DIR}/runs
cd ${PROJECT_DIR}

echo "Starting training at $(date)"
nh-run train --config-file configs/camels531_daymet_benchmark_seed2.yml

# Evaluate on the test period immediately after training finishes.
# nh-run evaluate picks up the most recent run matching the experiment name.
LATEST_RUN=$(ls -td ${PROJECT_DIR}/runs/camels531_daymet_h256_seed2* | head -1)
echo "Evaluating run: ${LATEST_RUN}"
nh-run evaluate --run-dir "${LATEST_RUN}"

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
