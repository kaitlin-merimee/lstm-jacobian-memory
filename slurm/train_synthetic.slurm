#!/bin/bash
#SBATCH --account <your-account>
#SBATCH --partition <your-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=syn_train
#SBATCH --output=./logs/syn_%x_%j.out
#SBATCH --error=./logs/syn_%x_%j.err
# usage:  sbatch --export=ALL,SCEN=S2 -J syn_S2 slurm/train_synthetic.slurm
set -euo pipefail
: "${SCEN:?set SCEN=S1|S2|S3|G with --export}"

PROJECT_DIR=.
CFG=${PROJECT_DIR}/configs/camels531_syn_${SCEN}_n00.yml
TARGETS=${PROJECT_DIR}/data/synthetic/syn_${SCEN}_n00.p

module purge
# module load miniconda3/25.9.1   # load your cluster's conda/python module here
# module load cuda/12.1   # load your cluster's conda/python module here
source activate neuralhydrology
cd "${PROJECT_DIR}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST} | SCEN=${SCEN} | $(date)"
python -c 'import torch,sys; ok=torch.cuda.is_available(); print("CUDA available:", ok); sys.exit(0 if ok else 1)'
[ -s "${TARGETS}" ] || { echo "missing synthetic targets ${TARGETS} (run synthetic_1_generate)"; exit 1; }
grep -q "^dataset:" "${CFG}" || { echo "${CFG} is not a full config (run synthetic_2_configs.py --write)"; exit 1; }

nh-run train --config-file "${CFG}"
LATEST_RUN=$(ls -td ${PROJECT_DIR}/runs/camels531_syn_${SCEN}_n00_h256_seed1_* | head -1)
echo "Evaluating ${LATEST_RUN}"
nh-run evaluate --run-dir "${LATEST_RUN}"
echo "Done $(date)"
