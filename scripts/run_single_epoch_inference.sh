#!/usr/bin/env bash
# Run experiment.py for one fine-tuned checkpoint on member + non-member perturbations.
#
# Usage (from src_v3/):
#   bash scripts/run_single_epoch_inference.sh
#   EPOCH=4 bash scripts/run_single_epoch_inference.sh
#
# Outputs:
#   results/pythia_70m/run_<epoch>/non_members/records.json, summary.csv, ...
#   results/pythia_70m/run_<epoch>/members/records.json, summary.csv, ...
#   results/logs/inference_run_<epoch>_non_member.log
#   results/logs/inference_run_<epoch>_member.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

EPOCH="${EPOCH:-4}"
CKPT_BASE="${CKPT_BASE:-checkpoints/EleutherAI_pythia-70m_fine_tuned}"
CKPT_DIR="${CKPT_BASE}/checkpoint-epoch-${EPOCH}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"

PERT_NON_MEMBER="${PERT_NON_MEMBER:-data/perturbations_non_members.json}"
PERT_MEMBER="${PERT_MEMBER:-data/perturbations_members.json}"
OUTPUT_BASE="${OUTPUT_BASE:-results/pythia_70m}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_DIR}" >&2
  exit 1
fi

mkdir -p results/logs "${OUTPUT_BASE}/run_${EPOCH}"

echo "=== Epoch ${EPOCH} | checkpoint ${CKPT_DIR} | GPU ${CUDA_DEVICE} ==="

# Non-member run
echo "  → non-member split..."
env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" experiment.py \
  --config config.yml \
  --model "${CKPT_DIR}" \
  --perturbations-file "${PERT_NON_MEMBER}" \
  --output-dir "${OUTPUT_BASE}/run_${EPOCH}" \
  --dataset-split unleak \
  > "results/logs/inference_run_${EPOCH}_non_member.log" 2>&1

# Member run
echo "  → member split..."
env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" experiment.py \
  --config config.yml \
  --model "${CKPT_DIR}" \
  --perturbations-file "${PERT_MEMBER}" \
  --output-dir "${OUTPUT_BASE}/run_${EPOCH}" \
  --dataset-split leak \
  > "results/logs/inference_run_${EPOCH}_member.log" 2>&1

echo "Done epoch ${EPOCH} → ${OUTPUT_BASE}/run_${EPOCH}/{non_members,members}/"
