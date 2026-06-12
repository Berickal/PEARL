#!/usr/bin/env bash
# Sweep fine-tuned checkpoints (epoch 0–10) and run experiment.py on member / non-member
# perturbation files — all epochs launched simultaneously.
#
# Usage (from src_v3/):
#   bash scripts/run_epoch_inference_sweep.sh
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES=0      GPU id (default: 0)
#   PYTHON=python               Python interpreter
#   EPOCHS="0 1 2"              Subset of epochs (default: 0–10)
#   SKIP_MISSING=1              Skip missing checkpoint dirs (default: 1)
#   CKPT_BASE=checkpoints/...   Base checkpoint directory
#   OUTPUT_BASE=results/...     Base output directory
#   SEQUENTIAL=1                Fall back to sequential mode (default: 0)
#
# Outputs per epoch:
#   results/pythia_70m/run_<epoch>/members/{records.json,summary.csv,...}
#   results/pythia_70m/run_<epoch>/non_members/{records.json,summary.csv,...}
#   results/logs/inference_run_<epoch>_member.log
#   results/logs/inference_run_<epoch>_non_member.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

CKPT_BASE="${CKPT_BASE:-checkpoints/EleutherAI_pythia-410m_fine_tuned}"
OUTPUT_BASE="${OUTPUT_BASE:-results/pythia_410m}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SEQUENTIAL="${SEQUENTIAL:-0}"

# Epoch list (override with EPOCHS="3 4 5")
if [[ -n "${EPOCHS:-}" ]]; then
  read -ra EPOCH_LIST <<< "${EPOCHS}"
else
  EPOCH_LIST=(0 1 2 3 4 5 6 7 8 9 10)
fi

PERT_NON_MEMBER="${PERT_NON_MEMBER:-data/perturbations_non_members.json}"
PERT_MEMBER="${PERT_MEMBER:-data/perturbations_members.json}"

mkdir -p results/logs

# ── validate inputs before launching anything ─────────────────────────────────

if [[ ! -f "${PERT_NON_MEMBER}" ]]; then
  echo "ERROR: ${PERT_NON_MEMBER} not found" >&2; exit 1
fi
if [[ ! -f "${PERT_MEMBER}" ]]; then
  echo "ERROR: ${PERT_MEMBER} not found" >&2; exit 1
fi

valid_epochs=()
for epoch in "${EPOCH_LIST[@]}"; do
  ckpt_dir="${CKPT_BASE}/checkpoint-epoch-${epoch}"
  if [[ ! -d "${ckpt_dir}" ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "SKIP epoch ${epoch}: missing ${ckpt_dir}"
    else
      echo "ERROR: missing checkpoint ${ckpt_dir}" >&2; exit 1
    fi
  else
    valid_epochs+=("${epoch}")
  fi
done

echo ""
echo "Epochs to run : ${valid_epochs[*]}"
echo "Mode          : $([ "${SEQUENTIAL}" == "1" ] && echo "sequential" || echo "parallel (all at once)")"
echo "GPU           : ${CUDA_DEVICE}"
echo "Checkpoint    : ${CKPT_BASE}"
echo "Output        : ${OUTPUT_BASE}"
echo ""

# ── worker function ───────────────────────────────────────────────────────────

run_one() {
  local epoch="$1"
  local split_label="$2"   # non_member | member
  local dataset_split="$3" # unleak | leak
  local pert_file="$4"
  local ckpt_dir="${CKPT_BASE}/checkpoint-epoch-${epoch}"
  local out_dir="${OUTPUT_BASE}/run_${epoch}"
  local log_file="results/logs/inference_run_${epoch}_${split_label}.log"

  mkdir -p "${out_dir}"

  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" experiment.py \
    --config             config.yml \
    --model              "${ckpt_dir}" \
    --perturbations-file "${pert_file}" \
    --output-dir         "${out_dir}" \
    --dataset-split      "${dataset_split}" \
    > "${log_file}" 2>&1

  echo "[done] epoch=${epoch} split=${split_label}"
}

# ── launch ────────────────────────────────────────────────────────────────────

pids=()

for epoch in "${valid_epochs[@]}"; do
  echo "Launching epoch ${epoch} (non_member + member)..."

  if [[ "${SEQUENTIAL}" == "1" ]]; then
    run_one "${epoch}" "non_member" "unleak" "${PERT_NON_MEMBER}"
    run_one "${epoch}" "member"     "leak"   "${PERT_MEMBER}"
  else
    run_one "${epoch}" "non_member" "unleak" "${PERT_NON_MEMBER}" &
    pids+=($!)
    run_one "${epoch}" "member"     "leak"   "${PERT_MEMBER}" &
    pids+=($!)
  fi
done

# ── wait for all background jobs ──────────────────────────────────────────────

if [[ "${SEQUENTIAL}" != "1" ]]; then
  echo ""
  echo "All ${#pids[@]} jobs launched. Waiting for completion..."
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "WARNING: job PID ${pid} exited with an error — check results/logs/" >&2
      failed=$((failed + 1))
    fi
  done

  if [[ "${failed}" -gt 0 ]]; then
    echo ""
    echo "WARNING: ${failed} job(s) failed. Check results/logs/ for details." >&2
  fi
fi

echo ""
echo "All done. Results under ${OUTPUT_BASE}/run_<epoch>/ ; logs under results/logs/"
