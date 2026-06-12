#!/usr/bin/env bash
# Run CDD / CoDeC / ACR baselines across all fine-tuned checkpoints.
#
# Unlike the IPA sweep, baselines need no perturbation files — they operate
# directly on the original input texts.  Each epoch loads one checkpoint and
# scores both members and non-members, saving results to:
#   results/<MODEL_TAG>/run_<epoch>/members/baseline_results.csv
#   results/<MODEL_TAG>/run_<epoch>/non_members/baseline_results.csv
#
# Usage (from src_v3/):
#   bash scripts/run_baseline_epoch_sweep.sh
#
# Key overrides:
#   METHODS="cdd codec"         Baselines to run (default: cdd codec)
#                               Add "acr" only for small datasets — it's very slow.
#   EPOCHS="0 1 2"              Subset of epochs (default: 0–10)
#   CKPT_BASE=checkpoints/...   Base checkpoint dir (default: pythia-70m)
#   MODEL_TAG=pythia_70m        Output subdirectory name
#   OUTPUT_BASE=results/...     Base output directory
#   SEQUENTIAL=1                Run epochs sequentially (default: 0 = parallel)
#   MAX_SAMPLES=100             Limit records per JSON file (members & non-members)
#   MAX_SAMPLES_MEMBERS=50      Override limit for members only
#   MAX_SAMPLES_NON_MEMBERS=50  Override limit for non-members only
#   CONFIG=config.yml           Config with baseline.max_samples_per_split
#   BATCH_SIZE=16               Samples per GPU batch for CDD and CoDeC (default: 8)
#   CUDA_VISIBLE_DEVICES=0      GPU id (default: 0)
#   PYTHON=python               Python interpreter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"
BASELINE_SWEEP="${SCRIPT_DIR}/scripts/run_baseline_sweep.py"

CKPT_BASE="${CKPT_BASE:-checkpoints/EleutherAI_pythia-70m_fine_tuned}"
MODEL_TAG="${MODEL_TAG:-pythia_70m}"
OUTPUT_BASE="${OUTPUT_BASE:-results/${MODEL_TAG}}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SEQUENTIAL="${SEQUENTIAL:-0}"
METHODS="${METHODS:-cdd codec}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_SAMPLES_MEMBERS="${MAX_SAMPLES_MEMBERS:-}"
MAX_SAMPLES_NON_MEMBERS="${MAX_SAMPLES_NON_MEMBERS:-}"
CONFIG="${CONFIG:-config.yml}"

if [[ -n "${EPOCHS:-}" ]]; then
  read -ra EPOCH_LIST <<< "${EPOCHS}"
else
  EPOCH_LIST=(0 1 2 3 4 5 6 7 8 9 10)
fi

mkdir -p results/logs

if [[ ! -f "${BASELINE_SWEEP}" ]]; then
  echo "ERROR: missing ${BASELINE_SWEEP}" >&2
  echo "Copy scripts/run_baseline_sweep.py from the repo to this machine." >&2
  exit 1
fi

# ── validate checkpoints ──────────────────────────────────────────────────────

valid_epochs=()
for epoch in "${EPOCH_LIST[@]}"; do
  ckpt_dir="${SCRIPT_DIR}/${CKPT_BASE}/checkpoint-epoch-${epoch}"
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
echo "Methods       : ${METHODS}"
echo "Mode          : $([ "${SEQUENTIAL}" == "1" ] && echo "sequential" || echo "parallel")"
echo "GPU           : ${CUDA_DEVICE}"
echo "Checkpoint    : ${CKPT_BASE}"
echo "Output        : ${OUTPUT_BASE}"
echo "Batch size    : ${BATCH_SIZE}"
if [[ -n "${MAX_SAMPLES}" ]]; then
  echo "Max samples   : ${MAX_SAMPLES} per file (members & non-members)"
elif [[ -n "${MAX_SAMPLES_MEMBERS}" || -n "${MAX_SAMPLES_NON_MEMBERS}" ]]; then
  echo "Max samples   : members=${MAX_SAMPLES_MEMBERS:-all} non-members=${MAX_SAMPLES_NON_MEMBERS:-all}"
else
  echo "Max samples   : (full datasets — set MAX_SAMPLES=N to cap)"
fi
echo ""

# ── worker function ───────────────────────────────────────────────────────────

run_one() {
  local epoch="$1"
  local ckpt_dir="${SCRIPT_DIR}/${CKPT_BASE}/checkpoint-epoch-${epoch}"
  ckpt_dir="$(cd "${ckpt_dir}" && pwd)"
  local log_file="results/logs/baseline_run_${epoch}.log"

  local extra_args=(--config "${CONFIG}")
  [[ -n "${MAX_SAMPLES}" ]] && extra_args+=(--max-samples "${MAX_SAMPLES}")
  [[ -n "${MAX_SAMPLES_MEMBERS}" ]] && extra_args+=(--max-samples-members "${MAX_SAMPLES_MEMBERS}")
  [[ -n "${MAX_SAMPLES_NON_MEMBERS}" ]] && extra_args+=(--max-samples-non-members "${MAX_SAMPLES_NON_MEMBERS}")

  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" "${BASELINE_SWEEP}" \
    --model      "${ckpt_dir}" \
    --epoch      "${epoch}" \
    --split      both \
    --methods    ${METHODS} \
    --output-dir "${OUTPUT_BASE}" \
    --batch-size "${BATCH_SIZE}" \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1

  echo "[done] epoch=${epoch} → ${log_file}"
}

# ── launch ────────────────────────────────────────────────────────────────────

failed=0

for epoch in "${valid_epochs[@]}"; do
  echo "Launching epoch ${epoch}…"
  if [[ "${SEQUENTIAL}" == "1" ]]; then
    run_one "${epoch}" || failed=$((failed + 1))
  else
    run_one "${epoch}" &
  fi
done

# ── wait for parallel jobs ────────────────────────────────────────────────────

if [[ "${SEQUENTIAL}" != "1" && ${#valid_epochs[@]} -gt 0 ]]; then
  echo ""
  echo "All ${#valid_epochs[@]} jobs launched. Waiting for completion…"
  for job in $(jobs -p); do
    if ! wait "${job}"; then
      echo "WARNING: job PID ${job} failed — check results/logs/" >&2
      failed=$((failed + 1))
    fi
  done
fi

if [[ "${failed}" -gt 0 ]]; then
  echo "WARNING: ${failed} job(s) failed." >&2
fi

echo ""
echo "All done. Results under ${OUTPUT_BASE}/run_<epoch>/{members,non_members}/baseline_results.csv"
echo "Logs under results/logs/baseline_run_<epoch>.log"
