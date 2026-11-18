#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${1:?Usage: bash scripts/run.sh MSPOV|RAPOV2 train|stage1|stage2|test}"
ACTION="${2:-train}"
GPU="${GPU:-0}"
SEED="${SEED:-605}"
DRY_RUN="${DRY_RUN:-0}"
case "${DATASET}" in
  MSPOV)
    dataset_pkl="${PROJECT_ROOT}/dataset/msp_ov.pkl"
    image_root="${PROJECT_ROOT}/data/MSP60K/images"
    bank="${PROJECT_ROOT}/instruction/mspov_basetrain_novelall_group_combos.json"
    stage1_default=11; stage2_default=2 ;;
  RAPOV2)
    dataset_pkl="${PROJECT_ROOT}/dataset/rap_ov2.pkl"
    image_root="${PROJECT_ROOT}/data/RAPv2/RAP_dataset"
    bank="${PROJECT_ROOT}/instruction/2/rapov2_basetrain_novelall_group_combos.json"
    stage1_default=12; stage2_default=25 ;;
  *) printf 'Unsupported dataset: %s\n' "${DATASET}" >&2; exit 2 ;;
esac
case "${ACTION}" in train|stage1|stage2|test) ;; *) printf 'Unsupported action: %s\n' "${ACTION}" >&2; exit 2 ;; esac

STAGE1_EPOCHS="${STAGE1_EPOCHS:-${stage1_default}}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-${stage2_default}}"
STAGE1_LOAD_EPOCH="${STAGE1_LOAD_EPOCH:-${STAGE1_EPOCHS}}"
if (( STAGE1_LOAD_EPOCH > STAGE1_EPOCHS || STAGE1_LOAD_EPOCH < 1 )); then
  printf 'STAGE1_LOAD_EPOCH must be between 1 and STAGE1_EPOCHS\n' >&2
  exit 2
fi
EXP_PREFIX="${EXP_PREFIX:-srenet_seed${SEED}}"
stage1_exp="${EXP_PREFIX}_stage1"
stage2_exp="${EXP_PREFIX}_stage2"
stage1_checkpoint="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/logs/${DATASET}/${stage1_exp}/epoch${STAGE1_LOAD_EPOCH}.pth}"
stage2_checkpoint="${PROJECT_ROOT}/logs/${DATASET}/${stage2_exp}/epoch${STAGE2_EPOCHS}.pth"
common=("${DATASET}" --dataset_pkl "${DATASET_PKL:-${dataset_pkl}}"
  --image_root "${IMAGE_ROOT:-${image_root}}" --semantic_bank_path "${SEMANTIC_BANK:-${bank}}"
  --seed "${SEED}" --vis_gpus "${GPU}" --lr "${LR:-1e-5}" --save_freq 1)

run_cmd() {
  printf 'RUN'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != 1 ]]; then "$@"; fi
}
cd "${PROJECT_ROOT}"
if [[ "${ACTION}" == train || "${ACTION}" == stage1 ]]; then
  run_cmd "${PYTHON_BIN}" train.py "${common[@]}" --stage stage1 \
    --epoch "${STAGE1_EPOCHS}" --batchsize "${STAGE1_BATCHSIZE:-64}" --exp "${stage1_exp}"
fi
if [[ "${ACTION}" == train || "${ACTION}" == stage2 ]]; then
  if [[ "${DRY_RUN}" != 1 && ! -f "${stage1_checkpoint}" ]]; then
    printf 'Missing Stage-1 checkpoint: %s\n' "${stage1_checkpoint}" >&2; exit 2
  fi
  run_cmd "${PYTHON_BIN}" train_stage2.py "${common[@]}" --stage stage2 \
    --epoch "${STAGE2_EPOCHS}" --batchsize "${STAGE2_BATCHSIZE:-32}" \
    --checkpoint "${stage1_checkpoint}" --exp "${stage2_exp}"
fi
if [[ "${ACTION}" == train || "${ACTION}" == test ]]; then
  run_cmd "${PYTHON_BIN}" evaluate.py "${common[@]}" --stage stage2 \
    --batchsize "${EVAL_BATCHSIZE:-32}" --checkpoint "${CHECKPOINT:-${stage2_checkpoint}}" \
    --results_dir "${RESULTS_DIR:-${PROJECT_ROOT}/results}" --exp "${EXP_PREFIX}_test"
fi
