#!/usr/bin/env bash
# DLC entry point for RoboCasa GR1 tabletop evaluation (24 tasks x 50 trials).
#
# Usage (inside the FluxVLA image on a GPU node):
#   EVAL_TASK=1 bash scripts/dlc_eval.sh                # TASK 1 (RoboCasa-only LoRA)
#   EVAL_TASK=2 bash scripts/dlc_eval.sh                # TASK 2 (ego SFT -> RoboCasa LoRA)
#   EVAL_TASK=3 bash scripts/dlc_eval.sh                # TASK 3 (ego + RoboCasa mixed)
#   CONFIG=configs/...py CKPT=/path/ckpt.safetensors bash scripts/dlc_eval.sh
#
# Common overrides:
#   DATA_ROOT=/mnt/data/cpfs/users/neri
#   TASK_IDS="0 1 2"            # task ids (default: all 24)
#   NUM_TRIALS_PER_TASK=50
#   NUM_GPUS=8                  # default: visible GPUs
#   MAX_TASKS_PER_GPU=1
#   MAX_EPISODE_STEPS=720
#   SAVE_VIDEO=False
#   OUTPUT_DIR=...              # default $DATA_ROOT/work_dirs/eval_runs/<run>_<ts>
#   ROBOCASA_ASSETS_DIR=...     # default $DATA_ROOT/env/robocasa_gr1_assets/assets
#   SMOKE=1                     # 1 task, 1 trial, 30 steps (write to *_smoke_* dir)
#   DRY_RUN=1                   # print the command without running the sim
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/mnt/data/cpfs/users/neri}"
EVAL_TASK="${EVAL_TASK:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSETS_TARGET="/opt/fluxvla/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets"

die() { echo "ERROR: $*" >&2; exit 1; }

case "${EVAL_TASK}" in
  1)
    CONFIG="${CONFIG:-configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py}"
    CKPT="${CKPT:-${DATA_ROOT}/work_dirs/pi05_robocasa_lora_full_expert/checkpoints/latest-checkpoint.safetensors}"
    ;;
  2)
    CONFIG="${CONFIG:-configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py}"
    CKPT="${CKPT:-${DATA_ROOT}/work_dirs/pi05_robocasa_lora_from_ego/checkpoints/latest-checkpoint.safetensors}"
    ;;
  3)
    CONFIG="${CONFIG:-configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py}"
    CKPT="${CKPT:-${DATA_ROOT}/work_dirs/pi05_robocasa_ego_mixed/checkpoints/latest-checkpoint.safetensors}"
    ;;
  *)
    die "EVAL_TASK must be 1, 2 or 3 (got '${EVAL_TASK}')"
    ;;
esac

SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
TASK_IDS="${TASK_IDS:-}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"

if [[ "${SMOKE}" == "1" ]]; then
  TASK_IDS="${TASK_IDS:-0}"
  NUM_TRIALS_PER_TASK="${SMOKE_TRIALS:-1}"
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-30}"
  SAVE_VIDEO="False"
fi

export DATA_ROOT
export ROBOCASA_ASSETS_DIR="${ROBOCASA_ASSETS_DIR:-${DATA_ROOT}/env/robocasa_gr1_assets/assets}"
export PI05_TOKENIZER="${PI05_TOKENIZER:-${DATA_ROOT}/checkpoints/pi05_base}"
export PI05_CHECKPOINT="${PI05_CHECKPOINT:-${DATA_ROOT}/checkpoints/pi05_base/model.safetensors}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-7}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PATH="/opt/venv/bin:${PATH}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
[[ -n "${PYTHON_BIN}" ]] || die "no python interpreter found"

if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    NUM_GPUS="${NPROC_PER_NODE}"
  else
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi
[[ "${NUM_GPUS:-0}" =~ ^[0-9]+$ && "${NUM_GPUS}" -ge 1 ]] || die "could not resolve GPU count; set NUM_GPUS"

echo "[dlc-eval] repo=${REPO_ROOT}"
echo "[dlc-eval] eval_task=${EVAL_TASK} config=${CONFIG}"
echo "[dlc-eval] ckpt=${CKPT}"
echo "[dlc-eval] gpus=${NUM_GPUS} trials=${NUM_TRIALS_PER_TASK} tasks=${TASK_IDS:-all} max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
echo "[dlc-eval] assets=${ROBOCASA_ASSETS_DIR}"

# The image is environment-only: install the project wheel for this node GPU.
bash scripts/install_project.sh

assets_complete() {
  local dir="$1"
  local rel
  for rel in fixtures textures arenas scenes objects/objaverse objects/sketchfab objects/lightwheel; do
    [[ -d "${dir}/${rel}" ]] || return 1
  done
  return 0
}

# The image ships only stub assets; point the robocasa package at the CPFS tree.
if ! assets_complete "${ASSETS_TARGET}"; then
  [[ "${ASSETS_TARGET}" == "/opt/fluxvla/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets" ]] \
    || die "unexpected asset target path: ${ASSETS_TARGET}"
  assets_complete "${ROBOCASA_ASSETS_DIR}" \
    || die "RoboCasa assets incomplete at ${ROBOCASA_ASSETS_DIR} (need fixtures, textures, generative_textures, objects/{objaverse,sketchfab,lightwheel}); run scripts/download_robocasa_assets.py"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dlc-eval] DRY_RUN: would link ${ROBOCASA_ASSETS_DIR} -> ${ASSETS_TARGET}"
  else
    rm -rf "${ASSETS_TARGET}"
    ln -s "${ROBOCASA_ASSETS_DIR}" "${ASSETS_TARGET}"
    echo "[dlc-eval] linked assets: ${ASSETS_TARGET} -> ${ROBOCASA_ASSETS_DIR}"
  fi
fi

"${PYTHON_BIN}" scripts/_preflight_eval.py "${CONFIG}" \
  --ckpt-path "${CKPT}" --assets-dir "${ROBOCASA_ASSETS_DIR}" \
  || die "eval preflight failed"

STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="$(basename "$(dirname "$(dirname "${CKPT}")")")"
if [[ "${SMOKE}" == "1" ]]; then
  DEFAULT_OUT="${DATA_ROOT}/work_dirs/eval_runs/${TAG}_smoke_${STAMP}"
else
  DEFAULT_OUT="${DATA_ROOT}/work_dirs/eval_runs/${TAG}_${STAMP}"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUT}}"
mkdir -p "${OUTPUT_DIR}"

export TASK_IDS NUM_TRIALS_PER_TASK MAX_TASKS_PER_GPU MAX_EPISODE_STEPS SAVE_VIDEO

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dlc-eval] DRY_RUN: NUM_GPUS=${NUM_GPUS} TASK_IDS=${TASK_IDS:-all} trials=${NUM_TRIALS_PER_TASK} output=${OUTPUT_DIR}"
  echo "[dlc-eval] DRY_RUN: bash scripts/eval_robocasa_manager.sh ${CONFIG} ${CKPT}"
  exit 0
fi

set +e
bash scripts/eval_robocasa_manager.sh "${CONFIG}" "${CKPT}" 2>&1 \
  | tee "${OUTPUT_DIR}/manager.log"
RC=${PIPESTATUS[0]}
set -e

echo "[dlc-eval] exit=${RC} output=${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/summary.csv" ]]; then
  echo "[dlc-eval] summary.csv:"
  head -5 "${OUTPUT_DIR}/summary.csv"
fi
exit "${RC}"
