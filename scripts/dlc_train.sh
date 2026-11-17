#!/usr/bin/env bash
# Alibaba Cloud PAI-DLC entrypoint for the ego -> RoboCasa GR1 experiments.
#
# Usage (DLC custom-task command line):
#   TASK=1 bash scripts/dlc_train.sh
#   TASK=2 bash scripts/dlc_train.sh
#   TASK=3 bash scripts/dlc_train.sh
#   TASK=4 bash scripts/dlc_train.sh
#
# TASK=1  VLM LoRA (adapters on the expert too) on RoboCasa 24 tasks, 40k steps
# TASK=2  stage 1: ego VLM SFT  ->  stage 2: same as TASK=1
# TASK=3  parallel mixed training (RoboCasa VLA stream : ego VLM stream = 3:1)
# TASK=4  VLM LoRA + FULL action expert, 100k steps, lr/warmup of the
#         published PI0.5 RoboCasa recipe (51.42%). Use this to separate
#         "LoRA/40k undertrained" from the official full-finetune recipe.
#
# Topology is auto-detected from the variables DLC / PAI injects:
#   WORLD_SIZE (number of nodes), RANK (node rank), MASTER_ADDR, MASTER_PORT,
#   NPROC_PER_NODE (GPUs per node)
# with fallbacks to PAI_WORKER_NUM / PAI_WORKER_INDEX, MLP_* (Volc), then a
# single-node default. GPUs per node: NPROC_PER_NODE, MLP_WORKER_GPU,
# CUDA_VISIBLE_DEVICES, nvidia-smi -L, in that order.
#
# Overridable from the DLC job env:
#   PER_DEVICE_BS=4 TARGET_GLOBAL_BS=256 MAX_STEPS=40000 bash scripts/dlc_train.sh
#   TASK=1 DRY_RUN=1 bash scripts/dlc_train.sh      # print commands, run nothing
#   REPO_DIR=/path/to/FluxVLA (defaults to the parent of this script)
#   DATA_ROOT=/mnt/data/cpfs/users/neri (root of data/, checkpoints/, work_dirs/)
#
# Data layout assumed (see docs/dlc_training.md):
#   DATA_ROOT/data/robocasa_lerobot_V2.1   DATA_ROOT/egodex_vlm/ego.jsonl
#   DATA_ROOT/checkpoints/pi05_base        DATA_ROOT/work_dirs/

set -euo pipefail

TASK="${TASK:-}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }

if [[ -z "${TASK}" ]]; then
  die "set TASK=1|2|3 (1: LoRA+full expert on RoboCasa, 2: ego SFT then 1, 3: parallel mixed)"
fi
case "${TASK}" in 1|2|3|4|5) ;; *) die "TASK must be 1, 2, 3, 4 or 5 (got '${TASK}')" ;; esac

# ---------------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:-/mnt/data/cpfs/users/neri}"
ROBOCASA_DATA_ROOT="${ROBOCASA_DATA_ROOT:-${DATA_ROOT}/data/robocasa_lerobot_V2.1}"
EGO_VLM_DATA_ROOT="${EGO_VLM_DATA_ROOT:-${DATA_ROOT}/egodex_vlm}"
# Byte-offset index for the JSONL annotations: node-local, so the ranks of a
# job do not share one CPFS index file (and its fingerprint) across nodes.
VLM_INDEX_CACHE_DIR="${VLM_INDEX_CACHE_DIR:-/tmp/fluxvla_vlm_index}"
# Prefer an episode/task split produced by tools/split_egodex_vlm_jsonl.py so
# the SFT stream never sees validation frames; fall back to the single file.
EGO_TRAIN_JSONL="${EGO_TRAIN_JSONL:-${EGO_VLM_DATA_ROOT}/ego_train.jsonl}"
if [[ -z "${EGO_JSONL:-}" ]]; then
  if [[ -f "${EGO_TRAIN_JSONL}" ]]; then
    EGO_JSONL="${EGO_TRAIN_JSONL}"
  else
    EGO_JSONL="${EGO_VLM_DATA_ROOT}/ego.jsonl"
  fi
fi
export EGO_VLM_JSONL="${EGO_VLM_JSONL:-${EGO_JSONL}}"
# Unified EEF action space (docs/eef_action_space.md): both datasets converted
# offline by tools/convert_{robocasa,egodex}_to_eef.py.
EGO_EEF_ROOT="${EGO_EEF_ROOT:-${DATA_ROOT}/egodex_eef_v1}"
EGO_EEF_JSONL="${EGO_EEF_JSONL:-${EGO_EEF_ROOT}/ego_eef.jsonl}"
EGO_EEF_STATS="${EGO_EEF_STATS:-${EGO_EEF_ROOT}/stats.json}"
ROBOCASA_EEF_DATA_ROOT="${ROBOCASA_EEF_DATA_ROOT:-${DATA_ROOT}/data/robocasa_eef_V0.1}"
ROBOCASA_EEF_STATS="${ROBOCASA_EEF_STATS:-${ROBOCASA_EEF_DATA_ROOT}/stats.json}"
CKPT_ROOT="${CKPT_ROOT:-${DATA_ROOT}/checkpoints}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-${CKPT_ROOT}/pi05_base/model.safetensors}"
PI05_TOKENIZER="${PI05_TOKENIZER:-${CKPT_ROOT}/pi05_base}"
WORK_ROOT="${WORK_ROOT:-${DATA_ROOT}/work_dirs}"

mkdir -p "${WORK_ROOT}"

# ---------------------------------------------------------------- topology
NNODES="${WORLD_SIZE:-${PAI_WORKER_NUM:-${MLP_WORKER_NUM:-1}}}"
NODE_RANK="${RANK:-${PAI_WORKER_INDEX:-${MLP_ROLE_INDEX:-${NODE_RANK:-0}}}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  GPUS_PER_NODE="${NPROC_PER_NODE}"
elif [[ -n "${MLP_WORKER_GPU:-}" ]]; then
  GPUS_PER_NODE="${MLP_WORKER_GPU}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  GPUS_PER_NODE="$(printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '[0-9]')"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPUS_PER_NODE=1
fi
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "cannot resolve GPUs per node (got '${GPUS_PER_NODE}')"

if command -v nvidia-smi >/dev/null 2>&1; then
  SMI_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${SMI_GPUS}" =~ ^[1-9][0-9]*$ && "${SMI_GPUS}" != "${GPUS_PER_NODE}" ]]; then
    echo "WARN: nvidia-smi reports ${SMI_GPUS} GPUs but resolved ${GPUS_PER_NODE}; set NPROC_PER_NODE explicitly if wrong" >&2
  fi
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
  else
    die "MASTER_ADDR is not injected for a multi-node job (WORLD_SIZE=${NNODES})"
  fi
fi

TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# ---------------------------------------------------------------- batch math
PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
TARGET_GLOBAL_BS="${TARGET_GLOBAL_BS:-256}"
# Per-task default: TASK 4 matches the published 100k-step recipe.
case "${TASK}" in
  4) MAX_STEPS="${MAX_STEPS:-100000}" ;;
  *) MAX_STEPS="${MAX_STEPS:-40000}" ;;
esac
NUM_WORKERS="${NUM_WORKERS:-4}"
# Checkpoint cadence. The base runner saves on `global_step % save_iter_interval`,
# so without this a 40k-step run only writes at 10k/20k/30k/40k and early
# weights (before overfitting) are lost.
SAVE_ITER="${SAVE_ITER:-1000}"
# Stage 1 (ego VLM SFT) has 4.1k samples and ~17 steps per epoch: loss is
# already memorised by ~150 steps, so 40k steps is 2300 epochs of overfitting.
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-20000}"
STAGE1_SAVE_ITER="${STAGE1_SAVE_ITER:-1000}"

MICRO=$((PER_DEVICE_BS * TOTAL_GPUS))
GA=$(( (TARGET_GLOBAL_BS + MICRO - 1) / MICRO ))
[[ "${GA}" -ge 1 ]] || GA=1
GLOBAL_BS=$((MICRO * GA))

# ---------------------------------------------------------------- helpers
log() { echo "[dlc] $*"; }

check_dir() {
  [[ -d "$1" ]] || die "missing directory: $1
  -> $2"
}

check_file() {
  [[ -f "$1" ]] || die "missing file: $1
  -> $2"
}

print_env() {
  log "task=${TASK} repo=${REPO_DIR}"
  log "topology: nodes=${NNODES} rank=${NODE_RANK} gpus/node=${GPUS_PER_NODE} total_gpus=${TOTAL_GPUS}"
  log "master: ${MASTER_ADDR}:${MASTER_PORT}"
  log "batch: per_device=${PER_DEVICE_BS} micro=${MICRO} grad_accum=${GA} global=${GLOBAL_BS} (target ${TARGET_GLOBAL_BS})"
  log "steps=${MAX_STEPS} save_iter=${SAVE_ITER} workers=${NUM_WORKERS}"
  log "robocasa=${ROBOCASA_DATA_ROOT}"
  log "pi05_ckpt=${PI05_CHECKPOINT}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/[dlc] gpu /'
  fi
  if git -C "${REPO_DIR}" rev-parse --short HEAD >/dev/null 2>&1; then
    log "git=$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)@$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
  fi
}

run_train() {
  # run_train <config> <work_dir> [extra cfg-options ...]
  # RUN_MAX_STEPS / RUN_SAVE_ITER allow a per-call budget (stage 1 != stage 2).
  local config="$1"; local work_dir="$2"; shift 2
  local max_steps="${RUN_MAX_STEPS:-${MAX_STEPS}}"
  local save_iter="${RUN_SAVE_ITER:-${SAVE_ITER}}"
  local cfg_opts=(
    "train_dataloader.per_device_batch_size=${PER_DEVICE_BS}"
    "train_dataloader.per_device_num_workers=${NUM_WORKERS}"
    "runner.grad_accumulation_steps=${GA}"
    "runner.max_steps=${max_steps}"
    "runner.save_iter_interval=${save_iter}"
    # Base runner asserts that max_steps and max_epochs are not both set; the
    # launcher is always step-based, so clear any epoch budget from the config.
    "runner.max_epochs=None"
    "runner.metric.active_trackers=('jsonl',)"
    "$@"
  )
  if [[ "${DRY_RUN}" != "1" ]]; then
    log "preflight: loading ${config}"
    "${PYTHON_BIN}" scripts/_preflight_config.py "${config}" \
      || die "config/data preflight failed for ${config} (see messages above)"
    # Build the JSONL index once per node: a stale CPFS metadata view otherwise
    # kills the whole job inside build_dataset_from_cfg with
    # "Annotation changed during indexing".
    mkdir -p "${VLM_INDEX_CACHE_DIR}"
    log "preflight: JSONL index (cache=${VLM_INDEX_CACHE_DIR})"
    "${PYTHON_BIN}" scripts/_preflight_index.py "${config}" \
      || die "JSONL annotation/index preflight failed for ${config}"
  fi
  mkdir -p "${work_dir}"
  local cmd=(bash scripts/train.sh "${config}" "${work_dir}" --cfg-options "${cfg_opts[@]}")
  log "run: ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

print_env >&2
{
  echo "task=${TASK} time=$(date -Is) host=$(hostname)"
  echo "nodes=${NNODES} rank=${NODE_RANK} gpus_per_node=${GPUS_PER_NODE} total_gpus=${TOTAL_GPUS} master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "per_device_bs=${PER_DEVICE_BS} grad_accum=${GA} global_bs=${GLOBAL_BS} max_steps=${MAX_STEPS} save_iter=${SAVE_ITER} ego_jsonl=${EGO_VLM_JSONL}"
  env | grep -E '^(WORLD_SIZE|RANK|NODE_RANK|NPROC_PER_NODE|MASTER_ADDR|MASTER_PORT|NVIDIA_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|PAI_|MLP_|DLC_)' | sort
} > "${WORK_ROOT}/dlc_env_TASK${TASK}_rank${NODE_RANK}.txt" 2>/dev/null || true

# ---------------------------------------------------------------- launch
export NPROC_PER_NODE="${GPUS_PER_NODE}"
export WORLD_SIZE="${NNODES}"
export RANK="${NODE_RANK}"
export MASTER_ADDR
export MASTER_PORT
export ROBOCASA_DATA_ROOT EGO_VLM_DATA_ROOT EGO_VLM_JSONL
export PI05_CHECKPOINT PI05_TOKENIZER VLM_INDEX_CACHE_DIR
export EGO_EEF_ROOT EGO_EEF_JSONL EGO_EEF_STATS ROBOCASA_EEF_DATA_ROOT ROBOCASA_EEF_STATS
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

check_dir "${ROBOCASA_DATA_ROOT}" "sync the RoboCasa dataset to ${DATA_ROOT}/data/robocasa_lerobot_V2.1"
check_file "${PI05_CHECKPOINT}" "put the Pi0.5 base checkpoint at ${CKPT_ROOT}/pi05_base (see docs/dlc_training.md)"

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN="python3"
log "python: $(command -v "${PYTHON_BIN}") $("${PYTHON_BIN}" -V 2>&1)"
if [[ "${DRY_RUN}" != "1" ]]; then
  if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    die "torch is not importable by ${PYTHON_BIN}; use the FluxVLA training image (see scripts/install_env.sh)"
  fi
  if ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    die "torch cannot see any GPU in this container; check the DLC resource spec / nvidia-smi"
  fi
  # The image is environment-only: install the project wheel for this node GPU.
  bash scripts/install_project.sh
  if ! "${PYTHON_BIN}" -c "import fluxvla" >/dev/null 2>&1; then
    die "fluxvla is not importable after scripts/install_project.sh"
  fi
fi

case "${TASK}" in
  1)
    run_train \
      "configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py" \
      "${WORK_ROOT}/pi05_robocasa_lora_full_expert"
    ;;
  2)
    # Unified EEF action space: ego full fine-tune -> RoboCasa VLM LoRA.
    # See docs/eef_action_space.md and docs/dlc_training.md.
    check_file "${EGO_EEF_JSONL}" \
      "run tools/convert_egodex_to_eef.py --root <egodex> --out ${EGO_EEF_ROOT} --stats-out ${EGO_EEF_STATS}"
    check_file "${EGO_EEF_STATS}" "the EgoDex EEF conversion must write stats.json"
    check_dir "${ROBOCASA_EEF_DATA_ROOT}" \
      "run tools/convert_robocasa_to_eef.py for all 24 task dirs into ${ROBOCASA_EEF_DATA_ROOT}"
    check_file "${ROBOCASA_EEF_STATS}" "the RoboCasa EEF conversion must write stats.json"
    log "stage 1/2: ego EEF full fine-tune (${EGO_EEF_JSONL}, max_steps=${STAGE1_MAX_STEPS}, save_iter=${STAGE1_SAVE_ITER})"
    RUN_MAX_STEPS="${STAGE1_MAX_STEPS}" RUN_SAVE_ITER="${STAGE1_SAVE_ITER}" run_train \
      "configs/pi05/pi05_paligemma_egodex_eef_full_finetune.py" \
      "${WORK_ROOT}/pi05_egodex_eef_sft"
    STAGE1_CKPT="${WORK_ROOT}/pi05_egodex_eef_sft/checkpoints/latest-checkpoint.safetensors"
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "stage 2/2 (dry-run): would export EGO_STAGE1_CHECKPOINT=${STAGE1_CKPT}"
    else
      check_file "${STAGE1_CKPT}" "stage 1 finished without writing ${STAGE1_CKPT}"
      export EGO_STAGE1_CHECKPOINT="${STAGE1_CKPT}"
      log "stage 2/2: RoboCasa EEF LoRA from ${STAGE1_CKPT}"
    fi
    # Stage 2 reads the converted (EEF) RoboCasa tree.
    export ROBOCASA_DATA_ROOT="${ROBOCASA_EEF_DATA_ROOT}"
    run_train \
      "configs/pi05/pi05_paligemma_robocasa_eef_lora.py" \
      "${WORK_ROOT}/pi05_robocasa_eef_lora_from_ego"
    ;;
  3)
    check_file "${EGO_JSONL}" "generate it with tools/egodex_to_vlm_jsonl.py (see docs/dlc_training.md)"
    log "parallel mixed training (vla:vlm = 3:1)"
    run_train \
      "configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py" \
      "${WORK_ROOT}/pi05_robocasa_ego_mixed"
    ;;
  4)
    log 'VLM LoRA + full action expert, published 100k-step recipe'
    run_train \
      "configs/pi05/pi05_paligemma_robocasa_vlm_lora_expert_full.py" \
      "${WORK_ROOT}/pi05_robocasa_lora_vlm_expert_full"
    ;;
  5)
    # Legacy joint-space serial recipe (kept for comparison with TASK 2).
    check_file "${EGO_JSONL}" "generate it with tools/egodex_to_vlm_jsonl.py"
    log "stage 1/2: ego VLM SFT (${EGO_JSONL}, max_steps=${STAGE1_MAX_STEPS})"
    RUN_MAX_STEPS="${STAGE1_MAX_STEPS}" RUN_SAVE_ITER="${STAGE1_SAVE_ITER}" run_train \
      "configs/pi05/pi05_paligemma_egodex_vlm_sft.py" \
      "${WORK_ROOT}/pi05_egodex_vlm_sft"
    STAGE1_CKPT="${WORK_ROOT}/pi05_egodex_vlm_sft/checkpoints/latest-checkpoint.safetensors"
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "stage 2/2 (dry-run): would export EGO_STAGE1_CHECKPOINT=${STAGE1_CKPT}"
    else
      check_file "${STAGE1_CKPT}" "stage 1 finished without writing ${STAGE1_CKPT}"
      export EGO_STAGE1_CHECKPOINT="${STAGE1_CKPT}"
    fi
    run_train \
      "configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py" \
      "${WORK_ROOT}/pi05_robocasa_lora_from_ego"
    ;;
esac

log "done task=${TASK}"
