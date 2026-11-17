#!/usr/bin/env bash
# Install the FluxVLA python package from the mounted repo into the current
# venv. The docker image is environment-only (see docker/Dockerfile), so this
# runs at job start.
#
# Why a wheel instead of pip install -e:
#   * the 3 CUDA extensions must be compiled for the GPUs of *this* node
#     (sm_80 / sm_90 / sm_120 differ), so compiling inside the image cannot
#     cover every cluster;
#   * a per-(commit, arch) wheel cached on the shared mount makes the usual
#     job start a few seconds instead of a few minutes.
#
# Usage: bash scripts/install_project.sh
#
# Environment:
#   PYTHON_BIN               python to install into (default: python)
#   DATA_ROOT                shared root for the wheel cache
#                            (default: /mnt/data/cpfs/users/neri)
#   FLUXVLA_WHEEL_DIR        override the wheel cache directory
#   SKIP_PROJECT_INSTALL     set to 1 to keep an existing install
#   PROJECT_INSTALL_DRY_RUN  set to 1 to print the plan only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/cpfs/users/neri}"
WHEEL_DIR="${FLUXVLA_WHEEL_DIR:-${DATA_ROOT}/env/fluxvla_wheels}"
DRY_RUN="${PROJECT_INSTALL_DRY_RUN:-0}"

log() { echo "[install-project] $*"; }
die() { echo "[install-project] ERROR: $*" >&2; exit 1; }

if [[ "${SKIP_PROJECT_INSTALL:-0}" == "1" ]]; then
  log "SKIP_PROJECT_INSTALL=1, keeping the existing install"
  exit 0
fi

# Keep a dev editable install of *this* repo (path inside REPO_DIR). A fluxvla
# baked into an older image lives outside the repo and must be replaced, so
# the mounted code always wins.
if [[ "${DRY_RUN}" != "1" ]]; then
  INSTALLED="$("${PYTHON_BIN}" -c "import fluxvla; print(fluxvla.__file__)" 2>/dev/null | tail -1 || true)"
  if [[ -n "${INSTALLED}" && "${INSTALLED}" == "${REPO_DIR}"/* ]]; then
    log "using editable install of the mounted repo: ${INSTALLED}"
    exit 0
  fi
  if [[ -n "${INSTALLED}" ]]; then
    log "replacing install outside the repo: ${INSTALLED}"
    "${PYTHON_BIN}" -m pip uninstall -y fluxvla >/dev/null 2>&1 || true
  fi
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python not found: ${PYTHON_BIN}"

# CUDA arch of the visible GPUs (torch comes from the image environment).
ARCH="$("${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('no CUDA device visible')
major, minor = torch.cuda.get_device_capability(0)
print(f'{major}.{minor}')
PY
)" || die "cannot detect CUDA arch (is a GPU visible?)"

COMMIT="$(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if git -C "${REPO_DIR}" diff --quiet HEAD -- fluxvla setup.py setup.cfg 2>/dev/null; then
  WHEEL_NAME="fluxvla-${COMMIT}-cu${ARCH}.whl"
else
  WHEEL_NAME=""
fi
CACHED_WHEEL="${WHEEL_DIR}/${WHEEL_NAME}"

log "repo=${REPO_DIR}"
log "python=$("${PYTHON_BIN}" -V 2>&1) arch=${ARCH} commit=${COMMIT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY RUN plan:"
  if [[ -n "${WHEEL_NAME}" ]]; then
    log "  wheel cache: ${CACHED_WHEEL}"
  else
    log "  working tree has FluxVLA changes, wheel will be built uncached"
  fi
  log "  stage  : mktemp; copy fluxvla setup.py setup.cfg README.md LICENSE"
  log "  build  : pip wheel --no-deps --no-build-isolation (TORCH_CUDA_ARCH_LIST=${ARCH})"
  log "  install: pip install --no-deps --no-build-isolation --force-reinstall <wheel>"
  exit 0
fi

install_wheel() {
  local wheel="$1"
  log "installing $(basename "${wheel}")"
  "${PYTHON_BIN}" -m pip install --no-deps --no-build-isolation --force-reinstall "${wheel}"
}

if [[ -n "${WHEEL_NAME}" && -f "${CACHED_WHEEL}" ]]; then
  log "using cached wheel ${CACHED_WHEEL}"
  install_wheel "${CACHED_WHEEL}"
else
  STAGE="$(mktemp -d)"
  BUILD_DIR="$(mktemp -d)"
  trap "rm -rf ${STAGE} ${BUILD_DIR}" EXIT
  # tar instead of rsync: the training image does not ship rsync.
  tar -C "${REPO_DIR}" -cf - --exclude=build --exclude='*.egg-info' --exclude=__pycache__ --exclude='*.so' fluxvla setup.py setup.cfg README.md LICENSE | tar -xf - -C "${STAGE}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export FORCE_CUDA=1
  export TORCH_CUDA_ARCH_LIST="${ARCH}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  log "building wheel for CUDA arch ${ARCH} (3 extensions, a few minutes)"
  "${PYTHON_BIN}" -m pip wheel --no-deps --no-build-isolation -w "${BUILD_DIR}" "${STAGE}"
  WHEEL="$(ls "${BUILD_DIR}"/*.whl 2>/dev/null | head -1 || true)"
  [[ -n "${WHEEL}" ]] || die "wheel build produced no artifact"
  install_wheel "${WHEEL}"
  if [[ -n "${WHEEL_NAME}" ]] && mkdir -p "${WHEEL_DIR}" 2>/dev/null && [[ -w "${WHEEL_DIR}" ]]; then
    cp -f "${WHEEL}" "${CACHED_WHEEL}" 2>/dev/null || log "could not cache the wheel at ${CACHED_WHEEL}"
  fi
fi

"${PYTHON_BIN}" -c "import fluxvla; print('[install-project] fluxvla', fluxvla.__file__)"
log "ok"
