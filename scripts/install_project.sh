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

# Every probe below must run from a NEUTRAL directory. `python -c` puts the
# current working directory on sys.path[0], so running `import fluxvla` from
# the repo root imports the *source tree* (${REPO_DIR}/fluxvla/__init__.py) and
# looks like a valid install even when site-packages has nothing. That false
# positive made this script skip the install while torchrun (sys.path[0] =
# scripts/, so no source-tree shadowing) failed with ModuleNotFoundError.
PROBE_DIR="$(mktemp -d)"
cleanup_probe() { rm -rf "${PROBE_DIR}"; }
trap cleanup_probe EXIT

# Prints the location fluxvla would be imported from, '' when not installed.
probe_install() {
  ( cd "${PROBE_DIR}" && "${PYTHON_BIN}" -c "import fluxvla; print(fluxvla.__file__)" 2>/dev/null | tail -1 || true )
}

# Where site-packages *thinks* fluxvla lives (no import, no CWD shadowing).
probe_dist() {
  ( cd "${PROBE_DIR}" && "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import importlib.metadata as md
try:
    dist = md.distribution('fluxvla')
except md.PackageNotFoundError:
    raise SystemExit(0)
print(dist._path)
PY
  )
}

# Keep a dev editable install of *this* repo (path inside REPO_DIR). A fluxvla
# baked into an older image lives outside the repo and must be replaced, so
# the mounted code always wins.
if [[ "${DRY_RUN}" != "1" ]]; then
  INSTALLED="$(probe_install)"
  if [[ -n "${INSTALLED}" && "${INSTALLED}" == "${REPO_DIR}"/* ]]; then
    log "using editable install of the mounted repo: ${INSTALLED}"
    exit 0
  fi
  if [[ -n "${INSTALLED}" || -n "$(probe_dist)" ]]; then
    log "replacing existing install (import=${INSTALLED:-none}, dist=$(probe_dist))"
    "${PYTHON_BIN}" -m pip uninstall -y fluxvla >/dev/null 2>&1 || true
  fi
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python not found: ${PYTHON_BIN}"

# CUDA arch of the visible GPUs (torch comes from the image environment).
# CPU-only conversion/validation hosts can pass TORCH_CUDA_ARCH_LIST instead.
ARCH="$("${PYTHON_BIN}" - <<'PY'
import os
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f'{major}.{minor}')
else:
    archs = os.environ.get('TORCH_CUDA_ARCH_LIST', '').split()
    if not archs:
        raise SystemExit('no CUDA device visible and TORCH_CUDA_ARCH_LIST is unset')
    print(archs[0].replace('+PTX', ''))
PY
)" || die "cannot resolve the CUDA arch to build for"

# Cache key for the wheel. `safe.directory=*` is required because the mounted
# repo is owned by the cluster user while the container runs as root (git
# otherwise refuses with "detected dubious ownership" and caching silently
# degrades to a rebuild per job). Without git, fall back to a content hash of
# the python sources, so cache hits still work inside minimal images.
git_repo() { git -c safe.directory='*' -C "${REPO_DIR}" "$@" 2>/dev/null; }
COMMIT_KEY=""
if git_repo rev-parse --short HEAD >/dev/null; then
  if git_repo diff --quiet HEAD -- fluxvla setup.py setup.cfg; then
    COMMIT_KEY="$(git_repo rev-parse --short HEAD)"
  fi
else
  COMMIT_KEY="$( cd "${REPO_DIR}" && { find fluxvla -type f -name '*.py' -not -path '*/__pycache__/*' | sort; echo setup.py; echo setup.cfg; } | xargs -r sha256sum | sha256sum | cut -c1-12 )" 2>/dev/null || COMMIT_KEY=""
fi
# Cache as a per-(key, arch) DIRECTORY holding the real wheel filename: pip
# rejects a renamed wheel ("Invalid wheel filename (wrong number of parts)").
if [[ -n "${COMMIT_KEY}" ]]; then
  CACHE_DIR="${WHEEL_DIR}/fluxvla-${COMMIT_KEY}-cu${ARCH}"
  CACHED_WHEEL="$(ls "${CACHE_DIR}"/*.whl 2>/dev/null | head -1 || true)"
else
  CACHE_DIR=""
  CACHED_WHEEL=""
fi

log "repo=${REPO_DIR}"
log "python=$("${PYTHON_BIN}" -V 2>&1) arch=${ARCH} key=${COMMIT_KEY:-<dirty worktree>}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY RUN plan:"
  if [[ -n "${CACHE_DIR}" ]]; then
    log "  wheel cache: ${CACHE_DIR}/<wheel>"
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

if [[ -n "${CACHED_WHEEL}" && -f "${CACHED_WHEEL}" ]]; then
  log "using cached wheel ${CACHED_WHEEL}"
  install_wheel "${CACHED_WHEEL}"
else
  STAGE="$(mktemp -d)"
  BUILD_DIR="$(mktemp -d)"
  trap 'rm -rf "${STAGE}" "${BUILD_DIR}" "${PROBE_DIR}"' EXIT
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
  if [[ -n "${CACHE_DIR}" ]] && mkdir -p "${CACHE_DIR}" 2>/dev/null && [[ -w "${CACHE_DIR}" ]]; then
    # Publish atomically: several nodes of a multi-node job may build the same
    # (commit, arch) wheel at once, and a half-written .whl would poison every
    # later job that trusts the cache.
    DEST="${CACHE_DIR}/$(basename "${WHEEL}")"
    PARTIAL="${DEST}.partial.$$"
    if cp -f "${WHEEL}" "${PARTIAL}" 2>/dev/null && mv -f "${PARTIAL}" "${DEST}" 2>/dev/null; then
      log "cached wheel at ${DEST}"
    else
      rm -f "${PARTIAL}" 2>/dev/null || true
      log "could not cache the wheel at ${DEST}"
    fi
  fi
fi

# `import fluxvla` needs a GPU (eagle2 probes CUDA at import time), so CPU
# conversion/validation hosts verify the install spec instead. The check runs
# from PROBE_DIR so a source-tree ${REPO_DIR}/fluxvla can never satisfy it.
( cd "${PROBE_DIR}" && "${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch

spec = importlib.util.find_spec('fluxvla')
assert spec is not None, 'fluxvla is not installed'
print('[install-project] fluxvla spec:', spec.origin or spec.submodule_search_locations)
if torch.cuda.is_available():
    import fluxvla
    print('[install-project] fluxvla', fluxvla.__file__)
else:
    print('[install-project] no GPU: skipped import check (CPU node)')
PY
)
log "ok"
