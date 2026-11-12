#!/usr/bin/env bash
# Build and push the FluxVLA environment image.
#
#   bash docker/build_and_push.sh                 # build, tag, push
#   PUSH=0 bash docker/build_and_push.sh          # build + tag only
#   VERSION=0.0.2 bash docker/build_and_push.sh   # bump the trailing version
#   NO_CACHE=1 bash docker/build_and_push.sh
#
# Tag convention: <os>-<python>-<cuda>-<pytorch>-<version>
#   e.g. ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
#
# Registry login is expected to be done already:
#   docker login limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env}"
VERSION="${VERSION:-0.0.1}"
LOCAL_TAG="${LOCAL_TAG:-fluxvla_env:build}"
PUSH="${PUSH:-1}"
NO_CACHE="${NO_CACHE:-0}"
PLATFORM="${PLATFORM:-linux/amd64}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
NO_PROXY="${NO_PROXY:-}"
EXTRA_BUILD_ARGS="${EXTRA_BUILD_ARGS:-}"

cd "${REPO_DIR}"

echo "[1/4] building ${LOCAL_TAG} from docker/Dockerfile"
BUILD_ARGS=(
  -f docker/Dockerfile
  -t "${LOCAL_TAG}"
  --platform "${PLATFORM}"
  --network host
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}"
  --build-arg "HTTP_PROXY=${HTTP_PROXY}"
  --build-arg "HTTPS_PROXY=${HTTPS_PROXY}"
  --build-arg "NO_PROXY=${NO_PROXY}"
  --label "org.opencontainers.image.title=fluxvla_env"
  --label "org.opencontainers.image.version=${VERSION}"
  --label "org.opencontainers.image.source=https://github.com/limxdynamics/FluxVLA"
)
if [[ "${NO_CACHE}" == "1" ]]; then
  BUILD_ARGS+=(--no-cache)
fi
if [[ -n "${EXTRA_BUILD_ARGS}" ]]; then
  # shellcheck disable=SC2206
  BUILD_ARGS+=(${EXTRA_BUILD_ARGS})
fi
docker build "${BUILD_ARGS[@]}" .

echo "[2/4] deriving tag components from the built image"
OS_ID="${OS_ID:-$(docker run --rm --entrypoint bash "${LOCAL_TAG}" -lc '. /etc/os-release; echo "$ID$VERSION_ID"')}"
PY_VER="$(docker run --rm --entrypoint python "${LOCAL_TAG}" -c 'import sys; print("py%d.%d" % sys.version_info[:2])')"
TORCH_FULL="$(docker run --rm --entrypoint python "${LOCAL_TAG}" -c 'import torch; print(torch.__version__)')"
CUDA_FULL="$(docker run --rm --entrypoint python "${LOCAL_TAG}" -c 'import torch; print(torch.version.cuda or "none")')"
TORCH_SHORT="$(printf '%s' "${TORCH_FULL}" | cut -d+ -f1 | awk -F. '{print $1 "." $2}')"
CUDA_SHORT="cu$(printf '%s' "${CUDA_FULL}" | tr -d '.')"

TAG="${OS_ID}-${PY_VER}-${CUDA_SHORT}-pytorch${TORCH_SHORT}-${VERSION}"
FULL_REF="${IMAGE_NAME}:${TAG}"

echo "      os=${OS_ID} python=${PY_VER} cuda=${CUDA_SHORT} (torch reports ${CUDA_FULL}) torch=${TORCH_FULL}"
echo "      tagged as ${FULL_REF}"

docker tag "${LOCAL_TAG}" "${FULL_REF}"

if [[ "${PUSH}" != "1" ]]; then
  echo "[3/4] PUSH=0, skipping push"
  echo "[4/4] done: ${FULL_REF}"
  exit 0
fi

echo "[3/4] pushing ${FULL_REF}"
docker push "${FULL_REF}"

echo "[4/4] verifying the pushed manifest"
docker buildx imagetools inspect "${FULL_REF}" | head -20

docker image inspect "${FULL_REF}" --format '{{json .Config.Labels}}' > "/tmp/fluxvla_env_${TAG}.labels.json" || true
echo "task image: ${FULL_REF}"
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' | head -5
