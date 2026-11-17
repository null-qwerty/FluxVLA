# FluxVLA *environment* image - no project code is baked in.
# (No `# syntax=` frontend pin: BuildKit's built-in frontend supports the
# `RUN --mount=type=cache` usage below and needs no registry round-trip.)
#
# Tag convention (see docker/build_and_push.sh):
#   <os>-<python>-<cuda>-<pytorch>-<version>
#   e.g. ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.3
#
# Contents: CUDA 12.8 devel + Python 3.10 venv + PyTorch 2.8 (cu128) +
# requirements-base/sim + robosuite/libero from the pinned forks +
# robocasa-gr1-tabletop-tasks + Isaac-GR00T source + flash-attn.
# The FluxVLA python package is NOT installed here; the training/eval
# launcher installs a prebuilt wheel from the mounted repo at job start
# (scripts/install_project.sh), so code changes never need an image rebuild.
# RoboCasa simulation assets, datasets and checkpoints are mounted at runtime
# (see docs/dlc_training.md).

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PYTHON_VERSION=3.10
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=mirrors.aliyun.com,mirrors.cloud.aliyuncs.com,.aliyuncs.com,archive.ubuntu.com,security.ubuntu.com

ENV DEBIAN_FRONTEND=noninteractive \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# System packages for MuJoCo/robosuite/robocasa (headless EGL), ffmpeg and
# runtime compilation of the FluxVLA CUDA extensions (nvcc lives in /usr/local/cuda).
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
               -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
               /etc/apt/sources.list.d/ubuntu.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget unzip ca-certificates ffmpeg \
        libegl1 libgl1 libglib2.0-0 libosmesa6 libx11-6 libglew-dev \
        libxi6 libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 \
        cmake ninja-build patchelf pkg-config \
    && rm -rf /var/lib/apt/lists/*

# EGL vendor ICD for NVIDIA headless rendering (install_env.sh --skip-egl-setup
# expects it to exist already).
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Python 3.10: Ubuntu 24.04 ships 3.12, while the pinned TF/open3d/sapien
# wheels need 3.10. uv installs a standalone CPython into /opt/venv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && /usr/local/bin/uv --version \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" --seed /opt/venv \
    && /opt/venv/bin/python -V \
    && /opt/venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

# Build-time source staging. Only the installer and its requirement files are
# copied; the directory is deleted before the image is committed, so the
# runtime image contains no FluxVLA code. Third-party sources cloned by the
# installer stay under /opt/fluxvla/src (environment, used by robocasa eval).
WORKDIR /build
COPY requirements-base.txt requirements-sim.txt requirements-real.txt requirements-test.txt ./
COPY scripts/install_env.sh ./scripts/install_env.sh

# Canonical environment installer, pinned by the repo itself:
#   sim-only       -> requirements-base + requirements-sim (robosuite/libero)
#   skip egl setup -> already created above
#   skip assets    -> 8.4 GB RoboCasa assets are mounted at runtime
#   skip robotwin  -> no cuRobo rebuild (not needed for RoboCasa)
#   skip project   -> the FluxVLA package is installed at job start
# OpenCV is pinned because requirements-base leaves it unpinned while
# requirements-sim pins 4.11.0.86, which otherwise makes pip backtrack through
# several 60-80 MB wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/fluxvla \
    printf 'opencv-python==4.11.0.86\nopencv-python-headless==4.11.0.86\nprotobuf==3.20.3\n' > /tmp/pip-constraints.txt \
    && export CUDA_HOME=/usr/local/cuda \
    && export PATH="/usr/local/cuda/bin:${PATH}" \
    && export CPATH="/usr/local/cuda/include:${CPATH}" \
    && export FORCE_CUDA=1 \
    && export TORCH_CUDA_ARCH_LIST="8.0 9.0 12.0" \
    && export FLUXVLA_ROBOCASA_SRC_ROOT=/opt/fluxvla/src \
    && PIP_CONSTRAINT=/tmp/pip-constraints.txt \
       bash scripts/install_env.sh sim-only --profile cu128 \
        --skip-egl-setup --skip-robocasa-assets --skip-robotwin --skip-project \
    && /opt/venv/bin/python -c "import torch, robosuite, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'robosuite', robosuite.__version__, 'flash-attn', flash_attn.__version__)" \
    && /opt/venv/bin/python -c "import google.protobuf as p; from sentencepiece import sentencepiece_model_pb2 as m; print('protobuf', p.__version__, 'sentencepiece pb2 ok:', bool(m.ModelProto))" \
    && cd / && rm -rf /build

# The build-time staging directory is gone; keep a valid workdir.
WORKDIR /opt/fluxvla

# RoboCasa GR1 task repo is installed by the script; keep the asset mount point.
RUN mkdir -p /opt/fluxvla/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets

# Runtime compilation of the FluxVLA CUDA extensions needs the toolkit on PATH.
ENV CUDA_HOME=/usr/local/cuda
# Do not bake build-time proxies into the runtime image.
ENV http_proxy= https_proxy= no_proxy= HTTP_PROXY= HTTPS_PROXY=

# Sanity check without a GPU (the FluxVLA package is installed by the launcher):
#   docker run --rm IMAGE python -c "import torch, robosuite, flash_attn"
# The training entrypoint is scripts/dlc_train.sh from the mounted repo.

CMD ["bash"]
