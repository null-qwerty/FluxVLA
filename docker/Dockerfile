# syntax=docker/dockerfile:1
# FluxVLA training/eval environment image.
#
# Tag convention (see docker/build_and_push.sh):
#   <os>-<python>-<cuda>-<pytorch>-<version>
#   e.g. ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
#
# Contents: CUDA 12.8 devel + Python 3.10 venv + PyTorch (cu128) +
# requirements-base/sim + robosuite/libero from the pinned forks +
# robocasa-gr1-tabletop-tasks + Isaac-GR00T source + editable FluxVLA.
# RoboCasa simulation assets, datasets and checkpoints are NOT baked in;
# mount them at runtime (see docs/dlc_training.md).

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_EXTRA_INDEX_URL=
ARG PYTHON_VERSION=3.10
ARG UV_VERSION=0.12.13
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=

ENV DEBIAN_FRONTEND=noninteractive \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# System packages needed by MuJoCo/robosuite/robocasa (headless EGL), ffmpeg,
# robosuite mesh loading, and building small C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget unzip ca-certificates ffmpeg \
        libegl1 libgl1 libglib2.0-0 libosmesa6 libx11-6 libglew-dev \
        libxi6 libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 \
        cmake ninja-build patchelf pkg-config \
    && rm -rf /var/lib/apt/lists/*

# EGL vendor ICD for NVIDIA headless rendering (same content as
# install_env.sh --skip-egl-setup expects to be pre-created).
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Python 3.10 (Ubuntu 24.04 ships 3.12; the pinned TF/open3d/sapien wheels
# need 3.10). uv installs a standalone CPython into /opt/venv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && /usr/local/bin/uv --version \
    && PYTHON_VERSION=${PYTHON_VERSION} uv venv --seed /opt/venv \
    && /opt/venv/bin/python -V \
    && /opt/venv/bin/pip install -q "uv==${UV_VERSION}"

WORKDIR /opt/fluxvla

# Only what the canonical installer needs; heavy/derived paths are excluded
# by .dockerignore.
COPY requirements-base.txt requirements-sim.txt requirements-real.txt requirements-test.txt \
     setup.py setup.cfg pytest.ini README.md LICENSE ./
COPY fluxvla ./fluxvla
COPY scripts ./scripts
COPY configs ./configs
COPY tools ./tools

# Canonical environment installer, pinned by the repo itself:
#   sim-only        -> requirements-base + requirements-sim (robosuite/libero)
#   skip egl setup  -> already created above
#   skip assets     -> 8.4 GB RoboCasa assets are mounted at runtime
#   skip robotwin   -> no cuRobo rebuild (not needed for RoboCasa)
#   skip flash-attn -> only the optional DreamZero model imports it
RUN bash scripts/install_env.sh sim-only \
        --skip-egl-setup --skip-robocasa-assets --skip-robotwin --skip-flash-attn \
    && /opt/venv/bin/python -c "import torch, robosuite, fluxvla; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Do not bake build-time proxies into the runtime image.
ENV http_proxy= https_proxy= no_proxy= HTTP_PROXY= HTTPS_PROXY=

# Make `python`/`pip` resolve to the venv even in non-login shells, and keep
# the editable FluxVLA checkout importable.
ENV PYTHONPATH=/opt/fluxvla

CMD ["bash"]
