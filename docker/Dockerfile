# syntax=docker/dockerfile:1
# FluxVLA training/eval environment image.
#
# Tag convention (see docker/build_and_push.sh):
#   <os>-<python>-<cuda>-<pytorch>-<version>
#   e.g. ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
#
# Contents: CUDA 12.8 devel + Python 3.10 venv + PyTorch 2.8 (cu128) +
# requirements-base/sim + robosuite/libero from the pinned forks +
# robocasa-gr1-tabletop-tasks + Isaac-GR00T source + flash-attn +
# editable FluxVLA.
# RoboCasa simulation assets, datasets and checkpoints are NOT baked in;
# mount them at runtime (see docs/dlc_training.md).

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

# System packages for MuJoCo/robosuite/robocasa (headless EGL), ffmpeg and the
# CUDA extension build.
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

WORKDIR /opt/fluxvla

COPY requirements-base.txt requirements-sim.txt requirements-real.txt requirements-test.txt \
     setup.py setup.cfg pytest.ini README.md LICENSE ./
COPY fluxvla ./fluxvla
COPY scripts ./scripts
COPY configs ./configs
COPY tools ./tools

# Canonical environment installer, pinned by the repo itself:
#   sim-only       -> requirements-base + requirements-sim (robosuite/libero)
#   skip egl setup -> already created above
#   skip assets    -> 8.4 GB RoboCasa assets are mounted at runtime
#   skip robotwin  -> no cuRobo rebuild (not needed for RoboCasa)
#   skip project   -> editable install is done by the next command: the
#                     installer's own verify step does "import fluxvla", which
#                     needs a GPU at import time and cannot run in a build
#                     container.
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
    && PIP_CONSTRAINT=/tmp/pip-constraints.txt \
       bash scripts/install_env.sh sim-only --profile cu128 \
        --skip-egl-setup --skip-robocasa-assets --skip-robotwin --skip-project \
    && /opt/venv/bin/pip install --no-build-isolation -e /opt/fluxvla \
    && /opt/venv/bin/python -c "import importlib.util, torch, robosuite, flash_attn; assert importlib.util.find_spec('fluxvla') is not None; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'robosuite', robosuite.__version__, 'flash-attn', flash_attn.__version__)" \
    && /opt/venv/bin/python -c "import google.protobuf as p; from sentencepiece import sentencepiece_model_pb2 as m; print('protobuf', p.__version__, 'sentencepiece pb2 ok:', bool(m.ModelProto))"

# RoboCasa GR1 task repo is installed by the script; keep the asset mount point.
RUN mkdir -p /opt/fluxvla/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets

# Do not bake build-time proxies into the runtime image.
ENV http_proxy= https_proxy= no_proxy= HTTP_PROXY= HTTPS_PROXY=
ENV PYTHONPATH=/opt/fluxvla
# CUDA toolkit stays on PATH for runtime compilation of torch extensions.
ENV CUDA_HOME=/usr/local/cuda

# NOTE: "import fluxvla" needs a GPU at import time (the Eagle2 radio module
# calls torch.cuda.get_device_capability() at module scope). Verify with:
#   docker run --rm --gpus all IMAGE python -c "import fluxvla; print(fluxvla.__file__)"

CMD ["bash"]
