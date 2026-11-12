# FluxVLA environment image

Image: `limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env`

## Tag convention

```text
<os>-<python>-<cuda>-<pytorch>-<version>
ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
```

The tag is derived from the built image itself (not hand-written), so it always
matches the real contents: OS from `/etc/os-release`, python minor version,
`cuXYZ` from `torch.version.cuda`, `pytorch<major>.<minor>` from
`torch.__version__`, and `VERSION` (default `0.0.1`).

## Build and push

```bash
# registry login once
docker login limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com

cd FluxVLA
bash docker/build_and_push.sh            # build + tag + push
PUSH=0 bash docker/build_and_push.sh     # build + tag only
VERSION=0.0.2 bash docker/build_and_push.sh
```

Build args: `PIP_INDEX_URL`, `HTTP_PROXY`/`HTTPS_PROXY` (passed through to the
build container, not baked into the runtime image), `PYTHON_VERSION` (3.10).

## What is inside

- `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- Python 3.10 venv at `/opt/venv` (on `PATH`), created with `uv`
- `bash scripts/install_env.sh sim-only --skip-egl-setup --skip-robocasa-assets --skip-robotwin --skip-flash-attn`
  - `requirements-base.txt` + `requirements-sim.txt` (torch 2.8 cu128, mujoco 3.2.6,
    robosuite/libero from the pinned forks, bddl, open3d, sapien, tensorflow 2.15, ...)
  - `robocasa-gr1-tabletop-tasks` and `Isaac-GR00T` source checkouts (pinned)
  - editable FluxVLA at `/opt/fluxvla`
- EGL vendor ICD for headless rendering, `MUJOCO_GL=egl`

Deliberately **not** baked in: RoboCasa simulation assets (8.4 GB), datasets,
checkpoints, flash-attn (only the optional DreamZero model imports it).
Mount them at runtime, e.g. the DLC training job mounts
`/mnt/data/cpfs/users/neri` and runs `scripts/dlc_train.sh`.

## Smoke test

```bash
docker run --rm --gpus all limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1 \
  python -c "import torch, robosuite, fluxvla; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
