# FluxVLA environment image

Image: `limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env`

**This image contains the environment only - no FluxVLA project code.** The
python package is installed at job start from the repo mounted on CPFS
(`scripts/install_project.sh`: builds/caches a per-(commit, CUDA arch) wheel,
~1 min cold, seconds warm), so code changes never require an image rebuild.

## Published versions

| tag | notes |
|---|---|
| `ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.3` | env-only (no repo code); the launcher installs the project wheel. See digest in `docs/dlc_training.md` |
| `ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2` | first fix release: protobuf 3.20.3, project code frozen at 2026-11-13 |
| `ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1` | first build; protobuf 4.25.9 (tokenizer conversion broken) |

## Tag convention

```text
<os>-<python>-<cuda>-<pytorch>-<version>
ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.3
```

The tag is derived from the built image itself, not hand-written: OS from
`/etc/os-release`, python minor version, `cuXYZ` from `torch.version.cuda`,
`pytorch<major>.<minor>` from `torch.__version__`, and `VERSION` (default
`0.0.3`). `bash docker/build_and_push.sh` does build -> derive tag -> tag ->
push -> verify.

## Contents

| item | value |
|---|---|
| base | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| OS | ubuntu24.04 |
| python | 3.10.22 (`/opt/venv`, on PATH) |
| torch | 2.8.0+cu128 (CUDA runtime 12.8) |
| robosuite | 1.5.1 (pinned fork `yinchimaoliang/robosuite@e293cc3`) |
| robocasa / Isaac-GR00T | source checkouts under `/opt/fluxvla/src` (assets mounted at runtime) |
| flash-attn | 2.8.3.post1 (prebuilt wheel) |
| mmengine | 0.10.7 |
| protobuf | 3.20.3 (sentencepiece pb2 compatible) |
| image size | ~24.9 GB |

Installed with `bash scripts/install_env.sh sim-only --profile cu128`, plus
`--skip-egl-setup` (the EGL ICD is created in the Dockerfile),
`--skip-robocasa-assets` (assets are mounted at runtime), `--skip-robotwin`,
and `--skip-project` (the project wheel is installed by the launcher).

Not baked in: FluxVLA project code, RoboCasa simulation assets (8.4 GB),
datasets, checkpoints.

## Runtime usage (DLC / GPU node)

```bash
docker run --rm --gpus all \
  limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.3 \
  python -c "import torch, robosuite, flash_attn; print(torch.__version__)"
```

On DLC, mount the CPFS tree `/mnt/data/cpfs/users/neri`, run
`TASK=1 bash scripts/dlc_train.sh` from `/mnt/data/cpfs/users/neri/code/FluxVLA`
as described in `docs/dlc_training.md`. The launcher installs the project
wheel for the node GPU before starting `torchrun`.

## Gotchas found while building

1. `import fluxvla` requires a GPU at import time
   (`fluxvla/models/third_party_models/eagle2_hg_model/radio_model.py` calls
   `torch.cuda.get_device_capability()` at module scope), so the project wheel
   can only be import-checked on a GPU node. The wheel itself builds fine on
   CPU with `FORCE_CUDA=1`.
2. The project wheel compiles three CUDA extensions
   (`fluxvla/ops/cuda/{gemma_rotary_embedding,rotary_pos_embedding,matmul_bias}`)
   for the local GPU arch; `scripts/install_project.sh` sets
   `TORCH_CUDA_ARCH_LIST` from `torch.cuda.get_device_capability()`. A cold
   build measured ~60 s inside this image.
3. flash-attn is required: the eagle2 radio module imports it unconditionally.
   The prebuilt wheel comes through gh-proxy; GitHub release URLs are
   rate-limited (HTTP 429) and need a retry.
4. robosuite warns that the `mink` whole-body IK plugin is missing. That only
   affects the example `WHOLE_BODY_MINK_IK` controller; the core
   `WHOLE_BODY_IK` referenced by `default_gr1.json` is present.
5. protobuf 4.25.9 + sentencepiece 0.1.99 broke the transformers 5
   slow-tokenizer converter in 0.0.1; 0.0.2+ pin `protobuf==3.20.3`.
6. Do not add `# syntax=docker/dockerfile:1`: the daemon then fetches the
   external Dockerfile frontend from docker.io, which fails behind the proxy.
   The built-in BuildKit frontend supports `RUN --mount=type=cache`.
7. Build network: pip/apt use the Aliyun mirrors directly (NO_PROXY) while
   GitHub/uv/astral go through the build-time HTTP proxy. Proxies are cleared
   from the runtime image.
