# FluxVLA environment image

Image: `limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env`

Built and pushed version:

```text
limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
```

## Tag convention

```text
<os>-<python>-<cuda>-<pytorch>-<version>
ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
```

The tag is derived from the built image itself, not hand-written: OS from
`/etc/os-release`, python minor version, `cuXYZ` from `torch.version.cuda`,
`pytorch<major>.<minor>` from `torch.__version__`, and `VERSION` (default
`0.0.1`). `bash docker/build_and_push.sh` does build -> derive tag -> tag ->
push -> verify.

## Contents

| item | value |
|---|---|
| base | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| OS | ubuntu24.04 |
| python | 3.10.22 (`/opt/venv`, on PATH) |
| torch | 2.8.0+cu128 (CUDA runtime 12.8) |
| robosuite | 1.5.1 (pinned fork `yinchimaoliang/robosuite@e293cc3`) |
| flash-attn | 2.8.3.post1 (prebuilt wheel) |
| mmengine | 0.10.7 |
| fluxvla | editable install at `/opt/fluxvla` |
| image size | 24.9 GB |

Installed with `bash scripts/install_env.sh sim-only --profile cu128`, plus
`--skip-egl-setup` (the EGL ICD is created in the Dockerfile),
`--skip-robocasa-assets` (assets are mounted at runtime), `--skip-robotwin`,
and `--skip-project` (the editable install is done explicitly because the
installer verify step does `import fluxvla`, which needs a GPU). RoboCasa GR1
source and Isaac-GR00T are checked out by the installer.

Not baked in: RoboCasa simulation assets (8.4 GB), datasets, checkpoints.

## Runtime usage (DLC / GPU node)

```bash
docker run --rm --gpus all \
  limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1 \
  python -c "import fluxvla; print(fluxvla.__file__)"
```

On DLC, mount the CPFS tree `/mnt/data/cpfs/users/neri` and run
`TASK=1 bash scripts/dlc_train.sh` as described in `docs/dlc_training.md`.

## Gotchas found while building

1. `import fluxvla` requires a GPU at import time.
   `fluxvla/models/third_party_models/eagle2_hg_model/radio_model.py:182` calls
   `torch.cuda.get_device_capability()` at module scope, so a GPU-less import
   (including a build-time smoke test) dies with `Found no NVIDIA driver`.
   Verify the image on a GPU node, not during build.
2. The editable install compiles a CUDA kernel
   (`fluxvla/ops/cuda/gemma_rotary_embedding`) through
   `torch.utils.cpp_extension`. A GPU-less build must set `FORCE_CUDA=1` and
   keep `/usr/local/cuda/bin` on PATH, otherwise `cuda.h` is not found.
   `TORCH_CUDA_ARCH_LIST="8.0 9.0 12.0"` covers A800/H100 and the RTX PRO
   5000 (Blackwell, sm_120).
3. flash-attn is required: the eagle2 radio module imports it
   unconditionally, so `--skip-flash-attn` yields an image where
   `import fluxvla` fails. The prebuilt wheel comes through gh-proxy; GitHub
   release URLs are rate-limited (HTTP 429) and need a retry.
4. robosuite prints a warning that the `mink` whole-body IK plugin is missing.
   That only affects the example `WHOLE_BODY_MINK_IK` controller; the core
   `WHOLE_BODY_IK` referenced by `default_gr1.json` is present.
5. Known latent issue (not fixed in the published 0.0.1 image): protobuf 4.25.9 +
   sentencepiece 0.1.99. Importing `sentencepiece.sentencepiece_model_pb2`
   raises `Descriptors cannot be created directly`, which breaks the
   transformers 5 slow-tokenizer converter when a checkpoint dir contains only
   `tokenizer.model`. The RoboCasa workaround is to ship the full HF tokenizer
   files (`tokenizer.json` etc.) in the checkpoint dir; the next image build
   should also pin `protobuf==3.20.3` (TF 2.15 allows <5) or
   `sentencepiece>=0.2.0`.
6. Build network: pip/apt use the Aliyun mirrors directly (NO_PROXY) while
   GitHub/uv go through the build-time HTTP proxy. Proxies are cleared from
   the runtime image.
