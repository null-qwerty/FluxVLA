# FluxVLA environment image

Image: `limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env`

## Published versions

| tag | digest | notes |
|---|---|---|
| `ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2` | `sha256:f30e8861e2a6777730f5e19cf644decfd769140396ec7b6d2b224a505144a9ef` | current: protobuf 3.20.3 (sentencepiece pb2 works), preflight + config fixes |
| `ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1` | `sha256:13edd7140be508e4c04d8de72bd5daead6de0cd1118b7f4b280b23635d5f13bf` | first build; protobuf 4.25.9 (tokenizer conversion broken) |

```text
limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2
```

## Tag convention

```text
<os>-<python>-<cuda>-<pytorch>-<version>
ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2
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
  limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2 \
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
5. Fixed in 0.0.2: 0.0.1 shipped protobuf 4.25.9 + sentencepiece 0.1.99, so
   `import sentencepiece.sentencepiece_model_pb2` raised `Descriptors cannot be
   created directly` and the transformers 5 slow-tokenizer converter failed for
   a checkpoint dir with only `tokenizer.model`. 0.0.2 pins `protobuf==3.20.3`
   (TF 2.15 still allows <5) - verified that `sentencepiece_model_pb2`, TF 2.15,
   `GemmaTokenizer` from the full HF files, and even a dir with only
   `tokenizer.model` all load.
6. Build network: pip/apt use the Aliyun mirrors directly (NO_PROXY) while
   GitHub/uv go through the build-time HTTP proxy. Proxies are cleared from
   the runtime image.
7. 0.0.2 was post-processed with `docker commit` after the build to pick up
   config/script fixes that landed while the 45-minute pip layer was being
   built (`static_graph=False` for the LoRA arm, the broader config-load
   fallback, the component-key preflight). The `fluxvla` package, CUDA
   extension and the protobuf pin all come from the Dockerfile build;
   `bash docker/build_and_push.sh` on this commit reproduces the same
   contents from scratch.
