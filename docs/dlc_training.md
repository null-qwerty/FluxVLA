# 在阿里云 PAI-DLC 上启动 FluxVLA 训练（ego + RoboCasa GR1）

`scripts/dlc_train.sh` 是 DLC 自定义任务的唯一入口。它自己识别 DLC 注入的拓扑变量，
不需要在命令里写死卡数/节点数。

## 1. DLC 自定义任务怎么填

| 项 | 填法 |
|---|---|
| 镜像 | 用 FluxVLA 训练镜像（torch 2.8 + cu128 + flash-attn，见 `scripts/install_env.sh`） |
| 代码目录 | 挂载 CPFS：`/mnt/data/cpfs/users/neri/code/FluxVLA`（已含分支 `feat/ego-robocasa-pretrain`） |
| 数据目录 | 挂载 CPFS：`/mnt/data/cpfs/users/neri`（data/、checkpoints/、work_dirs/ 都在这里） |
| 启动命令 | `TASK=1 bash scripts/dlc_train.sh`（工作目录设为 `/mnt/data/cpfs/users/neri/code/FluxVLA`） |
| 节点数/卡数 | 按需填；脚本能识别 1~N 节点、每节点任意卡数 |

三个任务（`TASK` 环境变量）：

| TASK | 内容 | 配置文件 | 输出目录（`DATA_ROOT/work_dirs/`） |
|---|---|---|---|
| 1 | VLM backbone LoRA + action expert 全量，RoboCasa 24 任务微调 | `pi05_paligemma_robocasa_ego_vlm_mixed_lora.py` | `pi05_robocasa_lora_full_expert` |
| 2 | 先 ego VLM SFT，再同 TASK 1 | `pi05_paligemma_egodex_vlm_sft.py` → `..._lora.py` | `pi05_egodex_vlm_sft`、`pi05_robocasa_lora_from_ego` |
| 3 | 并行混训（RoboCasa VLA : ego VLM = 3:1） | `pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py` | `pi05_robocasa_ego_mixed` |

TASK 1 不需要 ego 数据；TASK 2/3 需要 `DATA_ROOT/egodex_vlm/ego.jsonl`（见第 2 节）。

## 2. 一次性准备（已经做完，换机器时照做）

```bash
export DATA=/mnt/data/cpfs/users/neri

# 2.1 base checkpoint（Pi0.5，约 13.8 GB）
mkdir -p $DATA/checkpoints
cp -r /mnt/data/cpfs/users/eira/checkpoints/pi05_base $DATA/checkpoints/pi05_base   # model.safetensors + tokenizer.model

# 2.2 ego.jsonl + 抽帧（TASK 2/3 才需要；16 进程约 20-40 分钟）
$DATA/envs/egoprep/bin/python $DATA/code/FluxVLA/tools/egodex_to_vlm_jsonl.py \
  --root $DATA/data/egodex --out $DATA/egodex_vlm \
  --fps-sample 2 --max-frames-per-episode 8 --workers 16
# --out 是目录：产物为 <out>/ego.jsonl 与 <out>/images/；--max-episodes N 是**每任务**上限

# 2.3 RoboCasa 仿真资产（评测用；训练不需要）
mkdir -p $DATA/code/FluxVLA/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets
cp -r $DATA/env/robocasa_gr1_assets/assets/* $DATA/code/FluxVLA/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets/
tar -xf $DATA/env/robocasa_gr1_assets/objects.tar -C $DATA/code/FluxVLA/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets/
```

## 3. 拓扑识别规则

按下列顺序读，第一个命中的生效（都可被同名环境变量直接覆盖）：

| 量 | 来源（按优先级） |
|---|---|
| 节点数 | `NNODES` → `WORLD_SIZE`（DLC 官方定义：分布式作业的节点总数）→ `PAI_WORKER_NUM` → `MLP_WORKER_NUM` → 1 |
| 节点序号 | `RANK` → `PAI_WORKER_INDEX` → `MLP_ROLE_INDEX` → `NODE_RANK` → 0 |
| 每节点卡数 | `NPROC_PER_NODE` → `MLP_WORKER_GPU` → `CUDA_VISIBLE_DEVICES` 列表长度 → `nvidia-smi -L` → 1 |
| master | `MASTER_ADDR`（rank0 缺失时取本机首个 IP）+ `MASTER_PORT`（默认 29500） |

识别结果会打印在日志里，并写入 `DATA_ROOT/work_dirs/dlc_env_TASK<n>_rank<m>.txt`
（含所有 `WORLD_SIZE/RANK/NPROC_PER_NODE/CUDA_VISIBLE_DEVICES/PAI_*/DLC_*`，便于事后核对）。

如果 `nvidia-smi` 报的卡数与识别值不一致，会打印 `WARN`——此时显式传 `NPROC_PER_NODE=<卡数>`。

## 4. 全局 batch 自适应

默认保持官方配方的 **global batch = 256**：

```text
micro_batch = PER_DEVICE_BS * 总卡数
grad_accum  = ceil(TARGET_GLOBAL_BS / micro_batch)
```

所以 8 卡时 `grad_accum=4`、16 卡时 2、32 卡时 1；卡数很少时会自动加大梯度累积（例如 1 卡 → 32），
训练步数与有效 batch 仍对齐 100k/256 的配方。可在 DLC 环境变量里改：

```bash
PER_DEVICE_BS=4 TARGET_GLOBAL_BS=256 MAX_STEPS=40000 NUM_WORKERS=8 bash scripts/dlc_train.sh
```

> 注意：`PER_DEVICE_BS=8` 是 80 GB 卡的配方。RTX PRO 5000 72 GB 通常也放得下（Pi0.5 3B + bf16 +
> gradient checkpointing），若 OOM 先降到 4 再试。

## 5. 冒烟 → 正式

启动脚本在 `torchrun` 之前会跑 `scripts/_preflight_config.py`（单进程），依次检查：

1. 配置能解析（带 lazy-import 回退）
2. 所有组件字典的键与其类签名一致（退出码 5）
3. `runner.max_steps` / `runner.max_epochs` 不同时设置（退出码 6）
4. 24 个 RoboCasa 数据根存在（退出码 2）
5. 配置引用的本地 checkpoint / tokenizer 路径存在（退出码 7）
6. tokenizer 目录完整（退出码 4）
7. `ego.jsonl` 等 VLM 标注文件存在（退出码 3）

任何一项不满足都在几秒内退出并打印缺失路径，不会等到 8/16 个 rank 全部起来才报错。


第一次跑先缩短步数验证环境与数据（约 5-10 分钟）：

```bash
TASK=1 MAX_STEPS=20 bash scripts/dlc_train.sh
# 或只打印命令不执行：
TASK=1 DRY_RUN=1 bash scripts/dlc_train.sh
```

确认 `DATA_ROOT/work_dirs/<run>/` 下出现 `config.json`、`dataset_statistics.json`、`checkpoints/` 后再全量跑。

## 6. 失败排查

| 现象 | 原因/处理 |
|---|---|
| `missing file: .../ego.jsonl` | TASK 2/3 需要先生成 ego.jsonl（第 2.2 节），或 `EGO_JSONL=` 指到已有文件 |
| `missing directory: .../robocasa_lerobot_V2.1` | 数据没挂上；用 `ROBOCASA_DATA_ROOT=` 指到实际路径 |
| `MASTER_ADDR is not injected for a multi-node job` | 多节点但没注入 master；手动 `MASTER_ADDR=<rank0 ip>` |
| `WARN: nvidia-smi reports N GPUs ...` | 显式 `NPROC_PER_NODE=<实际卡数>` |
| `AssertionError: Only one of max_steps or max_epochs can be set!` | stage1（ego SFT）继承的 LLaVA-665K 配方是 epoch 制（`max_epochs=1`），而启动器总是传 `max_steps`。已修复：stage1 配置改为 `max_epochs=None, max_steps=40000`；启动器同时传 `runner.max_epochs=None` 兜底；preflight 现在会检查两者不能同时为真（退出码 6） |
| `missing local checkpoint/tokenizer paths`（preflight 退出码 7） | stage1 从 LLaVA 配方继承了写死的 `./checkpoints/pi05_base`。已修复：stage1 的 `model.pretrained_name_or_path` / 数据 transform 的 `tokenizer_path` / TextVQA eval 的 `preprocessor.tokenizer_path` 都改为读 `PI05_CHECKPOINT` / `PI05_TOKENIZER`（启动器已导出）；preflight 会静态检查所有本地路径是否存在 |
| `Descriptors cannot be created directly` / `tiktoken is required` / `Could not extract SentencePiece model` | checkpoint 目录里只有 `tokenizer.model`，缺 `tokenizer.json`/`tokenizer_config.json`。transformers 5 会走 sentencepiece 慢转快路径，而镜像里 protobuf 4.25 + sentencepiece 0.1.99 的 pb2 不兼容；失败后回退 tiktoken 也没装。已修复：`checkpoints/pi05_base` 补齐 HF 完整 tokenizer 文件（tokenizer.json、tokenizer_config.json、special_tokens_map.json、added_tokens.json、config.json），`AutoTokenizer` 直接读 fast tokenizer。`scripts/_preflight_config.py` 现在也会检查该目录，缺失即在启动前报错 |
| `ConfigParsingError: ... "lazy_import" or non-"lazy_import"` | 配方里把 `_base_` 当对象使用（`_base_.pop(...)`），mmengine 会判定为 lazy-import，而继承链里又用了 `_base_ = "...py"` 普通写法，解析器直接报错。已修复：`scripts/train.py` / `scripts/eval.py` 现在先按默认方式加载，遇到该错误自动回退到 `lazy_import=False`；`scripts/_preflight_config.py` 在 torchrun 之前做同样的加载 + 数据路径检查，配置/数据问题会在几秒内报错而不是等到 8 个 rank 起来 |
| OOM | `PER_DEVICE_BS=4`（或 2），其余不变 |

## 7. 产物与评测

- 训练产物：`DATA_ROOT/work_dirs/<run>/{config.json, dataset_statistics.json, checkpoints/latest-checkpoint.safetensors}`
- 评测需要 ckpt 在 `<run>/checkpoints/` 下（runner 默认从 ckpt 的上两级目录读 `dataset_statistics.json`）
- 评测命令（在装好 robocasa 的机器上）：

```bash
CONFIG=configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py \
CKPT=/mnt/data/cpfs/users/neri/work_dirs/pi05_robocasa_lora_full_expert/checkpoints/latest-checkpoint.safetensors \
NUM_GPUS=8 bash scripts/eval_robocasa_manager.sh "$CONFIG" "$CKPT"
```

## 8. 三个任务的对照关系

- TASK 1 是「纯 RoboCasa 数据 + LoRA 微调」的基线（无 ego）。
- TASK 3 是「ego 与 RoboCasa 同时进 batch」的并行混训。
- TASK 2 是「ego 先改 VLM，再在 RoboCasa 上 LoRA」的串行方案。
- 三者步数预算一致（40k updates、global 256），可直接比较 24×50 的成功率；
  另建议补一条 `LLaVA-665K` 混训对照，用于区分「混训本身」与「ego 内容」的贡献。
## 9. 训练镜像

DLC 自定义任务请使用下面这个镜像（已构建并推送，2026-11-13）：

```text
limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2
```

- digest: `sha256:f30e8861e2a6777730f5e19cf644decfd769140396ec7b6d2b224a505144a9ef`
- 0.0.2 相比 0.0.1：protobuf 固定为 3.20.3（修 sentencepiece pb2 / tokenizer 慢转快失败）、
  内置 `scripts/_preflight_config.py` 与配置加载回退、LoRA 配置关闭 DDP `static_graph`
- 内容：ubuntu24.04 / Python 3.10.22 / torch 2.8.0+cu128 / robosuite 1.5.1 /
  flash-attn 2.8.3.post1 / mmengine 0.10.7 / fluxvla 可编辑安装于 `/opt/fluxvla`
- 未内置：RoboCasa 资产、数据集、checkpoint（运行时挂载 `/mnt/data/cpfs/users/neri`）
- 镜像约 24.9 GB（未压缩），首次在 DLC 拉取需要几分钟

镜像构建脚本：`docker/Dockerfile`、`docker/build_and_push.sh`；细节与坑见
`docker/README.md`。

冒烟（在 GPU 节点上，因为 `import fluxvla` 会在导入期查询 GPU 计算能力）：

```bash
docker run --rm --gpus all \
  limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.2 \
  python -c "import fluxvla; print(fluxvla.__file__)"
```

DLC 里镜像填上面的地址，启动命令仍是 `TASK=1 bash scripts/dlc_train.sh`
（工作目录 `/mnt/data/cpfs/users/neri/code/FluxVLA`）。

## 10. 监控（TensorBoard）

DLC 任务把指标写到 `<DATA_ROOT>/work_dirs/<run>/`：

- `*.jsonl`：每步一行（`jsonl` tracker，启动器默认开启）
- `tensorboard/<run_id>/`：原生 TensorBoard 事件（启动器现默认同时开 `tensorboard` tracker）

在**有 CPFS 挂载的 CPU 节点**（例如 data_process）上起一个 TensorBoard，用 SSH 端口转发看：

```bash
# 在 data_process 上（幂等，可重复执行；stop 可停掉）
bash /mnt/data/cpfs/users/neri/code/FluxVLA/tools/serve_tensorboard.sh
# 输出: TensorBoard: http://127.0.0.1:6006/

# 在你本地机器上
ssh -N -L 6006:127.0.0.1:6006 data_process
# 浏览器打开 http://localhost:6006
```

旧任务只有 `*.jsonl` 时，`tools/metrics_jsonl_to_tensorboard.py` 会把它增量转成
`<run>/tensorboard_jsonl/<run_id>/` 事件文件（`serve_tensorboard.sh` 已自动以 30s 轮询后台运行），
所以正在跑的任务也能实时看到曲线，无需重启训练；新任务则直接写原生事件。

## 11. 本地已验证 / 待 DLC 冒烟

在本机（1×RTX 3090 24 GB，与镜像同版本环境）用裁剪到 20 条 episode 的 RoboCasa 子集验证过：

| 路径 | 状态 |
|---|---|
| TASK 1（LoRA + 全量 action expert） | ✅ 全尺寸模型 2 步训练 + 保存 `step-*.safetensors`（bf16 参数配方） |
| ego VLM 数据链路（`ego.jsonl` → LLaVAFormatDataset → PaliGemma SFT 样本） | ✅ 20 条样本、13 个监督 token、`images` 3×224×224 |
| TASK 2 / TASK 3（FSDP 冻结臂、并行混训） | ⏳ 24 GB 显存放不下 3B 的 FSDP 副本，未在本机跑完；建议在 DLC 先跑 `MAX_STEPS=20` 冒烟 |

若 72 GB 卡仍 OOM（正常不会），可退到 0.0.2 里验证过的 bf16 参数配方：

```bash
TASK=1 MAX_STEPS=20 bash scripts/dlc_train.sh
# 仍 OOM 时追加：
# --cfg-options runner.keep_params_fp32=False model.preserve_fp32_residuals=False \
#   model.vision_backbone.openpi_stem_fp32=False model.openpi_fp32_flow=False
```
