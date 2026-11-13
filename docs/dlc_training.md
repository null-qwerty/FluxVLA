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

启动脚本在 `torchrun` 之前会跑 `scripts/_preflight_config.py`（单进程）：
解析配置、检查 24 个 RoboCasa 数据根、检查 ego.jsonl。任何一项不满足会立刻退出并打印缺失路径，
不会等到 8/16 个 rank 全部起来才报错。


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

DLC 自定义任务请使用下面这个镜像（已构建并推送，2026-11-12）：

```text
limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1
```

- digest: `sha256:13edd7140be508e4c04d8de72bd5daead6de0cd1118b7f4b280b23635d5f13bf`
- 内容：ubuntu24.04 / Python 3.10.22 / torch 2.8.0+cu128 / robosuite 1.5.1 /
  flash-attn 2.8.3.post1 / mmengine 0.10.7 / fluxvla 可编辑安装于 `/opt/fluxvla`
- 未内置：RoboCasa 资产、数据集、checkpoint（运行时挂载 `/mnt/data/cpfs/users/neri`）
- 镜像约 24.9 GB（未压缩），首次在 DLC 拉取需要几分钟

镜像构建脚本：`docker/Dockerfile`、`docker/build_and_push.sh`；细节与坑见
`docker/README.md`。

冒烟（在 GPU 节点上，因为 `import fluxvla` 会在导入期查询 GPU 计算能力）：

```bash
docker run --rm --gpus all \
  limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.1 \
  python -c "import fluxvla; print(fluxvla.__file__)"
```

DLC 里镜像填上面的地址，启动命令仍是 `TASK=1 bash scripts/dlc_train.sh`
（工作目录 `/mnt/data/cpfs/users/neri/code/FluxVLA`）。
