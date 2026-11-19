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

任务列表（`TASK` 环境变量）：

| TASK | 动作空间 | 内容 | 配置文件 | 输出目录（`DATA_ROOT/work_dirs/`） |
|---|---|---|---|---|
| 1 | 关节（29D） | VLM LoRA（适配器也盖在 action expert 上），40k 步 | `pi05_paligemma_robocasa_ego_vlm_mixed_lora.py` | `pi05_robocasa_lora_full_expert` |
| 2 | **EEF** | 先 ego EEF 全量微调，再用 EEF RoboCasa 数据做 VLM LoRA | `pi05_paligemma_egodex_eef_full_finetune.py` → `..._eef_lora.py` | `pi05_egodex_eef_sft`、`pi05_robocasa_eef_lora_from_ego` |
| 3 | 关节 | 并行混训（RoboCasa VLA : ego VLM = 3:1） | `pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py` | `pi05_robocasa_ego_mixed` |
| 4 | 关节 | VLM LoRA + **action expert 全量**，100k 步、lr 2.5e-5 / warmup 5000（对齐官方 51.42% 配方） | `pi05_paligemma_robocasa_vlm_lora_expert_full.py` | `pi05_robocasa_lora_vlm_expert_full` |
| 5 | **EEF** | TASK 1 的配方（VLM LoRA 含 expert），40k 步 —— EEF vs 关节对照 | `pi05_paligemma_robocasa_eef_lora.py` | `pi05_robocasa_eef_lora_from_base` |
| 6 | **EEF** | TASK 4 的配方（VLM LoRA + expert 全量），100k 步 —— EEF vs 关节对照 | `pi05_paligemma_robocasa_eef_lora_expert_full.py` | `pi05_robocasa_eef_lora_expert_full` |
| 7 | 关节 | 旧串行方案（ego VLM SFT → RoboCasa 关节 LoRA），仅作对照 | `pi05_paligemma_egodex_vlm_sft.py` → `..._lora.py` | `pi05_egodex_vlm_sft`、`pi05_robocasa_lora_from_ego` |

TASK 1/4/5/6 不需要 ego 数据（都从 PI0.5 base 直接后训练，5/6 会忽略
`EGO_STAGE1_CHECKPOINT`，保证只比较动作空间）；TASK 3/7 需要
`DATA_ROOT/egodex_vlm/ego.jsonl`（见第 2 节）；TASK 2/5/6 需要 EEF 转换后的
数据（见第 2.3 节：`egodex_eef_v1` 与 `data/robocasa_eef_V0.1`）。

## 2. 一次性准备（已经做完，换机器时照做）

```bash
export DATA=/mnt/data/cpfs/users/neri

# 2.1 base checkpoint（Pi0.5，约 13.8 GB）
mkdir -p $DATA/checkpoints
cp -r /mnt/data/cpfs/users/eira/checkpoints/pi05_base $DATA/checkpoints/pi05_base   # model.safetensors + tokenizer.model

# 2.3 统一 EEF 数据（TASK 2 才需要）
# robot 侧：24 个任务目录 -> ROBOCASA_EEF_DATA_ROOT（parquet 约 4GB，videos 用 symlink）
#   python tools/convert_robocasa_to_eef.py \
#     --src $DATA_ROOT/data/robocasa_lerobot_V2.1 \
#     --dst $DATA_ROOT/data/robocasa_eef_V0.1 --workers 16 \
#     --stats-out $DATA_ROOT/data/robocasa_eef_V0.1/stats.json
# ego 侧：EgoDex hdf5 -> EGO_EEF_ROOT（jsonl + 抽帧图像，20Hz）
#   python tools/convert_egodex_to_eef.py --root $DATA_ROOT/data/egodex \
#     --out $DATA_ROOT/egodex_eef --fps-sample 20 --max-episodes 20 \
#     --stats-out $DATA_ROOT/egodex_eef/stats.json

# 2.2 ego.jsonl + 抽帧（TASK 3/7 才需要；16 进程约 20-40 分钟）
# 生成后在同一台机器（数据机）上写 manifest，供每节点 preflight 校验数据一致性：
# python tools/write_annotation_manifest.py <out>/ego.jsonl \
#                                               <out>/ego_train.jsonl <out>/ego_val.jsonl
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

随后 `scripts/_preflight_index.py` 再做两件事（每节点一次，退出码 8）：

8. 用 `<annotation>.manifest.json` 里记录的 sha256/size 校验节点看到的是**最新**副本
   （CPFS 元数据视图偶尔会短暂停留在旧版本，见排错表）
9. 用数据集同一份代码预热 JSONL 的 byte-offset 索引，避免 16 个 rank 在训练启动时
   并发建索引；索引缓存目录由 `VLM_INDEX_CACHE_DIR` 指定（启动器默认设为节点本地
   `/tmp/fluxvla_vlm_index`）

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
| `missing file: .../ego.jsonl` | TASK 3/7 需要先生成 ego.jsonl（第 2.2 节），或 `EGO_JSONL=` 指到已有文件 |
| `missing directory: .../robocasa_lerobot_V2.1` | 数据没挂上；用 `ROBOCASA_DATA_ROOT=` 指到实际路径 |
| `MASTER_ADDR is not injected for a multi-node job` | 多节点但没注入 master；手动 `MASTER_ADDR=<rank0 ip>` |
| `WARN: nvidia-smi reports N GPUs ...` | 显式 `NPROC_PER_NODE=<实际卡数>` |
| `RuntimeError: Annotation changed during indexing.` | `ego.jsonl` 的 byte-offset 索引按 `(path, size, mtime_ns, ctime_ns)` 指纹缓存，CPFS 在跨节点的元数据缓存里偶尔短暂停留在旧版本，导致建索引过程中指纹变化。已修复：启动器把 `VLM_INDEX_CACHE_DIR` 指到节点本地 `/tmp`，并在 `torchrun` 之前用 `scripts/_preflight_index.py` 预热索引；同时用 `ego.jsonl.manifest.json`（sha256+size）校验节点看到的是最新副本，不一致会重试 6 次×5s 后以退出码 8 明确报错。重新生成 manifest（在写数据的机器上执行）：`python tools/write_annotation_manifest.py /mnt/data/cpfs/users/neri/egodex_vlm/ego*.jsonl` |
| `AssertionError: Only one of max_steps or max_epochs can be set!` | stage1（ego SFT）继承的 LLaVA-665K 配方是 epoch 制（`max_epochs=1`），而启动器总是传 `max_steps`。已修复：stage1 配置改为 `max_epochs=None, max_steps=40000`；启动器同时传 `runner.max_epochs=None` 兜底；preflight 现在会检查两者不能同时为真（退出码 6） |
| `missing local checkpoint/tokenizer paths`（preflight 退出码 7） | stage1 从 LLaVA 配方继承了写死的 `./checkpoints/pi05_base`。已修复：stage1 的 `model.pretrained_name_or_path` / 数据 transform 的 `tokenizer_path` / TextVQA eval 的 `preprocessor.tokenizer_path` 都改为读 `PI05_CHECKPOINT` / `PI05_TOKENIZER`（启动器已导出）；preflight 会静态检查所有本地路径是否存在 |
| `Descriptors cannot be created directly` / `tiktoken is required` / `Could not extract SentencePiece model` | checkpoint 目录里只有 `tokenizer.model`，缺 `tokenizer.json`/`tokenizer_config.json`。transformers 5 会走 sentencepiece 慢转快路径，而镜像里 protobuf 4.25 + sentencepiece 0.1.99 的 pb2 不兼容；失败后回退 tiktoken 也没装。已修复：`checkpoints/pi05_base` 补齐 HF 完整 tokenizer 文件（tokenizer.json、tokenizer_config.json、special_tokens_map.json、added_tokens.json、config.json），`AutoTokenizer` 直接读 fast tokenizer。`scripts/_preflight_config.py` 现在也会检查该目录，缺失即在启动前报错 |
| `ConfigParsingError: ... "lazy_import" or non-"lazy_import"` | 配方里把 `_base_` 当对象使用（`_base_.pop(...)`），mmengine 会判定为 lazy-import，而继承链里又用了 `_base_ = "...py"` 普通写法，解析器直接报错。已修复：`scripts/train.py` / `scripts/eval.py` 现在先按默认方式加载，遇到该错误自动回退到 `lazy_import=False`；`scripts/_preflight_config.py` 在 torchrun 之前做同样的加载 + 数据路径检查，配置/数据问题会在几秒内报错而不是等到 8 个 rank 起来 |
| OOM | `PER_DEVICE_BS=4`（或 2），其余不变 |

## 7. 产物与 RoboCasa 评测

训练产物：`DATA_ROOT/work_dirs/<run>/{config.json, dataset_statistics.json, checkpoints/*.safetensors}`。
评测必须拿到 `<run>/checkpoints/` 下的 checkpoint：runner 会从 ckpt 的上两级目录（即 `<run>/`）
读 `dataset_statistics.json` 做反归一化；ckpt 里的权重是**合并后的普通 state_dict**（不是 PEFT 适配器），
所以 LoRA 臂和混训臂用同一套评测流程。

### 7.1 一键启动（DLC GPU 节点，镜像内）

```bash
cd /mnt/data/cpfs/users/neri/code/FluxVLA

# 先冒烟：1 个任务 x 1 trial x 30 步，约 5-8 分钟（含 14 GB ckpt 加载）
EVAL_TASK=1 SMOKE=1 bash scripts/dlc_eval.sh

# 全量：24 任务 x 50 trials x 720 步
EVAL_TASK=1 bash scripts/dlc_eval.sh        # TASK 1：RoboCasa-only LoRA
EVAL_TASK=3 bash scripts/dlc_eval.sh        # TASK 3：ego + RoboCasa 并行混训
EVAL_TASK=4 bash scripts/dlc_eval.sh        # TASK 4：LoRA + expert 全量
EVAL_TASK=7 bash scripts/dlc_eval.sh        # TASK 7：旧串行（ego SFT -> 关节 LoRA）

# EEF 动作空间的三条臂（2/5/6）经由部署适配层评测：
# 观测 29D 关节 -> FK -> 相机系 EEF；动作 EEF delta -> 施加 -> DLS IK -> 29D 关节目标。
# 上机前先离线校验适配层（几秒钟）：
#   python tools/validate_eef_eval_adapter.py --parquet \
#     $DATA_ROOT/data/robocasa_eef_V0.1/PnPBottleToCabinetClose/data/chunk-000/episode_000000.parquet
EVAL_TASK=2 bash scripts/dlc_eval.sh        # TASK 2：ego EEF SFT -> RoboCasa EEF LoRA
EVAL_TASK=5 bash scripts/dlc_eval.sh        # TASK 5：EEF LoRA（40k）
EVAL_TASK=6 bash scripts/dlc_eval.sh        # TASK 6：EEF LoRA + expert 全量（100k）
```

`EVAL_TASK` 只是默认 ckpt/config 的映射；也可以显式指定：

```bash
CONFIG=configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py \
CKPT=$DATA_ROOT/work_dirs/pi05_robocasa_lora_full_expert/checkpoints/step-040000-epoch-000-loss=0.0275.safetensors \
bash scripts/dlc_eval.sh
```

常用覆盖项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `TASK_IDS` | 全部 24 个 | 只跑子集，如 `TASK_IDS="0 1 2"` |
| `NUM_TRIALS_PER_TASK` | 50 | 每个任务 trial 数（对齐榜单） |
| `NUM_GPUS` | 可见卡数 | 每卡一个 eval worker |
| `MAX_TASKS_PER_GPU` | 1 | 单卡并发任务数 |
| `MAX_EPISODE_STEPS` | 720 | 冒烟时调小 |
| `SAVE_VIDEO` | False | 存 rollout 视频比较占盘 |
| `OUTPUT_DIR` | `$DATA_ROOT/work_dirs/eval_runs/<run>_<ts>` | 结果目录 |
| `SMOKE` | 0 | 1 = 1 任务/1 trial/30 步 |
| `DRY_RUN` | 0 | 1 = 只打印命令与预检 |

`scripts/dlc_eval.sh` 会先做 4 项预检（`scripts/_preflight_eval.py`）：配置能解析且有
`RobocasaEvalRunner` 的 24 任务列表、ckpt 存在、`dataset_statistics.json` 存在、RoboCasa 资产齐全；
任一项不满足都会在几秒内报错退出。

### 7.2 资产与渲染环境

- 镜像里 `robocasa` 包的资产目录是 `/opt/fluxvla/src/robocasa-gr1-tabletop-tasks/robocasa/models/assets`，
  但镜像只带了 `fixtures/arenas/scenes` 的空壳；8.4 GB 物体资产在
  `$DATA_ROOT/env/robocasa_gr1_assets/assets`（`objects/{objaverse,sketchfab,lightwheel}`、`textures`、`fixtures` 等）。
- `scripts/dlc_eval.sh` 会自动把该目录软链到镜像内的资产路径（容器内 root，可直接改），
  无需在 DLC 里额外配置挂载；`ROBOCASA_ASSETS_DIR` 可覆盖来源。
- 渲染走 EGL：脚本导出 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`；
  `generative_textures/` 只在 `generative_textures="100p"` 纹理随机化时用到，GR1 评测 `gym.make()` 不传该参数，缺失不影响评测。

### 7.3 结果在哪看

```text
$DATA_ROOT/work_dirs/eval_runs/<run>_<ts>/
├── manager.log                    # 汇总进度（每个 task 的成功数/总数）
├── summary.csv                    # 最终 per-task / per-group / overall 成功率
├── task_logs/task<id>_gpu<g>.log  # 单个任务的完整日志
├── robocasa/gpu<g>_task<id>_results.json
└── task_status/, failed_tasks.txt, manager_config.yaml
```

- 每个 task 的 json/csv 里 `successes/total_episodes` 即该任务成功率；`summary.csv` 末行是 overall。
- manager 结束时打印 `summary: <OUTPUT_DIR>/summary.csv`；失败任务写进 `failed_tasks.txt`。
- 参考：PI0.5（官方 51.42%）。24×50 全量在 8 卡约 1.5-3 小时（每卡同时 1 个任务）。

### 7.4 常见问题

| 现象 | 原因/处理 |
|---|---|
| `checkpoint not found` | 确认 `<run>/checkpoints/latest-checkpoint.safetensors` 存在；中途评测可显式传某个 `step-*.safetensors` |
| `missing dataset statistics` | ckpt 必须放在 `<run>/checkpoints/` 下，且 `<run>/dataset_statistics.json` 存在（训练脚本自动写） |
| `incomplete RoboCasa assets` | 资产没同步/路径不对；用 `ROBOCASA_ASSETS_DIR=` 指定，或先跑 `scripts/download_robocasa_assets.py` |
| 某个 task 失败 | 看 `task_logs/task<id>_gpu<g>.log`；常见是显存/渲染（EGL）或该任务资产缺失 |
| 渲染报 EGL/driver 错误 | 确认节点有 GPU 且镜像有 EGL ICD；不要设 `MUJOCO_GL=osmesa` |
| 想跑更快 | 只跑子集 `TASK_IDS`，或减少 `NUM_TRIALS_PER_TASK`（报告里注明） |

## 8. 任务的对照关系与「分数偏低」排查

- TASK 1 是「纯 RoboCasa 数据 + LoRA 微调」的基线（无 ego，40k 步）。
- TASK 3 是「ego 与 RoboCasa 同时进 batch」的并行混训。
- TASK 2 是**统一 EEF 动作空间**的串行方案（见 `docs/eef_action_space.md`）：
  ego 数据（相机系绝对 EEF + delta EEF）全量微调 → RoboCasa 数据 VLM LoRA；
  手/腰部维度保留 robot 关节空间，ego 样本用逐维 action mask 只监督手臂。
- TASK 5/6 是 EEF 空间的两个对照臂：TASK 5 = TASK 1 配方（VLM LoRA 含 expert，40k），
  TASK 6 = TASK 4 配方（VLM LoRA + expert 全量，100k）。三者共享同一份 EEF RoboCasa 数据、
  统计量、batch 256 与评测协议，只有动作空间不同，因此
  「1 vs 5」「4 vs 6」是干净的 EEF-vs-关节 对照。
- TASK 7 是旧的关节空间串行方案（ego VLM SFT → RoboCasa 关节 LoRA），保留作对照。
- TASK 4 是 TASK 1 的修正版：VLM 仍用 LoRA，但 **action expert 全量训练**，步数/LR/预热
  对齐官方发布配方（`pi05_paligemma_robocasa_full_data_full_finetune_bs256`，24×50 评测 51.42%）。
- 三者（1/2/3）步数预算一致（40k updates、global 256），可直接比较 24×50 的成功率；
  另建议补一条 `LLaVA-665K` 混训对照，用于区分「混训本身」与「ego 内容」的贡献。

### TASK 1 实测评测 32.42% 时该看什么

24×50 实测：Cabinet 34 / Drawer 34 / Microwave 28 / Generalization 32.56 / overall 32.42（389/1200）
（结果目录 `work_dirs/eval_runs/<run>_<ts>/`）。与官方 51.42% 的差异来自**训练配方**，
不是评测或数据链路：

| 项 | 官方发布 run（51.42%） | TASK 1（32.42%） |
|---|---|---|
| 可训练参数 | 全量 4.18B | LoRA 适配器 + 4 个投影头，0.57B（13.5%） |
| action expert | 全量训练 | 只有 LoRA 适配器（专家内部仅 13.8% 可训练） |
| 步数 | 100000（约 2 epoch） | 40000（约 0.8 epoch） |
| lr / warmup / decay | 2.5e-5 / 5000 / 50000 | 2e-5 / 1200 / 40000（继承自 LLaVA 配方） |
| 数据 / 变换 / 归一化统计 / batch=256 / 评测协议 | 相同 | 相同 |

排查顺序建议：

1. **确认评测链路**：用同一套 `scripts/dlc_eval.sh` 评测官方发布的 51.42% checkpoint
   （`checkpoints/baseline_pi05_robocasa_bs256/checkpoints/step-100000-...safetensors`，
   已按官方仓库结构下载到 CPFS），应复现 ~51%。若复现不了，再查评测协议而不是训练。
2. **TASK 4**（改配方重训）是最直接的修复：expert 全量 + 100k 步 + 官方 lr。
3. 只想单独验证「步数」假设：`TASK=1 MAX_STEPS=100000 bash scripts/dlc_train.sh`
   （lr 调度仍按 40k 衰减到 min_lr，可近似看作数据量对照）。

### stage 1（ego VLM SFT）的预算与留出集

`ego.jsonl` 只有 4111 条样本、118 种唯一答案、4 种 prompt，按 global batch 256 算约
17 step/epoch。实测 loss 在 ~140 step（约 8 epoch）就到 token accuracy 0.999、
~235 step 后 <1e-3，之后完全是记忆训练集，跑满 40k step ≈ 2300 epoch 只会过拟合并可能
损伤基座 VLM。因此：

- 启动器默认 `STAGE1_MAX_STEPS=300`（≈17 epoch）、`STAGE1_SAVE_ITER=100`（能留下早期权重，
  `save_iter_interval` 默认 10000 会一个中间 checkpoint 都不存）；
- `tools/split_egodex_vlm_jsonl.py` 可按 episode 或 task 划分留出集：

  ```bash
  python tools/split_egodex_vlm_jsonl.py \
      --input /mnt/data/cpfs/users/neri/egodex_vlm/ego.jsonl \
      --val-fraction 0.15 --group-by episode   # 或 --group-by task 整任务留出
  ```

  生成 `ego_train.jsonl` / `ego_val.jsonl` / `split_manifest.json`；启动器检测到
  `ego_train.jsonl` 会自动用它跑 SFT（也可用 `EGO_VLM_JSONL=` 显式指定）；
- 判断「学没学到东西」要用留出集指标（后续可用 `ego_val.jsonl` 离线算 loss/token acc），
  不能再拿 train loss 当依据。

## 9. 训练镜像

DLC 自定义任务请使用下面这个镜像（**纯环境镜像，不含项目代码**）：

```text
limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.3
```

- digest: `sha256:deddc65758274c05636614e5c9039d9d75792c5b5a04d9890f45ced76d23f7ad`
- 0.0.3 起镜像只装环境：torch / robosuite / robocasa / Isaac-GR00T / flash-attn /
  mmengine / protobuf 3.20.3 等；**不包含** FluxVLA 的 `fluxvla/`、`configs/`、`scripts/`、
  `tools/`，也不做 editable 安装
- 项目代码在任务启动时从挂载的 CPFS 仓库安装：`scripts/install_project.sh`
  按「git commit + 本机 CUDA 架构」构建/复用 wheel（首次约 1 分钟，之后命中缓存只需几秒），
  因此**改代码不需要重建镜像**
- 内容：ubuntu24.04 / Python 3.10.22 / torch 2.8.0+cu128 / robosuite 1.5.1 /
  flash-attn 2.8.3.post1 / mmengine 0.10.7 / protobuf 3.20.3
- 未内置：项目代码、RoboCasa 资产、数据集、checkpoint（运行时挂载 `/mnt/data/cpfs/users/neri`）
- 镜像约 24.9 GB（未压缩），首次在 DLC 拉取需要几分钟

wheel 缓存目录：`$DATA_ROOT/env/fluxvla_wheels/fluxvla-<commit>-cu<arch>.whl`
（可用 `FLUXVLA_WHEEL_DIR` 覆盖；`SKIP_PROJECT_INSTALL=1` 可跳过安装，供本地开发用）。

镜像构建脚本：`docker/Dockerfile`、`docker/build_and_push.sh`；细节与坑见
`docker/README.md`。

冒烟（无需 GPU，验证环境本体）：

```bash
docker run --rm \
  limx-shanghai-registry.cn-shanghai.cr.aliyuncs.com/limx_emb/fluxvla_env:ubuntu24.04-py3.10-cu128-pytorch2.8-0.0.3 \
  python -c "import torch, robosuite, flash_attn; print(torch.__version__, robosuite.__version__)"
```

DLC 里镜像填上面的地址，启动命令仍是 `TASK=1 bash scripts/dlc_train.sh`
（工作目录 `/mnt/data/cpfs/users/neri/code/FluxVLA`）。

## 10. 本地已验证 / 待 DLC 冒烟

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
## 11. TASK 2：统一 EEF 动作空间（ego 全量微调 → RoboCasa LoRA）

动作空间（`docs/eef_action_space.md`）：

```text
state  (29): [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]   # 输入绝对 EEF（相机系）
action (29): [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)] # 输出 delta EEF
```

ego 只有腕部标签，因此阶段 1 用逐维 `action_masks` 只监督 12 个手臂维度；
手部/腰部在阶段 2 由 RoboCasa 数据监督。

### 11.1 数据准备（已在本仓库同一台 CPFS 上完成）

```text
$DATA_ROOT/data/robocasa_eef_V0.1/     # 24 任务转换后 parquet（videos 软链到原始 robocasa_lerobot_V2.1）
$DATA_ROOT/egodex_eef_v1/             # ego_eef.jsonl + images/<task>.tar + stats.json
```

重新生成：

```bash
# robot 侧（CPU + mujoco，无需 RoboCasa 资产）
python tools/convert_robocasa_to_eef.py \
  --src $DATA_ROOT/data/robocasa_lerobot_V2.1 \
  --dst $DATA_ROOT/data/robocasa_eef_V0.1 --workers 16 \
  --stats-out $DATA_ROOT/data/robocasa_eef_V0.1/stats.json

# ego 侧（hdf5 + mp4 抽帧，建议写到节点本地再同步回 CPFS）
python tools/convert_egodex_to_eef.py --root $DATA_ROOT/data/egodex \
  --out /tmp/egodex_eef --fps-sample 20 --max-episodes 6 --image-max-side 256 \
  --workers 16 --stats-out /tmp/egodex_eef/stats.json
rsync -a /tmp/egodex_eef/ $DATA_ROOT/egodex_eef/
```

### 11.2 DLC 启动

```bash
# 冒烟（20 步，约 10 分钟）
TASK=2 MAX_STEPS=20 STAGE1_MAX_STEPS=20 STAGE1_SAVE_ITER=20 bash scripts/dlc_train.sh

# 正式训练：阶段 1 ego 20k 步全量微调 -> 阶段 2 RoboCasa 40k 步 LoRA
TASK=2 bash scripts/dlc_train.sh
```

可选覆盖：`PER_DEVICE_BS`、`TARGET_GLOBAL_BS`、`STAGE1_MAX_STEPS`、`MAX_STEPS`、
`EGO_EEF_ROOT`/`EGO_EEF_JSONL`/`EGO_EEF_STATS`、`ROBOCASA_EEF_DATA_ROOT`/`ROBOCASA_EEF_STATS`。

### 11.3 跨机房数据同步（pro5000 ↔ 算力中心）

算力中心 DLC 挂载的是上海 CPFS `/mnt/data/cpfs/users/neri`，本仓库的转换结果是
**直接写 CPFS**（本地/pro5000 转换 → rsync 到 data_process 的 CPFS 挂载点）。
若需要在 pro5000 上处理再同步：

```bash
# pro5000 上转换（有自己的 CPFS/数据），完成后推到 OSS
ossutil cp -r /mnt/data/cpfs/mnt/data/<user>/robocasa_eef_V0.1 oss://<bucket>/users/neri/data/robocasa_eef_V0.1

# data_process（上海）从 OSS 拉到 CPFS，DLC 即可直接读
ossutil cp -r oss://<bucket>/users/neri/data/robocasa_eef_V0.1 /mnt/data/cpfs/users/neri/data/robocasa_eef_V0.1
```

### 11.4 已验证 / 待验证

在本机 3090（24 GB）+ 真实数据切片上逐段验证过（脚本无需 GPU 集群即可复现）：

| 项 | 结果 |
|---|---|
| Gate1：joint → EEF → DLS IK → joint | MAE 0.011–0.013 rad（≈0.6–0.75°），位姿残差 ~1e-7 |
| stage1 数据集 / 8 个 transform / collator | states (32,)、actions (16,32)、action_masks (16,32)、images (3,224,224)、lang 200；batch (2,32)/(2,16,32) |
| stage1 前向 + flow-matching loss（真实 ego 切片） | loss 可算；mask 生效：监督目标数 768 vs 2048 = 12/32 |
| stage2 数据集（parquet + mp4 解码）/ collator | action_masks (16,) 时间掩码，batch (2,16,32) |
| stage1 → stage2 交接（`EGO_STAGE1_CHECKPOINT`） | stage2 以 `name_mapping=None, strict_mapping=True` 完整加载；可训练参数 566.8M（13.5%） |
| stage2 前向 + loss（LoRA） | loss 1.84（单卡 24 GB 只够前向） |
| 评测适配器（EEF→关节 注入 robocasa_eval_runner） | ⏳ 下一步（IK/坐标变换已验证，仅剩管线接入） |

未在本机验证的部分只有「反向 + 优化器步」：3B 全量微调与 DDP LoRA 都要 >24 GB 显存。
这两条路径此前已在 DLC 上跑通过（`work_dirs/pi05_robocasa_lora_full_expert`、
`pi05_robocasa_ego_mixed` 有完整 checkpoint），本次改动的数据/配置/EEF 空间已由上面逐段覆盖。

### 11.5 TASK 8：把 ego 数据混进 RoboCasa 的每一个 batch

TASK 2 的串行做法（ego 全量微调 20k 步 -> RoboCasa LoRA）被证明零迁移：
TASK 5（无 ego 阶段）在 2k/8k/19k/20k 步的 loss 与 TASK 2 完全相同
（0.312/0.200/0.176/0.173 vs 0.312/0.204/0.179/0.175）。因此新增 TASK 8，
改为**批内混训**：

* 一个 dataloader、两个数据源：24 任务 RoboCasa EEF parquet + EgoDex EEF 帧；
* 每个源用自己的统计量做 mean/std 归一化（样本自带 stats）；
* RoboCasa 样本监督全部维度（时间 mask 展开成逐维），EgoDex 样本只监督 12 个
  手臂维（逐维 mask）；
* 采样权重由 EGO_MIX_WEIGHT 控制（默认 0.25，即 3:1）。

```bash
# 正式：stage1 ego 只训动作头（VLM 冻结，4k 步）-> stage2 混训 40k 步
TASK=8 bash scripts/dlc_train.sh
TASK=8 EGO_MIX_WEIGHT=0.25 MAX_STEPS=40000 bash scripts/dlc_train.sh

# 冒烟
TASK=8 DRY_RUN=1 bash scripts/dlc_train.sh
TASK=8 MAX_STEPS=20 STAGE1_MAX_STEPS=20 bash scripts/dlc_train.sh

# 评测（与 TASK 2/5/6 同一套 EEF 部署适配器）
EVAL_TASK=8 NUM_TRIALS_PER_TASK=10 bash scripts/dlc_eval.sh
```

实现：fluxvla/datasets/eef_mixed_dataset.py（加权交错）、
fluxvla/transforms/normalize.py::ExpandActionMasks、
configs/pi05/pi05_paligemma_robocasa_eef_mixed_vla.py、
configs/pi05/pi05_paligemma_egodex_eef_expert_only.py。

### 11.6 检查点开销

全量微调每个 checkpoint = 41 GB（`.pt`，含优化器状态）+ 14.5 GB（`.safetensors`）。
默认 `SAVE_ITER=1000`、`MAX_KEEP_CKPTS=10`：阶段 1 每 1000 步写 ~55 GB，保留 10 个 ≈ 550 GB。
显存/带宽/空间紧张时可这样调：

```bash
TASK=2 STAGE1_SAVE_ITER=2000 SAVE_ITER=2000 MAX_KEEP_CKPTS=4 bash scripts/dlc_train.sh
```

另外 stage1 的 20,000 步 ≈ 26 个 epoch（192,478 帧 / 全局 batch 256 ≈ 752 步/epoch），
全量微调容易过拟合；想要更保守可 `STAGE1_MAX_STEPS=8000`（≈10 epoch），
配合 `STAGE1_SAVE_ITER=1000` 保留早期权重。
