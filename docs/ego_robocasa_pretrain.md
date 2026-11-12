# EgoDex 预训练 → RoboCasa 29D 后训练 / VLM 混训

本文档是 EgoDex 数据在 FluxVLA 中的两条完整实验路线：**串行两阶段**
（stage1 ego VLM SFT → stage2 RoboCasa 29D 后训练，冻结 VLM 或 LoRA）与
**并行 mixed**（RoboCasa VLA 与 ego VLM 每个 optimizer step 轮流，3:1 / 7:1）。
命令均在 FluxVLA 仓库根目录执行，配置位于 `configs/pi05/`，数据转换工具是
`tools/egodex_to_vlm_jsonl.py`。

## 0. 关键事实（先读）

- EgoDex **没有机器人关节动作、没有机器人 state、没有语言指令**。它只能进 VLM
  分支：mixed runner 每步只跑 `vla` 或 `vlm` 一路，VLM 批次不含
  state/action，VLA loss 只有 action flow MSE。**不要把 ego 行加进
  `train_dataloader`**。
- 因此 ego 的作用上限是"视觉-语言先验 / 正则"，不是"动作数据规模"。判读问题：
  相比 LLaVA-665K 混训（免费对照），ego 内容本身是否带来正收益；相比纯
  RoboCasa VLA，VLM 混训本身是否掉点。
- 评测口径不变：`gr1_unified/<task>_GR1ArmsAndWaistFourierHands_Env`，24 任务、
  每任务 50 trials，29D 关节（arm/waist 相对当前 state 的 delta，手 6D 为绝对
  离散命令）。
- Pi05 里 **`freeze_llm_expert` 冻结的是 action expert（policy head），不是
  VLM**。VLM 侧开关是 `freeze_llm_backbone`（PaliGemma 语言模型）和
  `freeze_vision_backbone`（SigLIP 视觉塔），两者与 action expert 相互独立。

## 1. 三种训练形态与 4 个配置

| 形态 | 配置 | 说明 |
| --- | --- | --- |
| 串行 stage1：ego VLM SFT | `pi05_paligemma_egodex_vlm_sft.py` | 纯 VLM SFT（train_task=vlm），ego.jsonl，1 epoch；产物给 stage2 当 init |
| 串行 stage2：冻结 VLM | `pi05_paligemma_robocasa_ego_vlm_mixed_frozen_vlm.py` | RoboCasa 24 任务 VLA-only；冻结 PaliGemma+SigLIP，只训 projector/action 投影/action expert |
| 串行 stage2：LoRA | `pi05_paligemma_robocasa_ego_vlm_mixed_lora.py` | RoboCasa 24 任务 VLA-only；VLM 上 LoRA(r256/alpha512)，modules_to_save 保 action head |
| 并行 mixed：VLA+VLM 3:1 / 7:1 | `pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py` | RoboCasa 24 任务 VLA + ego VLM 每步轮流，默认 3:1，可切 7:1 |

- 串行与并行的区别：串行是**先只学 VLM、再只学动作**；并行是**同一 run 内两种
  任务交替 optimizer step**，VLM 与动作互不拼接批次。
- 三个 `_base_` 覆盖配置都叠在既有配方上，只覆盖数据流 / run_dir /
  可训练参数集，机器人数据、16 步 action chunk、29D 统计量、24 任务评测全部继承。

### 1.1 freeze 语义（stage2 冻结 VLM 臂）

`PI05FlowMatching` / `BaseVLA.freeze_backbones()` 的确切含义：

| 配置项 | 冻结对象 |
| --- | --- |
| `model.freeze_llm_backbone=True` | PaliGemma 语言模型（VLM 的 LLM 侧） |
| `model.freeze_vision_backbone=True` | SigLIP 视觉塔 |
| `model.freeze_llm_expert=False` | 不冻结 action expert（policy head） |
| `model.freeze_projector=False`（默认） | 不冻结 vision->LLM projector 与 action/time 投影 |

stage2 冻结 VLM 臂就是上面四行。注意：VLM 主干被冻结后，语言批次上只剩
projector 可学，所以该臂必须是 **VLA-only**（`train_task='vla'`，配置已删除
继承来的 `vlm_dataloader` / `mixed_training`）；ego 数据只在 stage1 使用。
反过来若写 `freeze_llm_expert=True`，冻结的是 action expert，与 VLM 无关，
不构成"冻结 VLM 后训练"。

### 1.2 LoRA 语义（stage2 LoRA 臂）

- `use_lora=True / lora_rank=256 / lora_alpha=512 / lora_dropout=0.0`，
  target modules 照抄 `pi05_paligemma_libero_10_lora_finetune.py`；
  `modules_to_save=[action_in_proj, action_out_proj, time_mlp_in, time_mlp_out]`
  保证 flow-matching head 仍可训练。
- 继承来的 `freeze_llm_backbone / freeze_vision_backbone / freeze_llm_expert`
  保持 `False`：`peft.get_peft_model` 自己会冻结所有非 adapter 权重，这些
  预注入开关只描述 base 模型，本臂真正可训练集 = LoRA adapter + modules_to_save。
  想要硬冻结而不是 adapter，用冻结 VLM 臂；不要把两种配方叠加。
- LoRA 注入只在 `DDPTrainRunner`（`peft.get_peft_model`）；
  `FSDPTrainRunner` 不注入 adapter，所以 LoRA 配置显式切到 DDP runner 并删掉
  `fsdp_wrap_policy` / `sharding_strategy`。每卡一份完整模型，72GB 卡上因为
  只有 adapter + modules_to_save 有优化器状态，可以放下。

## 2. 数据准备（代码与数据分开）

代码：分支 feat/ego-robocasa-pretrain 同步到
`pro5000_4_dev:/mnt/data/oss/users/neri/code/FluxVLA`（保留 .git）。

RoboCasa 数据由数据同步任务负责，默认在
`/mnt/data/oss/users/neri/data/robocasa_lerobot_V2.1`：

    export ROBOCASA_DATA_ROOT=/mnt/data/oss/users/neri/data/robocasa_lerobot_V2.1

EgoDex 转换产物放 `/mnt/data/oss/users/neri/data/egodex_vlm`（不要写 raw 数据目录）：

    export EGO_VLM_DATA_ROOT=/mnt/data/oss/users/neri/data/egodex_vlm

### 2.1 先确认 HDF5 字段

EgoDex README 的字段约定是转换工具的契约（都以真实数据为准，先 `--inspect`）：

- `camera/intrinsic`：`(3,3)` 相机内参 `[[fx,0,cx],[0,fy,cy],[0,0,1]]`
  （也接受 `(T,3,3)` 逐帧栈）；工具只记录形状，不做去畸变。
- `transforms/camera`：`(T,4,4)` camera-to-world 变换，`T` 用来界定抽帧范围。
- `leftHand/transforms` / `rightHand/transforms`：`(T,4,4)` MANO 手腕位姿，
  无 confidence 时用于判断手是否在动。
- `leftHand/confidences` / `rightHand/confidences`：`(T,)` 手部跟踪置信度
  （0~1），用于 caption 里"哪只手在操作"的细节。

    python tools/egodex_to_vlm_jsonl.py --inspect \
      /mnt/data/oss/limx_mani_data/raw_data/EgoDex_test/test/pour/29.hdf5

缺失字段不会中断转换，只会减少 caption 细节。

### 2.2 转成 LLaVA 格式 ego.jsonl

    python tools/egodex_to_vlm_jsonl.py \
      --root /mnt/data/oss/limx_mani_data/raw_data/EgoDex_test \
      --out  /mnt/data/oss/users/neri/data/egodex_vlm \
      --fps-sample 1 --max-frames-per-episode 8 \
      --tasks 'pour,open_drawer,*pick*' --max-episodes 50 --workers 8

- 输入布局：`<root>/train/part*/<task>/<idx>.hdf5 + .mp4` 与
  `<root>/extra/<task>/<idx>.hdf5 + .mp4`；工具递归扫描，任何有同名 `.mp4`
  的 `.hdf5` 都会转换（`test/` / `val/` / 扁平目录同样适用）。
  任务名 = HDF5 父目录名，`--tasks` 支持 fnmatch 模式。
- 输出：`<out>/ego.jsonl`（每帧一条）、`<out>/images/<task>/*.jpg`、
  `<out>/convert_summary.json`。
- 每条记录是 LLaVA `from/value` 两轮对话：第一轮 `<image>\n<question>`，
  第二轮纯文本追问；Pi05 要求图像只出现在第一个 user turn，工具已满足。
- caption 默认由任务名模板 + 手部活动确定性生成；
  `--captioner vlm --captioner-module mymod:Captioner --captioner-model <path>`
  是预留钩子：模块提供 `captioner(image_paths, meta, task) -> [q1,a1,q2,a2]`，
  自带模型与显存预算，本仓库不内置。
- 抽帧默认 ffmpeg（一次调用按帧号 select），不可用时自动退回 imageio.v3；
  `--image-max-side 448` 把长边缩到 448，控制 OSS 容量。
- 性能量级：单 episode 一次 ffmpeg，8 workers 下 3k episodes 约 1~2 小时。

转换后抽查一条记录（真实 transform，含 tokenizer）：

    python - <<'PY'
    import json
    from fluxvla.transforms.vlm_transforms import BuildVLMMessages
    from fluxvla.transforms.paligemma_vlm_transforms import (
        ProcessPaliGemmaVLMSFTInputs)
    root = '/mnt/data/oss/users/neri/data/egodex_vlm'
    row = json.loads(open(root + '/ego.jsonl', encoding='utf-8').readline())
    data = BuildVLMMessages()(dict(row, data_root=root))
    sample = ProcessPaliGemmaVLMSFTInputs(
        tokenizer_path='./checkpoints/pi05_base', image_size=224,
        max_length=1024)(data)
    print(sample['input_ids'].shape, sample['images'].shape,
          int((sample['labels'] != -100).sum()))
    PY

## 3. 公共环境与 batch 对齐

    export ROBOCASA_DATA_ROOT=/mnt/data/oss/users/neri/data/robocasa_lerobot_V2.1
    export EGO_VLM_DATA_ROOT=/mnt/data/oss/users/neri/data/egodex_vlm
    # checkpoints/pi05_base 需已同步到代码目录（见 README）

配方默认按 2 卡 global batch 16 设计（per-device batch × accum × 卡数）。4 卡时
用下面的覆盖保持 global batch 16，否则各臂不可比：

    # mixed 臂（per-device 1，两个流一起改）
    MIXED_OPTS="runner.grad_accumulation_steps=4 \
    vlm_dataloader.grad_accumulation_steps=4"

    # stage1 VLM SFT（per-device 4 -> 2；accum 2）
    STAGE1_OPTS="train_dataloader.per_device_batch_size=2"

    # stage2 VLA（per-device 1，accum 4）
    STAGE2_OPTS="runner.grad_accumulation_steps=4"

mixed 的全量 resume 契约会记录 schedule/batch/accum/worker/可训练参数，冒烟和
正式跑必须一致；不一致时只能用权重初始化新 run，不能 resume。

## 4. 路线 A：串行两阶段（推荐先跑）

### 4.1 stage1：ego VLM SFT

    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_egodex_vlm_sft.py \
      work_dirs/pi05_egodex_vlm_sft --cfg-options $STAGE1_OPTS

- 默认 1 epoch（`runner.max_epochs=1`，步数由 ego.jsonl 行数推导）；想固定步数加
  `runner.max_epochs=None runner.max_steps=5000`。
- 产物在 `work_dirs/pi05_egodex_vlm_sft/checkpoints/`：`.pt` 是全量 resume
  状态，`.safetensors` 是纯权重，`latest-checkpoint.safetensors` 是软链。
  stage2 只用 `.safetensors`。
- 可选语言回归评测（TextVQA，需另外准备 val 标注，见 docs/vlm_data.md）：

    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_egodex_vlm_sft.py \
      work_dirs/pi05_egodex_vlm_sft --eval-after-train

- E0 冒烟建议：先转 ~5k 帧（如 30 个任务 × 20 episodes × 8 帧），stage1 跑
  1000 步，确认 VLM loss 下降、token_accuracy 上升。

### 4.2 stage2：RoboCasa 29D 后训练（冻结 VLM 或 LoRA）

stage2 用 stage1 的 `.safetensors` 初始化。Pi05 训练入口按
`model.pretrained_name_or_path` 加载权重；FluxVLA 格式 checkpoint 用的是
FluxVLA 参数名，因此配置在设置 `EGO_STAGE1_CHECKPOINT` 时自动关闭 OpenPI
`name_mapping` 并保持 `strict_mapping=True`：

    export EGO_STAGE1_CHECKPOINT=\
    work_dirs/pi05_egodex_vlm_sft/checkpoints/latest-checkpoint.safetensors

    # 冻结 VLM：只训 projector / action 投影 / action expert
    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_frozen_vlm.py \
      work_dirs/robocasa_stage2_frozen --cfg-options $STAGE2_OPTS

    # LoRA：VLM adapter + modules_to_save（DDP runner，每卡一份完整模型）
    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py \
      work_dirs/robocasa_stage2_lora --cfg-options $STAGE2_OPTS

- 不设 `EGO_STAGE1_CHECKPOINT` 时，stage2 回退到继承的 PI05 base
  checkpoint + OpenPI name_mapping，这就是"没有 ego 预训练的 stage2"对照臂。
  想显式换 init 也可 `--cfg-options model.pretrained_name_or_path=<path>
  model.name_mapping=None`。
- stage2 是 VLA-only（`train_task='vla'`，无语言流），默认 40k steps，
  24 任务数据，评测口径与并行臂一致。
- 冻结 VLM 臂的 trainable 列表应为：`projector.*`、`action_in_proj.*`、
  `action_out_proj.*`、`time_mlp_in/out.*`、`llm_expert.*`；
  `llm_backbone.*` 与 `vision_backbone.*` 不应出现。
- LoRA 臂启动日志会打印 peft 的 trainable 参数表，只应包含 adapter 与
  modules_to_save；保存时会把 adapter 合并回权重。

## 5. 路线 B：并行 mixed（VLA+VLM 同 run 交替）

3:1（默认 `('vla','vla','vla','vlm')`，40k 总步）：

    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py \
      work_dirs/ego_mixed_3to1 --cfg-options $MIXED_OPTS

7:1（只改 task_schedule，仍是 40k 总步：35k VLA + 5k VLM）：

    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py \
      work_dirs/ego_mixed_7to1 --cfg-options $MIXED_OPTS \
      "runner.mixed_training.task_schedule=('vla','vla','vla','vla','vla','vla','vla','vlm')" \
      runner.metric.run_dir=work_dirs/ego_mixed_7to1/metrics

E0 冒烟（1000 步，确认 VLM loss 下降且 VLA loss 与对照同量级）：

    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py \
      work_dirs/ego_mixed_smoke --cfg-options $MIXED_OPTS \
      runner.max_steps=1000 runner.save_iter_interval=500 \
      runner.metric.run_dir=work_dirs/ego_mixed_smoke/metrics

mixed 冒烟检查项：`train/vla/loss` 与 `train/vlm/loss` 同时出现（3:1 时每 4
步一条 VLM 记录），VLM loss 稳定下降、token_accuracy 上升；`train/vla/loss`
与 LLaVA 混训臂同 step 量级一致。

### 对照臂（必须同 4 卡、同 40k、同 global batch 16）

    # A. 纯 RoboCasa VLA（无语言流）
    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py \
      work_dirs/robocasa_vla_40k --cfg-options \
      train_dataloader.per_device_batch_size=1 \
      runner.grad_accumulation_steps=4 runner.max_steps=40000 \
      runner.save_iter_interval=4000 runner.max_keep_ckpts=3

    # B. LLaVA-665K 混训（"VLM 混训本身"的免费对照）
    NPROC_PER_NODE=4 bash scripts/train.sh \
      configs/pi05/pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py \
      work_dirs/robocasa_llava_mixed_40k --cfg-options $MIXED_OPTS

## 6. 24 × 50 官方评测

    NUM_GPUS=4 NUM_TRIALS_PER_TASK=50 \
    CONFIG=<训练该 checkpoint 的同一份配置> \
    CKPT=<run>/checkpoints/latest-checkpoint.safetensors \
    OUTPUT_DIR=<run>/eval_24x50 \
    bash scripts/eval_robocasa_manager.sh

24 个任务各 50 trials（每臂 1200 episodes）。配置必须与训练时一致（frozen /
lora 臂用各自配置名），因为 eval.task_list / unnorm_key / action_order 共享。
报告时分开写 Generalization 18 个 novel 任务与 Seen 6 个任务的平均，并给出相对
A、B 两臂的差值；不要只看 24 任务总平均。

## 7. 判读与风险

- 串行：stage2 冻 VLM vs LoRA 比的是"如何在不破坏 VLM 的前提下适配动作"；
  要判断 ego 预训练是否值得，必须和"stage2 直接由 pi05_base 初始化"对照。
- 并行：只有 ego mixed > LLaVA mixed 才说明 ego 内容本身有价值；ego mixed ≈
  LLaVA mixed 说明收益来自 VLM 混训；ego mixed 明显低于 LLaVA mixed（尤其
  novel 任务）说明域差在伤害策略，应减 ego 权重（`EGO_VLM_WEIGHT`，可用
  `EGO_LLAVA_WEIGHT` 增加第二源）或改走串行。
- VLM 源权重只影响 VLM 步内的数据重复次数，不改变 3:1/7:1 的步数配比。
- EgoDex 是真实第一人称视频、RoboCasa 是仿真渲染，外观域差是已知风险：
  先小预算冒烟、先看 novel 任务。
- 不要把 `datasets/.cache/vlm` 指向只读 OSS 目录；annotation index cache 需要
  可写目录。

## 8. 验证清单

    # 4 个配置可加载且关键字段正确（本机 .venv/FluxVLA，CPU 即可）
    .venv/FluxVLA/bin/python -m pytest \
      test/test_configs/test_ego_robocasa_pretrain_config.py -q

    # 转换工具语法与 CLI
    .venv/FluxVLA/bin/python -m py_compile tools/egodex_to_vlm_jsonl.py
    .venv/FluxVLA/bin/python tools/egodex_to_vlm_jsonl.py --help
