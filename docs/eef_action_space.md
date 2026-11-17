# 方案 B：统一 EEF 动作空间（TASK 2 替换）

## 目标

- ego 阶段 1 与 robocasa 阶段 2 使用同一套机械臂动作表示：
  **输入（state）= 绝对 EEF，输出（action）= delta EEF**。
- 阶段 1：ego 数据全量微调（VLM + action expert）；阶段 2：robocasa 数据微调，VLM 用 LoRA。
- 手部 / 腰部沿用 robot 关节空间，ego 样本只监督手臂维度（per-dim loss mask）。

## 动作/状态布局（flat vector）

```text
state  (29): [L_eef_abs(7: pos3+quat4), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
action (29): [L_eef_delta(6: dpos3+drotvec3), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]
mask   (29): ego 样本 = [1]*6 + [0]*6 + [1]*6 + [0]*6 + [0]*3；robot 样本 = 全 1
```

`delta_mask` 与现有 `robocasa-joint-delta` 一致：手臂/腰部存绝对目标 → 训练用 delta；
手部是离散绝对命令 → 保持绝对。新增 profile：`robocasa-eef-delta`。

## 坐标系

- 统一到 **头戴相机系（ego_view）**：ego 数据天然是相机系；robocasa 的 `ego_view` 也是头相机，
  sim 里可直接取相机位姿完成 base↔camera 变换。
- 阶段 2 部署时：预测 delta EEF（相机系）→ 旋到基座系 → 累加成绝对 EEF 目标 → robosuite IK → 关节目标。

## 数据转换

1. `robot → EEF`（离线，精确 FK）：
   - 绝对关节目标（arms/waist）+ FK → 绝对 EEF；相邻帧差分 → delta EEF。
   - 相机系：用 sim 中 ego_view 相机位姿（由 qpos 前向计算）。
   - 手部/腰部原样保留。
2. `ego → EEF`（mocap，精确标签）：
   - `transforms/{left,right}Hand` 4×4 → 相机系位姿；差分 → delta EEF。
   - 逐帧 `confidences/*` 过滤低置信帧。
   - 30 fps → 20 Hz 重采样，与 robot 对齐。
3. 统计量：用新 profile 重新计算（mean/std/quantile），或从 robot+ego 混合集统计。

## 需要的代码改动（image 已改为纯环境镜像，改动仅需同步 CPFS 代码）

| 文件 | 改动 |
|---|---|
| `fluxvla/datasets/utils/transformed_statistics.py` | 新增 `robocasa-eef-delta` profile（state/action key + delta_mask 29） |
| `fluxvla/models/vlas/pi0_flowmatching.py` / `pi05_flowmatching.py` | flow-matching loss 支持 per-dim `action_mask`（ego 样本屏蔽手部/腰部） |
| `fluxvla/transforms/robocasa_transforms.py` | 新增 `RobocasaEEFToJointAction`（部署适配：相机系 delta → 基座系 → IK → 29D） |
| `fluxvla/engines/runners/robocasa_eval_runner.py` | 适配器接入点（`action_keys` 之后） |
| `configs/pi05/pi05_paligemma_egodex_eef_full_finetune.py` | 新阶段 1 配置（ego，全量微调，29D EEF 布局，ego mask） |
| `configs/pi05/pi05_paligemma_robocasa_eef_lora.py` | 新阶段 2 配置（robocasa，VLM LoRA，EEF 布局） |
| `tools/convert_robocasa_to_eef.py` | robot 数据 → EEF parquet 转换 |
| `tools/convert_egodex_to_eef_jsonl.py` | ego hdf5 → EEF 样本（jsonl 索引 + 图像帧） |
| `tools/validate_eef_roundtrip.py` | 往返误差 / 适配器上界验证 |
| `scripts/dlc_train.sh` | TASK 2 改为新两阶段（ego 全量 → robocasa LoRA） |

## 验证门槛（先跑，再训练）

1. **Gate 1**：robot 数据 `joint → EEF → IK → joint` 往返误差分布（逐关节 / 位姿）。
2. **Gate 2**：ego `transforms` 与统计量的合理性（腕部速度、delta 分布、置信度覆盖率）。
3. **Gate 3**：1–2 个任务 × 20 episodes 的 oracle-EEF 冒烟（用真值 EEF 走适配器执行）。

## 训练对照

- baseline：现有 29D joint（TASK 1 lora / frozen）
- 对照 A：EEF 空间，仅 robocasa 数据
- 实验 B：EEF 空间，ego 全量 → robocasa LoRA（本方案）

## 关键风险

- delta EEF 的旋转用 rotvec，跨 ±π 需连续性处理（相邻帧最短路径）。
- 相机系依赖 sim 中 ego_view 相机位姿；实机需头部位姿。
- ego 无手部标签：阶段 1 手部维度 loss 屏蔽，阶段 2 才学习手部。
