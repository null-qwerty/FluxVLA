# Copyright 2026 Limx Dynamics
"""TASK 4: VLM-backbone LoRA + full action expert, full RoboCasa budget.

Why this exists: TASK 1 (pi05_paligemma_robocasa_ego_vlm_mixed_lora) puts LoRA
adapters on the action expert too (PEFT matches leaf names such as q_proj
everywhere), so only ~14% of the expert is trainable. The published PI0.5
RoboCasa baseline (51.42%, pi05_paligemma_robocasa_full_data_full_finetune_bs256)
trains every parameter for 100k steps with lr 2.5e-5 / warmup 5000.

This arm keeps the VLM backbone as LoRA but trains the action expert fully
(PEFT modules_to_save wraps the whole llm_expert, i.e. a fully trainable copy),
and matches the published step/LR schedule. Everything else (data, transforms,
normalisation statistics, batch 256, eval protocol) is identical to TASK 1.

Overhead: the expert copy adds ~0.58B parameters; 12.3B/24.1% of the 4.76B
trainable set, which fits a 72-GB RTX PRO 5000 with DDP.
"""

_base_ = './pi05_paligemma_robocasa_ego_vlm_mixed_lora.py'

model = dict(
    # Full (trainable+merged-on-save) copy of the action expert; VLM stays LoRA.
    modules_to_save=[
        'llm_expert',
        'action_in_proj',
        'action_out_proj',
        'time_mlp_in',
        'time_mlp_out',
    ])

# Match the published full-finetune recipe (100k steps, lr 2.5e-5,
# warmup 5000, cosine decay over 50k steps).
runner = dict(
    max_steps=100000,
    optimizer=dict(lr=2.5e-05),
    lr_scheduler=dict(
        warmup_steps=5000, decay_steps=50000, min_lr=2.5e-06))

eval = dict(output_dir='work_dirs/pi05_robocasa_lora_vlm_expert_full')
