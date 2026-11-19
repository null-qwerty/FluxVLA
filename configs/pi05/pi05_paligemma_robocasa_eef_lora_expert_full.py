# Copyright 2026 Limx Dynamics
"""TASK 6: unified EEF RoboCasa post-training, VLM LoRA + FULL action expert.

The EEF-space counterpart of TASK 4 (pi05_paligemma_robocasa_vlm_lora_expert_full)
and the second half of the EEF comparison pair:

  TASK 1  joint space, VLM LoRA (adapters on the expert too), 40k steps
  TASK 4  joint space, VLM LoRA + fully trainable action expert, 100k steps
  TASK 5  EEF space,   same recipe as TASK 1
  TASK 6  EEF space,   same recipe as TASK 4   <- this config

The action space is the unified EEF one from docs/eef_action_space.md, produced
by tools/convert_robocasa_to_eef.py (set ROBOCASA_EEF_DATA_ROOT):

  state  : [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
  action : [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]

Everything except the action space and the expert trainability matches TASK 1 /
TASK 4: same 24 task roots, statistics, batch 256, transforms and eval key.

No stage-1 (EgoDex) initialisation: like TASK 1 this is a pure post-training arm
started from the PI0.5 base checkpoint, so the pair isolates the action space.
Export EGO_STAGE1_CHECKPOINT to start from an ego checkpoint instead.
"""

_base_ = './pi05_paligemma_robocasa_eef_lora.py'

model = dict(
    # Full (trainable + merged-on-save) copy of the action expert; VLM stays
    # LoRA. Same list as TASK 4.
    modules_to_save=[
        'llm_expert',
        'action_in_proj',
        'action_out_proj',
        'time_mlp_in',
        'time_mlp_out',
    ])

# Published PI0.5 full-finetune budget: 100k steps, lr 2.5e-5, warmup 5000.
runner = dict(
    type='DDPTrainRunner',
    train_task='vla',
    max_steps=100000,
    max_epochs=None,
    static_graph=False,
    optimizer=dict(lr=2.5e-05),
    lr_scheduler=dict(
        warmup_steps=5000, decay_steps=50000, min_lr=2.5e-06),
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_eef_lora_expert_full'))

eval = dict(output_dir='work_dirs/pi05_robocasa_eef_lora_expert_full')
