# Copyright 2026 Limx Dynamics
"""Stage 1 variant: EgoDex EEF, action expert only (VLM frozen).

TASK 2's stage 1 fine-tuned the whole 3B backbone for 26 epochs on ego data and
then transferred nothing to RoboCasa (TASK 5, which starts from the base model,
matched its loss at 2k/8k/19k/20k steps). This variant keeps the base model's
visual and language priors intact and lets ego data teach only what it is
actually informative about: the EEF arm-delta mapping in the action expert
(plus the projector / in-out heads, which stay trainable).

Use it as a drop-in replacement for
configs/pi05/pi05_paligemma_egodex_eef_full_finetune.py, with a much shorter
budget (a few thousand steps).
"""
_base_ = './pi05_paligemma_egodex_eef_full_finetune.py'

model = dict(
    freeze_llm_backbone=True,
    freeze_vision_backbone=True,
)

runner = dict(
    max_steps=4000,
    metric=dict(run_dir='work_dirs/pi05_egodex_eef_expert_only'))
