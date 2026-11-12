# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stage 2 of the serial recipe: RoboCasa 29D training with a frozen VLM.

This is the frozen-VLM post-training arm: the PaliGemma language backbone and
the SigLIP vision tower keep their stage-1 weights and only the
vision-to-LLM projector, the action/time projections and the Pi05 action
expert are trained on the converted RoboCasa 24-task dataset.

Freeze semantics (see PI05FlowMatching docstring and BaseVLA.freeze_backbones):

* model.freeze_llm_backbone=True     -> PaliGemma language model frozen.
* model.freeze_vision_backbone=True  -> SigLIP vision tower frozen.
* model.freeze_llm_expert=False      -> action expert (policy head) TRAINABLE.
* model.freeze_projector default False -> the vision->LLM projector and the
  action_in/action_out/time projections remain trainable.

freeze_llm_expert is NOT a VLM switch: it controls the action expert only.
With both VLM backbones frozen, a language batch could only update the
projector, so this arm is VLA-only (train_task='vla'); it has no language
stream and the inherited ego vlm_dataloader/mixed_training are removed. Stage 1
(pi05_paligemma_egodex_vlm_sft.py) is where ego data is used.

Initialize from the stage-1 checkpoint instead of the OpenPI base:

    export EGO_STAGE1_CHECKPOINT=\
      work_dirs/pi05_egodex_vlm_sft/checkpoints/latest-checkpoint.safetensors

A FluxVLA checkpoint already uses FluxVLA parameter names, so the OpenPI
name_mapping is disabled for that load (strict_mapping stays on). Without the
environment variable the recipe falls back to the inherited PI05 base
checkpoint and OpenPI mapping.
"""

import os

_base_ = './pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py'

model = dict(
    freeze_llm_backbone=True,
    freeze_vision_backbone=True,
    freeze_llm_expert=False,
    # Explicit: the vision->LLM projector and action/time projections must
    # keep training; a frozen action projector would break the flow head.
    freeze_projector=False)

_stage1_checkpoint = os.environ.get('EGO_STAGE1_CHECKPOINT')
if _stage1_checkpoint:
    model.update(
        pretrained_name_or_path=_stage1_checkpoint,
        name_mapping=None,
        strict_mapping=True)

# Serial stage 2 trains RoboCasa actions only: no ego language stream and no
# mixed task schedule (a VLM update would have no trainable parameters).
_base_.pop('vlm_dataloader', None)
_base_.runner.pop('mixed_training', None)

runner = dict(
    type='FSDPTrainRunner',
    train_task='vla',
    # Keep the 40,000-update budget of the mixed arms for comparability.
    max_steps=40000,
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_stage2_frozen_vlm'))

eval = dict(output_dir='work_dirs/pi05_robocasa_stage2_frozen_vlm')
