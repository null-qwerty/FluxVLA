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
"""Stage 2 of the serial recipe: RoboCasa 29D training with LoRA adapters.

Same serial stage-2 role as the frozen-VLM arm, but instead of a hard freeze
the VLM receives LoRA adapters: rank/alpha/dropout/targets follow
pi05_paligemma_libero_10_lora_finetune.py and modules_to_save keeps the
action/time heads trainable.

Freeze combination: the inherited freeze_llm_backbone / freeze_vision_backbone
/ freeze_llm_expert flags stay False. That is intentional: peft.get_peft_model
freezes every non-adapter weight itself, so those pre-injection flags only
describe the base model. The trainable set of this arm is exactly the LoRA
adapters plus modules_to_save (action_in_proj, action_out_proj, time_mlp_in,
time_mlp_out). If you want a hard freeze instead of adapters, use
pi05_paligemma_robocasa_ego_vlm_mixed_frozen_vlm.py; do not combine both
recipes.

LoRA injection lives in DDPTrainRunner (peft.get_peft_model);
FSDPTrainRunner does not inject adapters, so this arm replaces the inherited
FSDP runner with the DDP runner and drops the FSDP-only fields. Each GPU then
holds one full replica, which fits a 72-GB RTX PRO 5000 because only adapters
and modules_to_save carry optimizer state.

Stage 2 is VLA-only (train_task='vla'): the inherited ego vlm_dataloader and
mixed_training are removed. Initialize from stage 1 (ego VLM SFT) with

    export EGO_STAGE1_CHECKPOINT=\
      work_dirs/pi05_egodex_vlm_sft/checkpoints/latest-checkpoint.safetensors

FluxVLA checkpoints use plain model parameter names, so the OpenPI
name_mapping is disabled for that load; LoRA is injected after the weights are
loaded.
"""

import os

_base_ = './pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py'

# DDPTrainRunner rejects these FSDP-only fields and owns LoRA injection.
for _fsdp_only in ('fsdp_wrap_policy', 'sharding_strategy'):
    _base_.runner.pop(_fsdp_only, None)

# Serial stage 2 trains RoboCasa actions only: no ego language stream.
_base_.pop('vlm_dataloader', None)
_base_.runner.pop('mixed_training', None)

model = dict(
    use_lora=True,
    lora_rank=256,
    lora_alpha=512,
    lora_dropout=0.0,
    lora_target_modules=[
        # Attention projections (llm_backbone & llm_expert)
        'q_proj',
        'v_proj',
        'k_proj',
        'o_proj',
        # MLP layers (llm_backbone & llm_expert)
        'gate_proj',
        'up_proj',
        'down_proj',
        # Vision projector (image -> LLM embedding)
        'projector.projector',
        # Vision backbone attention (SigLIP)
        'out_proj',
        'fc1',
        'fc2',
    ],
    # PEFT freezes every non-adapter parameter; keep the action/time heads
    # trainable so the VLA flow-matching head can still adapt.
    modules_to_save=[
        'action_in_proj',
        'action_out_proj',
        'time_mlp_in',
        'time_mlp_out',
    ])

_stage1_checkpoint = os.environ.get('EGO_STAGE1_CHECKPOINT')
if _stage1_checkpoint:
    model.update(
        pretrained_name_or_path=_stage1_checkpoint,
        name_mapping=None,
        strict_mapping=True)

runner = dict(
    type='DDPTrainRunner',
    train_task='vla',
    # Keep the 40,000-update budget of the mixed arms for comparability.
    max_steps=40000,
    static_graph=True,
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_stage2_lora'))

eval = dict(output_dir='work_dirs/pi05_robocasa_stage2_lora')
