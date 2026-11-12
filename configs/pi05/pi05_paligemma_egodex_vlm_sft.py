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
"""Stage 1 of the serial recipe: pure EgoDex VLM SFT.

EgoDex contributes only first-person frames plus generated captions, so the
serial recipe starts here: adapt the Pi05 PaliGemma VLM on ego.jsonl with the
plain PI05 VLM SFT objective (one language task per optimizer step, no robot
actions). This is the same pure-VLM form as
pi05_paligemma_llava665k_full_finetune.py; only the annotation source, run
directory and TextVQA evaluation output change.

Stage 2 (RoboCasa 29D post-training, frozen VLM or LoRA) then initializes its
model from this stage's checkpoint, e.g.

    export EGO_STAGE1_CHECKPOINT=\
      work_dirs/pi05_egodex_vlm_sft/checkpoints/latest-checkpoint.safetensors

and runs pi05_paligemma_robocasa_ego_vlm_mixed_frozen_vlm.py or
pi05_paligemma_robocasa_ego_vlm_mixed_lora.py. FluxVLA safetensors use plain
model parameter names, so those recipes disable the OpenPI name_mapping when
EGO_STAGE1_CHECKPOINT is set.

Prepare ego.jsonl with:

    python tools/egodex_to_vlm_jsonl.py \
        --root /mnt/data/oss/limx_mani_data/raw_data/EgoDex_test \
        --out ./datasets/egodex_vlm --fps-sample 1

The default is repository-relative; on a shared mount set EGO_VLM_DATA_ROOT
instead of editing this file.
"""

import os

_base_ = './pi05_paligemma_llava665k_full_finetune.py'

ego_root = os.environ.get('EGO_VLM_DATA_ROOT', './datasets/egodex_vlm')
ego_jsonl = ego_root + '/ego.jsonl'
ego_weight = float(os.environ.get('EGO_VLM_WEIGHT', '1.0'))

# Same dataset/transform/collator contract as the LLaVA-665K VLM SFT recipe:
# one image + a two-turn text conversation per row, no robot state or action.
train_dataloader = dict(
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=[dict(path=ego_jsonl, weight=ego_weight)],
        data_root_path=ego_root,
        index_cache_dir='./datasets/.cache/vlm',
        # Fail fast on a missing extracted frame instead of silently
        # resampling another EgoDex frame.
        max_retry_times=0,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessPaliGemmaVLMSFTInputs',
                image_size=224,
                max_length=1024,
                tokenizer_path='./checkpoints/pi05_base'),
        ]))

runner = dict(
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_egodex_vlm_sft'))

# TextVQA stays as a held-out language check; it is only executed with
# --eval-after-train or scripts/eval.sh. Keep its artifacts inside this run.
eval = dict(
    checkpoint_dir='./work_dirs/pi05_egodex_vlm_sft/checkpoints',
    run_dir='./work_dirs/pi05_egodex_vlm_sft/textvqa')
