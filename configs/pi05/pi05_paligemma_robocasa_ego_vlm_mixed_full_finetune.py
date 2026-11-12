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
"""Pi05 RoboCasa 29D action training + EgoDex egocentric VLM SFT.

Same mixed contract as the LLaVA-665K recipe (3 VLA updates : 1 VLM update,
40,000 updates, global batch 16), but the language stream is converted from
EgoDex episodes instead of LLaVA-665K. EgoDex contributes no robot state or
action, so it only enters the VLM branch: a VLM batch has no states/actions
and the VLA loss remains action flow MSE.

Prepare the language stream with:

    python tools/egodex_to_vlm_jsonl.py \
        --root /mnt/data/oss/limx_mani_data/raw_data/EgoDex_test \
        --out ./datasets/egodex_vlm --fps-sample 1

The recipe keeps repository-relative paths. On a shared mount set
EGO_VLM_DATA_ROOT (and, if needed, ROBOCASA_DATA_ROOT) instead of editing this
file, e.g. EGO_VLM_DATA_ROOT=/mnt/data/oss/users/neri/data/egodex_vlm.

Optional anti-forgetting arm: set EGO_LLAVA_WEIGHT > 0 to append raw
LLaVA-665K as a second weighted source of the same VLM loader (weight is a
repeat factor, not a sampling probability).

Robot evaluation is the inherited 24-task RoboCasa protocol (24 x 50
trials), not a language benchmark.
"""

import os

_base_ = './pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py'

# EgoDex -> LLaVA-format JSONL (see tools/egodex_to_vlm_jsonl.py). Relative by
# default so the recipe stays portable; override on shared/OSS mounts.
ego_root = os.environ.get('EGO_VLM_DATA_ROOT', './datasets/egodex_vlm')
ego_jsonl = ego_root + '/ego.jsonl'
llava_root = './datasets/llava665k'
llava_jsonl = llava_root + '/llava_v1_5_mix665k.json'

# Source weights are repeat factors. Keep one ego source by default so this
# arm measures the ego content itself; the LLaVA-665K recipe is the separate
# "VLM mixing" control arm.
ego_weight = float(os.environ.get('EGO_VLM_WEIGHT', '1.0'))
llava_weight = float(os.environ.get('EGO_LLAVA_WEIGHT', '0.0'))

_vlm_sources = [dict(path=ego_jsonl, weight=ego_weight)]
_vlm_roots = [ego_root]
if llava_weight > 0:
    _vlm_sources.append(dict(path=llava_jsonl, weight=llava_weight))
    _vlm_roots.append(llava_root)

# The robot stream, action/state statistics, 24-task evaluation and model
# architecture are inherited unchanged from the mixed LLaVA recipe.
vlm_dataloader = dict(
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=_vlm_sources,
        data_root_path=_vlm_roots,
        index_cache_dir='./datasets/.cache/vlm',
        # Fail fast on a missing extracted frame instead of silently
        # resampling a different EgoDex frame.
        max_retry_times=0,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessPaliGemmaVLMSFTInputs',
                image_size=224,
                max_length=1024,
                tokenizer_path=_base_.runner.tokenizer.model_path),
        ]))

runner = dict(
    train_task='mixed',
    mixed_training=dict(
        task_schedule=('vla', 'vla', 'vla', 'vlm'),
        sync_each_microbatch=True,
        loss_weights=dict(vla=1.0, vlm=1.0)),
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_ego_vlm_mixed'))

eval = dict(output_dir='work_dirs/pi05_robocasa_ego_vlm_mixed')
