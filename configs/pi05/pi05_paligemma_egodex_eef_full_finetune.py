# Copyright 2026 Limx Dynamics
"""EgoDex in the unified EEF action space - stage 1 (full fine-tune).

Stage 1 of TASK 2 (see docs/eef_action_space.md): train the VLM + action expert
on EgoDex frames converted by tools/convert_egodex_to_eef.py. The action space
is the same 29D layout as the RoboCasa stage-2 recipe:

  state  : [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
  action : [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]

EgoDex only labels the two wrist poses, so the per-dimension action mask keeps
the 12 arm EEF dimensions and masks the hand/waist dimensions out of the flow
matching loss (pi0/pi05 reduce_action_bc_loss honours action_masks).

Environment (names are set by scripts/dlc_train.sh):
  EGO_EEF_ROOT        output directory of tools/convert_egodex_to_eef.py
                      (EGO_EEF_DATA_ROOT is accepted as an alias)
  EGO_EEF_JSONL       annotation file (default <root>/ego_eef.jsonl)
  EGO_EEF_STATS       statistics json produced by the same tool
                      (default <root>/stats.json)
"""
import json
import os

_base_ = './pi05_paligemma_robocasa_full_data_full_finetune.py'

# The launcher exports EGO_EEF_ROOT; EGO_EEF_DATA_ROOT is the older alias and
# kept for manual runs. Resolve now: the dataset joins the record paths
# ("images/<task>.tar") against this root, so a relative fallback silently
# pointed at <cwd>/datasets/egodex_eef instead of the converted data.
_EGO_EEF_ROOT = os.path.abspath(
    os.environ.get('EGO_EEF_ROOT')
    or os.environ.get('EGO_EEF_DATA_ROOT', './datasets/egodex_eef'))
_EGO_EEF_JSONL = os.environ.get('EGO_EEF_JSONL',
                                  _EGO_EEF_ROOT + '/ego_eef.jsonl')
_EGO_EEF_STATS_PATH = os.environ.get('EGO_EEF_STATS',
                                      _EGO_EEF_ROOT + '/stats.json')
_EGO_EEF_STATS = json.loads(open(_EGO_EEF_STATS_PATH).read())
# The base recipe owns _PI05_TOKENIZER; child configs cannot see it.
_PI05_TOKENIZER_PATH = os.environ.get('PI05_TOKENIZER',
                                      './checkpoints/pi05_base')

train_dataloader = dict(
    _delete_=True,
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='EgoEEFDataset',
        anno_json_path=_EGO_EEF_JSONL,
        data_root_path=_EGO_EEF_ROOT,
        action_horizon=16,
        statistics=_EGO_EEF_STATS,
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                action_dim=None,
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std',
                output_dtype='float32'),
            dict(
                type='PreparePromptWithState',
                lowercase_task_description=False,
                add_action_prefix=True),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_PI05_TOKENIZER_PATH)),
            dict(type='PadStatesAndActions', model_action_dim=32),
            dict(type='RandomCropImages', scale=0.95),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='ColorJitterImages',
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08),
            dict(type='SimpleNormalizeImages'),
        ]))

runner = dict(
    train_task='vla',
    max_steps=40000,
    max_epochs=None,
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_egodex_eef_sft'))

eval = dict(output_dir='work_dirs/pi05_egodex_eef_sft')
