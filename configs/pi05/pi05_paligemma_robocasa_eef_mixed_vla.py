# Copyright 2026 Limx Dynamics
"""TASK 2b: RoboCasa EEF + EgoDex EEF in ONE batch ("mixed VLA" recipe).

The serial recipe (stage 1 ego full fine-tune -> stage 2 RoboCasa LoRA, TASK 2)
showed zero transfer: TASK 5 (no ego stage) matched TASK 2's loss curve at 2k,
8k, 19k and 20k steps. This config keeps the ego supervision alive during the
whole RoboCasa run instead of relying on that fragile hand-off:

* one dataloader, two sources: 24-task RoboCasa EEF parquet + EgoDex EEF frames
* every source keeps its own statistics (each sample carries "stats"), so the
  RoboCasa and EgoDex deltas are normalised with their own mean/std
* the EgoDex samples only supervise the 12 arm dimensions (per-dimension mask),
  the RoboCasa samples supervise all dimensions (time mask expanded to dims)

Environment:
  EGO_EEF_ROOT / EGO_EEF_JSONL / EGO_EEF_STATS   EgoDex EEF dataset (stage 1)
  ROBOCASA_EEF_DATA_ROOT / ROBOCASA_EEF_STATS    converted RoboCasa tree
  EGO_MIX_WEIGHT                                  ego share of the batch
                                                  (default 0.25 = 3:1)
"""
import copy
import json
import os

_base_ = './pi05_paligemma_robocasa_eef_lora.py'

_EGO_ROOT = os.path.abspath(
    os.environ.get('EGO_EEF_ROOT',
                   os.environ.get('EGO_EEF_DATA_ROOT',
                                  '/mnt/data/cpfs/users/neri/egodex_eef_v1')))
_EGO_JSONL = os.environ.get('EGO_EEF_JSONL', _EGO_ROOT + '/ego_eef.jsonl')
_EGO_STATS_PATH = os.environ.get('EGO_EEF_STATS', _EGO_ROOT + '/stats.json')
_EGO_STATS = json.loads(open(_EGO_STATS_PATH).read())
_TOKENIZER = os.environ.get('PI05_TOKENIZER',
                            '/mnt/data/cpfs/users/neri/checkpoints/pi05_base')
_EGO_WEIGHT = float(os.environ.get('EGO_MIX_WEIGHT', '0.25'))

# --- RoboCasa branch: the 24-task parquet dataset, statistics attached
# directly because the balanced wrapper that normally injects them is gone.
_robo = copy.deepcopy(dict(_base_.train_dataloader.dataset.datasets))
_robo['statistic_name'] = 'robocasa_eef'
# ParquetDataset takes the statistics blob per call (the balanced wrapper
# normally supplies it), so pass it through EEFMixedDataset instead.
_ROBO_STATS = {'robocasa_eef': json.loads(
    open(os.environ.get('ROBOCASA_EEF_STATS',
                        os.environ.get('ROBOCASA_EEF_DATA_ROOT',
                                       '/mnt/data/cpfs/users/neri/data/'
                                       'robocasa_eef_V0.1') + '/stats.json')).read())}
_robo_transforms = list(_robo['transforms'])
# The parquet branch carries a (horizon,) time mask; the ego branch carries a
# (horizon, dim) mask. Expand it so one collator can batch both.
_robo_transforms.append(dict(type='ExpandActionMasks', action_dim=32))
_robo['transforms'] = _robo_transforms

# --- EgoDex branch: same contract the stage-1 recipe used.
_ego = dict(
    type='EgoEEFDataset',
    anno_json_path=_EGO_JSONL,
    data_root_path=_EGO_ROOT,
    action_horizon=16,
    statistics=_EGO_STATS,
    transforms=[
        dict(
            type='NormalizeStatesAndActions',
            action_dim=None,
            state_dim=None,
            state_key='proprio',
            action_key='action',
            norm_type='mean_std',
            state_norm_type='mean_std',
            action_norm_type='mean_std',
            output_dtype='float32'),
        dict(
            type='PreparePromptWithState',
            lowercase_task_description=False,
            add_action_prefix=True),
        dict(
            type='ProcessPrompts',
            max_len=200,
            tokenizer=dict(
                type='PretrainedTokenizer', model_path=_TOKENIZER)),
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
    ])

train_dataloader = dict(
    _delete_=True,
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='EEFMixedDataset',
        datasets=[_robo, _ego],
        weights=[1.0 - _EGO_WEIGHT, _EGO_WEIGHT],
        statistics=[_ROBO_STATS, None],
    ))

# Statistic bookkeeping for evaluation/unnormalisation stays the EEF one.
_base_.eval.unnorm_key = 'robocasa_eef'
