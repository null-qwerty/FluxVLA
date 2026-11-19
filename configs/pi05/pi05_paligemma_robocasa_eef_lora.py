# Copyright 2026 Limx Dynamics
"""RoboCasa GR1 in the unified EEF action space - stage 2 (VLM LoRA).

Stage 2 of TASK 2 in docs/eef_action_space.md. The dataset is the RoboCasa
lerobot tree converted by tools/convert_robocasa_to_eef.py:

  observation.state : [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
  action            : [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]

EEF poses are in the head-camera (egoview) frame and the deltas are already
applied, so the joint-space RelativeActions transform is disabled and
normalization runs in mean_std mode with statistics produced by the converter.

Environment:
  ROBOCASA_EEF_DATA_ROOT   converted 24-task root (default
                           $DATA_ROOT/data/robocasa_eef_V0.1)
  ROBOCASA_EEF_STATS       stats json produced by the converter
                           (default <root>/stats.json)
  EGO_STAGE1_CHECKPOINT    stage-1 ego checkpoint (optional)
"""
import json
import os

_base_ = './pi05_paligemma_robocasa_ego_vlm_mixed_lora.py'

_EEF_ROOT = os.path.abspath(
    os.environ.get('ROBOCASA_EEF_DATA_ROOT', './datasets/robocasa_eef_V0.1'))
_EEF_STATS_PATH = os.environ.get('ROBOCASA_EEF_STATS',
                                _EEF_ROOT + '/stats.json')
if not os.path.isfile(_EEF_STATS_PATH):
    raise FileNotFoundError(
        'RoboCasa EEF statistics not found: '
        f'{_EEF_STATS_PATH} (ROBOCASA_EEF_DATA_ROOT={_EEF_ROOT}). Point '
        'ROBOCASA_EEF_DATA_ROOT/ROBOCASA_EEF_STATS at the tree produced by '
        'tools/convert_robocasa_to_eef.py, e.g. '
        '$DATA_ROOT/data/robocasa_eef_V0.1.')
_EEF_STATS = json.loads(open(_EEF_STATS_PATH).read())
_EEF_STAT_NAME = 'robocasa_eef'

# Point the 24 task roots at the converted tree (same task basenames).
_datasets = _base_.train_dataloader.dataset.datasets
_datasets.data_root_path = [
    '{}/{}'.format(_EEF_ROOT, os.path.basename(path))
    for path in _datasets.data_root_path
]
# Both the balanced wrapper and the inner ParquetDataset resolve the
# statistic name, so point both at the EEF statistics.
_base_.train_dataloader.dataset.statistic_name = _EEF_STAT_NAME
_base_.train_dataloader.dataset.dataset_statistics = {
    _EEF_STAT_NAME: _EEF_STATS,
}
_datasets.statistic_name = _EEF_STAT_NAME

# Actions are already deltas and the new space is normalized with mean_std.
_transforms = _datasets.transforms
for _transform in _transforms:
    if isinstance(_transform, dict):
        if _transform.get('type') == 'RelativeActions':
            _transform['mask'] = None
        elif _transform.get('type') == 'NormalizeStatesAndActions':
            _transform['norm_type'] = 'mean_std'
            _transform['state_norm_type'] = 'mean_std'
            _transform['action_norm_type'] = 'mean_std'

# Evaluation: the policy consumes absolute EEF state and emits delta EEF, while
# RoboCasa still takes absolute 29D joint targets. RobocasaEvalRunner converts
# with fluxvla/utils/eef_eval_adapter.py (FK for the observation, DLS IK for the
# action) when `eef_action_space` is set.
_base_.eval.unnorm_key = _EEF_STAT_NAME
# The joint recipe denormalizes into *absolute* targets by re-applying the
# delta to the current state (DenormalizeDeltaAction). In the EEF space the
# delta is applied by the deployment adapter (apply_eef_delta + DLS IK), so the
# denormalizer must only undo the mean/std normalization - and
# DenormalizeDeltaAction rejects a missing delta mask (None -> 0-d array).
_base_.eval.denormalize_action = dict(
    type='DenormalizePrivateAction',
    statistic_name=_EEF_STAT_NAME,
    norm_type='mean_std',
    action_dim=29,
)
_base_.eval.eef_action_space = dict(enabled=True)
# Eval-time state normalisation must match training (mean_std on the EEF stats,
# not the joint recipe's quantile).
for _eval_transform in _base_.eval.dataset.transforms:
    if (isinstance(_eval_transform, dict)
            and _eval_transform.get('type') == 'NormalizeStatesAndActions'):
        _eval_transform['norm_type'] = 'mean_std'
        _eval_transform['state_norm_type'] = 'mean_std'
        _eval_transform['action_norm_type'] = 'mean_std'

runner = dict(
    type='DDPTrainRunner',
    train_task='vla',
    max_steps=40000,
    max_epochs=None,
    static_graph=False,
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_eef_lora'))

eval = dict(output_dir='work_dirs/pi05_robocasa_eef_lora')
