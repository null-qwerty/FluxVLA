# Copyright 2026 Limx Dynamics
"""EgoDex stage-1/stage-2 and parallel mixed arms stay loadable."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
EGO_STAGE1 = 'pi05/pi05_paligemma_egodex_vlm_sft.py'
EGO_MIXED = ('pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py')
EGO_STAGE2_FROZEN = (
    'pi05/pi05_paligemma_robocasa_ego_vlm_mixed_frozen_vlm.py')
EGO_STAGE2_LORA = 'pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py'
EGO_CONFIGS = (EGO_STAGE1, EGO_MIXED, EGO_STAGE2_FROZEN, EGO_STAGE2_LORA)
SCHEDULE = ('vla', 'vla', 'vla', 'vlm')
LORA_TARGETS = (
    'q_proj',
    'v_proj',
    'k_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
    'projector.projector',
    'out_proj',
    'fc1',
    'fc2',
)
LORA_SAVE = (
    'action_in_proj',
    'action_out_proj',
    'time_mlp_in',
    'time_mlp_out',
)
EGO_JSONL = './datasets/egodex_vlm/ego.jsonl'
EGO_ROOT = './datasets/egodex_vlm'


@pytest.fixture(autouse=True)
def portable_config_environment(monkeypatch):
    for key in ('EGO_VLM_DATA_ROOT', 'EGO_VLM_WEIGHT', 'EGO_LLAVA_WEIGHT',
                'EGO_STAGE1_CHECKPOINT', 'ROBOCASA_DATA_ROOT',
                'PI05_CHECKPOINT', 'PI05_TOKENIZER'):
        monkeypatch.delenv(key, raising=False)


def load(name):
    # This interpreter may freeze the os module, which makes mmengine's
    # lazy-import sniffing treat "import os" as a third-party import. The
    # explicit flag parses the same files with the non-lazy parser.
    return Config.fromfile(ROOT / 'configs' / name, lazy_import=False)


def assert_relative_paths(value, key='config'):
    if isinstance(value, Mapping):
        for name, item in value.items():
            assert_relative_paths(item, '%s.%s' % (key, name))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_relative_paths(item, '%s[%d]' % (key, index))
    elif isinstance(value, str):
        assert not Path(value).is_absolute(), '%s: %s' % (key, value)
        assert not value.startswith('~/'), '%s: %s' % (key, value)


def test_stage1_is_pure_ego_vlm_sft():
    cfg = load(EGO_STAGE1)
    runner = cfg.runner
    assert runner.type == 'FSDPTrainRunner'
    assert runner.train_task == 'vlm'
    # Stage 1 is step-based like the other ego arms; the inherited Llava-665K
    # recipe was epoch-based (max_epochs=1), which conflicts with the
    # launcher-provided max_steps and trips the BASE runner assertion.
    assert runner.max_epochs is None and runner.max_steps is not None
    assert 'mixed_training' not in runner
    dataset = cfg.train_dataloader.dataset
    assert dataset.type == 'LLaVAFormatDataset'
    assert list(dataset.anno_json_path)[0]['path'] == EGO_JSONL
    assert dataset.data_root_path == EGO_ROOT
    assert dataset.transforms[0].type == 'BuildVLMMessages'
    assert dataset.transforms[1].type == 'ProcessPaliGemmaVLMSFTInputs'
    # The base LLaVA-665K VLM recipe already uses freeze_llm_expert correctly
    # (the action expert is unused in pure language SFT).
    assert cfg.model.freeze_llm_expert is True
    assert cfg.model.freeze_llm_backbone is False
    assert cfg.model.freeze_vision_backbone is False


def test_parallel_mixed_arm_keeps_the_robocasa_contract():
    cfg = load(EGO_MIXED)
    runner = cfg.runner
    assert runner.train_task == 'mixed'
    assert tuple(runner.mixed_training.task_schedule) == SCHEDULE
    assert dict(runner.mixed_training.loss_weights) == dict(vla=1.0, vlm=1.0)
    assert runner.max_epochs is None
    assert runner.max_steps == 40000
    assert runner.grad_accumulation_steps == 8
    dataset = cfg.vlm_dataloader.dataset
    assert dataset.type == 'LLaVAFormatDataset'
    assert list(dataset.anno_json_path)[0]['path'] == EGO_JSONL
    assert list(dataset.data_root_path) == [EGO_ROOT]
    assert dataset.max_retry_times == 0
    assert dataset.transforms[0].type == 'BuildVLMMessages'
    assert dataset.transforms[1].type == 'ProcessPaliGemmaVLMSFTInputs'
    # The robot stream and the 24-task evaluation are inherited unchanged.
    assert len(cfg.train_dataloader.dataset.datasets.data_root_path) == 24
    assert cfg.eval.total_tasks == 24
    assert cfg.eval.num_trials_per_task == 50


@pytest.mark.parametrize('name', (EGO_STAGE2_FROZEN, EGO_STAGE2_LORA))
def test_stage2_arms_are_vla_only_robocasa_runs(name):
    cfg = load(name)
    runner = cfg.runner
    assert runner.train_task == 'vla'
    assert 'mixed_training' not in runner
    assert 'vlm_dataloader' not in cfg
    assert runner.max_steps == 40000
    # Stage 2 trains the 24-task RoboCasa dataset only.
    assert len(cfg.train_dataloader.dataset.datasets.data_root_path) == 24
    assert cfg.eval.type == 'RobocasaEvalRunner'
    assert cfg.eval.total_tasks == 24
    assert cfg.eval.num_trials_per_task == 50


def test_frozen_stage2_freezes_vlm_not_the_action_expert():
    cfg = load(EGO_STAGE2_FROZEN)
    model = cfg.model
    # VLM side frozen, action expert trainable, projections still trainable.
    assert model.freeze_llm_backbone is True
    assert model.freeze_vision_backbone is True
    assert model.freeze_llm_expert is False
    assert model.freeze_projector is False
    assert model.freeze_vision_backbone is True
    assert cfg.runner.type == 'FSDPTrainRunner'
    assert 'fsdp_wrap_policy' in cfg.runner


def test_lora_stage2_matches_the_libero_lora_recipe():
    cfg = load(EGO_STAGE2_LORA)
    model = cfg.model
    assert model.use_lora is True
    assert model.lora_rank == 256
    assert model.lora_alpha == 512
    assert model.lora_dropout == pytest.approx(0.0)
    assert tuple(model.lora_target_modules) == LORA_TARGETS
    assert tuple(model.modules_to_save) == LORA_SAVE
    # The inherited freeze flags stay False: peft freezes the base itself.
    assert model.freeze_llm_backbone is False
    assert model.freeze_vision_backbone is False
    assert model.freeze_llm_expert is False
    # LoRA injection lives in DDPTrainRunner; FSDP-only fields must be gone.
    assert cfg.runner.type == 'DDPTrainRunner'
    assert 'fsdp_wrap_policy' not in cfg.runner
    assert 'sharding_strategy' not in cfg.runner


@pytest.mark.parametrize('name', EGO_CONFIGS)
def test_ego_config_run_directories_are_distinct(name):
    cfg = load(name)
    run_dir = cfg.runner.metric.run_dir
    assert run_dir.startswith('work_dirs/')
    if cfg.get('eval') is not None:
        eval_dir = cfg.eval.get('output_dir') or cfg.eval.get('run_dir')
        if eval_dir is not None:
            assert Path(eval_dir).name in (Path(run_dir).name,
                                           'textvqa') or eval_dir == run_dir


@pytest.mark.parametrize('name', EGO_CONFIGS)
def test_ego_config_runtime_paths_stay_relative(name):
    cfg = load(name)
    assert_relative_paths({
        key: value
        for key, value in cfg.to_dict().items() if not key.startswith('_')
    })


def test_stage1_data_and_weight_overrides(monkeypatch):
    monkeypatch.setenv('EGO_VLM_DATA_ROOT',
                       '/mnt/data/oss/users/neri/data/egodex_vlm')
    monkeypatch.setenv('EGO_VLM_WEIGHT', '2.0')
    dataset = load(EGO_STAGE1).train_dataloader.dataset
    source = list(dataset.anno_json_path)[0]
    assert source['path'] == (
        '/mnt/data/oss/users/neri/data/egodex_vlm/ego.jsonl')
    assert source['weight'] == pytest.approx(2.0)
    assert dataset.data_root_path == (
        '/mnt/data/oss/users/neri/data/egodex_vlm')


def test_mixed_arm_llava_replay_override(monkeypatch):
    monkeypatch.setenv('EGO_LLAVA_WEIGHT', '1.0')
    dataset = load(EGO_MIXED).vlm_dataloader.dataset
    sources = list(dataset.anno_json_path)
    roots = list(dataset.data_root_path)
    assert len(sources) == len(roots) == 2
    assert sources[0]['path'] == EGO_JSONL
    assert sources[1]['path'].endswith('llava_v1_5_mix665k.json')
    assert roots[1] == './datasets/llava665k'


@pytest.mark.parametrize('name', (EGO_STAGE2_FROZEN, EGO_STAGE2_LORA))
def test_stage2_can_initialize_from_stage1_checkpoint(name, monkeypatch):
    checkpoint = ('work_dirs/pi05_egodex_vlm_sft/checkpoints/'
                  'latest-checkpoint.safetensors')
    monkeypatch.setenv('EGO_STAGE1_CHECKPOINT', checkpoint)
    model = load(name).model
    assert model.pretrained_name_or_path == checkpoint
    # FluxVLA checkpoints already use FluxVLA names: no OpenPI mapping.
    assert model.name_mapping is None
    assert model.strict_mapping is True
