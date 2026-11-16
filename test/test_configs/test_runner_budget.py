# Copyright 2026 Limx Dynamics
"""Guard the runner budget and env-driven checkpoint paths of the ego arms."""

from pathlib import Path

import pytest
from mmengine import Config
from mmengine.config.utils import ConfigParsingError

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    'pi05/pi05_paligemma_egodex_vlm_sft.py',
    'pi05/pi05_paligemma_robocasa_ego_vlm_mixed_full_finetune.py',
    'pi05/pi05_paligemma_robocasa_ego_vlm_mixed_frozen_vlm.py',
    'pi05/pi05_paligemma_robocasa_ego_vlm_mixed_lora.py',
)


def load(name):
    path = ROOT / 'configs' / name
    try:
        return Config.fromfile(path)
    except (ConfigParsingError, RuntimeError):
        return Config.fromfile(path, lazy_import=False)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ('PI05_CHECKPOINT', 'PI05_TOKENIZER', 'EGO_VLM_DATA_ROOT',
                'EGO_STAGE1_CHECKPOINT', 'ROBOCASA_DATA_ROOT'):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize('name', CONFIGS)
def test_runner_has_a_single_budget(name):
    cfg = load(name)
    runner = cfg.runner
    max_steps = runner.get('max_steps')
    max_epochs = runner.get('max_epochs')
    assert not (max_steps is not None and max_epochs is not None), (
        f'{name}: max_steps={max_steps} and max_epochs={max_epochs} are both set')


def test_egodex_sft_is_step_based_and_env_driven(monkeypatch):
    cfg = load('pi05/pi05_paligemma_egodex_vlm_sft.py')
    assert cfg.runner.max_epochs is None
    assert cfg.runner.max_steps is not None
    tokenizer = cfg.train_dataloader.dataset.transforms[-1]['tokenizer_path']
    assert tokenizer == './checkpoints/pi05_base'

    monkeypatch.setenv('PI05_TOKENIZER', '/data/tok')
    monkeypatch.setenv('PI05_CHECKPOINT', '/data/model.safetensors')
    cfg = load('pi05/pi05_paligemma_egodex_vlm_sft.py')
    assert cfg.train_dataloader.dataset.transforms[-1]['tokenizer_path'] == '/data/tok'
    assert cfg.model.pretrained_name_or_path == '/data/model.safetensors'
