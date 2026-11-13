# Copyright 2026 Limx Dynamics
"""Ego configs must survive the lazy-import parser conflict.

The ego recipes reference `_base_` as an object (for example
`_base_.pop('vlm_dataloader')`), which makes mmengine pick the lazy parser,
while the inherited files use the plain `_base_ = "...py"` form. The
production loaders in scripts/train.py and scripts/eval.py therefore fall back
to `lazy_import=False`; this test keeps that fallback honest.
"""

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


def load_with_fallback(path):
    try:
        return Config.fromfile(path)
    except ConfigParsingError:
        return Config.fromfile(path, lazy_import=False)


@pytest.mark.parametrize('name', CONFIGS)
def test_ego_config_uses_lazy_import_fallback(monkeypatch, name):
    for key in ('EGO_VLM_DATA_ROOT', 'EGO_STAGE1_CHECKPOINT', 'ROBOCASA_DATA_ROOT'):
        monkeypatch.delenv(key, raising=False)
    cfg = load_with_fallback(ROOT / 'configs' / name)
    assert getattr(cfg.runner, 'train_task', None) in ('vla', 'vlm', 'mixed')
    assert cfg.model is not None
