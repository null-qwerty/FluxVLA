#!/usr/bin/env python3
"""Fail fast on RoboCasa eval config / checkpoint / asset problems.

Usage:
  python scripts/_preflight_eval.py CONFIG [--ckpt-path CKPT] [--assets-dir DIR]

Checks:
  1. config parses (lazy-import fallback) and has a RobocasaEvalRunner eval block
  2. the checkpoint file exists
  3. <ckpt>/../../dataset_statistics.json exists (eval de-norm source)
  4. the RoboCasa asset tree is complete

Exit codes: 2 eval config, 3 checkpoint, 4 statistics, 5 assets.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mmengine import Config
from mmengine.config.utils import ConfigParsingError

REQUIRED_ASSETS = (
    'fixtures',
    'textures',
    'arenas',
    'scenes',
    'objects/objaverse',
    'objects/sketchfab',
    'objects/lightwheel',
    'fixtures/toasters/basic_popup_2/visuals/model_0.obj',
)
# Only used when a task asks for generative_textures="100p"; gym.make() for
# the GR1 tabletop suites leaves it None, so a missing dir is not fatal.
OPTIONAL_ASSETS = ('generative_textures',)


def load(path):
    try:
        return Config.fromfile(path)
    except (ConfigParsingError, RuntimeError):
        return Config.fromfile(path, lazy_import=False)


def eval_runner_cfg(cfg):
    return cfg.eval.runner if hasattr(cfg.eval, 'runner') else cfg.eval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--assets-dir', default=None)
    args = parser.parse_args()

    cfg = load(args.config)
    runner = eval_runner_cfg(cfg)
    runner_type = getattr(runner, 'type', None)
    if runner_type != 'RobocasaEvalRunner':
        print(f'[preflight-eval] {args.config} eval.type={runner_type!r}, '
              'expected RobocasaEvalRunner', file=sys.stderr)
        return 2
    task_list = list(getattr(runner, 'task_list', []) or [])
    if not task_list:
        print('[preflight-eval] empty eval task_list', file=sys.stderr)
        return 2
    print(f'[preflight-eval] eval config ok: {len(task_list)} tasks, '
          f'num_trials_per_task={getattr(runner, "num_trials_per_task", None)}')

    ckpt = args.ckpt_path or getattr(runner, 'ckpt_path', None)
    if ckpt:
        ckpt_path = Path(ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            print(f'[preflight-eval] checkpoint not found: {ckpt_path}',
                  file=sys.stderr)
            return 3
        print(f'[preflight-eval] checkpoint ok: {ckpt_path} '
              f'({ckpt_path.stat().st_size / 1e9:.1f} GB)')
        norm_stats = getattr(runner, 'norm_stats_path', None)
        if norm_stats:
            stats_path = Path(norm_stats).expanduser().resolve()
        else:
            stats_path = ckpt_path.parent.parent / 'dataset_statistics.json'
        if not stats_path.is_file():
            print('[preflight-eval] missing dataset statistics: '
                  f'{stats_path}', file=sys.stderr)
            print('  the eval runner reads dataset_statistics.json from the '
                  'training run dir (two levels above the checkpoint)',
                  file=sys.stderr)
            return 4
        print(f'[preflight-eval] statistics ok: {stats_path}')

    assets_dir = (args.assets_dir or os.environ.get('ROBOCASA_ASSETS_DIR'))
    if not assets_dir:
        print('[preflight-eval] ROBOCASA_ASSETS_DIR is not set',
              file=sys.stderr)
        return 5
    assets = Path(assets_dir)
    missing = [name for name in REQUIRED_ASSETS
               if not (assets / name).exists()]
    if missing:
        print(f'[preflight-eval] incomplete RoboCasa assets under {assets}:',
              file=sys.stderr)
        for name in missing:
            print(f'  missing {name}', file=sys.stderr)
        return 5
    optional_missing = [name for name in OPTIONAL_ASSETS
                        if not (assets / name).exists()]
    if optional_missing:
        print('[preflight-eval] assets ok (optional missing: '
              + ', '.join(optional_missing) + ')')
    else:
        print(f'[preflight-eval] assets ok: {assets}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
