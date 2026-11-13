#!/usr/bin/env python3
"""Fail fast on config / data problems before torchrun starts N ranks.

Usage: python scripts/_preflight_config.py <config.py>

Loading uses the same lazy-import fallback as scripts/train.py: configs that
reference ``_base_`` as an object make MMEngine pick the lazy parser, which
rejects the plain ``_base_ = "...py"`` inheritance form.
"""
from __future__ import annotations

import os
import sys

from mmengine import Config
from mmengine.config.utils import ConfigParsingError


def load(path: str) -> Config:
    try:
        return Config.fromfile(path)
    except ConfigParsingError:
        return Config.fromfile(path, lazy_import=False)


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def main(path: str) -> int:
    cfg = load(path)
    task = getattr(cfg.runner, 'train_task', None)
    print(f'[preflight] config ok: {path} train_task={task}')

    roots = []
    loader = getattr(cfg, 'train_dataloader', None)
    if loader is not None:
        dataset = getattr(loader, 'dataset', None)
        inner = getattr(dataset, 'datasets', None) if dataset else None
        if inner is not None:
            roots = [str(r) for r in (getattr(inner, 'data_root_path', []) or [])]
    missing_roots = [r for r in roots if not os.path.isdir(r)]
    if missing_roots:
        print('[preflight] missing data roots:', file=sys.stderr)
        for root in missing_roots:
            print(f'  {root}', file=sys.stderr)
        return 2
    print(f'[preflight] data roots ok: {len(roots)}')

    sources = []
    # Language streams show up either as a separate vlm_dataloader (mixed
    # training) or directly on train_dataloader (pure VLM SFT).
    for loader_key in ('vlm_dataloader', 'train_dataloader'):
        loader = cfg.get(loader_key, None)
        if loader is None:
            continue
        dataset = loader.get('dataset', None) if hasattr(loader, 'get') else getattr(loader, 'dataset', None)
        if dataset is None:
            continue
        if hasattr(dataset, 'get'):
            sources.append(dataset.get('anno_json_path', []))
        else:
            sources.append(getattr(dataset, 'anno_json_path', []))
    jsonl = [s for s in _iter_strings(sources) if s.endswith(('.jsonl', '.ndjson'))]
    missing_jsonl = [s for s in jsonl if not os.path.isfile(s)]
    if missing_jsonl:
        print('[preflight] missing VLM annotation files:', file=sys.stderr)
        for name in missing_jsonl:
            print(f'  {name}', file=sys.stderr)
        return 3
    if jsonl:
        print(f'[preflight] vlm annotations ok: {len(jsonl)}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: _preflight_config.py <config.py>', file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
