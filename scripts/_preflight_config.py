#!/usr/bin/env python3
"""Fail fast on config / data problems before torchrun starts N ranks.

Usage: python scripts/_preflight_config.py <config.py>

Checks, in order:
  1. config parses (lazy-import fallback)
  2. component dict keys match their class signatures (catches e.g.
     "Unexpected runner config field(s)")
  3. exactly one of runner.max_steps / runner.max_epochs is set
  4. RoboCasa data roots exist
  5. local checkpoint / tokenizer paths referenced by the config exist
  6. tokenizer directories are complete
  7. referenced VLM annotation files exist

All checks are static: class signatures are read from fluxvla/**/*.py with ast,
so this script does not import fluxvla and works on a GPU-less node.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

from mmengine import Config
from mmengine.config.utils import ConfigParsingError

REPO_ROOT = Path(__file__).resolve().parents[1]
STRICT_EXTRA_KEYS_MARKER = 'Unexpected runner config field'
LOCAL_PREFIXES = ('./', '../', '/')
TOKENIZER_SUFFIXES = ('.model',)
PATH_KEYS = ('tokenizer_path', 'model_path')


def load(path: str) -> Config:
    try:
        return Config.fromfile(path)
    except (ConfigParsingError, RuntimeError):
        # Lazy parsing either reports the inheritance-chain error or, when the
        # config reads os.environ, a bare RuntimeError from mmengine's env
        # proxy. The plain parser handles both.
        return Config.fromfile(path, lazy_import=False)


def _as_dict(value):
    if isinstance(value, Config):
        return value._cfg_dict
    return value


def _iter_pairs(value, prefix=''):
    value = _as_dict(value)
    if isinstance(value, dict):
        for key, item in value.items():
            path = '.'.join([prefix, str(key)]) if prefix else str(key)
            yield path, key, item
            yield from _iter_pairs(item, path)
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            yield from _iter_pairs(item, progress_join(prefix, position))


def progress_join(prefix, position):
    return ''.join([prefix, '[', str(position), ']'])


def _is_local_path(value):
    return isinstance(value, str) and value.startswith(LOCAL_PREFIXES)


def _iter_strings(value):
    value = _as_dict(value)
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _init_signature(class_node):
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            args = [a.arg for a in item.args.args][1:]
            return args, item.args.kwarg is not None
    return None, None


def _component_index():
    index = {}
    for file in sorted((REPO_ROOT / 'fluxvla').rglob('*.py')):
        if '__pycache__' in file.parts:
            continue
        try:
            text = file.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        strict_extra = STRICT_EXTRA_KEYS_MARKER in text
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            params, has_kwargs = _init_signature(node)
            if params is None:
                continue
            index.setdefault(node.name, []).append(
                (params, has_kwargs, strict_extra))
    # Ambiguous class names (defined in several modules) are skipped.
    return {name: hits[0] for name, hits in index.items() if len(hits) == 1}


def _check_components(value, index, path, errors):
    value = _as_dict(value)
    if isinstance(value, dict):
        type_name = value.get('type')
        if isinstance(type_name, str) and type_name in index:
            params, has_kwargs, strict_extra = index[type_name]
            for key in value:
                if key == 'type' or str(key).startswith('_'):
                    continue
                if key not in params and (strict_extra or not has_kwargs):
                    errors.append(
                        f'{path}.{type_name}: does not accept {key!r}')
        for key, item in value.items():
            _check_components(item, index, f'{path}.{key}', errors)
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _check_components(item, index, f'{path}[{position}]', errors)


def _tokenizer_dir_is_complete(path):
    directory = Path(path)
    if not directory.is_dir():
        return True
    has_fast = (directory / 'tokenizer.json').is_file()
    has_slow = ((directory / 'tokenizer.model').is_file()
                and (directory / 'tokenizer_config.json').is_file())
    return has_fast or has_slow


def main(path: str) -> int:
    cfg = load(path)
    task = getattr(cfg.runner, 'train_task', None)
    print(f'[preflight] config ok: {path} train_task={task}')

    errors = []
    _check_components(cfg, _component_index(), 'cfg', errors)
    if errors:
        print('[preflight] invalid component config keys:', file=sys.stderr)
        for message in errors:
            print('  ' + message, file=sys.stderr)
        return 5
    print('[preflight] component keys ok')

    runner = _as_dict(getattr(cfg, 'runner', None))
    if isinstance(runner, dict):
        max_steps = runner.get('max_steps')
        max_epochs = runner.get('max_epochs')
        if max_steps is not None and max_epochs is not None:
            print('[preflight] runner.max_steps and runner.max_epochs are both set:',
                  file=sys.stderr)
            print(f'  max_steps={max_steps} max_epochs={max_epochs}',
                  file=sys.stderr)
            print('  only one may be set; the launcher passes max_epochs=None',
                  file=sys.stderr)
            return 6
    print('[preflight] runner budget ok')

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
            print('  ' + root, file=sys.stderr)
        return 2
    print(f'[preflight] data roots ok: {len(roots)}')

    local_paths = []
    for config_path, key, item in _iter_pairs(cfg):
        if key in PATH_KEYS and _is_local_path(item):
            local_paths.append((config_path, item))
    pretrained = _as_dict(getattr(getattr(cfg, 'model', None),
                                 'pretrained_name_or_path', None))
    if _is_local_path(pretrained):
        local_paths.append(('cfg.model.pretrained_name_or_path', pretrained))
    missing_paths = [(name, value) for name, value in local_paths
                     if not os.path.exists(value)]
    if missing_paths:
        print('[preflight] missing local checkpoint/tokenizer paths:',
              file=sys.stderr)
        for name, value in missing_paths:
            print(f'  {name} = {value}', file=sys.stderr)
        print('  set PI05_CHECKPOINT / PI05_TOKENIZER or copy the files',
              file=sys.stderr)
        return 7
    print(f'[preflight] local paths ok: {len(local_paths)}')

    tokenizer_dirs = sorted({value for name, value in local_paths
                             if 'tokenizer' in name.lower()})
    for directory in tokenizer_dirs:
        if not _tokenizer_dir_is_complete(directory):
            print('[preflight] incomplete tokenizer directory: ' + directory,
                  file=sys.stderr)
            print('  need tokenizer.json, or tokenizer.model + tokenizer_config.json',
                  file=sys.stderr)
            return 4

    sources = []
    for loader_key in ('vlm_dataloader', 'train_dataloader'):
        stream = cfg.get(loader_key, None)
        if stream is None:
            continue
        dataset = (stream.get('dataset', None)
                   if hasattr(stream, 'get')
                   else getattr(stream, 'dataset', None))
        if dataset is None:
            continue
        if hasattr(dataset, 'get'):
            sources.append(dataset.get('anno_json_path', []))
        else:
            sources.append(getattr(dataset, 'anno_json_path', []))

    jsonl = [s for s in _iter_strings(sources)
             if s.endswith(('.jsonl', '.ndjson'))]
    missing_jsonl = [s for s in jsonl if not os.path.isfile(s)]
    if missing_jsonl:
        print('[preflight] missing VLM annotation files:', file=sys.stderr)
        for name in missing_jsonl:
            print('  ' + name, file=sys.stderr)
        return 3
    if jsonl:
        print(f'[preflight] vlm annotations ok: {len(jsonl)}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: _preflight_config.py <config.py>', file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
