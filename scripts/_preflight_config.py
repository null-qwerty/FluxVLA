#!/usr/bin/env python3
"""Fail fast on config / data problems before torchrun starts N ranks.

Usage: python scripts/_preflight_config.py <config.py>

Checks, in order:
  1. config parses (same lazy-import fallback as scripts/train.py)
  2. every component dict (runner, transforms, optimizer, ...) only passes
     keyword arguments its class declares, which catches errors such as
     "Unexpected runner config field(s): change_key_name" before torchrun
  3. all RoboCasa data roots exist
  4. the tokenizer directory is complete
  5. referenced VLM annotation files exist

Component-key validation is static: class signatures are read from
fluxvla/**/*.py with ast, so this script does not import fluxvla and works on
a GPU-less node.
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


def load(path: str) -> Config:
    try:
        return Config.fromfile(path)
    except (ConfigParsingError, RuntimeError):
        # Lazy parsing either reports the inheritance-chain error or, when the
        # config reads os.environ, a bare RuntimeError from mmengine's env
        # proxy. The plain parser handles both.
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
    if isinstance(value, Config):
        value = value._cfg_dict
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
            _check_components(item, index, '.'.join([path, str(key)]), errors)
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _check_components(item, index, ''.join([path, str(position)]), errors)


def main(path: str) -> int:
    cfg = load(path)
    task = getattr(cfg.runner, 'train_task', None)
    print('[preflight] config ok: ' + path + ' train_task=' + str(task))

    errors = []
    index = _component_index()
    _check_components(cfg, index, 'cfg', errors)
    if errors:
        print('[preflight] invalid component config keys:', file=sys.stderr)
        for message in errors:
            print('  ' + message, file=sys.stderr)
        return 5
    print('[preflight] component keys ok, classes scanned: ' + str(len(index)))

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
    print('[preflight] data roots ok: ' + str(len(roots)))

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

    tokenizer = getattr(getattr(cfg, 'runner', None), 'tokenizer', None)
    tokenizer_path = (getattr(tokenizer, 'model_path', None)
                      if tokenizer is not None else None)
    if tokenizer_path:
        path_obj = Path(tokenizer_path)
        if path_obj.is_dir():
            has_fast = (path_obj / 'tokenizer.json').is_file()
            has_slow = ((path_obj / 'tokenizer.model').is_file()
                        and (path_obj / 'tokenizer_config.json').is_file())
            if not (has_fast or has_slow):
                print('[preflight] incomplete tokenizer directory: ' + tokenizer_path,
                      file=sys.stderr)
                print('  need tokenizer.json, or tokenizer.model + tokenizer_config.json',
                      file=sys.stderr)
                print('  a dir with only tokenizer.model forces the sentencepiece converter',
                      file=sys.stderr)
                print('  copy the full HF tokenizer files into that directory',
                      file=sys.stderr)
                return 4
            print('[preflight] tokenizer ok: ' + tokenizer_path)

    jsonl = [s for s in _iter_strings(sources)
             if s.endswith(('.jsonl', '.ndjson'))]
    missing_jsonl = [s for s in jsonl if not os.path.isfile(s)]
    if missing_jsonl:
        print('[preflight] missing VLM annotation files:', file=sys.stderr)
        for name in missing_jsonl:
            print('  ' + name, file=sys.stderr)
        return 3
    if jsonl:
        print('[preflight] vlm annotations ok: ' + str(len(jsonl)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: _preflight_config.py <config.py>', file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
