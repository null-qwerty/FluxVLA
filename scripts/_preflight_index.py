#!/usr/bin/env python3
"""Warm the JSONL byte-offset indexes before torchrun starts N ranks.

LLaVAFormatDataset indexes every JSONL annotation source as a 16-byte-per-row
file keyed by a fingerprint of (path, size, mtime_ns, ctime_ns). On a shared
CPFS mount a node can briefly see stale metadata for a recently written file,
which makes the index build abort with 'Annotation changed during indexing'
(or, worse, publish an index for a truncated view of the file).

Building the index once per node here - with the same code the dataset uses -
turns that race into a clear preflight failure instead of a dead N-rank job.
The optional sibling file '<annotation>.manifest.json' records the
authoritative sha256/size, so a stale node view is detected instead of being
silently trained on.

Usage: python scripts/_preflight_index.py <config.py>

Environment:
  VLM_INDEX_CACHE_DIR   override the config's index_cache_dir (the DLC
                        launcher points this at node-local /tmp)
  PREFLIGHT_INDEX_RETRIES / PREFLIGHT_INDEX_SLEEP   retry tuning (6 / 5s)

Exit codes: 0 ok, 8 annotation/index problem.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

from mmengine import Config
from mmengine.config.utils import ConfigParsingError

RETRIES = int(os.environ.get('PREFLIGHT_INDEX_RETRIES', '6'))
SLEEP = float(os.environ.get('PREFLIGHT_INDEX_SLEEP', '5'))
MANIFEST_SUFFIX = '.manifest.json'


def load(path: str) -> Config:
    try:
        return Config.fromfile(path)
    except (ConfigParsingError, RuntimeError):
        return Config.fromfile(path, lazy_import=False)


def _as_dict(value):
    if isinstance(value, Config):
        return value._cfg_dict
    return value


def _dataset_streams(cfg):
    """Yield (stream_name, dataset_cfg) for every annotation-bearing loader."""
    for stream_name in ('vlm_dataloader', 'train_dataloader'):
        stream = cfg.get(stream_name, None)
        if stream is None:
            continue
        dataset = _as_dict(stream.get('dataset', None))
        if isinstance(dataset, dict):
            yield stream_name, dataset


def _jsonl_sources(dataset):
    sources = dataset.get('anno_json_path', [])
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    for source in sources or []:
        if isinstance(source, dict):
            source = source.get('path')
        if isinstance(source, str) and source.lower().endswith('.jsonl'):
            yield source


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_manifest(path):
    """Return an error string if the node sees a stale copy of path."""
    manifest_path = path + MANIFEST_SUFFIX
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, 'r', encoding='utf-8') as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as error:
        return f'cannot read {manifest_path}: {error}'
    size = os.path.getsize(path)
    expected_size = manifest.get('size')
    if expected_size is not None and size != expected_size:
        return (f'size {size} != {expected_size} recorded in '
                f'{manifest_path} (stale CPFS view?)')
    expected_hash = manifest.get('sha256')
    if expected_hash:
        actual = _sha256(path)
        if actual != expected_hash:
            return (f'sha256 {actual[:16]}... != {expected_hash[:16]}... '
                    f'recorded in {manifest_path} (stale CPFS view?)')
    return None


def _resolve_cache_dir(dataset):
    cache = dataset.get('index_cache_dir')
    cache = os.environ.get('VLM_INDEX_CACHE_DIR', cache)
    return os.path.expanduser(cache) if cache else None


def main(config_path: str) -> int:
    cfg = load(config_path)
    jobs = []
    for stream_name, dataset in _dataset_streams(cfg):
        cache_dir = _resolve_cache_dir(dataset)
        for source in _jsonl_sources(dataset):
            jobs.append((stream_name, os.path.expanduser(source), cache_dir))
    if not jobs:
        print('[preflight-index] no JSONL annotation sources')
        return 0
    print(f'[preflight-index] JSONL sources: {len(jobs)}')

    for stream_name, path, _ in jobs:
        if not os.path.isfile(path):
            print(f'[preflight-index] missing annotation: {path}', file=sys.stderr)
            return 8
        error = None
        for attempt in range(1, RETRIES + 1):
            error = _check_manifest(path)
            if error is None:
                break
            print(f'[preflight-index] {path}: {error} (attempt {attempt}/'
                  f'{RETRIES}, waiting {SLEEP:g}s)', file=sys.stderr)
            time.sleep(SLEEP)
        if error is not None:
            print('[preflight-index] annotation integrity check failed; the '
                  'node keeps seeing a stale copy. Rerun the job before '
                  f'training. {error}', file=sys.stderr)
            return 8
        print(f'[preflight-index] integrity ok: {path}')

    try:
        from fluxvla.datasets.utils.vlm_annotations import IndexedJSONL
    except Exception as error:  # pragma: no cover - GPU-less nodes
        print(f'[preflight-index] skip index warm-up ({error})')
        return 0

    for stream_name, path, cache_dir in jobs:
        for attempt in range(1, RETRIES + 1):
            try:
                index = IndexedJSONL(path, cache_dir)
                print(f'[preflight-index] {stream_name}: {path} '
                      f'rows={len(index)} cache={cache_dir or "default"}')
                break
            except RuntimeError as error:
                if attempt == RETRIES:
                    print(f'[preflight-index] {path}: {error}', file=sys.stderr)
                    return 8
                print(f'[preflight-index] {path}: {error} '
                      f'(attempt {attempt}/{RETRIES}, waiting {SLEEP:g}s)',
                      file=sys.stderr)
                time.sleep(SLEEP)
    print('[preflight-index] ok')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: _preflight_index.py <config.py>', file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
