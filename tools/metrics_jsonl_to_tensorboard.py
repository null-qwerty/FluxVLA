#!/usr/bin/env python3
"""Mirror FluxVLA JSONL training metrics into TensorBoard event files.

VLAMetric's JSONLinesTracker writes one JSON object per logged step to
"<work_dir>/<run_id>.jsonl". TensorBoard cannot read that format, so this
script tails those files and writes equivalent scalar events under
"<work_dir>/tensorboard_jsonl/<run_id>"; TensorBoard then serves them live.

Usage:
    python tools/metrics_jsonl_to_tensorboard.py --root /path/to/work_dirs
    python tools/metrics_jsonl_to_tensorboard.py --root /path/to/work_dirs --watch 30

It only needs the "tensorboard" package (no torch), so it can run on a
CPU/login node next to the shared filesystem.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

STATE_NAME = ".tb_jsonl_state.json"
OUT_DIR_NAME = "tensorboard_jsonl"
SKIP_FILES = ("run-metrics.jsonl",)


def as_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def step_of(metrics):
    for key, value in metrics.items():
        if key.endswith("/Step") or key == "Step":
            number = as_float(value)
            if number is not None:
                return int(number)
    return None


def find_metric_files(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in SKIP_FILES:
            continue
        parts = path.parts
        if OUT_DIR_NAME in parts or "tensorboard" in parts:
            continue
        yield path


def output_dir_for(jsonl: Path) -> Path:
    return jsonl.parent / OUT_DIR_NAME / jsonl.stem


class Mirror:

    def __init__(self, jsonl: Path, out_dir: Path, offset: int, step: int):
        self.jsonl = jsonl
        self.out_dir = out_dir
        self.offset = offset
        self.step = step
        self.writer = None
        self.written = 0

    def _ensure_writer(self):
        if self.writer is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.writer = EventFileWriter(str(self.out_dir))

    def poll(self) -> bool:
        try:
            size = self.jsonl.stat().st_size
        except OSError:
            return False
        if size < self.offset:
            # File was truncated or rotated: restart from the beginning.
            self.offset = 0
        if size == self.offset:
            return False
        with self.jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(self.offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    metrics = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(metrics, dict):
                    continue
                step = step_of(metrics)
                if step is None:
                    step = self.step + 1
                self.step = max(self.step, step)
                values = []
                for key, value in metrics.items():
                    if key.endswith("/Step") or key == "Step":
                        continue
                    number = as_float(value)
                    if number is None:
                        continue
                    values.append(Summary.Value(tag=key, simple_value=number))
                if not values:
                    continue
                self._ensure_writer()
                self.writer.add_event(
                    Event(
                        wall_time=time.time(),
                        step=step,
                        summary=Summary(value=values)))
                self.written += 1
            self.offset = handle.tell()
        if self.writer is not None:
            self.writer.flush()
        return True

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def load_state(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(path: Path, mirrors):
    payload = {
        str(mirror.jsonl): {
            "offset": mirror.offset,
            "step": mirror.step,
            "out": str(mirror.out_dir),
            "written": mirror.written,
        }
        for mirror in mirrors.values()
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="work_dirs root")
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        help="poll interval in seconds (0 = single pass)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    state_path = root / STATE_NAME
    state = load_state(state_path)
    mirrors = {}

    def refresh():
        for jsonl in find_metric_files(root):
            key = str(jsonl)
            if key in mirrors:
                continue
            saved = state.get(key, {})
            out_dir = Path(saved.get("out") or output_dir_for(jsonl))
            mirror = Mirror(
                jsonl,
                out_dir,
                int(saved.get("offset", 0)),
                int(saved.get("step", 0)))
            mirrors[key] = mirror

    refresh()
    for mirror in mirrors.values():
        mirror.poll()
    save_state(state_path, mirrors)
    total = sum(m.written for m in mirrors.values())
    print(
        f"[tb-mirror] {len(mirrors)} run(s), {total} point(s) converted",
        flush=True)

    if args.watch <= 0:
        for mirror in mirrors.values():
            mirror.close()
        return

    try:
        while True:
            time.sleep(args.watch)
            refresh()
            changed = [mirror for mirror in mirrors.values() if mirror.poll()]
            if changed:
                save_state(state_path, mirrors)
                print(
                    f"[tb-mirror] {len(changed)} run(s) updated, "
                    f"{sum(m.written for m in mirrors.values())} point(s) total",
                    flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for mirror in mirrors.values():
            mirror.close()


if __name__ == "__main__":
    main()
