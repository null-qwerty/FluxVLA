#!/usr/bin/env python3
"""Split an EgoDex VLM jsonl into train/val parts without frame leakage.

Rows written by tools/egodex_to_vlm_jsonl.py carry "task", "episode" and
"frame_index" fields. Splitting by episode (default) keeps all frames of one
recording on the same side, so validation measures generalisation instead of
memorisation of individual frames. --group-by task holds out whole tasks, a
much harder split that measures task transfer.

Usage:
    python tools/split_egodex_vlm_jsonl.py \
        --input /mnt/data/cpfs/users/neri/egodex_vlm/ego.jsonl \
        --val-fraction 0.15

Outputs (next to the input by default):
    ego_train.jsonl, ego_val.jsonl, split_manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_RE = re.compile(r"images/(?P<task>[^/]+)/(?P<episode>[^/]+?)(?:_f\d+)?\.jpg$")


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def row_group(row: dict, group_by: str):
    task = row.get("task")
    episode = row.get("episode")
    if task is None or episode is None:
        image = row.get("image")
        if isinstance(image, str):
            match = IMAGE_RE.search(image)
            if match:
                task = task or match.group("task")
                episode = episode or match.group("episode")
    task = str(task) if task is not None else "unknown"
    if group_by == "task":
        return task
    episode = str(episode) if episode is not None else "unknown"
    return f"{task}/{episode}"


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ego.jsonl to split")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: the input file's directory)")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="fraction of samples to put into validation (default 0.15)")
    parser.add_argument(
        "--group-by",
        choices=("episode", "task"),
        default="episode",
        help="episode keeps recordings intact; task holds out whole tasks")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-name", default="ego_train.jsonl")
    parser.add_argument("--val-name", default="ego_val.jsonl")
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")

    source = Path(args.input).resolve()
    rows = read_rows(source)
    if not rows:
        parser.error(f"no rows parsed from {source}")

    groups = defaultdict(list)
    for row in rows:
        groups[row_group(row, args.group_by)].append(row)

    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)

    target = args.val_fraction * len(rows)
    val_keys, val_count = [], 0
    for key in keys:
        if val_count >= target:
            break
        val_keys.append(key)
        val_count += len(groups[key])
    if not val_keys:
        val_keys = [keys[0]]
    # Keep a non-empty train side even for tiny inputs.
    if len(val_keys) == len(keys):
        removed = val_keys.pop()
        val_count -= len(groups[removed])

    val_set = set(val_keys)
    train_rows = [row for row in rows if row_group(row, args.group_by) not in val_set]
    val_rows = [row for row in rows if row_group(row, args.group_by) in val_set]

    out_dir = Path(args.out_dir).resolve() if args.out_dir else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    def task_counts(items):
        return Counter(str(row.get("task", "unknown")) for row in items)

    train_tasks = task_counts(train_rows)
    val_tasks = task_counts(val_rows)
    manifest = {
        "input": str(source),
        "group_by": args.group_by,
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "rows": {"total": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "groups": {
            "total": len(keys),
            "train": len(keys) - len(val_keys),
            "val": len(val_keys),
        },
        "tasks": {
            "train": len(train_tasks),
            "val": len(val_tasks),
            "held_out_entirely": sorted(set(val_tasks) - set(train_tasks)),
        },
        "files": {"train": str(train_path), "val": str(val_path)},
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    if not val_rows:
        print("[split] WARNING: no validation rows; need at least 2 "
              "episodes/tasks to build a split", flush=True)
    print(f"[split] rows: total={len(rows)} train={len(train_rows)} val={len(val_rows)}")
    print(f"[split] groups({args.group_by}): train={len(keys) - len(val_keys)} val={len(val_keys)}")
    print(f"[split] tasks: train={len(train_tasks)} val={len(val_tasks)}")
    if manifest["tasks"]["held_out_entirely"]:
        print(f"[split] tasks only in val: {manifest['tasks']['held_out_entirely'][:8]}")
    print(f"[split] wrote {train_path}")
    print(f"[split] wrote {val_path}")
    print(f"[split] wrote {out_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
