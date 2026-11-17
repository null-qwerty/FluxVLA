"""Convert RoboCasa GR1 lerobot parquet files to the unified EEF action space.

Output parquet keeps the same row/video layout but rewrites two columns:

  observation.state : [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
  action            : [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]

where the EEF pose is expressed in the head-camera (egoview) frame at the
current frame and the delta is a 6D (translation + rotation-vector) delta in
the previous EEF frame. Hand/waist dimensions are copied unchanged (Fourier
hand actions are 6D commands; the 6->11 joint expansion is applied by the
robot controller, not the data).

The original joint-space values are preserved in additional columns
(``observation.state.joint`` / ``action.joint``) so IK-based deployment and
offline diagnostics can use them.

Usage (one task directory per call; videos are symlinked, meta is copied):

    python tools/convert_robocasa_to_eef.py \
        --src /mnt/data/cpfs/users/neri/data/robocasa_lerobot_V2.1/PnPBottleToCabinetClose \
        --dst /mnt/data/cpfs/users/neri/data/robocasa_eef_V0.1/PnPBottleToCabinetClose \
        --workers 8

``--stats-out`` collects per-dimension moment/quantile statistics over every
converted frame so a training recipe can normalize the new space.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robocasa_eef_fk import (  # noqa: E402
    GR1Kinematics, apply_eef_delta, eef_abs_vector, eef_delta_vector,
    joint29_from_parquet_row, to_camera_frame)

STATE_COL = "observation.state"
ACTION_COL = "action"


def convert_episode(path, kin):
    """Return (new_state, new_action, old_state, old_action) arrays."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    states = np.stack([joint29_from_parquet_row(v) for v in
                       table.column(STATE_COL).to_pylist()])
    actions = np.stack([joint29_from_parquet_row(v) for v in
                        table.column(ACTION_COL).to_pylist()])
    n = len(states)
    new_state = np.zeros_like(states)
    new_action = np.zeros_like(actions)
    for i in range(n):
        cam, eef = kin.state_and_eef(states[i])
        cam_next, eef_next = kin.state_and_eef(actions[i])
        for side, (abs_slice, delta_slice) in (
                ("left", (slice(0, 7), slice(0, 6))),
                ("right", (slice(13, 20), slice(13, 19)))):
            pose = to_camera_frame(eef[side], cam)
            pose_next = to_camera_frame(eef_next[side], cam)
            new_state[i, abs_slice] = eef_abs_vector(cam, pose)
            new_action[i, delta_slice] = eef_delta_vector(pose, pose_next)
        # hands + waist are unchanged
        new_state[i, 7:13] = states[i, 7:13]
        new_state[i, 20:26] = states[i, 20:26]
        new_state[i, 26:29] = states[i, 26:29]
        new_action[i, 7:13] = actions[i, 7:13]
        new_action[i, 20:26] = actions[i, 20:26]
        new_action[i, 26:29] = actions[i, 26:29]
    return new_state, new_action, table


def convert_file(job):
    src, dst, = job
    kin = GR1Kinematics()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        new_state, new_action, table = convert_episode(src, kin)
        arrays = {name: table.column(name) for name in table.schema.names}
        arrays[STATE_COL] = pa.array([row.tolist() for row in new_state],
                                     type=table.schema.field(STATE_COL).type)
        arrays[ACTION_COL] = pa.array([row.tolist() for row in new_action],
                                      type=table.schema.field(ACTION_COL).type)
        arrays["observation.state.joint"] = pa.array(
            [row.tolist() for row in np.stack([
                joint29_from_parquet_row(v) for v in
                table.column(STATE_COL).to_pylist()])],
            type=table.schema.field(STATE_COL).type)
        arrays["action.joint"] = pa.array(
            [row.tolist() for row in np.stack([
                joint29_from_parquet_row(v) for v in
                table.column(ACTION_COL).to_pylist()])],
            type=table.schema.field(ACTION_COL).type)
        out = pa.table(arrays)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, dst)
        return {"file": dst, "frames": len(new_state),
                "state_abs": new_state, "action_delta": new_action}
    finally:
        kin.close()


def link_videos(src_task, dst_task):
    src_videos = Path(src_task) / "videos"
    dst_videos = Path(dst_task) / "videos"
    if not src_videos.is_dir():
        return
    for chunk in sorted(src_videos.iterdir()):
        for cam in sorted(chunk.iterdir()):
            target = dst_videos / chunk.name / cam.name
            target.mkdir(parents=True, exist_ok=True)
            for video in sorted(cam.iterdir()):
                link = target / video.name
                if not link.exists():
                    os.symlink(video.resolve(), link)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="task directory")
    parser.add_argument("--dst", required=True, help="output task directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="convert at most N episodes (debug)")
    parser.add_argument("--stats-out", default=None,
                        help="write EEF statistics json here")
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args(argv)

    src_task = Path(args.src)
    dst_task = Path(args.dst)
    data_dirs = sorted((src_task / "data").iterdir())
    jobs = []
    for chunk in data_dirs:
        for parquet in sorted(chunk.glob("*.parquet")):
            rel = parquet.relative_to(src_task)
            jobs.append((str(parquet), str(dst_task / rel)))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[convert] {src_task.name}: {len(jobs)} episodes", flush=True)

    (dst_task / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl",
                 "episodes_stats.jsonl"):
        src = src_task / "meta" / name
        if src.is_file():
            shutil.copy2(src, dst_task / "meta" / name)
    if not args.no_videos:
        link_videos(src_task, dst_task)

    states, actions = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(convert_file, jobs):
            states.append(result["state_abs"])
            actions.append(result["action_delta"])
            done += 1
            if done % 50 == 0:
                print(f"[convert] {done}/{len(jobs)}", flush=True)

    if args.stats_out:
        state = np.concatenate(states)
        action = np.concatenate(actions)
        stats = {
            "proprio": {"mean": state.mean(0).tolist(),
                        "std": state.std(0).tolist(),
                        "min": state.min(0).tolist(),
                        "max": state.max(0).tolist()},
            "action": {"mean": action.mean(0).tolist(),
                       "std": action.std(0).tolist(),
                       "min": action.min(0).tolist(),
                       "max": action.max(0).tolist()},
            "frames": int(len(state)),
        }
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))
        print(f"[convert] stats -> {args.stats_out}")
    print(f"[convert] done {dst_task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
