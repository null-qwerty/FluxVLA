"""Convert EgoDex episodes to unified EEF action-space records (TASK 2 stage 1).

Each JSONL record stores one ego frame:

  image        extracted frame (relative to --out)
  state        29D absolute EEF state, camera frame:
               [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
  action       29D EEF delta to the next selected frame:
               [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]
  action_mask  29D: 1 for the 12 arm EEF dims (hands/waist are unlabelled)
  task         language instruction from the episode attributes

EgoDex supplies per-frame 4x4 transforms for the camera and both wrists; the
wrist pose is mapped into the camera frame exactly like the RoboCasa robot
data (egoview), so both datasets share one representation. Frames are
resampled from 30 fps to 20 fps to match RoboCasa control frequency.

Usage:
  python tools/convert_egodex_to_eef.py --root /mnt/data/cpfs/users/neri/data/egodex \
      --out /mnt/data/cpfs/users/neri/egodex_eef --fps-sample 20 --max-episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EFF_DIM_MASK = np.zeros(29, dtype=np.float32)
EFF_DIM_MASK[0:6] = 1.0
EFF_DIM_MASK[13:19] = 1.0


def quat_to_mat(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(mat):
    m = np.asarray(mat, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def mat_to_rotvec(mat):
    m = np.asarray(mat, dtype=np.float64)
    cos = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(m) + 1.0) / 2.0, 0.0))
        return axis / max(np.linalg.norm(axis), 1e-12) * theta
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0],
                     m[1, 0] - m[0, 1]]) / (2 * np.sin(theta))
    return axis * theta


def camera_frame_pose(world_pose, camera_pose):
    """world_pose and camera_pose are 4x4; returns 4x4 pose in camera frame."""
    return np.linalg.inv(camera_pose) @ world_pose


def resize_save(array, path, max_side):
    from PIL import Image

    image = Image.fromarray(array)
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, int(image.size[0] * scale)),
             max(1, int(image.size[1] * scale))), Image.BICUBIC)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)


def convert_episode(job):
    hdf5_path, video_path, task, episode_id, out_dir, args = job
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        camera = np.asarray(handle["transforms/camera"][:], dtype=np.float64)
        left = np.asarray(handle["transforms/leftHand"][:], dtype=np.float64)
        right = np.asarray(handle["transforms/rightHand"][:], dtype=np.float64)
    frames = len(camera)
    if frames < 2:
        return None
    src_fps = float(args.ego_fps)
    target_fps = float(args.fps_sample)
    step = src_fps / target_fps
    indices = np.unique(np.round(np.arange(0, frames, step)).astype(int))
    indices = indices[indices < frames]
    if len(indices) < 2:
        return None

    states = np.zeros((len(indices), 29), dtype=np.float32)
    actions = np.zeros((len(indices), 29), dtype=np.float32)
    poses = []
    for slot, frame_idx in enumerate(indices):
        cam_pose = camera[frame_idx]
        left_pose = camera_frame_pose(left[frame_idx], cam_pose)
        right_pose = camera_frame_pose(right[frame_idx], cam_pose)
        poses.append((left_pose, right_pose))
        states[slot, 0:3] = left_pose[:3, 3]
        states[slot, 3:7] = mat_to_quat_wxyz(left_pose[:3, :3])
        states[slot, 13:16] = right_pose[:3, 3]
        states[slot, 16:20] = mat_to_quat_wxyz(right_pose[:3, :3])
    for slot in range(len(indices)):
        next_slot = min(slot + 1, len(indices) - 1)
        for (prev, nxt, delta_slice) in (
                (poses[slot][0], poses[next_slot][0], slice(0, 6)),
                (poses[slot][1], poses[next_slot][1], slice(13, 19))):
            prev_r = prev[:3, :3]
            delta_r = prev_r.T @ nxt[:3, :3]
            delta_t = prev_r.T @ (nxt[:3, 3] - prev[:3, 3])
            actions[slot, delta_slice] = np.concatenate(
                [delta_t, mat_to_rotvec(delta_r)])

    image_dir = Path(out_dir) / "images" / task / str(episode_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    # One ffmpeg pass per episode (fps filter + scale) is far faster than a
    # Python per-frame loop, which dominated the wall clock for EgoDex's 69 GB
    # of 1080p videos. Output frame k corresponds to source frame
    # round(k * ego_fps / fps_sample), matching the index grid above.
    import subprocess

    ffmpeg = "ffmpeg"
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - system ffmpeg fallback
        pass
    raw_dir = image_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video_path,
        "-vf", f"fps={args.fps_sample:g},scale={args.image_max_side}:-2",
        "-q:v", "3", str(raw_dir / "f_%05d.jpg"),
    ]
    subprocess.run(command, check=True)
    files = sorted(raw_dir.glob("f_*.jpg"))
    images = [None] * len(indices)
    for slot in range(min(len(files), len(indices))):
        destination = image_dir / f"frame_{slot:05d}.jpg"
        os.replace(files[slot], destination)
        images[slot] = str(destination.relative_to(out_dir))
    for leftover in files[len(images):]:
        leftover.unlink()
    raw_dir.rmdir()

    description = str(task)
    records = []
    for slot in range(len(indices)):
        if images[slot] is None:
            continue
        records.append({
            "id": f"{task}/{episode_id}/{slot:05d}",
            "episode_id": f"{task}/{episode_id}",
            "image": images[slot],
            "task": description,
            "state": states[slot].tolist(),
            "action": actions[slot].tolist(),
            "action_mask": EFF_DIM_MASK.tolist(),
        })
    return {"records": records, "states": states, "actions": actions}


def discover(root):
    root = Path(root)
    episodes = []
    for hdf5 in sorted(root.rglob("*.hdf5")):
        if "extra" in hdf5.parts:
            continue
        video = hdf5.with_suffix(".mp4")
        if not video.is_file():
            continue
        episodes.append({
            "task": hdf5.parent.name,
            "episode_id": hdf5.stem,
            "hdf5": str(hdf5),
            "video": str(video),
        })
    return episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps-sample", type=float, default=20.0,
                        help="target frame rate (RoboCasa uses 20 Hz)")
    parser.add_argument("--ego-fps", type=float, default=30.0)
    parser.add_argument("--max-episodes", type=int, default=20,
                        help="per task (0 = all)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats-out", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = discover(args.root)
    if args.tasks:
        wanted = set(args.tasks)
        episodes = [e for e in episodes if e["task"] in wanted]
    if args.max_episodes:
        selected, counts = [], {}
        for episode in episodes:
            count = counts.get(episode["task"], 0)
            if count >= args.max_episodes:
                continue
            counts[episode["task"]] = count + 1
            selected.append(episode)
        episodes = selected
    print(f"[ego-eef] {len(episodes)} episodes from {args.root}", flush=True)

    jobs = [(e["hdf5"], e["video"], e["task"], e["episode_id"], str(out_dir),
             args) for e in episodes]
    all_records, state_chunks, action_chunks = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(convert_episode, jobs), 1):
            if result is None:
                continue
            all_records.extend(result["records"])
            state_chunks.append(result["states"])
            action_chunks.append(result["actions"])
            if done % 20 == 0:
                print(f"[ego-eef] {done}/{len(jobs)} episodes", flush=True)

    jsonl_path = out_dir / "ego_eef.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record) + "\n")
    print(f"[ego-eef] wrote {len(all_records)} frames -> {jsonl_path}")

    if args.stats_out and state_chunks:
        states = np.concatenate(state_chunks)
        actions = np.concatenate(action_chunks)
        stats = {
            "proprio": {"mean": states.mean(0).tolist(),
                        "std": states.std(0).tolist(),
                        "min": states.min(0).tolist(),
                        "max": states.max(0).tolist()},
            "action": {"mean": actions.mean(0).tolist(),
                       "std": actions.std(0).tolist(),
                       "min": actions.min(0).tolist(),
                       "max": actions.max(0).tolist()},
            "frames": int(len(states)),
        }
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))
        print(f"[ego-eef] stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
