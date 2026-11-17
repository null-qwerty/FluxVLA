"""Gate 1: validate EEF -> joint inversion on RoboCasa episodes.

For every sampled frame the ground-truth joint action is converted to a
world-frame EEF target with forward kinematics, then the deployment-style
damped-least-squares IK is solved from the current joint state. The report
shows how much joint error the EEF action interface introduces, i.e. the
upper bound of stage-2 controllability before any policy error.

Usage:
  python tools/validate_eef_roundtrip.py --src <task dir> --episodes 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robocasa_eef_fk import GR1Kinematics, joint29_from_parquet_row  # noqa: E402


def locate_parquets(src: Path):
    if (src / "data").is_dir():
        return sorted((src / "data").rglob("*.parquet"))
    return sorted((src / "data").rglob("*.parquet"))


def evaluate_episode(kin, path, stride=5):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = table.schema.names
    state_col = "observation.state.joint" if "observation.state.joint" in names else "observation.state"
    action_col = "action.joint" if "action.joint" in names else "action"
    states = np.stack([joint29_from_parquet_row(v) for v in table.column(state_col).to_pylist()])
    actions = np.stack([joint29_from_parquet_row(v) for v in table.column(action_col).to_pylist()])
    rows = []
    for frame in range(0, len(states), stride):
        state, action = states[frame], actions[frame]
        target_world = kin.eef_poses(action)
        solved = state.copy()
        for side, arm_slice in (("left", slice(0, 7)), ("right", slice(13, 20))):
            solved, residual, iterations = kin.solve_ik(
                side, target_world[side], solved)
            error = np.abs(solved[arm_slice] - action[arm_slice])
            rows.append(dict(
                side=side,
                frame=frame,
                joint_abs_error=error.tolist(),
                joint_abs_error_mean=float(error.mean()),
                pose_residual_norm=float(np.linalg.norm(residual)),
                iterations=int(iterations)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="task dir or parent")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    src = Path(args.src)
    parquets = locate_parquets(src)
    if not parquets:
        raise SystemExit(f"no parquet files under {src}")
    parquets = parquets[:args.episodes]
    kin = GR1Kinematics()
    all_rows = []
    try:
        for path in parquets:
            rows = evaluate_episode(kin, path, stride=args.stride)
            all_rows.extend(rows)
            mean_error = np.mean([r["joint_abs_error_mean"]
                                  for r in rows])
            pose = np.mean([r["pose_residual_norm"] for r in rows])
            print(f"[gate1] {path.name}: frames={len(rows)} "
                  f"joint_mae={mean_error:.4f} rad pose_residual={pose:.4f}")
    finally:
        kin.close()

    errors = np.array([r["joint_abs_error"] for r in all_rows])
    residuals = np.array([r["pose_residual_norm"] for r in all_rows])
    summary = {
        "samples": int(len(all_rows)),
        "joint_mae_rad": float(errors.mean()),
        "joint_p95_rad": float(np.percentile(errors, 95)),
        "joint_max_rad": float(errors.max()),
        "pose_residual_mean": float(residuals.mean()),
        "pose_residual_p95": float(np.percentile(residuals, 95)),
        "per_joint_mae_rad": errors.mean(0).tolist(),
    }
    print("[gate1] " + json.dumps({k: v for k, v in summary.items()
                                  if k != "per_joint_mae_rad"}, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
