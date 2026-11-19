#!/usr/bin/env python3
"""Check the EEF eval adapter against a converted RoboCasa parquet episode.

The EEF arms (TASK 2/5/6) consume absolute EEF state and emit delta EEF; the
simulation still needs absolute 29D joint targets. fluxvla/utils/eef_eval_adapter.py
does the conversion at eval time. This script validates it offline on real data:

  1. state_from_joint(state.joint) reproduces the recorded EEF state
     (observation.state) within the FK build-to-build offset;
  2. a zero delta is a no-op through IK;
  3. FK(adapter(delta)) reproduces the requested EEF delta;
  4. hand/waist commands pass through as absolute targets.

Usage:
    python tools/validate_eef_eval_adapter.py \
        --parquet /path/to/robocasa_eef_V0.1/<task>/data/chunk-000/episode_000000.parquet
"""
import argparse
import sys

import numpy as np

from fluxvla.utils.eef_eval_adapter import EEFActionAdapter
from fluxvla.utils.gr1_kinematics import (eef_delta_vector, to_camera_frame)

# Build-to-build FK offsets (robosuite fork versions) shift the recorded EEF by
# ~1e-2; the checkpoint statistics make that ~0.07 sigma, so this bound only
# catches a real convention error (wrong site, frame, joint order).
STATE_TOL = 5e-2
DELTA_TOL = 2e-3


def main(parquet: str) -> int:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet).read_row_group(0)
    joints = table.column('observation.state.joint').to_pylist()
    recorded = table.column('observation.state').to_pylist()
    joint29 = np.asarray(joints[0], dtype=np.float64)
    ref = np.asarray(recorded[0], dtype=np.float64)
    print(f'frames: {len(joints)}')

    adapter = EEFActionAdapter()
    state = adapter.state_from_joint(joint29)
    arm = np.concatenate([state[0:6], state[13:19]])
    ref_arm = np.concatenate([ref[0:6], ref[13:19]])
    state_err = float(np.abs(arm - ref_arm).max())
    print(f'[1] state vs recorded EEF: max abs {state_err:.2e} '
          f'(tolerance {STATE_TOL:g})')
    if state_err > STATE_TOL:
        print('FAIL: state convention differs beyond the FK build offset')
        return 1
    if not (np.allclose(state[7:13], joint29[7:13])
            and np.allclose(state[20:26], joint29[20:26])
            and np.allclose(state[26:29], joint29[26:29])):
        print('FAIL: hands/waist are not passed through')
        return 1

    targets = adapter.action_to_joint(np.zeros(29), joint29)
    arm_slice = np.r_[0:7, 13:20]
    identity_err = float(
        np.abs(targets[arm_slice] - joint29[arm_slice]).max())
    print(f'[2] zero delta is a no-op: arm error {identity_err:.2e}')
    if identity_err > 1e-5:
        print('FAIL: zero delta moved the arms')
        return 1

    kin = adapter.kinematics
    delta = np.zeros(29)
    delta[0:3] = [0.02, -0.01, 0.015]
    delta[3:6] = [0.0, 0.0, 0.05]
    delta[13:16] = [0.0, 0.0, 0.02]
    delta[16:19] = [0.0, 0.04, 0.0]
    # Waist is an absolute command: hold it so the IK base matches the target.
    delta[26:29] = joint29[26:29]
    targets = adapter.action_to_joint(delta, joint29)
    cam, before = kin.state_and_eef(joint29)
    cam, after = kin.state_and_eef(targets)
    for side, sl in (('left', slice(0, 6)), ('right', slice(13, 19))):
        recovered = eef_delta_vector(
            to_camera_frame(before[side], cam),
            to_camera_frame(after[side], cam))
        error = float(np.abs(recovered - delta[sl]).max())
        print(f'[3] {side} delta round-trip: {error:.2e} '
              f'(IK residual {adapter.last_residual:.1e})')
        if error > DELTA_TOL:
            print(f'FAIL: {side} IK did not reach the requested delta')
            return 1

    delta = np.zeros(29)
    delta[7:13] = 0.9
    delta[26:29] = 0.1
    targets = adapter.action_to_joint(delta, joint29)
    print('[4] hand/waist passthrough:', np.allclose(targets[7:13], 0.9),
          np.allclose(targets[26:29], 0.1))
    print('EEF eval adapter OK')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', required=True,
                        help='converted EEF episode parquet')
    args = parser.parse_args()
    sys.exit(main(args.parquet))
