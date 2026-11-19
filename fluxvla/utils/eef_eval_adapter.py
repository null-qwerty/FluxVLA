# Copyright 2026 Limx Dynamics
"""Deployment adapter: unified EEF action space <-> RoboCasa 29D joint actions.

The EEF training arms (TASK 2/5/6, see docs/eef_action_space.md) predict a 29D
vector in the egoview camera frame:

    state  : [L_eef_abs(7), L_hand(6), R_eef_abs(7), R_hand(6), waist(3)]
    action : [L_eef_delta(6), L_hand(6), R_eef_delta(6), R_hand(6), waist(3)]

RoboCasa's environment however takes absolute 29D joint targets
(action.left_arm ... action.waist). This adapter bridges the two at evaluation
time, with the same conventions the offline conversion used
(tools/convert_robocasa_to_eef.py):

* observation: joint state -> FK -> EEF poses -> world-to-camera -> position +
  quaternion (x, y, z, qw, qx, qy, qz). Hands and waist pass through unchanged.
* action: EEF delta -> apply to the current camera-frame pose -> back to the
  world frame -> damped least-squares IK per arm -> absolute joint targets.
  Hands (absolute 6D command) and waist pass through unchanged.

FK/IK use GR1Kinematics (robosuite robot model only, no task assets), which is
the same code path validated by Gate 1 (joint -> EEF -> IK -> joint, MAE
0.011-0.013 rad).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fluxvla.utils.gr1_kinematics import (GR1Kinematics, apply_eef_delta,
                                          eef_abs_vector, pose_compose,
                                          to_camera_frame)

STATE_DIM = 29
# Slices of the unified 29D vector (identical to the training layout).
ARM_SLICES = {"left": slice(0, 7), "right": slice(13, 20)}
ARM_DELTA_SLICES = {"left": slice(0, 6), "right": slice(13, 19)}
HAND_SLICES = {"left": slice(7, 13), "right": slice(20, 26)}
WAIST_SLICE = slice(26, 29)


class EEFActionAdapter:
    """Convert between 29D joint state/actions and the unified EEF space.

    Args:
        ik_iters: Damped least-squares iterations per arm (60 reaches the
            Gate-1 tolerance; raise it if the residual log shows >1e-3).
        ik_damping: DLS damping factor.
    """

    def __init__(self,
                 ik_iters: int = 60,
                 ik_damping: float = 0.05) -> None:
        self.ik_iters = int(ik_iters)
        self.ik_damping = float(ik_damping)
        self._kin: Optional[GR1Kinematics] = None
        self.last_residual = 0.0

    # ------------------------------------------------------------ kinematics
    @property
    def kinematics(self) -> GR1Kinematics:
        if self._kin is None:
            self._kin = GR1Kinematics()
        return self._kin

    def close(self) -> None:
        if self._kin is not None:
            self._kin.close()
            self._kin = None

    # ---------------------------------------------------------- observations
    def state_from_joint(self, joint29) -> np.ndarray:
        """29D joint observation -> 29D absolute EEF state (camera frame)."""
        joint29 = np.asarray(joint29, dtype=np.float64).reshape(-1)
        if joint29.shape[0] != STATE_DIM:
            raise ValueError(
                f"joint state must have {STATE_DIM} dims, got {joint29.shape}")
        cam, eef_world = self.kinematics.state_and_eef(joint29)
        # Keep the dataset layout: left arm, left hand, right arm, right hand.
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[ARM_SLICES["left"]] = eef_abs_vector(
            cam, to_camera_frame(eef_world["left"], cam))
        state[HAND_SLICES["left"]] = joint29[HAND_SLICES["left"]]
        state[ARM_SLICES["right"]] = eef_abs_vector(
            cam, to_camera_frame(eef_world["right"], cam))
        state[HAND_SLICES["right"]] = joint29[HAND_SLICES["right"]]
        state[WAIST_SLICE] = joint29[WAIST_SLICE]
        return state

    # --------------------------------------------------------------- actions
    def action_to_joint(self, delta29, joint29) -> np.ndarray:
        """29D EEF delta (denormalised) -> 29D absolute joint targets.

        joint29 must be the joint state used to build the policy observation
        (every action in a chunk is relative to that snapshot).
        """
        delta29 = np.asarray(delta29, dtype=np.float64).reshape(-1)
        joint29 = np.asarray(joint29, dtype=np.float64).reshape(-1)
        if delta29.shape[0] < STATE_DIM:
            raise ValueError(
                f"action must have >= {STATE_DIM} dims, got {delta29.shape}")
        if joint29.shape[0] != STATE_DIM:
            raise ValueError(
                f"joint state must have {STATE_DIM} dims, got {joint29.shape}")

        cam, eef_world = self.kinematics.state_and_eef(joint29)
        targets = joint29.copy()
        residual = 0.0
        for side in ("left", "right"):
            previous = to_camera_frame(eef_world[side], cam)
            desired_cam = apply_eef_delta(
                previous, delta29[ARM_DELTA_SLICES[side]])
            desired_world = pose_compose(cam, desired_cam)
            targets, res, _ = self.kinematics.solve_ik(
                side,
                desired_world,
                joint29=targets,
                iters=self.ik_iters,
                damping=self.ik_damping)
            residual = max(residual, float(np.linalg.norm(res)))
        self.last_residual = residual
        # Hands (absolute 6D command) and waist (absolute) pass through.
        targets[HAND_SLICES["left"]] = delta29[HAND_SLICES["left"]]
        targets[HAND_SLICES["right"]] = delta29[HAND_SLICES["right"]]
        targets[WAIST_SLICE] = delta29[WAIST_SLICE]
        return targets.astype(np.float32)

    # ------------------------------------------------------------------ misc
    @staticmethod
    def obs_with_eef_state(obs: Dict, eef_state: np.ndarray) -> Dict:
        """Copy obs with the state keys replaced by the EEF state.

        ProcessRobocasaEvalInputs concatenates state.left_arm ... state.waist
        in exactly this order, so the standard eval transform normalises the
        EEF state with the EEF statistics without any change.
        """
        out = dict(obs)
        out["state.left_arm"] = eef_state[ARM_SLICES["left"]]
        out["state.left_hand"] = eef_state[HAND_SLICES["left"]]
        out["state.right_arm"] = eef_state[ARM_SLICES["right"]]
        out["state.right_hand"] = eef_state[HAND_SLICES["right"]]
        out["state.waist"] = eef_state[WAIST_SLICE]
        return out
