"""GR1 (RoboCasa) kinematic helpers: joint space <-> EEF space.

Used by the unified EEF action-space recipe (docs/eef_action_space.md):

  robot side: 29D absolute joint targets --FK--> absolute EEF (base frame)
              --transform--> camera (egoview) frame --difference--> delta EEF

  ego side  : EgoDex wrist 4x4 transforms are already metric EEF poses;
              they are mapped into the same canonical camera frame.

The model is the same robosuite GR1ArmsAndWaistFourierHands robot used by
RoboCasa; no RoboCasa task assets are needed because the EEF/camera poses only
depend on the robot kinematics. The egoview camera offset is taken from
robocasa/environments/tabletop/tabletop.py:set_cameras (mounted on
robot0_head_pitch, fovy=77).

CLI:
    python tools/robocasa_eef_fk.py --selftest
"""
from __future__ import annotations

import argparse
import numpy as np

# ---------------------------------------------------------------- constants

# Dataset action/state vector layout (RoboCasa GR1 lerobot V2.1).
ARM_JOINTS = {
    "left": [
        "robot0_l_shoulder_pitch", "robot0_l_shoulder_roll",
        "robot0_l_shoulder_yaw", "robot0_l_elbow_pitch",
        "robot0_l_wrist_yaw", "robot0_l_wrist_roll",
        "robot0_l_wrist_pitch",
    ],
    "right": [
        "robot0_r_shoulder_pitch", "robot0_r_shoulder_roll",
        "robot0_r_shoulder_yaw", "robot0_r_elbow_pitch",
        "robot0_r_wrist_yaw", "robot0_r_wrist_roll",
        "robot0_r_wrist_pitch",
    ],
}
WAIST_JOINTS = [
    "robot0_torso_waist_yaw", "robot0_torso_waist_pitch",
    "robot0_torso_waist_roll",
]
# 6D hand command -> 11 actuated joints (Fourier hand format_action).
HAND_EXPAND = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
HAND_JOINTS = {
    "right": [
        "gripper0_right_R_thumb_proximal_yaw_joint",
        "gripper0_right_R_thumb_proximal_pitch_joint",
        "gripper0_right_R_thumb_distal_joint",
        "gripper0_right_R_index_proximal_joint",
        "gripper0_right_R_index_intermediate_joint",
        "gripper0_right_R_middle_proximal_joint",
        "gripper0_right_R_middle_intermediate_joint",
        "gripper0_right_R_ring_proximal_joint",
        "gripper0_right_R_ring_intermediate_joint",
        "gripper0_right_R_pinky_proximal_joint",
        "gripper0_right_R_pinky_intermediate_joint",
    ],
    "left": [
        "gripper0_left_L_thumb_proximal_yaw_joint",
        "gripper0_left_L_thumb_proximal_pitch_joint",
        "gripper0_left_L_thumb_distal_joint",
        "gripper0_left_L_index_proximal_joint",
        "gripper0_left_L_index_intermediate_joint",
        "gripper0_left_L_middle_proximal_joint",
        "gripper0_left_L_middle_intermediate_joint",
        "gripper0_left_L_ring_proximal_joint",
        "gripper0_left_L_ring_intermediate_joint",
        "gripper0_left_L_pinky_proximal_joint",
        "gripper0_left_L_pinky_intermediate_joint",
    ],
}
ARM_JOINT_SUFFIXES = {
    side: [name.split("robot0_", 1)[1] for name in joints]
    for side, joints in ARM_JOINTS.items()
}
WAIST_JOINTS_SUFFIXES = [name.split("robot0_", 1)[1] for name in WAIST_JOINTS]
EEF_SITES = {
    "left": "gripper0_left_grip_site",
    "right": "gripper0_right_grip_site",
}
HEAD_BODY = "robot0_head_pitch"
# robocasa tabletop.py set_cameras(): camera mounted on robot0_head_pitch.
EGOVIEW_POS = np.array([
    2.650 - 2.65017178 + 0.23, -1.944 + 2.174 - 0.23, 1.538 - 1.4475,
])
EGOVIEW_QUAT_WXYZ = np.array([0.676, 0.205, -0.205, -0.676])

# 29D layout: [left_arm(7), left_hand(6), right_arm(7), right_hand(6), waist(3)].
LAYOUT = (
    ("left_arm", 0, 7),
    ("left_hand", 7, 13),
    ("right_arm", 13, 20),
    ("right_hand", 20, 26),
    ("waist", 26, 29),
)


# ---------------------------------------------------------------- rotations

def quat_wxyz_to_mat(quat):
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
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


def rotvec_to_mat(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def mat_to_rotvec(mat):
    m = np.asarray(mat, dtype=np.float64)
    cos = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # near-pi: extract axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(m) + 1.0) / 2.0, 0.0))
        return axis / max(np.linalg.norm(axis), 1e-12) * theta
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0],
                     m[1, 0] - m[0, 1]]) / (2 * np.sin(theta))
    return axis * theta


def pose_compose(a, b):
    """Compose (R,t) poses: result = a * b."""
    ra, ta = a
    rb, tb = b
    return ra @ rb, ta + ra @ tb


def pose_inverse(p):
    r, t = p
    rt = r.T
    return rt, -rt @ t


# ---------------------------------------------------------------- kinematics

class GR1Kinematics:
    """Forward kinematics for the RoboCasa GR1 robot (no task assets needed)."""

    def __init__(self):
        import os
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        import robosuite
        import robocasa  # noqa: F401  registers GR1*FourierHands robots
        self.env = robosuite.make(
            env_name="Lift",
            robots="GR1ArmsAndWaistFourierHands",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            hard_reset=False)
        self.env.reset()
        self.sim = self.env.sim
        model = self.sim.model
        self.qpos_adr = {
            name: int(model.jnt_qposadr[i])
            for i, name in enumerate(model.joint_names)
        }
        self.site_id = {
            name: model.site_name2id(name) for name in EEF_SITES.values()
        }
        self.body_id = model.body_name2id(HEAD_BODY)
        # robot0_ (robot) joints are the ones we write; cube_joint0 is scene.
        self.qpos = np.zeros(model.nq)
        self.extra_qpos = {
            name: model.jnt_qposadr[i]
            for i, name in enumerate(model.joint_names)
            if not (name.startswith("robot0_") or name.startswith("gripper0_"))
        }

    def close(self):
        self.env.close()

    def _set_qpos(self, joint29):
        """Write a 29D joint vector (dataset layout) into the MuJoCo qpos."""
        joint29 = np.asarray(joint29, dtype=np.float64).reshape(-1)
        assert joint29.shape == (29,), joint29.shape
        qpos = np.zeros(self.sim.model.nq)
        for name, start, end in LAYOUT:
            if name == "waist":
                for offset, joint in enumerate(WAIST_JOINTS):
                    qpos[self.qpos_adr[joint]] = joint29[start + offset]
            elif name.endswith("_arm"):
                side = name.split("_")[0]
                for offset, joint in enumerate(ARM_JOINTS[side]):
                    qpos[self.qpos_adr[joint]] = joint29[start + offset]
            elif name.endswith("_hand"):
                side = name.split("_")[0]
                hand = joint29[start:end]
                expanded = hand[HAND_EXPAND]
                for offset, joint in enumerate(HAND_JOINTS[side]):
                    qpos[self.qpos_adr[joint]] = expanded[offset]
        self.sim.data.qpos[:] = qpos
        import mujoco
        mujoco.mj_forward(self.sim.model._model, self.sim.data._data)

    def _pose_of(self, kind, ident):
        import mujoco
        data = self.sim.data
        model = self.sim.model
        if kind == "site":
            pos = np.array(data.site_xpos[ident])
            mat = np.array(data.site_xmat[ident]).reshape(3, 3)
        elif kind == "body":
            pos = np.array(data.body_xpos[ident])
            mat = np.array(data.body_xmat[ident]).reshape(3, 3)
        else:
            raise ValueError(kind)
        return mat, pos

    def camera_pose(self):
        """World pose of the egoview camera (mounted on robot0_head_pitch)."""
        head = self._pose_of("body", self.body_id)
        local = (quat_wxyz_to_mat(EGOVIEW_QUAT_WXYZ), EGOVIEW_POS)
        return pose_compose(head, local)

    def eef_poses(self, joint29):
        """Absolute world-frame EEF poses for both grippers."""
        self._set_qpos(joint29)
        return {side: self._pose_of("site", self.site_id[EEF_SITES[side]])
                for side in ("left", "right")}

    def arm_dof_indices(self, side):
        """MuJoCo dof indices of one 7-joint arm."""
        indices = []
        for joint in ARM_JOINTS[side]:
            jid = self.sim.model.joint_name2id(joint)
            indices.append(int(self.sim.model.jnt_dofadr[jid]))
        return np.asarray(indices, dtype=np.int64)

    def solve_ik(self, side, target_pose_world, joint29=None, iters=60,
                 damping=0.05, step_clip=0.15, tol=1e-6):
        """Damped least-squares IK for one 7-DOF arm.

        Args:
            side: 'left' or 'right'.
            target_pose_world: (R, t) world-frame EEF pose to reach.
            joint29: starting 29D joint vector (defaults to zeros).
        Returns:
            tuple(updated 29D joint vector, final 6D pose residual, iterations)
        """
        import mujoco

        joint29 = (np.zeros(29) if joint29 is None
                   else np.asarray(joint29, dtype=np.float64).copy())
        target_r, target_t = target_pose_world
        arm_columns = self.arm_dof_indices(side)
        site = self.site_id[EEF_SITES[side]]
        self._set_qpos(joint29)
        model = self.sim.model._model
        data = self.sim.data._data
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        residual = np.zeros(6)
        for iteration in range(1, iters + 1):
            current_r, current_t = self._pose_of("site", site)
            dp = target_t - current_t
            rot_error = target_r @ current_r.T
            dr = mat_to_rotvec(rot_error)
            residual = np.concatenate([dp, dr])
            if np.linalg.norm(residual) < tol:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            jacobian = np.vstack([jacp, jacr])[:, arm_columns]
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
            normal = jacobian @ jacobian.T + (damping ** 2) * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(normal, residual)
            dq = np.clip(dq, -step_clip, step_clip)
            for local, joint in enumerate(ARM_JOINTS[side]):
                joint29[self._joint29_index(joint)] += dq[local]
            self._set_qpos(joint29)
        return joint29, residual, iteration

    def _joint29_index(self, joint_name):
        prefix = "robot0_"
        short = joint_name[len(prefix):] if joint_name.startswith(prefix) else joint_name
        for name, start, end in LAYOUT:
            if name == "waist" and short in WAIST_JOINTS_SUFFIXES:
                return start + WAIST_JOINTS_SUFFIXES.index(short)
            if name.endswith("_arm"):
                side = name.split("_")[0]
                suffixes = ARM_JOINT_SUFFIXES[side]
                if short in suffixes:
                    return start + suffixes.index(short)
        raise KeyError(joint_name)

    def state_and_eef(self, joint29):
        """Return camera pose + EEF poses (world) for one 29D joint vector."""
        cam = self.camera_pose()
        eef = self.eef_poses(joint29)
        return cam, eef


# ---------------------------------------------------------------- conversions

def to_camera_frame(pose_world, cam_world):
    """Express a world pose in the camera frame."""
    return pose_compose(pose_inverse(cam_world), pose_world)


def eef_abs_vector(cam, eef_cam):
    """Flatten an EEF pose into 7D (pos3 + quat4 wxyz)."""
    rot, pos = eef_cam
    return np.concatenate([pos, mat_to_quat_wxyz(rot)])


def eef_delta_vector(prev_eef_cam, next_eef_cam):
    """Flatten the delta between two camera-frame poses into 6D.

    Translation delta is expressed in the *previous* EEF frame; rotation delta
    is a rotation vector in the previous EEF frame (shortest path).
    """
    prev_r, prev_t = prev_eef_cam
    next_r, next_t = next_eef_cam
    delta_r = prev_r.T @ next_r
    delta_t = prev_r.T @ (next_t - prev_t)
    return np.concatenate([delta_t, mat_to_rotvec(delta_r)])


def apply_eef_delta(prev_eef_cam, delta6):
    """Inverse of eef_delta_vector: apply a 6D delta to a camera-frame pose."""
    prev_r, prev_t = prev_eef_cam
    delta_t = delta6[:3]
    delta_r = rotvec_to_mat(delta6[3:])
    return prev_r @ delta_r, prev_t + prev_r @ delta_t


def joint29_from_parquet_row(proprio):
    """Slice the 29D joint vector out of the dataset proprio column."""
    proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
    if proprio.shape[0] < 29:
        raise ValueError(f"proprio has {proprio.shape[0]} dims, need >= 29")
    return proprio[:29]


# ---------------------------------------------------------------- self test

def selftest(seed=0, samples=5):
    rng = np.random.default_rng(seed)
    kin = GR1Kinematics()
    errors = []
    try:
        for _ in range(samples):
            # stay inside a plausible joint range for the arms
            joint_a = np.zeros(29)
            joint_a[0:7] = rng.uniform(-1.2, 1.2, 7)
            joint_a[0 + 3] = rng.uniform(-1.6, -0.6)
            joint_a[13:20] = rng.uniform(-1.2, 1.2, 7)
            joint_a[13 + 3] = rng.uniform(-1.6, -0.6)
            joint_a[26:29] = rng.uniform(-0.4, 0.4, 3)
            joint_b = joint_a.copy()
            joint_b[0:7] += rng.uniform(-0.15, 0.15, 7)
            joint_b[13:20] += rng.uniform(-0.15, 0.15, 7)
            joint_b[26:29] += rng.uniform(-0.1, 0.1, 3)
            cam_a, eef_a = kin.state_and_eef(joint_a)
            cam_b, eef_b = kin.state_and_eef(joint_b)
            for side in ("left", "right"):
                abs_a = eef_abs_vector(cam_a, to_camera_frame(eef_a[side], cam_a))
                abs_b = eef_abs_vector(cam_b, to_camera_frame(eef_b[side], cam_b))
                # absolute EEF must be 7D with unit quaternion
                assert abs_a.shape == (7,), abs_a.shape
                assert abs(np.linalg.norm(abs_a[3:]) - 1.0) < 1e-6
                delta = eef_delta_vector(
                    to_camera_frame(eef_a[side], cam_a),
                    to_camera_frame(eef_b[side], cam_b))
                back = apply_eef_delta(
                    to_camera_frame(eef_a[side], cam_a), delta)
                err_t = np.linalg.norm(back[1] - to_camera_frame(
                    eef_b[side], cam_b)[1])
                err_r = np.linalg.norm(back[0] - to_camera_frame(
                    eef_b[side], cam_b)[0])
                errors.append((err_t, err_r))
        worst_t = max(e[0] for e in errors)
        worst_r = max(e[1] for e in errors)
        print(f"[selftest] samples={samples * 2} max_translation_err={worst_t:.3e} "
              f"max_rotation_err={worst_r:.3e}")
        assert worst_t < 1e-9 and worst_r < 1e-9
        print("[selftest] delta <-> absolute composition roundtrip OK")
        # sanity: camera is on the head, EEF in front of the robot
        cam_a, eef_a = kin.state_and_eef(np.zeros(29))
        print("[selftest] camera pos", np.round(cam_a[1], 3),
              "left eef pos", np.round(eef_a["left"][1], 3),
              "right eef pos", np.round(eef_a["right"][1], 3))
    finally:
        kin.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.selftest:
        selftest(seed=args.seed)
        return 0
    parser.error("nothing to do; pass --selftest")


if __name__ == "__main__":
    raise SystemExit(main())
