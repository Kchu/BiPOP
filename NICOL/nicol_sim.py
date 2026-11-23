#!/usr/bin/env python
"""
Stand-in driver for the NICOL robot in CoppeliaSim.

It covers only what this repo needs to run its experiments in simulation: arm and hand joint control,
arm inverse kinematics, the head's look-at and the eye cameras.

For a more complete implementation, for example ROS integration and control of the physical robot,
see the NICOL API developed by Jan-Gerrit Habekost (Knowledge Technology group, University of Hamburg).

The simulation environment builds on nicol_coppeliasim (https://github.com/knowledgetechnologyuhh/nicol_coppeliasim).

It talks to the running simulator through coppeliasim-zmqremoteapi-client.
The inverse kinematics of the arms is pluggable:

  ik="cycleik"   neural IK, CycleIK (github.com/jangerritha/CycleIK) with the public NICOL weights (default)
  ik="coppelia"  the simulator's own solver (simIK plugin, damped least squares); no extra Python packages
  ik="dls"       damped least squares on the URDF model with pytorch-kinematics; needs a NICOL URDF

Choose with create_nicol(ik=...) or the environment variable NICOL_IK. To add another solver, write a class
with `__init__(self, arm)` and `solve(self, pose) -> 8 joint angles` and register it in IK_SOLVERS.

Usage:
    nicol = create_nicol(ik="coppelia")
    head, left, right, sim = nicol.head(), nicol.left(), nicol.right(), nicol.sim
    left.set_pose_target(NicolPose([x, y, z], [qx, qy, qz, qw]))
"""
import atexit
import math
import os
import sys
import time
from typing import NamedTuple

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8")
# hand joint -> object alias in the scene (after the "l_" / "r_" prefix)
HAND_JOINTS = {"jointT0": "jointT00", "jointT1": "jointT", "jointI1": "jointI", "jointM1": "jointM", "jointLR1": "jointLR"}

# Neck kinematics [m], taken from the NICOL URDF. The yaw joint (head_z) turns about the vertical axis
# through NECK. The pitch joint (head_y) sits PITCH_OFFSET further, in the turned frame. The head looks
# along its x-axis, on a line VIEW_HEIGHT above the pitch axis.
NECK = np.array([0.187, -0.001, 1.3665])
PITCH_OFFSET = np.array([0.007, 0.0, 0.0605])
VIEW_HEIGHT = 0.1
HEAD_RANGE = {"head_y": (-0.583, 1.282), "head_z": (-1.5, 1.5)}


# ----------------------------------------------------------------------------------------------------
# Poses
# ----------------------------------------------------------------------------------------------------

class Point(NamedTuple):
    x: float
    y: float
    z: float

    def as_list(self):
        return list(self)


class Quaternion(NamedTuple):
    """Rotation as a quaternion (x, y, z, w)."""
    x: float
    y: float
    z: float
    w: float

    def as_list(self):
        return list(self)

    def normalized(self):
        norm = math.sqrt(sum(c * c for c in self))
        return [c / norm for c in self]


class NicolPose:
    """Pose in the world frame. The orientation may be left as zeros where only a position matters."""

    def __init__(self, position, orientation=(0.0, 0.0, 0.0, 1.0)):
        self.position = Point(*(float(c) for c in position))
        self.orientation = Quaternion(*(float(c) for c in orientation))

    def __repr__(self):
        return f"NicolPose(position={self.position.as_list()}, orientation={self.orientation.as_list()})"


# ----------------------------------------------------------------------------------------------------
# Joint control
# ----------------------------------------------------------------------------------------------------

class JointGroup:
    """Joints driven by their position controllers in the simulator.

    A blocking move returns once every joint is within its tolerance of the target, or once the joints
    have stopped moving (for example fingers closed on an object), or after a timeout.
    """

    def __init__(self, sim, handles, tolerances, settle_time=0.5, timeout=10.0):
        self.sim = sim
        self.handles = dict(handles)        # joint name -> handle, in a fixed order
        self.tolerances = dict(tolerances)  # joint name -> allowed error [rad]
        self.settle_time, self.timeout = settle_time, timeout

    @property
    def names(self):
        return list(self.handles)

    def read(self, names=None):
        return {name: self.sim.getJointPosition(self.handles[name]) for name in (names or self.handles)}

    def command(self, targets, wait=True):
        """Send targets ({name: angle}, or angles in the group's joint order); returns whether they were reached."""
        if not isinstance(targets, dict):
            targets = dict(zip(self.handles, targets))
        targets = {name: float(angle) for name, angle in targets.items()}
        for name, angle in targets.items():
            self.sim.setJointTargetPosition(self.handles[name], angle)
        return self.wait(targets) if wait else True

    def wait(self, targets):
        start = last_motion = time.monotonic()
        reference = None  # joint angles when the joints were last seen moving
        while True:
            current = self.read(targets)
            if all(abs(current[name] - angle) <= self.tolerances[name] for name, angle in targets.items()):
                return True
            now = time.monotonic()
            if reference is None or any(abs(current[name] - reference[name]) > 1e-3 for name in targets):
                reference, last_motion = current, now
            elif now - last_motion > self.settle_time:
                return False  # stopped short of the target
            if now - start > self.timeout:
                return False


# ----------------------------------------------------------------------------------------------------
# Inverse kinematics for the arms
# ----------------------------------------------------------------------------------------------------

def _nicol_urdf():
    """Text of the NICOL URDF: the file in $NICOL_URDF, otherwise the one shipped with a CycleIK clone."""
    path = os.environ.get("NICOL_URDF")
    if not path:
        try:
            import cycleik_pytorch
        except ImportError:
            raise ImportError('ik="dls" needs a NICOL URDF: set NICOL_URDF=/path/to/NICOL.urdf or install CycleIK, '
                              'which ships one (assets/urdf/NICOL.urdf).') from None
        path = os.path.join(os.path.dirname(os.path.abspath(cycleik_pytorch.__file__)), "..", "assets", "urdf", "NICOL.urdf")
    with open(path) as f:
        return f.read()


class CycleIKSolver:
    """Neural IK: one forward pass of a network trained for the arm. Always answers, also for unreachable poses."""

    def __init__(self, arm):
        from cycleik_pytorch import CycleIK
        self._net = CycleIK(robot="nicol", chain=f"{arm.side}_arm", cuda_device=None)

    def solve(self, pose):
        query = np.array([pose.position.as_list() + pose.orientation.as_list()], dtype=np.float32)
        return [float(q) for q in self._net.inverse_kinematics(query)[0][0]]


class CoppeliaIKSolver:
    """The simulator's built-in solver (simIK plugin): damped least squares on a copy of the arm's
    kinematic chain taken from the scene. Starts from the current joint angles and, if that gets
    stuck, from random configurations."""

    def __init__(self, arm, damping=0.1, iterations=99, attempts=8):
        sim, self._ik, self._arm = arm.sim, arm.client.getObject("simIK"), arm
        self._attempts = attempts
        self._lower, self._upper = arm.joint_limits()
        self._env = self._ik.createEnvironment()
        self._group = self._ik.createGroup(self._env)
        self._ik.setGroupCalculation(self._env, self._group, self._ik.method_damped_least_squares, damping, iterations)
        target = sim.createDummy(0.01)  # only needed while the chain is copied
        try:
            _, sim_to_ik, _ = self._ik.addElementFromScene(self._env, self._group, -1, arm.eef, target, self._ik.constraint_pose)
        finally:
            sim.removeObject(target)
        self._joints = [sim_to_ik[arm.joints.handles[name]] for name in ARM_JOINTS]
        self._target = sim_to_ik[target]
        atexit.register(self._ik.eraseEnvironment, self._env)  # don't leave the copy behind in the simulator

    def solve(self, pose):
        start = self._arm.joint_angles()
        self._ik.setObjectPose(self._env, self._target, self._ik.handle_world, pose.position.as_list() + pose.orientation.normalized())
        rng = np.random.default_rng(0)
        first = None
        for _ in range(self._attempts):  # first from where the arm is now, then from random configurations
            for joint, angle in zip(self._joints, start):
                self._ik.setJointPosition(self._env, joint, float(angle))
            result = self._ik.handleGroup(self._env, self._group, {"syncWorlds": False})[0]
            solution = [self._ik.getJointPosition(self._env, joint) for joint in self._joints]
            if result == self._ik.result_success:
                return solution
            first = first or solution
            start = rng.uniform(self._lower, self._upper)
        print(f"[coppelia IK] pose {np.round(pose.position.as_list(), 3)} not reached exactly, using the closest solution.", file=sys.stderr)
        return first


class DampedLeastSquaresSolver:
    """Numerical IK on the URDF model: damped least squares steps with the Jacobian from pytorch-kinematics,
    inside the joint limits. Starts from the current joint angles (smallest motion) and, in case that
    gets stuck, from random configurations too."""

    def __init__(self, arm, seeds=16, iterations=100, damping=0.05, pos_tolerance=5e-4, rot_tolerance=5e-3):
        import pytorch_kinematics as pk
        import torch
        self._arm, self._torch = arm, torch
        self._chain = pk.build_serial_chain_from_urdf(_nicol_urdf(), f"{arm.side[0]}_laser").to(dtype=torch.float64)
        self._lower, self._upper = arm.joint_limits()
        self._seeds, self._iterations, self._damping = seeds, iterations, damping
        self._tolerance = (pos_tolerance, rot_tolerance)

    def solve(self, pose):
        from scipy.spatial.transform import Rotation
        target_pos, target_rot = np.array(pose.position.as_list()), Rotation.from_quat(pose.orientation.normalized())
        start = np.clip(self._arm.joint_angles(), self._lower, self._upper)
        rng = np.random.default_rng(0)
        q = np.vstack([start, rng.uniform(self._lower, self._upper, (self._seeds - 1, len(ARM_JOINTS)))])
        damping = self._damping ** 2 * np.eye(6)
        for _ in range(self._iterations):
            jacobian, eef = self._chain.jacobian(self._torch.tensor(q), ret_eef_pose=True)
            jacobian, eef = jacobian.numpy(), eef.numpy()
            pos_err = target_pos - eef[:, :3, 3]
            rot_err = (target_rot * Rotation.from_matrix(eef[:, :3, :3]).inv()).as_rotvec()
            pos_dist, rot_dist = np.linalg.norm(pos_err, axis=1), np.linalg.norm(rot_err, axis=1)
            converged = (pos_dist < self._tolerance[0]) & (rot_dist < self._tolerance[1])
            if converged[0]:
                break
            # limit the step, the Jacobian is only valid nearby
            pos_err *= np.minimum(1, 0.1 / np.maximum(pos_dist, 1e-9))[:, None]
            rot_err *= np.minimum(1, 0.5 / np.maximum(rot_dist, 1e-9))[:, None]
            err = np.hstack([pos_err, rot_err])[:, :, None]
            jacobian_t = jacobian.transpose(0, 2, 1)
            step = jacobian_t @ np.linalg.solve(jacobian @ jacobian_t + damping, err)
            q = np.where(converged[:, None], q, np.clip(q + step[:, :, 0], self._lower, self._upper))
        if converged.any():  # of all solutions take the one closest to where the arm is now
            candidates = np.flatnonzero(converged)
            best = candidates[np.argmin(np.linalg.norm(q[candidates] - start, axis=1))]
        else:
            best = np.argmin(pos_dist + 0.1 * rot_dist)
            print(f"[dls IK] pose {np.round(target_pos, 3)} not reached exactly (off by {pos_dist[best]:.3f} m, "
                  f"{rot_dist[best]:.3f} rad), using the closest solution.", file=sys.stderr)
        return [float(angle) for angle in q[best]]


IK_SOLVERS = {"cycleik": CycleIKSolver, "coppelia": CoppeliaIKSolver, "dls": DampedLeastSquaresSolver}


# ----------------------------------------------------------------------------------------------------
# Robot parts
# ----------------------------------------------------------------------------------------------------

class Arm:
    """One 8-joint arm with its 5-joint RH8D hand. The end effector is the laser frame in the palm."""

    def __init__(self, client, side, ik="cycleik"):
        if ik not in IK_SOLVERS:
            raise ValueError(f"Unknown IK solver '{ik}'. Available: {', '.join(IK_SOLVERS)}")
        self.client, self.sim, self.side = client, client.getObject("sim"), side
        prefix = side[0]
        self.joints = JointGroup(self.sim, {name: self.sim.getObject(f"/{prefix}_{name}") for name in ARM_JOINTS},
                                 {name: 0.002 if i < 6 else 0.02 for i, name in enumerate(ARM_JOINTS)})
        self.fingers = JointGroup(self.sim, {name: self.sim.getObject(f"/{prefix}_{alias}") for name, alias in HAND_JOINTS.items()},
                                  dict.fromkeys(HAND_JOINTS, 0.1))
        self.eef = self.sim.getObject(f"/{prefix}_laser_visual")
        self.ik_name, self._solver = ik, None

    @property
    def solver(self):
        if self._solver is None:  # setting up a solver takes a moment, only do it for arms that move
            self._solver = IK_SOLVERS[self.ik_name](self)
        return self._solver

    def joint_angles(self):
        return np.array(list(self.joints.read().values()))

    def joint_limits(self):
        """Lower and upper limits of the 8 arm joints, as set in the scene."""
        intervals = np.array([self.sim.getJointInterval(handle)[1] for handle in self.joints.handles.values()])
        return intervals[:, 0], intervals[:, 0] + intervals[:, 1]

    def solve_ik(self, pose):
        """The 8 arm joint angles that bring the end effector to `pose`."""
        return self.solver.solve(pose)

    def get_eef_pose(self):
        pose = self.sim.getObjectPose(self.eef, self.sim.handle_world)
        return NicolPose(pose[:3], pose[3:])

    def set_pose_target(self, pose, block=True):
        return self.joints.command(self.solve_ik(pose), wait=block)

    def set_joint_position(self, angles, block=True):
        """Arm joint angles, as a list in joint order or a {name: angle} dict."""
        return self.joints.command(angles, wait=block)

    def set_joint_position_for_hand(self, angles, path_time=None, block=True):
        """Hand joint angles: [thumb base, thumb, index, middle, ring + little]. path_time is ignored."""
        return self.fingers.command(angles, wait=block)


class Head:
    """The neck (pitch head_y, yaw head_z) and the two eye cameras."""

    def __init__(self, sim, image_dir, eye_resolution=(1024, 768)):
        self.sim, self.image_dir = sim, image_dir
        self.neck = JointGroup(sim, {"head_y": sim.getObject("/joint_head_y"), "head_z": sim.getObject("/joint_head_z")},
                               {"head_y": 0.002, "head_z": 0.002})
        self.eyes = {"left": sim.getObject("/left_eye_sensor"), "right": sim.getObject("/right_eye_sensor")}
        for eye in self.eyes.values():
            sim.setObjectInt32Param(eye, sim.visionintparam_resolution_x, eye_resolution[0])
            sim.setObjectInt32Param(eye, sim.visionintparam_resolution_y, eye_resolution[1])

    @staticmethod
    def look_at_angles(point):
        """Neck angles (head_y, head_z) that put `point` on the head's line of sight, clipped to the joint range.

        Turning (yaw) does not move the line of sight sideways, so yaw simply faces the point. In the
        vertical plane that is left, the line of sight must pass the point at VIEW_HEIGHT from the pitch axis.
        """
        offset = np.asarray(point, dtype=float) - NECK
        yaw = math.atan2(offset[1], offset[0])
        forward = math.hypot(offset[0], offset[1]) - PITCH_OFFSET[0]
        up = offset[2] - PITCH_OFFSET[2]
        distance = math.hypot(forward, up)
        pitch = math.asin(min(1.0, VIEW_HEIGHT / distance)) - math.atan2(up, forward)
        return float(np.clip(pitch, *HEAD_RANGE["head_y"])), float(np.clip(yaw, *HEAD_RANGE["head_z"]))

    def set_pose_target(self, pose, block=True):
        """Look at the position of `pose`; its orientation is ignored."""
        head_y, head_z = self.look_at_angles(pose.position.as_list())
        return self.neck.command({"head_y": head_y, "head_z": head_z}, wait=block)

    def camera_image(self, eye, save_as=None):
        """RGB image (height, width, 3) of one eye; optionally also saved as <image_dir>/images/<save_as>.png."""
        self.sim.handleVisionSensor(self.eyes[eye])
        data, (width, height) = self.sim.getVisionSensorImg(self.eyes[eye])
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()  # rows come bottom-up
        if save_as:
            from PIL import Image
            folder = os.path.join(self.image_dir, "images")
            os.makedirs(folder, exist_ok=True)
            Image.fromarray(image).save(os.path.join(folder, f"{save_as}.png"))
        return image

    def get_left_eye_camera_img(self, persistent=False, prefix=""):
        return self.camera_image("left", save_as=prefix if persistent else None)

    def get_right_eye_camera_img(self, persistent=False, prefix=""):
        return self.camera_image("right", save_as=f"{prefix}_right" if persistent else None)


class Nicol:
    """Connects to a running CoppeliaSim with NICOL/nicol.ttt loaded and starts the simulation."""

    def __init__(self, port=23000, image_dir=None, ik=None):
        ik = ik or os.environ.get("NICOL_IK", "cycleik")
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.getObject("sim")
        self.left_ = Arm(self.client, "left", ik)
        self.right_ = Arm(self.client, "right", ik)
        self.head_ = Head(self.sim, image_dir or os.getcwd())
        self.sim.startSimulation()
        self.client.setStepping(False)

    def left(self):
        return self.left_

    def right(self):
        return self.right_

    def head(self):
        return self.head_

    def set_pose_target_for_both_arms(self, left_pose, right_pose, block=True):
        """Move both arms at the same time."""
        left_angles, right_angles = self.left_.solve_ik(left_pose), self.right_.solve_ik(right_pose)
        self.left_.joints.command(left_angles, wait=False)
        self.right_.joints.command(right_angles, wait=False)
        if not block:
            return True
        reached_right = self.right_.joints.wait(dict(zip(ARM_JOINTS, right_angles)))
        reached_left = self.left_.joints.wait(dict(zip(ARM_JOINTS, left_angles)))
        return reached_left and reached_right


def create_nicol(port=23000, image_dir=None, ik=None):
    """ik: one of IK_SOLVERS ("cycleik", "coppelia", "dls"); default: $NICOL_IK or "cycleik".
    image_dir: eye-camera images saved with persistent=True go to <image_dir>/images (default: current directory)."""
    return Nicol(port=port, image_dir=image_dir, ik=ik)
