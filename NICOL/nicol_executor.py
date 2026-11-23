#!/usr/bin/env python
"""
Robot skills of NICOL in CoppeliaSim, executed for one hand per process.

nicol_controller.py starts one process per hand (and one for the head) for
every plan step, so the two hands act in parallel:

    python NICOL/nicol_executor.py --side left --command side_grasp --para '{"obj_name": "blue_cup"}' \
        --hand_state Vertical --finger_state Open

The process prints ``LEFT_RESULT: <message>. LEFT_HAND: <state> LEFT_FINGER: <state>``
(``RIGHT_...`` for the right hand), which the controller parses. A message
containing "failed" marks a failed step.

Needs CoppeliaSim running with NICOL/nicol.ttt loaded; the arm IK solver is
chosen with the NICOL_IK environment variable (see nicol_sim.py).
"""
import argparse
import json
import math
import time

import numpy as np
from transforms3d.euler import euler2quat

from nicol_sim import NicolPose, create_nicol

# =====================================================================
# INITIALIZATION
# =====================================================================
nicol = create_nicol()
head = nicol.head()
left = nicol.left()
right = nicol.right()
sim = nicol.sim

# Sensor and attach point handles
left_sensor = sim.getObject('/l_sensor')
left_attachpoint = sim.getObject('/l_palm_attachPoint')
right_sensor = sim.getObject('/r_sensor')
right_attachpoint = sim.getObject('/r_palm_attachPoint')

# =====================================================================
# ORIENTATION DICTIONARIES
# =====================================================================
ORIENTATION_DICT_LEFT = {
    'Horizontally_Up': euler2quat(0, np.pi/2, np.pi/2),
    'Horizontally_Down': euler2quat(np.pi, -np.pi/2, np.pi/2),
    'Vertical': euler2quat(np.pi/2, np.pi, np.pi),
    'Horizontally_Slanted_Up': euler2quat(2*np.pi/3, 5*np.pi/6, np.pi),
    'Horizontally_Slanted_Down': euler2quat(-np.pi, -np.pi/4, np.pi*5/8),
    'Vertically_Slanted': euler2quat(np.pi/2, np.pi, np.pi*4/3),
    'Vertically_Down': euler2quat(np.pi, -np.pi/8, np.pi/2)
}

ORIENTATION_DICT_RIGHT = {
    'Horizontally_Up': euler2quat(0, np.pi/2, np.pi/2),
    'Horizontally_Down': euler2quat(np.pi, -np.pi/2, np.pi/2),
    'Vertical': euler2quat(np.pi/2, 0, np.pi),
    'Horizontally_Slanted_Up': euler2quat(np.pi/4, np.pi/6, np.pi),
    'Horizontally_Slanted_Down': euler2quat(np.pi, -np.pi*6/8, np.pi*5/8),
    'Vertically_Slanted': euler2quat(np.pi/2, 0, np.pi*2/3),
    'Vertically_Down': euler2quat(np.pi, 0, 3*np.pi/8)
}

# =====================================================================
# GLOBAL STATE
# =====================================================================
class RobotState:
    """Manages robot hand states for both sides."""
    def __init__(self):
        self.left_hand = 'Vertical'
        self.left_finger = 'Open'
        self.right_hand = 'Vertical'
        self.right_finger = 'Open'

    def reset(self):
        self.left_hand = self.right_hand = 'Vertical'
        self.left_finger = self.right_finger = 'Open'

    def get_state(self, side):
        if side == 'left':
            return self.left_hand, self.left_finger
        return self.right_hand, self.right_finger

    def set_state(self, side, hand=None, finger=None):
        if side == 'left':
            if hand: self.left_hand = hand
            if finger: self.left_finger = finger
        else:
            if hand: self.right_hand = hand
            if finger: self.right_finger = finger

    def get_info_string(self, side):
        """Get formatted state string for controller parsing."""
        if side == 'left':
            return f' LEFT_HAND: {self.left_hand} LEFT_FINGER: {self.left_finger}'
        return f' RIGHT_HAND: {self.right_hand} RIGHT_FINGER: {self.right_finger}'

state = RobotState()

# Globals to track the OTHER hand's state (set by main)
other_hand_holding = None
other_hand_command = None  # What command the other hand is executing this step

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def get_side_context(side):
    """Get all side-specific objects in one call."""
    if side == 'left':
        return {
            'controller': left,
            'sensor': left_sensor,
            'attachpoint': left_attachpoint,
            'ori_dict': ORIENTATION_DICT_LEFT,
            'offset_suffix': '_left',
            'other_sensor': right_sensor,
            'boundary_check': lambda y: y < -0.3,
        }
    return {
        'controller': right,
        'sensor': right_sensor,
        'attachpoint': right_attachpoint,
        'ori_dict': ORIENTATION_DICT_RIGHT,
        'offset_suffix': '_right',
        'other_sensor': left_sensor,
        'boundary_check': lambda y: y > 0.3,
    }

def is_out_of_reach(side, position):
    """Check if position is out of reach for given side."""
    return (side == "right" and position[1] > 0.3) or (side == "left" and position[1] < -0.3)


# Keywords identifying task-relevant (manipulable) objects.
# Mirrors the graspable categories in nicol_controller.py (SIDE_/TOP_GRASP_OBJECTS).
TASK_OBJECT_KEYWORDS = ('block', 'cup', 'bowl', 'apple', 'banana')

# Suffixes that mark helper/reference objects (grasp-point dummies, offset
# dummies, sensors) rather than the actual manipulable body. These share the
# block/cup keyword in their name but must not be treated as task objects, or
# a block's own '_top' marker gets detected as something sitting on top of it.
NON_TASK_SUFFIXES = ('_top', '_left', '_right', '_sensor')

def is_task_object(name):
    """True if an object name belongs to a task-relevant (manipulable) object."""
    name = clean_obj_name(name)
    if name.endswith(NON_TASK_SUFFIXES):
        return False
    return any(keyword in name for keyword in TASK_OBJECT_KEYWORDS)

def get_task_object_handles(exclude=None):
    """Return handles of all task-relevant objects currently in the scene."""
    handles = []
    for handle in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0):
        if handle == exclude:
            continue
        if is_task_object(clean_obj_name(sim.getObjectAlias(handle, -1))):
            handles.append(handle)
    return handles

def check_object_above_block(target: str):
    """
    Check if a task-relevant object (block/cup/...) is sitting above a block,
    using the block's sensor. Irrelevant objects (hands, fingers, fixtures) are
    ignored, so a closer non-task object cannot mask the object actually
    resting on the block.
    Returns tuple: (has_object_above: bool, object_name: str or None)
    Used for StackBlocks task to prevent moving above a block that has another block on top.
    """
    # Cups are never stacked on each other (there are only two), so skip the
    # above-check for them. This also avoids a false positive where a cup's own
    # '_respondable' body (a distinct handle that cleans to the same name) is
    # detected as an object sitting on top of itself.
    if 'cup' in clean_obj_name(target):
        return False, None

    # Get the target block handle to exclude it from detection
    target_handle = get_object_handle_safe(target)
    if not target_handle:
        return False, None

    # Get the sensor for this block (e.g., red_block -> red_block_sensor)
    target_sensor = get_object_handle_safe(target + '_sensor')
    if not target_sensor:
        return False, None

    # Test the sensor against each task object individually instead of
    # sim.handle_all, which only reports the single closest detected object
    # (which may be an irrelevant hand/fixture that masks the real one).
    for handle in get_task_object_handles(exclude=target_handle):
        if sim.checkProximitySensor(target_sensor, handle)[0] and handle!=target_handle:
            return True, clean_obj_name(sim.getObjectAlias(handle, -1))

    return False, None

def get_object_handle_safe(obj_name):
    """Safely get object handle, return None if not found."""
    try:
        return sim.getObject('/' + obj_name)
    except Exception:
        return None

def get_object_body_handle(obj_name):
    """Return the collidable body a proximity sensor can actually detect.

    Some objects (e.g. cups) expose a '<obj>_respondable' shape under a plain
    '/<obj>' dummy that has no geometry; others (e.g. blocks) are the collidable
    shape themselves. Prefer the respondable body, fall back to the object
    itself when it has no '_respondable' variant.
    """
    respondable = get_object_handle_safe(obj_name + '_respondable')
    return respondable if respondable is not None else get_object_handle_safe(obj_name)

def clean_obj_name(name):
    """Remove '_respondable' suffix from object name."""
    return name.replace('_respondable', '') if '_respondable' in name else name

def is_expected_detected_object(expected_name, detected_handle):
    """Check whether the detected object name matches the expected object name."""
    detected_name = clean_obj_name(sim.getObjectAlias(detected_handle, -1))
    return detected_name == clean_obj_name(expected_name), detected_name

def get_grasped_object_from_sensor(side):
    """Detect the object currently in the side hand's sensor, if any."""
    _, finger_stat = state.get_state(side)
    if finger_stat == 'Open':
        return None

    ctx = get_side_context(side)
    if not sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]:
        return None

    read_result = sim.readProximitySensor(ctx['sensor'])
    if not read_result[0]:
        return None

    detected_handle = read_result[3]
    return clean_obj_name(sim.getObjectAlias(detected_handle, -1))

def other_hand_is_holding(side, obj_name):
    """True if the hand opposite `side` is currently holding obj_name.

    Two independent signals, so the guard works in both execution paths:
    - In-process (e.g. ipython): the other side's finger state is 'Closed',
      because the global `state` persists across calls.
    - Subprocess (nicol_controller.py and the ReAct-agent): the caller forwards
      what the other hand holds via --other_holding -> `other_hand_holding`. The subprocess
      never initializes the other side's finger state (it stays default
      'Open'), so the finger-state signal alone is unreliable there.
    """
    if other_hand_holding and clean_obj_name(other_hand_holding) == clean_obj_name(obj_name):
        return True
    other_side = 'right' if side == 'left' else 'left'
    _, other_finger_stat = state.get_state(other_side)
    return other_finger_stat == 'Closed'


def format_result(side, message):
    """Format result with state info for controller parsing."""
    return message + state.get_info_string(side)

# =====================================================================
# ROBOT SKILL FUNCTIONS
# =====================================================================
def look_at(obj_name: str):
    """Control head to look at specified location."""
    positions = {
        'left': [0.7, 0.2, 0.9],
        'right': [0.7, -0.2, 0.9],
        'center': [0.7, 0, 0.9]
    }

    if obj_name in positions:
        head.set_pose_target(NicolPose(positions[obj_name], [0, 0, 0, 0]))
        return f"The NICOL has looked at the {obj_name}."

    # Default: look at human
    human_pose = sim.getObjectPosition(sim.getObject('/Bill'), sim.handle_world)
    head.set_pose_target(NicolPose([human_pose[0], human_pose[1], human_pose[2] + 1.5], [0, 0, 0, 0]))
    return "The NICOL has looked at the human user."


def move_both_to_poses(obj_name: str):
    """Move both hands simultaneously to center on an object."""
    handle = get_object_handle_safe(obj_name)
    if not handle:
        return f"The robot's both hands failed to move to {obj_name}, object not found."

    if not (sim.checkProximitySensor(left_sensor, sim.handle_all)[0] and sim.checkProximitySensor(right_sensor, sim.handle_all)[0]):
        return f"The robot's both hands failed to move to {obj_name}, as the {obj_name} is not detected around both hands."

    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    left_pose = left.get_eef_pose().position.as_list()
    right_pose = right.get_eef_pose().position.as_list()
    mid_point = [(left_pose[i] + right_pose[i]) / 2 for i in range(3)]

    left_target = [left_pose[i] + new_pose[i] - mid_point[i] for i in range(3)]
    right_target = [right_pose[i] + new_pose[i] - mid_point[i] for i in range(3)]

    left_ori = left.get_eef_pose().orientation
    left_ori = [left_ori.x, left_ori.y, left_ori.z, left_ori.w]
    right_ori = right.get_eef_pose().orientation
    right_ori = [right_ori.x, right_ori.y, right_ori.z, right_ori.w]

    # Both hands shift by the same offset; move along it in waypoints at most
    # step_length apart, in whichever direction the object lies.
    step_length = 0.06
    offset = [new_pose[i] - mid_point[i] for i in range(3)]
    steps = max(1, math.ceil(max(abs(delta) for delta in offset) / step_length))

    head.set_pose_target(NicolPose([new_pose[0], new_pose[1], new_pose[2] + 0.5], [0, 0, 0, 0]))

    for i in range(1, steps):
        fraction = i / steps
        left_step = [left_pose[j] + offset[j] * fraction for j in range(3)]
        right_step = [right_pose[j] + offset[j] * fraction for j in range(3)]
        nicol.set_pose_target_for_both_arms(NicolPose(left_step, left_ori), NicolPose(right_step, right_ori))
        nicol.set_pose_target_for_both_arms(NicolPose(left_step, left_ori), NicolPose(right_step, right_ori))

    nicol.set_pose_target_for_both_arms(NicolPose(left_target, left_ori), NicolPose(right_target, right_ori))
    nicol.set_pose_target_for_both_arms(NicolPose(left_target, left_ori), NicolPose(right_target, right_ori))

    return f"The NICOL robot's both hands have moved simultaneously to the new position where the {obj_name} is at the middle point of two hands."


def side_grasp(side: str, obj_name: str):
    """Grasp object from the side."""
    ctx = get_side_context(side)
    hand_stat, finger_stat = state.get_state(side)

    handle = get_object_handle_safe(obj_name)
    if not handle:
        return format_result(side, f"The {side} hand failed to grasp {obj_name}, object not found.")

    handle_ori = sim.getObjectOrientation(handle, sim.handle_world)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)

    # Validation checks
    if is_out_of_reach(side, new_pose):
        return format_result(side, f"The robot's {side} hand failed to move to {obj_name}, out of its area.")

    detected = sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]
    if finger_stat == 'Closed' and detected:
        return format_result(side, f"The {side} hand failed to grasp {obj_name}, as the {side} hand is already occupied.")

    # Proximity sensors only detect respondable shapes. A cup's collidable body
    # is '<cup>_respondable'; the bare '/<cup>' dummy returned by
    # get_object_handle_safe has no geometry, so checking against it always
    # returns False and the "grasped by another hand" guard silently fails.
    obj_body = get_object_body_handle(obj_name) or handle
    if sim.checkProximitySensor(ctx['other_sensor'], obj_body)[0] and other_hand_is_holding(side, obj_name):
        return format_result(side, f"The {side} hand failed to grasp {obj_name}, as it is still grasped by another hand.")

    if hand_stat == 'Hold_Up':
        release(side)

    if "bowl" in obj_name:
        return format_result(side, "Failed. The bowl is not allowed to be grasped.")

    if "block" in obj_name:
        has_object_above, object_above = check_object_above_block(obj_name)
        grasped_obj = get_grasped_object_from_sensor(side)
        if has_object_above and object_above != grasped_obj:
            return format_result(
                side,
                f"The robot's {side} hand failed to grasp {obj_name}, as {object_above} is currently on top of {obj_name}."
            )

    result = move_single_to_pose(side, target=obj_name, obj_name=None, ori_angle='Vertical', off_set=None)
    if 'Failed' in result or 'failed' in result:
        return format_result(side, result.split('.')[0])

    # Grasp sequence
    quat = [
        [np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
        [np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
        [np.pi, -2.9, -1.8, -1.8, -1.8]
    ]

    detected = sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]
    if detected:
        read_result = sim.readProximitySensor(ctx['sensor'])
        detected_handle = read_result[3]
        print(f"Detected handle: {detected_handle}, Expected handle: {handle}")
        is_match, obj_name_grasped = is_expected_detected_object(obj_name, detected_handle)
        if is_match:
            sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
            sim.setObjectParent(detected_handle, ctx['attachpoint'], True)

            for sub_quat in quat:
                ctx['controller'].set_joint_position_for_hand(sub_quat, block=True)

            state.set_state(side, hand='Vertical', finger='Closed')
            result = f"The robot's {side} hand has grasped {obj_name_grasped}, and is holding the {obj_name_grasped}."
        else:
            result = f"The robot's {side} hand failed to grasp {obj_name}, detected {obj_name_grasped} instead."
    else:
        result = f"The robot's {side} hand failed to grasp anything, as there is no object detected."

    time.sleep(0.5)
    sim.setObjectOrientation(handle, -1, handle_ori)
    return format_result(side, result)


def top_grasp(side: str, obj_name: str):
    """Grasp object from the top."""
    ctx = get_side_context(side)
    hand_stat, finger_stat = state.get_state(side)

    handle = get_object_handle_safe(obj_name)
    if not handle:
        return format_result(side, f"The {side} hand failed to grasp {obj_name}, object not found.")

    handle_ori = sim.getObjectOrientation(handle, sim.handle_world)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)

    if is_out_of_reach(side, new_pose):
        return format_result(side, f"The robot's {side} hand failed to move to {obj_name} (with pose: {new_pose}), out of its area.")

    detected = sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]
    if finger_stat == 'Closed' and detected:
        return format_result(side, f"The {side} hand failed to grasp {obj_name}, as the {side} hand is already occupied.")

    # See side_grasp: check the respondable body, not the '/<cup>' dummy, or the
    # "grasped by another hand" guard silently fails for cups.
    obj_body = get_object_body_handle(obj_name) or handle
    if sim.checkProximitySensor(ctx['other_sensor'], obj_body)[0] and other_hand_is_holding(side, obj_name):
        return format_result(side, f"The {side} hand failed to grasp {obj_name}, as it is still grasped by another hand.")

    if hand_stat == 'Hold_Up':
        release(side)

    if "bowl" in obj_name:
        return format_result(side, "Failed. The bowl is not allowed to be grasped.")

    if "block" in obj_name:
        has_object_above, object_above = check_object_above_block(obj_name)
        grasped_obj = get_grasped_object_from_sensor(side)
        if has_object_above:
            if object_above and object_above != grasped_obj and "block" in object_above:
                return format_result(
                    side,
                    f"The robot's {side} hand failed to grasp {obj_name}, as {object_above} is currently on top of {obj_name}."
                )

    # Move above then to object
    move_single_to_pose(side, target=obj_name, obj_name=None, ori_angle='Horizontally_Down', off_set="up")
    move_single_to_pose(side, target=obj_name, obj_name=None, ori_angle='Horizontally_Down', off_set=None)

    quat = [
        [np.pi, -np.pi, -np.pi, -np.pi, -np.pi],
        [np.pi, -np.pi/2, -0.3, 0, 0],
        [np.pi, -2, -0.3, 0, 0]
    ]

    detected = sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]
    if detected:
        sim.setObjectParent(handle, ctx['attachpoint'], True)
        obj_name_clean = clean_obj_name(obj_name)

        for sub_quat in quat:
            ctx['controller'].set_joint_position_for_hand(sub_quat)

        state.set_state(side, hand='Horizontally_Down', finger='Closed')
        delta_move(side, 'up', 0.05, ori_single='Horizontally_Down')
        delta_move(side, side, 0.05, ori_single='Horizontally_Down')
        result = f"The robot's {side} hand has grasped {obj_name_clean} from the top, and is lifting the {obj_name_clean}."
    else:
        result = f"The robot's {side} hand failed to grasp anything from the top, as there is no appropriate object detected."

    time.sleep(0.5)
    sim.setObjectOrientation(handle, -1, handle_ori)

    return format_result(side, result)


def move_single_to_pose(side: str, target: str, obj_name=None, ori_angle=None, off_set=None):
    """Move single hand to target pose."""
    ctx = get_side_context(side)
    hand_stat, _ = state.get_state(side)

    # Get target position
    target_handle = get_object_handle_safe(target)
    if not target_handle:
        return format_result(side, f"The robot's {side} hand failed to move to {target} as the {target} is not detected in the scene.")

    new_pose = sim.getObjectPosition(target_handle, sim.handle_world)

    if is_out_of_reach(side, new_pose):
        return format_result(side, f"The robot's {side} hand failed to move to {target}, out of its area.")

    # Check if object is held
    if obj_name:
        obj_handle = get_object_handle_safe(obj_name)
        if not obj_handle:
            return format_result(side, f"The robot's {side} hand failed to move {obj_name} as the {obj_name} is not detected in the scene.")
        if not sim.checkProximitySensor(ctx['sensor'], obj_handle)[0]:
            return format_result(side, f"The robot's {side} hand failed to move {obj_name}, as the {obj_name} is not detected around the hand.")

    # Check if target point is occupied
    if "point" in target:
        point_sensor = get_object_handle_safe(target + '_sensor')
        if point_sensor and sim.checkProximitySensor(point_sensor, sim.handle_all)[0]:
            # Check if the blocking object is being held by the other hand AND
            # the other hand is executing a move command (moving away from this point)
            read_result = sim.readProximitySensor(point_sensor)
            if read_result[0]:
                detected_handle = read_result[3]
                detected_name = clean_obj_name(sim.getObjectAlias(detected_handle, -1))
                # Allow if detected object is not a task-relevant object (block/cup)
                if 'block' not in detected_name and 'cup' not in detected_name:
                    pass  # Allow - detected object is irrelevant (e.g. hand, fixture)
                else:
                    # Only allow if: other hand holds the blocking object AND is moving away
                    is_other_moving_away = other_hand_command and 'move' in other_hand_command
                    side_grasping = get_grasped_object_from_sensor(side)
                    if detected_name == side_grasping:
                        pass  # Allow - this hand is already holding the detected object
                    elif other_hand_holding and detected_name == other_hand_holding and is_other_moving_away:
                        pass  # Allow - other hand is actively moving this object away in parallel
                    else:
                        return format_result(side, f"The robot's {side} hand failed to move to {target}, as the {target} is already occupied by {detected_name}.")
            else:
                return format_result(side, f"The robot's {side} hand failed to move to {target}, as the {target} is already occupied by the other object.")

    # Check if there's any object above the target block (for StackBlocks task)
    if off_set == 'up':
        has_object_above, object_above = check_object_above_block(target)
        grasped_obj = get_grasped_object_from_sensor(side)
        if has_object_above and object_above is not None and object_above != grasped_obj:
            # Allow parallel actions: the detected object is being held by the
            # other hand, which is either moving it away from above the target or
            # placing it onto the target. In both cases the obstruction is
            # transient, so moving above the target is permitted.
            is_other_moving_away = other_hand_command and 'move' in other_hand_command
            is_other_placing = other_hand_command and 'place' in other_hand_command
            if other_hand_holding and object_above == other_hand_holding and (is_other_moving_away or is_other_placing):
                pass  # Allow - other hand is moving the object away or placing it on the target in parallel
            else:
                return format_result(side, f"The robot's {side} hand failed to move above {target}, as {object_above} is currently on top of {target}.")

    # Determine orientation
    ori_angle = ori_angle or hand_stat

    # Adjust target position based on orientation
    if ori_angle == 'Vertical':
        offset_handle = get_object_handle_safe(target + ctx['offset_suffix'])
        if offset_handle:
            new_pose = sim.getObjectPosition(offset_handle, sim.handle_world)
    elif ori_angle == 'Horizontally_Down':
        top_handle = get_object_handle_safe(target + '_top')
        if top_handle:
            new_pose = sim.getObjectPosition(top_handle, sim.handle_world)
        else:
            new_pose = [new_pose[0], new_pose[1], new_pose[2] + 0.015]

    new_quat = ctx['ori_dict'][ori_angle]

    # Apply offset
    offset_map = {
        'up': (0, 0, 0.3), 'above': (0, 0, 0.3),
        'down': (0, 0, -0.18), 'below': (0, 0, -0.12),
        'left': (0, 0.2, 0.02), 'right': (0, -0.2, 0.02)
    }

    dx, dy, dz = offset_map.get(off_set, (0, 0, 0))
    target_pose = [new_pose[0] + dx, new_pose[1] + dy, new_pose[2] + dz]

    if off_set:
        ctx['controller'].set_pose_target(NicolPose(target_pose, new_quat))
    else:
        if target != "serve_point":
            ctx['controller'].set_pose_target(NicolPose([new_pose[0], new_pose[1], new_pose[2] + 0.15], new_quat))
        ctx['controller'].set_pose_target(NicolPose(new_pose, new_quat))

    # Format result message
    if off_set == 'up':
        msg = f"The robot's {side} hand has moved above the {target}."
    elif off_set == 'down':
        msg = f"The robot's {side} hand has moved below the {target}."
    else:
        msg = f"The robot's {side} hand has moved to {target}."

    return format_result(side, msg)


def delta_move(side, direction, distance=None, ori_single=None):
    """Move hand by delta in specified direction."""
    ctx = get_side_context(side)
    controller = ctx['controller']

    ee_pose = controller.get_eef_pose().position.as_list()
    ori = controller.get_eef_pose().orientation
    ori = [ori.x, ori.y, ori.z, ori.w]

    distance = distance or 0.06

    direction_map = {
        'right': (0, -distance, 0), 'y-decrease': (0, -distance, 0),
        'left': (0, distance, 0), 'y-increase': (0, distance, 0),
        'front': (distance, 0, 0), 'x-increase': (distance, 0, 0),
        'behind': (-distance, 0, 0), 'back': (-distance, 0, 0), 'x-decrease': (-distance, 0, 0),
        'down': (0, 0, -distance),
        'up': (0, 0, distance)
    }

    dx, dy, dz = direction_map.get(direction, (0, 0, 0))
    new_pose = [ee_pose[0] + dx, ee_pose[1] + dy, ee_pose[2] + dz]

    if ori_single:
        controller.set_pose_target(NicolPose(new_pose, ctx['ori_dict'][ori_single]))
    else:
        controller.set_pose_target(NicolPose(new_pose, ori))

    return format_result(side, f"The robot's {side} hand has moved {direction} for {distance} meters.")


def release(side: str, obj_name=None, point=None):
    """Release grasped object."""
    ctx = get_side_context(side)
    hand_stat, _ = state.get_state(side)
    controller = ctx['controller']
    sensor = ctx['sensor']

    # All checks come first: a release that fails must leave the object in the hand.
    obj_handle = get_object_handle_safe(obj_name) if obj_name else None
    if obj_name:
        obj_respondable = get_object_handle_safe(obj_name + '_respondable')
        if obj_respondable and not sim.checkProximitySensor(sensor, obj_respondable)[0]:
            return format_result(side, f"The robot's {side} hand failed to release {obj_name}, as the {obj_name} is not detected around the hand.")

    if point and obj_name:
        if "cup" in obj_name and "serve" in point:
            return format_result(side, "Failed! The release command is not allowed here.")

        # Verify, via the point's own proximity sensor, that the object is
        # actually located at the point before releasing it there. Prefer the
        # respondable body, but fall back to the plain handle so the check
        # still runs when no '_respondable' variant exists.
        point_sensor = get_object_handle_safe(point + '_sensor')
        obj_to_check = get_object_handle_safe(obj_name + '_respondable') or get_object_handle_safe(obj_name)
        if point_sensor:
            if not obj_to_check or not sim.checkProximitySensor(point_sensor, obj_to_check)[0]:
                return format_result(side, f"The robot's {side} hand failed to release {obj_name} on {point}, as the {obj_name} is not at the {point}.")

    if obj_name and "block" in obj_name and obj_handle:
        sim.setObjectInt32Parameter(obj_handle, sim.shapeintparam_respondable, 0)
        sim.setObjectParent(obj_handle, -1, True)
        time.sleep(0.5)

    # Open hand
    quat = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
    controller.set_joint_position_for_hand(quat)

    # Detach object
    detected = sim.checkProximitySensor(sensor, sim.handle_all)[0]
    if detected:
        read_result = sim.readProximitySensor(sensor)
        detected_handle = read_result[3]
        sim.setObjectParent(detected_handle, -1, True)
        obj_name = clean_obj_name(sim.getObjectAlias(detected_handle, -1))
        if "block" in obj_name:
            # Align released blocks to the upright pose used during task reset.
            obj_pos = sim.getObjectPosition(detected_handle, -1)
            sim.setObjectPose(detected_handle, -1, list(obj_pos) + [1, 0, 0, 0])

    # Move away based on hand state
    if hand_stat == 'Vertical':
        delta_move(side, 'up', 0.15, ori_single=hand_stat)
        delta_move(side, side, 0.1, ori_single=hand_stat)
    elif hand_stat == 'Horizontally_Down':
        delta_move(side, 'up', 0.08, ori_single=hand_stat)
        delta_move(side, side, 0.3, ori_single=hand_stat)

    if obj_name:
        result = f"The robot's {side} hand has been opened, and the grasped {obj_name} has been released."
    else:
        result = f"The robot's {side} hand has been opened."

    state.set_state(side, finger='Open')
    return format_result(side, result)


def place(side: str, source_obj: str, target_obj: str):
    """Place held object on target."""
    ctx = get_side_context(side)
    controller = ctx['controller']
    sensor = ctx['sensor']
    other_sensor = ctx['other_sensor']
    ori_dict = ctx['ori_dict']

    ee_pose = controller.get_eef_pose().position.as_list()
    controller.set_pose_target(NicolPose(ee_pose, ori_dict['Horizontally_Down']))

    source_handle = get_object_handle_safe(source_obj)
    target_handle = get_object_handle_safe(target_obj)

    if not source_handle or not target_handle:
        return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, object not found.")

    source_pos = sim.getObjectPosition(source_handle, -1)
    target_pos = sim.getObjectPosition(target_handle, -1)
    source_ori = sim.getObjectOrientation(source_handle, -1)

    if is_out_of_reach(side, target_pos):
        return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as the {target_obj} is out of its area.")

    if not sim.checkProximitySensor(sensor, source_handle)[0]:
        return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as the {source_obj} is not detected in on the hand.")

    if sim.checkProximitySensor(other_sensor, target_handle)[0]:
        return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as the {source_obj} is grasped by another hand.")

    # For apple and banana, verify the source object is detected by the
    # target's proximity sensor, i.e. it is actually above the target.
    if source_obj in ("apple", "banana"):
        target_sensor = get_object_handle_safe(target_obj + '_sensor')
        source_body = get_object_body_handle(source_obj) or source_handle
        if not (target_sensor and sim.checkProximitySensor(target_sensor, source_body)[0]):
            return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as {source_obj} is not on top of {target_obj}.")
    # Check if there's any object above the target block (for StackBlocks task).
    # An object on top is only a problem if it is not the source object being placed.
    if "block" in target_obj:
        has_object_above, object_above = check_object_above_block(target_obj)
        if has_object_above and object_above != source_obj:
            return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as {source_obj} is not on top of {target_obj}.")

    if "bowl" not in target_obj and "block" not in target_obj:
        return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as only a bowl or a block can be a placing target.")

    quat = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi]

    if "bowl" in target_obj:
        bowl = sim.getObject('/bowl_respondable')
        # Validate placement with the bowl's own proximity sensor against the
        # source object's collidable body, rather than a fixed-distance test on
        # a position captured at function entry (which is the object's grasp
        # pose, not where it actually ends up). The bowl branch only lowers the
        # hand straight down, so this correctly fails when no prior step moved
        # the hand above the bowl.
        bowl_sensor = get_object_handle_safe('bowl_sensor')
        source_body = get_object_body_handle(source_obj) or source_handle
        placed_in_bowl = bowl_sensor and sim.checkProximitySensor(bowl_sensor, source_body)[0]

        if placed_in_bowl:
            delta_move(side, "down", 0.12)
            controller.set_joint_position_for_hand(quat)
            time.sleep(2)

            sim.setObjectInt32Parameter(source_handle, sim.shapeintparam_static, 1)
            sim.setObjectInt32Parameter(source_handle, sim.shapeintparam_respondable, 0)
            sim.setObjectParent(source_handle, bowl, True)

            if source_handle == get_object_handle_safe("banana"):
                sim.setObjectPosition(source_handle, bowl, [-0.05, -0.07, 0.01])
            elif source_handle == get_object_handle_safe("apple"):
                sim.setObjectPosition(source_handle, bowl, [-0.012, -0.146, 0.02])

            sim.setObjectOrientation(source_handle, bowl, source_ori)
            time.sleep(0.5)
        else:
            return format_result(side, f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as {source_obj} is not on top of {target_obj}.")

        delta_move(side, 'up', 0.05)
        delta_move(side, side, 0.2)

    elif "block" in target_obj:
        new_pose = [target_pos[0], target_pos[1], target_pos[2] + 0.12]
        controller.set_pose_target(NicolPose(new_pose, ori_dict['Horizontally_Down']))
        time.sleep(1)
        controller.set_joint_position_for_hand(quat)

        if abs(source_pos[0] - target_pos[0]) < 0.3 and abs(source_pos[1] - target_pos[1]) < 0.3 and source_pos[2] > target_pos[2]:
            sim.setObjectInt32Parameter(source_handle, sim.shapeintparam_static, 1)
            sim.setObjectParent(source_handle, -1, True)
            sim.setObjectParent(source_handle, target_handle, True)
            sim.setObjectPosition(source_handle, target_handle, [0, 0, -0.05])
            sim.setObjectOrientation(source_handle, -1, [-np.pi, 0, np.pi/2])

        delta_move(side, 'up', 0.075)
        delta_move(side, side, 0.2)

    state.set_state(side, finger='Open')
    return format_result(side, f"The robot's {side} hand has already placed the grasped {source_obj} on the {target_obj}.")


def reset(side: str):
    """Reset hand to original position."""
    release(side)

    ctx = get_side_context(side)
    controller = ctx['controller']

    controller.set_joint_position_for_hand([-np.pi] * 5, block=True)

    if side == 'left':
        controller.set_joint_position([-1.57] + [0.] * 7, block=True)
    else:
        controller.set_joint_position([1.57] + [0.] * 7, block=True)

    state.set_state(side, hand='Vertical', finger='Open')
    return format_result(side, f"The robot's {side} hand has been reset to its original position.")


def flip_down(side: str):
    """Flip hand down for pouring."""
    ctx = get_side_context(side)
    controller = ctx['controller']
    ori_dict = ctx['ori_dict']

    ball = get_object_handle_safe("big_ball")
    yellow_cup = sim.getObjectHandle('/yellow_cup_respondable')
    blue_cup = sim.getObjectHandle('/blue_cup_respondable')
    yellow_pos = sim.getObjectPosition(yellow_cup, -1)
    blue_pos = sim.getObjectPosition(blue_cup, -1)

    ee_pose = controller.get_eef_pose().position.as_list()

    if side == 'left':
        new_pose_1 = [ee_pose[0] - 0.01, ee_pose[1] - 0.01, ee_pose[2] - 0.04]
    else:
        new_pose_1 = [ee_pose[0] - 0.01, ee_pose[1] - 0.02, ee_pose[2] - 0.04]

    # Prepare to pour
    ori_1 = ori_dict['Horizontally_Slanted_Down']
    controller.set_pose_target(NicolPose(new_pose_1, ori_1))
    time.sleep(0.5)

    # Transfer ball if cups aligned
    if ball and abs(blue_pos[0] - yellow_pos[0]) < 0.04 and abs(blue_pos[1] - yellow_pos[1]) < 0.04 and blue_pos[2] > yellow_pos[2]:
        sim.setObjectInt32Parameter(ball, sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(ball, sim.shapeintparam_respondable, 0)
        sim.setObjectParent(ball, yellow_cup, True)
        sim.setObjectPosition(ball, yellow_cup, [0.03, -0.01, -0.02])
        time.sleep(0.5)

    # Return to vertical
    ori_2 = ori_dict['Vertical']
    new_pose_2 = [new_pose_1[0], new_pose_1[1], new_pose_1[2] + 0.02]
    controller.set_pose_target(NicolPose(new_pose_2, ori_2))

    return format_result(side, f"The robot's {side} hand's is fliped down for once for pouring out water.")


def hold_up_single(side: str, obj_name: str):
    """Hold up a large object from the side (co_hold: one call per hand)."""
    ctx = get_side_context(side)
    hand_stat, finger_stat = state.get_state(side)

    quat = [-np.pi, -np.pi * 0.75, -np.pi * 0.75, -np.pi * 0.75, -np.pi * 0.75]

    obj_handle = get_object_handle_safe(obj_name)
    if not obj_handle:
        return format_result(side, f"The {side} hand failed to hold {obj_name}, object not found.")

    obj_pose = sim.getObjectPosition(obj_handle, sim.handle_world)
    obj_pose = [obj_pose[0], obj_pose[1], obj_pose[2] + 0.18]

    # Boundary check
    if (side == 'left' and obj_pose[1] < -0.25) or (side == 'right' and obj_pose[1] > 0.25):
        return format_result(side, "hold_up is failed, out of its area.")

    detected = sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]
    if finger_stat == 'Closed' and detected:
        return format_result(side, f"The {side} hand failed to hold {obj_name}, as the {side} hand is already occupied for grasping and holding some object.")
    else:
        release(side)

    controller = ctx['controller']
    ori_dict = ctx['ori_dict']

    target_pose = sim.getObjectPosition(get_object_handle_safe(obj_name + ctx['offset_suffix']), sim.handle_world)

    if side == 'left' and hand_stat != 'Horizontally_Slanted_Up':
        poses = [
            [0.6, 0.3, 1.0],
            [target_pose[0], target_pose[1] + 0.01, target_pose[2] + 0.18],
            [target_pose[0] - 0.02, target_pose[1] + 0.04, target_pose[2]]
        ]
        for pose in poses:
            controller.set_pose_target(NicolPose(pose, ori_dict['Horizontally_Slanted_Up']))

        new_pose = [target_pose[0] + 0.02, target_pose[1] - 0.05, target_pose[2]]
        controller.set_pose_target(NicolPose(new_pose, ori_dict['Horizontally_Slanted_Up']))
        state.set_state(side, hand='Horizontally_Slanted_Up', finger='Hold_Up')
        controller.set_joint_position_for_hand(quat)

    elif side == 'right' and hand_stat != 'Horizontally_Slanted_Up':
        poses = [
            [0.6, -0.3, 1.0],
            [target_pose[0], target_pose[1] - 0.01, target_pose[2] + 0.18],
            [target_pose[0] - 0.02, target_pose[1] - 0.06, target_pose[2]]
        ]
        for pose in poses:
            controller.set_pose_target(NicolPose(pose, ori_dict['Horizontally_Slanted_Up']))

        new_pose = [target_pose[0] - 0.02, target_pose[1] + 0.05, target_pose[2]]
        controller.set_pose_target(NicolPose(new_pose, ori_dict['Horizontally_Slanted_Up']))
        controller.set_joint_position_for_hand(quat)
        state.set_state(side, hand='Horizontally_Slanted_Up', finger='Hold_Up')

    # Attach detected object
    detected = sim.checkProximitySensor(ctx['sensor'], sim.handle_all)[0]
    if detected:
        read_result = sim.readProximitySensor(ctx['sensor'])
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, ctx['attachpoint'], True)

    return format_result(side, f"The robot's {side} hand is prepared to hold the {obj_name} from the side.")


def push_to(side: str, source_obj: str, target_obj: str):
    """Push object to target area."""
    _, finger_stat = state.get_state(side)

    if finger_stat == 'Closed':
        return format_result(side, f"failed! The robot's {side} hand is already occupied for grasping some object.")

    if "area" not in target_obj:
        return format_result(side, f"failed! It is not allowed to push {source_obj} to {target_obj}.")

    source_handle = get_object_handle_safe(source_obj)
    source_handle_respond = get_object_handle_safe(source_obj + '_respondable')
    if not source_handle:
        return format_result(side, f"failed! The object {source_obj} is not in the scene.")

    target_handle = get_object_handle_safe(target_obj)
    if not target_handle:
        return format_result(side, f"failed! The object {target_obj} is not in the scene.")

    source_pose = np.array(sim.getObjectPosition(source_handle, sim.handle_world))
    target_pose = np.array(sim.getObjectPosition(target_handle, sim.handle_world))
    origin_orientation = sim.getObjectOrientation(source_handle, sim.handle_world)

    if is_out_of_reach(side, source_pose):
        return format_result(side, f"The robot's {side} hand failed to push {source_obj}, out of its area.")

    if is_out_of_reach(side, target_pose):
        return format_result(side, f"The robot's {side} hand failed to push to {target_obj}, out of its area.")

    ctx = get_side_context(side)
    controller = ctx['controller']
    ori_dict = ctx['ori_dict']
    attachpoint = ctx['attachpoint']

    release(side)
    state.set_state(side, hand='Vertical')

    # Determine approach side
    if (side == 'left' and target_pose[1] > 0) or (side == 'right' and target_pose[1] < 0):
        approach_suffix = '_right' if side == 'left' else '_left'
    else:
        approach_suffix = '_left' if side == 'left' else '_right'

    approach_handle = get_object_handle_safe(source_obj + approach_suffix)
    if approach_handle:
        new_pose = np.array(sim.getObjectPosition(approach_handle, sim.handle_world))
    else:
        new_pose = source_pose

    # Approach sequence
    poses = [
        [new_pose[0], new_pose[1], new_pose[2] + 0.3],
        [new_pose[0], new_pose[1], new_pose[2] + 0.1],
        [new_pose[0], new_pose[1], new_pose[2] + 0.01]
    ]

    for pos in poses:
        controller.set_pose_target(NicolPose(pos, ori_dict['Vertical']))

    # Attach and push
    sim.setObjectParent(source_handle_respond, attachpoint, True)

    offset_dir = 'left' if side == 'left' else 'right'
    move_single_to_pose(side, target=target_obj, ori_angle='Vertical', off_set=offset_dir)

    sim.setObjectParent(source_handle_respond, -1, True)

    # Move away
    if (side == 'left' and target_pose[1] > 0) or (side == 'right' and target_pose[1] < 0):
        away_dir = 'right' if side == 'left' else 'left'
    else:
        away_dir = side

    delta_move(side, away_dir, 0.1)
    delta_move(side, 'up', 0.2)

    sim.setObjectOrientation(source_handle, sim.handle_world, origin_orientation)
    return format_result(side, f"The robot's {side} hand has pushed the {source_obj} to the {target_obj}.")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main(args):
    """Main command dispatcher."""
    global other_hand_holding, other_hand_command

    command = args.command
    para = json.loads(args.para)

    # Set what the other hand is holding and doing (for parallel action awareness)
    other_hand_holding = args.other_holding if args.other_holding else None
    other_hand_command = args.other_command if args.other_command else None

    # Initialize state from arguments
    if args.side == 'left':
        state.set_state('left', hand=args.hand_state, finger=args.finger_state)
    elif args.side == 'right':
        state.set_state('right', hand=args.hand_state, finger=args.finger_state)

    # Command dispatch
    command_map = {
        'side_grasp': lambda: side_grasp(args.side, **para),
        'top_grasp': lambda: top_grasp(args.side, **para),
        'release': lambda: release(args.side, **para),
        'move_above': lambda: move_single_to_pose(args.side, **para),
        'pour': lambda: flip_down(args.side, **para),
        'hold_up': lambda: hold_up_single(args.side, **para),
        'move_single_to_pose': lambda: move_single_to_pose(args.side, **para),
        'reset': lambda: reset(args.side),
        'place': lambda: place(args.side, **para),
        'push_to': lambda: push_to(args.side, **para),
        'co_move_to': lambda: move_both_to_poses(**para) + state.get_info_string(args.side),
        'co_hold': lambda: hold_up_single(args.side, **para),
        'look_at': lambda: look_at(**para),
    }

    if command in ['wait', 'None']:
        return format_result(args.side, f"The robot's {args.side} hand is keeping the current status.")

    if command in command_map:
        return command_map[command]()

    return f"Unknown command: {command}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subprocess Controller Parameters")
    parser.add_argument('--side', type=str, help="Which side of the hand that executes the command")
    parser.add_argument('--command', type=str, help="The command for the side controller")
    parser.add_argument('--para', type=str, help="The parameters for the command")
    parser.add_argument('--hand_state', type=str, default='Vertical', help="The hand state for the side controller")
    parser.add_argument('--finger_state', type=str, default='Open', help="The finger state for the side controller")
    parser.add_argument('--other_holding', type=str, default='', help="Object held by the other hand (for parallel action awareness)")
    parser.add_argument('--other_command', type=str, default='', help="Command being executed by the other hand")
    args = parser.parse_args()

    result = main(args)
    result = f'LEFT_RESULT: {result}' if args.side == 'left' else f'RIGHT_RESULT: {result}'
    print(result)
