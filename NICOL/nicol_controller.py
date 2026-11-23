#!/usr/bin/env python
"""
Step-wise execution of a bimanual plan on NICOL.

A plan is a JSON file ``{"Left": [...], "Right": [...]}``; step k of both
hands runs at the same time. Every step starts one nicol_executor.py process
per hand (and one for the head), the processes report their result and the
hand/finger state. The plan stops at the first failed step; the task goal is
checked with ``task.check_success()``.

The shorter hand is padded with ``wait``, a step ``a || b`` (two actions of
one hand in the same step) runs as consecutive steps (bipop.plan_format.align_step_plan,
which also defines how plan steps are counted), and a step that cannot be
parsed fails the plan (same rules as scripts/gt_pddl_eval.py).
"""
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:  # also when run as a script
    sys.path.insert(0, str(ROOT_DIR))

from bipop.plan_format import align_step_plan, is_wait_action  # noqa: E402

# =====================================================================
# CONSTANTS
# =====================================================================
EXECUTOR_PATH = str(Path(__file__).resolve().parent / "nicol_executor.py")

# Objects that require specific grasp types
SIDE_GRASP_OBJECTS = ['cup']
TOP_GRASP_OBJECTS = ['apple', 'banana', 'block']

# Sentinel command emitted when a plan step cannot be parsed or is missing required
# parameters. nicol_control() short-circuits on it and feeds back False with the error message.
INVALID_COMMAND = '__invalid__'

# Executor commands that move a hand to a target; both hands must not share the target.
MOVE_COMMANDS = ('move_single_to_pose', 'move_above')

# --- DIRECT plan completeness ---
# A direct plan step must supply every parameter its skill signature in
# bipop/llm/prompts.py declares. These are the raw, positional plan parameters
# (para0, para1, ...) the LLM is required to provide, named as in prompts.py.
# e.g. move_to(source_obj, target_point) -> both para0 and para1 are required.
DIRECT_SKILL_PARAMS: Dict[str, List[str]] = {
    'move_to': ['source_obj', 'target_point'],
    'move_to_from': ['source_obj', 'target_point'],
    'move_above': ['source_obj', 'target_object'],
    'grasp': ['source_obj'],
    'release': ['source_obj', 'target_point'],
    'pour': ['source_obj', 'target_object'],
    'place': ['source_obj', 'target_object'],
    'push': ['source_obj', 'target_area'],
    'push_to': ['source_obj', 'target_area'],
    'co_hold': ['target_object'],
    'co_move_to': ['target_point'],
    'wait': [],
}

# --- PDDL plan completeness ---
# Parameters (besides `side`) each executor command needs, named with the executor
# keys the PDDL parser forwards. Only keys the PDDL parser actually produces are
# listed, so the check never fails spuriously on grounded planner actions.
REQUIRED_PARAMS: Dict[str, List[str]] = {
    'side_grasp': ['obj_name'],               # grasp(source_obj)
    'top_grasp': ['obj_name'],                # grasp(source_obj)
    'grasp': ['obj_name'],                    # grasp(source_obj)
    'release': ['obj_name', 'point'],         # release(source_obj, target_point)
    'move_single_to_pose': ['target'],        # move_to - PDDL forwards target only
    'move_above': ['target'],                 # move_above - PDDL forwards target only
    'place': ['source_obj', 'target_obj'],    # place(source_obj, target_object)
    'push_to': ['source_obj', 'target_obj'],  # push(source_obj, target_area)
    'co_move_to': ['obj_name'],               # co_move_to(target_point)
    'co_hold': ['obj_name'],                  # co_hold(target_object)
    'pour': [],                               # pour - executor flip_down takes none
    'reset': [],
    'wait': [],
}


# =====================================================================
# STATE MANAGEMENT
# =====================================================================
@dataclass
class HandState:
    """Tracks hand and finger states for one side."""
    hand: str = 'Vertical'
    finger: str = 'Open'
    holding: Optional[str] = None  # Object currently held by this hand


class RobotController:
    """Manages robot state and command execution."""

    def __init__(self):
        self.left = HandState()
        self.right = HandState()

    def reset(self):
        """Reset both hands to default state."""
        self.left = HandState()
        self.right = HandState()

    def get_state(self, side: str) -> HandState:
        return self.left if side == 'left' else self.right

    def get_other_holding(self, side: str) -> Optional[str]:
        """Get what the OTHER hand is holding."""
        other = self.right if side == 'left' else self.left
        return other.holding

    def update_state(self, side: str, hand: Optional[str], finger: Optional[str], holding: Optional[str] = None):
        """Update state for specified side."""
        state = self.get_state(side)
        if hand:
            state.hand = hand
        if finger:
            state.finger = finger
        # Update holding based on finger state and explicit holding
        if holding is not None:
            state.holding = holding
        elif finger == 'Open':
            state.holding = None  # Released


# =====================================================================
# RESULT PARSING
# =====================================================================
def extract_result(side: str, output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract result, hand state, and finger state from executor output.

    Returns:
        Tuple of (result_message, hand_state, finger_state)
    """
    prefix = side.upper()

    patterns = {
        'result': rf"{prefix}_RESULT: (.*?)\. {prefix}_HAND:",
        'hand': rf"{prefix}_HAND: (\w+)",
        'finger': rf"{prefix}_FINGER: (\w+)"
    }

    def extract(pattern: str) -> Optional[str]:
        match = re.search(pattern, output)
        return match.group(1) if match else None

    return (
        extract(patterns['result']),
        extract(patterns['hand']),
        extract(patterns['finger'])
    )


def extract_grasped_object(result: Optional[str]) -> Optional[str]:
    """
    Extract grasped object name from result message.
    E.g., "has grasped yellow_block from the top" -> "yellow_block"
    """
    if not result:
        return None
    match = re.search(r'has grasped (\w+)', result)
    return match.group(1) if match else None


# =====================================================================
# PLAN PARSING
# =====================================================================
def parse_action(action: str) -> Optional[Dict]:
    """Parse a single action string into command dict (None if it cannot be parsed)."""
    action = str(action).strip()
    if is_wait_action(action):
        action = "wait()"

    pattern = r'(\w+)\(\s*([\'"\w\s,]*)\s*\)?$'
    match = re.match(pattern, action)

    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2)

    if params_str:
        params = [p.strip().strip("'\"") for p in params_str.split(",") if p.strip()]
        param_dict = {f"para{i}": p for i, p in enumerate(params)}
    else:
        param_dict = {}

    return {"commands": func_name, "paras": param_dict}


def plan_to_commands(plan: List[str]) -> List[Dict]:
    """Convert plan list to command dictionaries, one per step.

    A step that cannot be parsed is kept as an INVALID_COMMAND entry (not
    dropped), so the two hands stay aligned and the plan fails at that step.
    """
    commands = []
    for action in plan:
        cmd = parse_action(action)
        if cmd is None:
            cmd = {"commands": INVALID_COMMAND, "paras": {},
                   "error": f"failed. Unreadable action '{action}'"}
        commands.append(cmd)
    return commands


def determine_grasp_type(obj_name: str) -> str:
    """Determine appropriate grasp type based on object."""
    if any(obj in obj_name for obj in SIDE_GRASP_OBJECTS):
        return 'side_grasp'
    if any(obj in obj_name for obj in TOP_GRASP_OBJECTS):
        return 'top_grasp'
    return 'grasp'


def format_params(**kwargs) -> str:
    """Format parameters as JSON string."""
    filtered = {k: v for k, v in kwargs.items() if v}
    return json.dumps(filtered)


def expected_param_type(param_name: str) -> Optional[str]:
    """Infer the type a skill parameter expects from its name in the signature."""
    if 'point' in param_name:
        return 'point'
    if 'area' in param_name:
        return 'area'
    if 'obj' in param_name:  # source_obj, target_object
        return 'object'
    return None


def actual_param_type(value: str) -> str:
    """Classify an actual parameter value by its name (points/areas are named)."""
    if 'point' in value:
        return 'point'
    if 'area' in value:
        return 'area'
    return 'object'


def check_direct_completeness(command: str, para: Dict) -> Optional[str]:
    """
    Verify a DIRECT plan step supplies every parameter its skill signature in
    bipop/llm/prompts.py declares, checked against the raw positional plan parameters
    (para0, para1, ...), and that each parameter is of the expected type
    (e.g. a target_point must actually be a point, not an object or area).

    Returns an error message when a required parameter is missing or has the
    wrong type, else None.
    """
    required = DIRECT_SKILL_PARAMS.get(command)
    if not required:
        return None

    signature = f"{command}({', '.join(required)})"

    missing = [name for i, name in enumerate(required) if not para.get(f'para{i}')]
    if missing:
        return (
            f"failed. Incomplete command '{command}': missing required "
            f"parameter(s) {missing}, expected signature {signature}."
        )

    type_errors = []
    for i, name in enumerate(required):
        value = para.get(f'para{i}')
        if not value:
            continue
        expected = expected_param_type(name)
        actual = actual_param_type(value)
        if expected and actual != expected:
            type_errors.append(f"'{value}' is a {actual}, but '{name}' must be a {expected}")
    if type_errors:
        return (
            f"failed. Wrong parameter type in command '{command}': "
            f"{'; '.join(type_errors)}. Expected signature {signature}."
        )
    return None


def check_completeness(command: str, exec_cmd: str, exec_para: str) -> Tuple[str, str]:
    """
    Verify a parsed step supplies every parameter its executor function requires.

    Returns the (exec_cmd, exec_para) pair unchanged when complete, otherwise
    returns (INVALID_COMMAND, error_message) so the controller can feed back False.
    """
    required = REQUIRED_PARAMS.get(exec_cmd)
    if not required:
        return exec_cmd, exec_para

    try:
        provided = json.loads(exec_para)
    except (json.JSONDecodeError, TypeError):
        provided = {}

    missing = [key for key in required if not provided.get(key)]
    if missing:
        return INVALID_COMMAND, (
            f"failed. Incomplete command '{command}': missing required "
            f"parameter(s) {missing} for '{exec_cmd}'."
        )
    return exec_cmd, exec_para


def parse_command_pddl(command: str, para: Dict) -> Tuple[str, str]:
    """Parse PDDL-style command into executor format."""
    def parse_grasp(p: Dict) -> Tuple[str, str]:
        return determine_grasp_type(p.get('para0', '')), format_params(obj_name=p.get('para0'))

    parsers = {
        'move_to': lambda p: ('move_single_to_pose', format_params(target=p.get('para2'))),
        'move_to_from': lambda p: ('move_single_to_pose', format_params(target=p.get('para2'))),
        'move_above': lambda p: ('move_above', format_params(target=p.get('para2'), off_set='up')),
        'pour': lambda p: ('pour', '{}'),
        'wait': lambda p: ('wait', '{}'),
        'grasp': parse_grasp,
        'grasp_from_air': parse_grasp,
        'grasp_from_block': parse_grasp,
        'grasp_a': parse_grasp,
        'grasp_b': parse_grasp,
        'release': lambda p: ('release', format_params(obj_name=p.get('para0'), point=p.get('para2'))),
        'push': lambda p: ('push_to', format_params(source_obj=p.get('para0'), target_obj=p.get('para3'))),
        'place': lambda p: ('place', format_params(source_obj=p.get('para0'), target_obj=p.get('para2'))),
        'co_move_to': lambda p: ('co_move_to', format_params(obj_name=p.get('para1'))),
        'co_hold': lambda p: ('co_hold', format_params(obj_name=p.get('para0'))),
    }

    if command in parsers:
        exec_cmd, exec_para = parsers[command](para)
        return check_completeness(command, exec_cmd, exec_para)
    return command, format_params(**para)


def parse_command_direct(command: str, para: Dict) -> Tuple[str, str]:
    """Parse direct-style command into executor format."""
    parsers = {
        'move_to': lambda p: ('move_single_to_pose', format_params(obj_name=p.get('para0'), target=p.get('para1'))),
        'move_to_from': lambda p: ('move_single_to_pose', format_params(obj_name=p.get('para0'), target=p.get('para1'))),
        'move_above': lambda p: ('move_above', format_params(target=p.get('para1'), off_set='up')),
        'pour': lambda p: ('pour', '{}'),
        'wait': lambda p: ('wait', '{}'),
        'grasp': lambda p: (determine_grasp_type(p.get('para0', '')), format_params(obj_name=p.get('para0'))),
        'release': lambda p: ('release', format_params(obj_name=p.get('para0'), point=p.get('para1'))),
        'push': lambda p: ('push_to', format_params(source_obj=p.get('para0'), target_obj=p.get('para1'))),
        'push_to': lambda p: ('push_to', format_params(source_obj=p.get('para0'), target_obj=p.get('para1'))),
        'place': lambda p: ('place', format_params(source_obj=p.get('para0'), target_obj=p.get('para1'))),
        'co_move_to': lambda p: ('co_move_to', format_params(obj_name=p.get('para0'))),
        'co_hold': lambda p: ('co_hold', format_params(obj_name=p.get('para0'))),
    }

    if command in parsers:
        # Completeness check against the skill signature in bipop/llm/prompts.py
        # (e.g. move_to needs both source_obj and target_point).
        error = check_direct_completeness(command, para)
        if error:
            return INVALID_COMMAND, error
        return parsers[command](para)
    return command, format_params(**para)


def transform_commands(command_dicts: List[Dict], is_pddl: bool) -> Tuple[List[str], List[str]]:
    """Transform command dictionaries to executor format."""
    parser = parse_command_pddl if is_pddl else parse_command_direct

    commands = []
    params = []

    for cmd_dict in command_dicts:
        if cmd_dict['commands'] == INVALID_COMMAND:
            cmd, para = INVALID_COMMAND, cmd_dict['error']
        else:
            cmd, para = parser(cmd_dict['commands'], cmd_dict['paras'])
        commands.append(cmd)
        params.append(para)

    return commands, params


def same_move_target(left_cmd: str, left_para: str, right_cmd: str, right_para: str) -> bool:
    """True if both hands are sent to the same target in the same step."""
    if left_cmd != right_cmd or left_cmd not in MOVE_COMMANDS:
        return False
    try:
        left_target = json.loads(left_para).get('target')
        right_target = json.loads(right_para).get('target')
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False
    return bool(left_target) and left_target == right_target


# =====================================================================
# SUBPROCESS EXECUTION
# =====================================================================
def run_executor(side: str, command: str, params: str, state: HandState,
                 other_holding: Optional[str] = None, other_command: Optional[str] = None) -> subprocess.Popen:
    """Launch executor subprocess for specified side."""
    args = [
        sys.executable, EXECUTOR_PATH,
        '--side', side,
        '--command', command,
        '--para', params,
        f'--hand_state={state.hand}',
        f'--finger_state={state.finger}',
        f'--other_holding={other_holding or ""}',
        f'--other_command={other_command or ""}'
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)


def run_head_command(look_at: str) -> subprocess.Popen:
    """Launch head movement subprocess."""
    args = [
        sys.executable, EXECUTOR_PATH,
        '--command', 'look_at',
        '--para', f'{{"obj_name":"{look_at}"}}'
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)


# =====================================================================
# MAIN CONTROLLER
# =====================================================================
def nicol_control(file_path: str, is_pddl: bool, task) -> Tuple[bool, int]:
    """
    Execute a bimanual robot plan.

    Args:
        file_path: Path to JSON plan file
        is_pddl: Whether plan is in PDDL format
        task: Task object with check_success() method, or None to only replay the
            plan (success then means that every step was executed without failure)

    Returns:
        Tuple of (success, steps_completed)
    """
    controller = RobotController()

    # Load and parse plan
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    left_plan, right_plan = align_step_plan(data.get('Left') or [], data.get('Right') or [])

    left_cmds, left_params = transform_commands(plan_to_commands(left_plan), is_pddl)
    right_cmds, right_params = transform_commands(plan_to_commands(right_plan), is_pddl)

    num_steps = len(left_cmds)  # both hands have the same number of steps

    for step in range(num_steps):
        left_cmd, left_para = left_cmds[step], left_params[step]
        right_cmd, right_para = right_cmds[step], right_params[step]

        # An unreadable step or a command missing required parameters is a
        # malformed plan step. Feed back False directly with the error message.
        if left_cmd == INVALID_COMMAND or right_cmd == INVALID_COMMAND:
            if left_cmd == INVALID_COMMAND:
                print(f"Left command invalid at step {step + 1}: {left_para}")
            if right_cmd == INVALID_COMMAND:
                print(f"Right command invalid at step {step + 1}: {right_para}")
            print(f"The task failed at step {step + 1} due to an invalid command!")
            return False, step + 1

        # Synchronize co_hold commands
        if 'co_hold' in left_cmd:
            right_cmd, right_para = left_cmd, left_para
        elif 'co_hold' in right_cmd:
            left_cmd, left_para = right_cmd, right_para

        # Determine head direction and handle wait commands
        if 'wait' in left_cmd:
            left_para = '{}'
            look_direction = 'right'
        elif 'wait' in right_cmd:
            right_para = '{}'
            look_direction = 'left'
        else:
            look_direction = 'center'

        # Handle co_move_to (only one side executes)
        if 'co_move_to' in left_cmd:
            right_cmd, right_para = 'wait', '{}'
        elif 'co_move_to' in right_cmd:
            left_cmd, left_para = 'wait', '{}'

        # Check for collision (both hands moving to the same target)
        if same_move_target(left_cmd, left_para, right_cmd, right_para):
            left_result = right_result = "failed. You can not move two objects to the same place!"
            left_hand = controller.left.hand
            left_finger = controller.left.finger
            right_hand = controller.right.hand
            right_finger = controller.right.finger
            print(f"Left: {left_cmd}, {left_para} Right: {right_cmd}, {right_para}")
            print(f"Left result:  {left_result}, {left_hand}, {left_finger}")
            print(f"Right result: {right_result}, {right_hand}, {right_finger}")
        else:
            # Execute commands
            print(f"Left: {left_cmd}, {left_para} Right: {right_cmd}, {right_para}")

            head_proc = run_head_command(look_direction)
            # Pass what the OTHER hand is holding AND doing to each executor
            left_proc = run_executor('left', left_cmd, left_para, controller.left,
                                     other_holding=controller.get_other_holding('left'),
                                     other_command=right_cmd)
            right_proc = run_executor('right', right_cmd, right_para, controller.right,
                                      other_holding=controller.get_other_holding('right'),
                                      other_command=left_cmd)

            # Wait for completion
            l_stdout, l_stderr = left_proc.communicate()
            r_stdout, r_stderr = right_proc.communicate()
            head_proc.communicate()

            # Parse results
            left_result, left_hand, left_finger = extract_result('left', l_stdout)
            right_result, right_hand, right_finger = extract_result('right', r_stdout)
            for side, result, stderr in (('Left', left_result, l_stderr), ('Right', right_result, r_stderr)):
                if result is None and stderr.strip():  # the executor crashed; show why
                    print(f"{side} executor error: {stderr.strip()[-800:]}")

            print(f"Left result:  {left_result}, {left_hand}, {left_finger}")
            print(f"Right result: {right_result}, {right_hand}, {right_finger}")

            # Extract grasped objects from results
            left_grasped = extract_grasped_object(left_result)
            right_grasped = extract_grasped_object(right_result)

            # Update state (including what's being held)
            controller.update_state('left', left_hand, left_finger, holding=left_grasped)
            controller.update_state('right', right_hand, right_finger, holding=right_grasped)

        # Check for failure
        if (left_result is None or right_result is None or
            'failed' in str(left_result).lower() or 'failed' in str(right_result).lower()):
            if task is not None and task.check_success():
                print("The task is successfully accomplished!")
                return True, step + 1
            print(f"The task failed at step {step + 1}!")
            return False, step + 1

    # Final check
    head_proc = run_head_command('human')
    head_proc.communicate()
    time.sleep(4)

    if task is None:
        print(f"All {num_steps} steps were executed (the task goal was not checked).")
        return True, num_steps

    if task.check_success():
        print("The task is successfully accomplished!")
        return True, num_steps

    print(f"The task failed after completing all {num_steps} steps!")
    return False, num_steps


# =====================================================================
# STANDALONE EXECUTION (replays a plan without checking the task goal)
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay a plan on NICOL (the task goal is not checked).")
    parser.add_argument('--plan', type=str, required=True, help="Path to the plan JSON file")
    parser.add_argument('--pddl', action='store_true', help="The plan comes from the BiPOP-agent")
    args = parser.parse_args()

    success, steps = nicol_control(args.plan, args.pddl, task=None)
    print(f"Execution {'finished' if success else 'failed'} after {steps} steps")
