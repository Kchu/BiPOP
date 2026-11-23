"""ReAct-agent: interleaves reasoning, one bimanual action and the observation from NICOL.

Based on "ReAct: Synergizing Reasoning and Acting in Language Models"
(https://arxiv.org/abs/2210.03629). Every step is executed in CoppeliaSim and
the result is fed back to the LLM before it chooses the next step.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from NICOL.nicol_controller import extract_grasped_object

from ..llm.client import build_client, chat
from ..llm.prompts import choose_skill_desc
from ..paths import EXECUTOR_PATH, ROOT_DIR
from ..plan_format import compute_plan_stats
from .base import AgentRunContext, BasePlanningAgent, EvaluationResult, GenerationResult

# =============================================================================
# Prompts
# =============================================================================

REACT_SYSTEM_PROMPT = """You are a ReAct agent controlling a bimanual robot. You solve tasks by interleaving Thought, Action, and Observation steps.

### WORKSPACE LAYOUT
Three table areas:
  - Left Area: only the left hand can reach;
  - Right Area: only the right hand can reach;
  - Overlap Area: at the center of the table, both hands can reach.
- The robot's hands are initialized vertically and open at the beginning.
- overlap_point_1 and overlap_point_2 are two points in the overlap area, accessible by both hands.

### SPATIO-TEMPORAL RULES
You should follow the spatio-temporal rules below:

**Uncoordination**: for task-related objects at left and right areas, two hands manipulate the object in their own area independently.
1). In left and right areas, each hand's manipulations are sequential;
2). At the same time, the manipulations of the left and right hand do not interfere with each other and can be performed in parallel.

**Coordination**: for task-related objects in overlap area (or moved from left/right to overlap area), two hands manipulate the object in the overlap area dependently.
1). *Synchronous*: both hands act symmetrically and simultaneously for mutual supportive collaboration. Left and right hand's action needs to be the same and executed at the same time.
2). *Asynchronous*: one hand's action relies on another hand's action first to meet spatial pre-conditions.

Based on the spatial relationships of the objects, the specific object can be transferred via the overlap area for further manipulations.

### REASONING PROCESS
For each step, you must:
1. **Thought**: Analyze the current state, what has been done, and what needs to be done next. Consider:
   - Current positions and states of objects
   - What each hand is holding (if anything)
   - Spatial constraints (which hand can reach which area)
   - Dependencies between actions (e.g., must grasp before moving)

2. **Action**: Output a single timestep action for both hands in the format:
   ```json
   {"Left": "command(params)", "Right": "command(params)"}
   ```
   Use "wait()" if a hand should not act this timestep.

3. **Observation**: You will receive feedback about the action's result.

### RULES
- Execute ONE timestep per turn (both hands can act in parallel if possible)
- For coordinated actions (co_hold, co_move_to), both hands must execute the same command simultaneously
- Think step-by-step about spatial relationships and preconditions
- If an action fails, analyze why and try an alternative approach
- Continue until the task goal is achieved or you determine it's impossible
- Generate the valid plan with the least number of steps for accomplishing the task goal based on the current task description.
"""

REACT_INITIAL_PROMPT = """
## Current Task
{task_description}

## Available Skills
{skill_description}

## Instructions
Solve this task step by step using the ReAct framework:
- Think about the current state and what action to take
- Output exactly ONE action (one timestep for both hands)
- Wait for the observation result before continuing

Begin by analyzing the initial state and outputting your first Thought and Action.
"""

REACT_CONTINUE_PROMPT = """
## Observation from previous action (Step {step}):
{observation}

## Current State Summary:
- Steps completed: {step}
- Left hand: {left_state}
- Right hand: {right_state}
{task_status}

Continue with your next Thought and Action to progress toward the goal.
If the task is complete (goal achieved), respond with:
```json
{{"status": "COMPLETE", "reason": "task goal achieved"}}
```
If the task cannot be completed (e.g., repeated failures), respond with:
```json
{{"status": "FAILED", "reason": "explanation of why"}}
```
"""

PARSE_RETRY_PROMPT = (
    "I couldn't parse your action. Please provide the action in the exact JSON format: "
    "```json\n{\"Left\": \"command(params)\", \"Right\": \"command(params)\"}\n```"
)

# Why a ReAct run that did not reach the goal failed, by the reason the loop stopped.
FAILURE_REASONS = {
    "llm_complete": "Misjudged",   # the LLM reported COMPLETE, but the goal is not reached
    "llm_failed": "Aborted",       # the LLM gave up (FAILED)
    "parse_error": "Aborted",      # no readable action, even after asking again
    "max_steps": "Incomplete",     # the step limit was reached
}


def react_failure_reason(task_achieved: bool, stop_reason: Optional[str]) -> str:
    """'' for a run that reached the goal, else Misjudged / Aborted / Incomplete."""
    if task_achieved:
        return ""
    return FAILURE_REASONS.get(stop_reason or "", "Aborted")


# =============================================================================
# Execution of one step on NICOL
# =============================================================================

def _failed_step(left_action: str, right_action: str, reason: str, error_msg: str) -> Dict[str, Any]:
    return {
        "left_result": f"FAILED - {reason}",
        "right_result": f"FAILED - {reason}",
        "left_success": False,
        "right_success": False,
        "task_success": False,
        "observation": f"Left hand [{left_action}]: FAILED\nRight hand [{right_action}]: FAILED\n{error_msg}",
        "execution_time": 0,
    }


class NICOLExecutor:
    """Executes one bimanual step on NICOL (one executor subprocess per hand) and reports the result."""

    def __init__(self, task: Any):
        self.task = task
        self.sim = task.sim
        self.left_hand_stat = "Vertical"
        self.left_finger_stat = "Open"
        self.right_hand_stat = "Vertical"
        self.right_finger_stat = "Open"
        # Object each hand holds, tracked like NICOL/nicol_controller.py does for the other agents.
        self.holding: Dict[str, Optional[str]] = {"left": None, "right": None}
        self.executed_steps: List[Dict[str, Any]] = []
        self.total_execution_time = 0.0

    @staticmethod
    def _extract_result(side: str, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """(result sentence, hand state, finger state) from the executor output."""
        if not text:
            return None, None, None
        prefix = side.upper()
        sentence = re.search(rf"{prefix}_RESULT:\s*(.*?)\.\s*{prefix}_HAND:", text, re.DOTALL)
        hand = re.search(rf"{prefix}_HAND:\s*(\w+)", text)
        finger = re.search(rf"{prefix}_FINGER:\s*(\w+)", text)
        return (
            sentence.group(1).strip() if sentence else None,
            hand.group(1) if hand else None,
            finger.group(1) if finger else None,
        )

    @staticmethod
    def _parse_action_string(action_str: str) -> Tuple[str, Dict[str, str]]:
        """``grasp(blue_cup)`` -> ``("grasp", {"para0": "blue_cup"})``."""
        if action_str in ["wait", "wait()", None, ""]:
            return "wait", {}
        match = re.match(r'(\w+)\(\s*([\'"\w\s,_-]*)\s*\)?$', action_str)
        if not match:
            return action_str, {}
        params_str = match.group(2)
        if not params_str:
            return match.group(1), {}
        params = [param.strip().strip("'\"") for param in params_str.split(",") if param.strip()]
        return match.group(1), {f"para{i}": param for i, param in enumerate(params)}

    @staticmethod
    def _convert_to_executor_command(command: str, params: Dict[str, str]) -> Tuple[str, str]:
        """Skill call -> (executor command, JSON parameters) of NICOL/nicol_executor.py."""
        if command == "move_to":
            return "move_single_to_pose", json.dumps({"obj_name": params.get("para0", ""), "target": params.get("para1", "")})
        if command == "move_above":
            return "move_above", json.dumps({"target": params.get("para1", ""), "off_set": "up"})
        if command == "pour":
            return "pour", "{}"
        if command in ("wait", "wait()"):
            return "wait", "{}"
        if command == "grasp":
            obj_name = params.get("para0", "")
            if "cup" in obj_name:
                executor_cmd = "side_grasp"
            elif any(item in obj_name for item in ["apple", "banana", "block"]):
                executor_cmd = "top_grasp"
            else:
                executor_cmd = "side_grasp"
            return executor_cmd, json.dumps({"obj_name": obj_name})
        if command == "release":
            return "release", json.dumps({"obj_name": params.get("para0", ""), "point": params.get("para1", "")})
        if command in ("push", "push_to"):
            return "push_to", json.dumps({"source_obj": params.get("para0", ""), "target_obj": params.get("para1", "")})
        if command == "place":
            return "place", json.dumps({"source_obj": params.get("para0", ""), "target_obj": params.get("para1", "")})
        if command in ("co_move_to", "co_hold"):
            return command, json.dumps({"obj_name": params.get("para0", "")})
        return command, json.dumps(params) if params else "{}"

    @staticmethod
    def _launch(args: List[str]) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, str(EXECUTOR_PATH), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT_DIR,
        )

    def execute_step(self, action: Dict[str, str]) -> Dict[str, Any]:
        """Execute one timestep for both hands and check the task goal."""
        start_time = time.time()
        left_action_str = action.get("Left", "wait()")
        right_action_str = action.get("Right", "wait()")
        left_cmd, left_params = self._parse_action_string(left_action_str)
        right_cmd, right_params = self._parse_action_string(right_action_str)

        # Conflicting bimanual actions are rejected without execution.
        if "grasp" in left_cmd and "grasp" in right_cmd and left_params.get("para0") == right_params.get("para0"):
            print("  ERROR: Both commands are grasping the same object - invalid action!")
            return _failed_step(
                left_action_str, right_action_str,
                "Cannot grasp the object while other hand is also grasping the same object",
                "ERROR: Both hands cannot grasp the same object simultaneously! ",
            )
        if left_cmd == "move_to" and right_cmd == "move_to" and left_params.get("para1") == right_params.get("para1"):
            print("  ERROR: Both hands are moving to the same point - invalid action!")
            return _failed_step(
                left_action_str, right_action_str,
                "Cannot move both hands to the same place simultaneously",
                "ERROR: Can not move the both hand to the same place at the same time!",
            )
        if left_cmd == "move_above" and right_cmd == "move_above" and left_params.get("para1") == right_params.get("para1"):
            print("  ERROR: Both hands are moving above the same object - invalid action!")
            return _failed_step(
                left_action_str, right_action_str,
                "Cannot move both hands above the same object simultaneously",
                "ERROR: Can not move the both hand to the same place at the same time!",
            )

        left_exec_cmd, left_exec_para = self._convert_to_executor_command(left_cmd, left_params)
        right_exec_cmd, right_exec_para = self._convert_to_executor_command(right_cmd, right_params)

        # co_hold is executed by both hands, co_move_to by one process that moves both arms.
        if "co_hold" in left_exec_cmd:
            right_exec_cmd, right_exec_para = left_exec_cmd, left_exec_para
        if "co_hold" in right_exec_cmd:
            left_exec_cmd, left_exec_para = right_exec_cmd, right_exec_para

        if left_exec_cmd == "wait" and right_exec_cmd != "wait":
            head_target = '{"obj_name":"right"}'
        elif right_exec_cmd == "wait" and left_exec_cmd != "wait":
            head_target = '{"obj_name":"left"}'
        else:
            head_target = '{"obj_name":"center"}'

        if "co_move_to" in left_exec_cmd:
            right_exec_cmd, right_exec_para = "wait", "{}"
        elif "co_move_to" in right_exec_cmd:
            left_exec_cmd, left_exec_para = "wait", "{}"

        print(f"  Executing: Left={left_exec_cmd}({left_exec_para}), Right={right_exec_cmd}({right_exec_para})")
        head_proc = self._launch(["--command", "look_at", "--para", head_target])
        # Each hand also gets what the other hand holds and does in this step, as in
        # NICOL/nicol_controller.py, so the skills apply the same rules for every agent.
        left_proc = self._launch([
            "--side", "left", "--command", left_exec_cmd, "--para", left_exec_para,
            f"--hand_state={self.left_hand_stat}", f"--finger_state={self.left_finger_stat}",
            f"--other_holding={self.holding['right'] or ''}", f"--other_command={right_exec_cmd}",
        ])
        right_proc = self._launch([
            "--side", "right", "--command", right_exec_cmd, "--para", right_exec_para,
            f"--hand_state={self.right_hand_stat}", f"--finger_state={self.right_finger_stat}",
            f"--other_holding={self.holding['left'] or ''}", f"--other_command={left_exec_cmd}",
        ])
        l_stdout, _ = left_proc.communicate()
        r_stdout, _ = right_proc.communicate()
        head_proc.communicate()

        left_result, left_hand, left_finger = self._extract_result("left", l_stdout)
        right_result, right_hand, right_finger = self._extract_result("right", r_stdout)
        print(f"  Parsed - Left: result={left_result}, hand={left_hand}, finger={left_finger}")
        print(f"  Parsed - Right: result={right_result}, hand={right_hand}, finger={right_finger}")

        self.left_hand_stat = left_hand or self.left_hand_stat
        self.left_finger_stat = left_finger or self.left_finger_stat
        self.right_hand_stat = right_hand or self.right_hand_stat
        self.right_finger_stat = right_finger or self.right_finger_stat
        for side, result, finger in (("left", left_result, left_finger), ("right", right_result, right_finger)):
            grasped = extract_grasped_object(result)
            if grasped is not None:
                self.holding[side] = grasped
            elif finger == "Open":
                self.holding[side] = None  # released

        left_success = left_result is not None and "failed" not in str(left_result).lower()
        right_success = right_result is not None and "failed" not in str(right_result).lower()
        task_success = self.task.check_success()
        execution_time = time.time() - start_time
        self.total_execution_time += execution_time

        observation_parts = []
        for side, action_str, success, result in (
            ("Left", left_action_str, left_success, left_result),
            ("Right", right_action_str, right_success, right_result),
        ):
            if action_str not in ["wait", "wait()"]:
                status = "SUCCESS" if success else "FAILED"
                observation_parts.append(f"{side} hand [{action_str}]: {status} - {result if result else 'No response'}")
            else:
                observation_parts.append(f"{side} hand: waiting")
        if task_success:
            observation_parts.append("*** TASK GOAL ACHIEVED! ***")
        observation = "\n".join(observation_parts)

        step_record = {
            "left_result": left_result,
            "right_result": right_result,
            "left_success": left_success,
            "right_success": right_success,
            "task_success": task_success,
            "execution_time": execution_time,
        }
        self.executed_steps.append({"action": action, **step_record})
        return {**step_record, "observation": observation}


# =============================================================================
# ReAct loop
# =============================================================================

class ReActAgent:
    """Reason -> act -> observe loop on NICOL until the goal is reached or ``max_steps``."""

    def __init__(self, model_name: str, task_name: str, temperature: float = 0, max_steps: int = 20):
        self.model_name = model_name
        self.task_name = task_name
        self.temperature = temperature
        self.max_steps = max_steps
        self.client = build_client(model_name)

        self.messages: List[Dict[str, str]] = []
        self.plan_history: List[Dict[str, str]] = []
        self.left_state = "free, vertical, open"
        self.right_state = "free, vertical, open"
        self.step_count = 0  # calls of step()
        self.llm_calls = 0
        self.total_time = 0.0  # LLM time
        self.stop_reason: Optional[str] = None  # llm_complete | llm_failed | parse_error | max_steps

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        start_time = time.time()
        response = chat(self.client, self.model_name, messages, self.temperature)
        self.total_time += time.time() - start_time
        self.llm_calls += 1
        return response

    @staticmethod
    def _load_action(json_str: str) -> Optional[Dict[str, str]]:
        """Parsed status or step JSON; None if it is malformed or a hand's action is not a string."""
        try:
            action = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        if not isinstance(action, dict):
            return None
        if "status" not in action and any(
            not isinstance(action.get(hand, ""), str) for hand in ("Left", "Right")
        ):
            return None
        return action

    @classmethod
    def _parse_action(cls, response: str) -> Optional[Dict[str, str]]:
        """The step JSON ({"Left", "Right"}) or a status JSON ({"status": COMPLETE|FAILED}).

        None if no well-formed JSON is found; the caller then asks the LLM again.
        """
        for pattern in (
            r'```json\s*\n?\{[^}]*"status"\s*:\s*"(COMPLETE|FAILED)"[^}]*\}\s*\n?```',
            r'```json\s*\n?\{[^}]*"Left"[^}]*\}\s*\n?```',
        ):
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = re.sub(r"```json\s*\n?", "", match.group(0))
                json_str = re.sub(r"\s*\n?```", "", json_str)
                action = cls._load_action(json_str)
                if action is not None:
                    return action

        for pattern in (
            r'\{[^{}]*"Left"\s*:\s*"[^"]*"[^{}]*"Right"\s*:\s*"[^"]*"[^{}]*\}',
            r'\{[^{}]*"Right"\s*:\s*"[^"]*"[^{}]*"Left"\s*:\s*"[^"]*"[^{}]*\}',
        ):
            match = re.search(pattern, response)
            if match:
                action = cls._load_action(match.group(0))
                if action is not None:
                    return action
        return None

    def _ask(self) -> Tuple[Optional[Dict[str, str]], str]:
        """Query the LLM; if the answer has no readable action, ask once more for the exact format."""
        response = self._call_llm(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        action = self._parse_action(response)

        if action is None:
            self.messages.append({"role": "user", "content": PARSE_RETRY_PROMPT})
            response = self._call_llm(self.messages)
            self.messages.append({"role": "assistant", "content": response})
            action = self._parse_action(response)
        return action, response

    def _record(self, action: Optional[Dict[str, str]]) -> bool:
        """Record a parsed answer; True if it ends the run (unreadable answer or a status)."""
        if action is None:
            self.stop_reason = "parse_error"
            return True
        if "status" in action:
            self.stop_reason = f"llm_{str(action['status']).lower()}"
            return True
        self.plan_history.append(action)
        return False

    def initialize(self, task_description: str) -> Tuple[Optional[Dict[str, str]], str, bool]:
        """Send the task, get the first action. Returns (action, response, is_done) like ``step``."""
        self.messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REACT_INITIAL_PROMPT.format(
                    task_description=task_description, skill_description=choose_skill_desc(self.task_name)
                ),
            },
        ]
        action, response = self._ask()
        return action, response, self._record(action)

    def step(self, observation: Optional[str] = None, task_achieved: bool = False) -> Tuple[Optional[Dict[str, str]], str, bool]:
        """Send the observation, get the next action. Returns (action, response, is_done)."""
        self.step_count += 1
        executed_steps = len(self.plan_history)  # every planned step has been executed
        if executed_steps >= self.max_steps:
            self.stop_reason = "max_steps"
            return None, "Maximum steps reached", True

        if observation:
            task_status = (
                "\n** The environment reports the TASK GOAL HAS BEEN ACHIEVED! You should respond with COMPLETE status. **"
                if task_achieved
                else ""
            )
            self.messages.append(
                {
                    "role": "user",
                    "content": REACT_CONTINUE_PROMPT.format(
                        step=executed_steps,
                        observation=observation,
                        left_state=self.left_state,
                        right_state=self.right_state,
                        task_status=task_status,
                    ),
                }
            )

        action, response = self._ask()
        return action, response, self._record(action)

    def run(self, task_description: str, executor: NICOLExecutor) -> Dict[str, Any]:
        """Run the ReAct loop with real execution on NICOL."""
        print(f"\n{'=' * 60}")
        print(f"ReAct Agent - Task: {self.task_name}")
        print(f"Model: {self.model_name}")
        print(f"{'=' * 60}\n")

        action, initial_response, is_done = self.initialize(task_description)
        print(f"[Initial Response]\n{initial_response}\n")

        observation = None
        task_achieved = False
        execution_results = []
        while not is_done:  # ``action`` is a bimanual step here; a status or no answer ends the run
            exec_result = executor.execute_step(action)
            observation = exec_result["observation"]
            task_achieved = exec_result["task_success"]
            execution_results.append(exec_result)
            self.left_state = f"{executor.left_hand_stat}, {executor.left_finger_stat}"
            self.right_state = f"{executor.right_hand_stat}, {executor.right_finger_stat}"
            if not exec_result["left_success"] or not exec_result["right_success"]:
                observation += "\n\n** WARNING: One or more actions failed! Consider adjusting your approach. **"
            print(f"[Observation]\n{observation}\n")
            if task_achieved:
                print("[TASK ACHIEVED] Environment reports success!")

            action, response, is_done = self.step(observation, task_achieved=task_achieved)
            print(f"[Step {self.step_count}]\n{response}\n")

        final_plan: Dict[str, List[str]] = {"Left": [], "Right": []}
        for step in self.plan_history:
            final_plan["Left"].append(step.get("Left", "wait()"))
            final_plan["Right"].append(step.get("Right", "wait()"))

        agent_status = action.get("status") if action and "status" in action else None
        if task_achieved:
            final_status = "COMPLETE"
        elif agent_status:
            final_status = agent_status
        elif self.plan_history:
            final_status = "INCOMPLETE"
        else:
            final_status = "UNKNOWN"

        execution_time = executor.total_execution_time
        return {
            # Success is the environment's goal check; the LLM's own claim is logged separately.
            "success": task_achieved,
            "task_achieved": task_achieved,
            "llm_claimed_complete": agent_status == "COMPLETE",
            "stop_reason": self.stop_reason,
            "failure_reason": react_failure_reason(task_achieved, self.stop_reason),
            "agent_status": agent_status,
            "final_status": final_status,
            "task_description": task_description,
            "plan": final_plan,
            "plan_length": len(self.plan_history),
            # Executed bimanual steps (same meaning as finished_steps of the other agents)
            "executed_steps": len(execution_results),
            "llm_calls": self.llm_calls,
            "llm_time": self.total_time,
            "execution_time": execution_time,
            "total_time": self.total_time + execution_time,
            "execution_results": execution_results,
            "conversation_history": self.messages,
        }


# =============================================================================
# Agent wrapper used by the experiment scripts
# =============================================================================

def _json_default(obj: Any) -> str:
    return str(obj) if hasattr(obj, "__dict__") else repr(obj)


def extract_step_log(result: Dict[str, Any]) -> Dict[str, Any]:
    """Step-by-step log (thought, action, observation, execution result) of a ReAct run."""
    conversation = result.get("conversation_history", [])
    execution_results = result.get("execution_results", [])
    steps = []
    exec_idx = 0

    for i, msg in enumerate(conversation):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")

        thought = ""
        thought_match = re.search(
            r"(?:^|\n)(?:##?\s*)?(?:Thought|Think(?:ing)?)[:\s]*(.+?)(?=\n(?:##?\s*)?(?:Action|```json)|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            json_start = content.find("```json")
            if json_start > 0:
                thought = content[:json_start].strip()

        action = None
        action_match = re.search(r'\{[^{}]*"Left"[^{}]*"Right"[^{}]*\}', content) or re.search(
            r'\{[^{}]*"Right"[^{}]*"Left"[^{}]*\}', content
        )
        if action_match:
            try:
                action = json.loads(action_match.group(0))
            except json.JSONDecodeError:
                action = action_match.group(0)

        status = None
        status_match = re.search(r'\{[^{}]*"status"[^{}]*\}', content)
        if status_match:
            try:
                status = json.loads(status_match.group(0))
            except json.JSONDecodeError:
                pass

        observation = ""
        if i + 1 < len(conversation) and conversation[i + 1].get("role") == "user":
            obs_content = conversation[i + 1].get("content", "")
            obs_match = re.search(r"Observation[^:]*:\s*(.+?)(?=\n##|$)", obs_content, re.DOTALL)
            observation = obs_match.group(1).strip() if obs_match else obs_content.strip()

        exec_result = None
        if isinstance(action, dict) and "Left" in action and exec_idx < len(execution_results):
            exec_result = execution_results[exec_idx]
            exec_idx += 1

        if action or status:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "thought": thought,
                    "action": action,
                    "status": status,
                    "observation": observation,
                    "execution_result": exec_result,
                }
            )

    return {
        "task_description": result.get("task_description", ""),
        "final_status": result.get("final_status"),
        "success": result.get("success"),
        "task_achieved": result.get("task_achieved"),
        "llm_claimed_complete": result.get("llm_claimed_complete"),
        "stop_reason": result.get("stop_reason"),
        "failure_reason": result.get("failure_reason"),
        "logged_llm_answers": len(steps),  # answers with an action or a status
        "plan_length": result.get("plan_length", 0),
        "executed_steps": result.get("executed_steps", 0),
        "llm_time": result.get("llm_time", 0),
        "execution_time": result.get("execution_time", 0),
        "steps": steps,
    }


class ReActExecutionAgent(BasePlanningAgent):
    agent_id = "react-agent"
    display_name = "ReAct-agent"
    is_pddl = False

    def __init__(self, model_name: str, task_name: str, temperature: float = 0.0, max_steps: int = 20):
        super().__init__(model_name, task_name)
        self.temperature = temperature
        self.max_steps = max_steps

    def generate(self, context: AgentRunContext) -> GenerationResult:
        """Plan and execute on NICOL; the plan, full result and step log are saved to ``react_plan_dir``."""
        if context.react_plan_dir is None:
            raise ValueError("react_plan_dir must be provided for react-agent")
        if context.task is None:
            raise ValueError("task must be provided for react-agent")

        context.react_plan_dir.mkdir(parents=True, exist_ok=True)
        agent = ReActAgent(self.model_name, self.task_name, self.temperature, self.max_steps)

        start = time.time()
        result = agent.run(context.task_description, executor=NICOLExecutor(context.task))
        elapsed = time.time() - start

        # Same format as the LLM-agent plans, so both can be evaluated the same way.
        plan = result.get("plan", {"Left": [], "Right": []})
        stem = f"{self.model_name}_{context.index}"
        plan_path = context.react_plan_dir / f"react_plan_{stem}.json"
        plan_path.write_text(json.dumps(plan, indent=4), encoding="utf-8")
        (context.react_plan_dir / f"react_result_{stem}.json").write_text(
            json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
        )
        (context.react_plan_dir / f"react_step_log_{stem}.json").write_text(
            json.dumps(extract_step_log(result), indent=2, default=_json_default), encoding="utf-8"
        )

        plan_steps, wait_count = compute_plan_stats(plan)
        outcome = {
            "final_status": result.get("final_status"),
            "agent_status": result.get("agent_status"),
            "llm_claimed_complete": bool(result.get("llm_claimed_complete", False)),
            "stop_reason": result.get("stop_reason") or "",
            "failure_reason": result.get("failure_reason") or "",
        }
        embedded_eval = EvaluationResult(
            success=bool(result.get("success", False)),
            finished_steps=int(result.get("executed_steps", 0)),
            execution_time=float(result.get("execution_time", 0.0)),
            plan_steps=plan_steps,
            number_of_waits=wait_count,
            task_achieved=bool(result.get("task_achieved", False)),
            metadata=outcome,
        )
        return GenerationResult(
            agent_id=self.agent_id,
            plan_path=plan_path,
            plan=plan,
            success=embedded_eval.success,
            llm_time_cost=float(result.get("llm_time", 0.0)),
            generation_time=elapsed,
            plan_length=plan_steps,
            metadata={
                "execution_time": float(result.get("execution_time", 0.0)),
                "total_time": float(result.get("total_time", 0.0)),
                **outcome,
            },
            embedded_evaluation=embedded_eval,
        )
