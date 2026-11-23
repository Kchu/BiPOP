"""ReAct-agent: robust answer parsing, environment-checked success and failure reasons (no LLM or simulator)."""

import json
from pathlib import Path
from typing import List

import pytest

from bipop.agents import react_agent
from bipop.agents.base import AgentRunContext
from bipop.agents.react_agent import PARSE_RETRY_PROMPT, ReActAgent, ReActExecutionAgent

STEP = '```json\n{"Left": "grasp(apple)", "Right": "wait()"}\n```'
COMPLETE = '```json\n{"status": "COMPLETE", "reason": "done"}\n```'
FAILED = '```json\n{"status": "FAILED", "reason": "cannot reach"}\n```'
GARBAGE = "I am not sure what to do."


class _Executor:
    """Executes nothing; the goal is reached after ``goal_after`` steps (never if None)."""

    def __init__(self, goal_after=None) -> None:
        self.goal_after = goal_after
        self.steps = 0
        self.left_hand_stat = self.right_hand_stat = "Vertical"
        self.left_finger_stat = self.right_finger_stat = "Open"
        self.total_execution_time = 0.0

    def execute_step(self, action):
        self.steps += 1
        achieved = self.goal_after is not None and self.steps >= self.goal_after
        return {"observation": "ok", "task_success": achieved, "left_success": True, "right_success": True}


def _agent(monkeypatch, replies: List[str], max_steps: int = 20) -> ReActAgent:
    """ReAct-agent whose LLM answers with ``replies``; ``agent.sent`` holds the last message of every call."""
    monkeypatch.setattr(react_agent, "build_client", lambda model_name: None)
    agent = ReActAgent("gpt-4.1", "ServeFruit", max_steps=max_steps)
    scripted = iter(replies)
    agent.sent = []

    def fake_chat(client, model_name, messages, temperature):
        agent.sent.append(messages[-1]["content"])
        return next(scripted)

    monkeypatch.setattr(react_agent, "chat", fake_chat)
    return agent


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"Left": "grasp(apple)", "Right": "wait()",}\n```',      # trailing comma
        '```json\n{"Left": ["grasp(apple)"], "Right": "wait()"}\n```',     # hand action is not a string
        "{'Left': 'grasp(apple)', 'Right': 'wait()'}",                     # single quotes
    ],
)
def test_malformed_answers_are_not_parsed(response: str) -> None:
    assert ReActAgent._parse_action(response) is None


def test_well_formed_answers_are_parsed() -> None:
    assert ReActAgent._parse_action(STEP) == {"Left": "grasp(apple)", "Right": "wait()"}
    assert ReActAgent._parse_action(COMPLETE)["status"] == "COMPLETE"


def test_malformed_answer_is_asked_again(monkeypatch) -> None:
    agent = _agent(monkeypatch, ['```json\n{"Left": "grasp(apple)", "Right": "wait()",}\n```', STEP])
    action, _, done = agent.step("observation")
    assert action == {"Left": "grasp(apple)", "Right": "wait()"} and not done
    assert agent.sent[-1] == PARSE_RETRY_PROMPT


@pytest.mark.parametrize(
    "replies, goal_after, max_steps, expected",
    [
        # (success, llm_claimed_complete, stop_reason, failure_reason)
        ([STEP, COMPLETE], 1, 20, (True, True, "llm_complete", "")),
        ([STEP, COMPLETE], None, 20, (False, True, "llm_complete", "Misjudged")),
        ([STEP, FAILED], None, 20, (False, False, "llm_failed", "Aborted")),
        ([STEP, GARBAGE, GARBAGE], None, 20, (False, False, "parse_error", "Aborted")),
        ([STEP, STEP, STEP], None, 2, (False, False, "max_steps", "Incomplete")),
        ([GARBAGE, STEP, COMPLETE], 1, 20, (True, True, "llm_complete", "")),  # first answer without action
        ([STEP, FAILED], 1, 20, (True, False, "llm_failed", "")),  # the environment decides
    ],
)
def test_success_is_checked_by_the_environment(monkeypatch, replies, goal_after, max_steps, expected) -> None:
    result = _agent(monkeypatch, replies, max_steps).run("task", _Executor(goal_after))
    assert (
        result["success"], result["llm_claimed_complete"], result["stop_reason"], result["failure_reason"]
    ) == expected
    assert result["success"] == result["task_achieved"]
    # finished steps are executed bimanual steps, never more than max_steps
    assert result["executed_steps"] == len(result["plan"]["Left"]) == len(result["execution_results"])
    assert 1 <= result["executed_steps"] <= max_steps
    assert result["llm_calls"] >= result["executed_steps"]


def test_outcome_reaches_the_evaluation_and_step_log(monkeypatch, tmp_path: Path) -> None:
    def fake_run(self, task_description, executor):
        return {
            "success": False, "task_achieved": False, "llm_claimed_complete": True,
            "stop_reason": "llm_complete", "failure_reason": "Misjudged",
            "agent_status": "COMPLETE", "final_status": "COMPLETE",
            "plan": {"Left": ["grasp(apple)"], "Right": ["wait()"]}, "executed_steps": 1, "llm_calls": 2,
            "execution_time": 1.0, "llm_time": 2.0, "total_time": 3.0, "conversation_history": [],
        }

    monkeypatch.setattr(react_agent, "build_client", lambda model_name: None)
    monkeypatch.setattr(ReActAgent, "run", fake_run)
    monkeypatch.setattr(react_agent, "NICOLExecutor", lambda task: None)
    generation = ReActExecutionAgent("gpt-4.1", "ServeFruit").generate(
        AgentRunContext(index=3, task_description="", react_plan_dir=tmp_path, task=object())
    )

    evaluation = generation.embedded_evaluation
    assert evaluation.success is False
    assert evaluation.metadata["llm_claimed_complete"] is True
    assert evaluation.metadata["failure_reason"] == "Misjudged"
    assert evaluation.metadata["stop_reason"] == "llm_complete"
    assert evaluation.finished_steps == 1 and evaluation.plan_steps == 1 and evaluation.number_of_waits == 1
    step_log = json.loads((tmp_path / "react_step_log_gpt-4.1_3.json").read_text(encoding="utf-8"))
    assert step_log["failure_reason"] == "Misjudged" and step_log["llm_claimed_complete"] is True


def test_max_steps_limits_the_executed_steps(monkeypatch) -> None:
    executor = _Executor()
    result = _agent(monkeypatch, [STEP] * 10, max_steps=3).run("task", executor)
    assert executor.steps == result["executed_steps"] == 3
    assert result["failure_reason"] == "Incomplete"


def test_steps_completed_in_the_prompt_counts_executed_steps(monkeypatch) -> None:
    agent = _agent(monkeypatch, [STEP, STEP, COMPLETE])
    agent.run("task", _Executor())
    assert "Steps completed: 1" in agent.sent[1] and "Steps completed: 2" in agent.sent[2]


@pytest.mark.parametrize(
    "replies, expected",
    [
        # (stop_reason, failure_reason, llm_claimed_complete)
        ([COMPLETE], ("llm_complete", "Misjudged", True)),
        ([FAILED], ("llm_failed", "Aborted", False)),
        ([GARBAGE, GARBAGE], ("parse_error", "Aborted", False)),
    ],
)
def test_first_answer_without_a_step_ends_the_run(monkeypatch, replies, expected) -> None:
    executor = _Executor()
    result = _agent(monkeypatch, replies).run("task", executor)
    assert (result["stop_reason"], result["failure_reason"], result["llm_claimed_complete"]) == expected
    assert executor.steps == result["executed_steps"] == 0


def test_unreadable_first_answer_is_asked_again(monkeypatch) -> None:
    agent = _agent(monkeypatch, [GARBAGE, STEP, COMPLETE])
    agent.run("task", _Executor(goal_after=1))
    assert agent.sent[1] == PARSE_RETRY_PROMPT


def test_each_hand_is_told_what_the_other_hand_holds_and_does(monkeypatch) -> None:
    """The ReAct executor passes the same hand context to the robot skills as NICOL/nicol_controller.py."""
    launched = []

    class _Process:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

        def communicate(self):
            return self.stdout, ""

    def fake_launch(args):
        options = dict(arg.split("=", 1) for arg in args if arg.startswith("--") and "=" in arg)
        side = args[args.index("--side") + 1] if "--side" in args else None
        command = args[args.index("--command") + 1]
        if side is None:
            return _Process("")
        launched.append((side, command, options["--other_holding"], options["--other_command"]))
        prefix, obj = side.upper(), json.loads(args[args.index("--para") + 1]).get("obj_name")
        if command == "top_grasp":
            result, finger = f"has grasped {obj} from the top", "Closed"
        elif command == "release":
            result, finger = f"has been opened, and the grasped {obj} has been released", "Open"
        else:  # wait: the executor reports the finger state it was started with
            result, finger = "is keeping the current status", options["--finger_state"]
        return _Process(f"{prefix}_RESULT: The robot's {side} hand {result}. {prefix}_HAND: Vertical {prefix}_FINGER: {finger}")

    class _Task:
        sim = None

        @staticmethod
        def check_success() -> bool:
            return False

    monkeypatch.setattr(react_agent.NICOLExecutor, "_launch", staticmethod(fake_launch))
    executor = react_agent.NICOLExecutor(_Task())
    executor.execute_step({"Left": "grasp(apple)", "Right": "wait()"})
    executor.execute_step({"Left": "release(apple, overlap_point_1)", "Right": "grasp(banana)"})
    executor.execute_step({"Left": "wait()", "Right": "wait()"})

    assert launched == [
        ("left", "top_grasp", "", "wait"),
        ("right", "wait", "", "top_grasp"),
        ("left", "release", "", "top_grasp"),
        ("right", "top_grasp", "apple", "release"),  # the left hand still holds the apple during this step
        ("left", "wait", "banana", "wait"),
        ("right", "wait", "", "wait"),  # the apple has been released
    ]
    assert executor.holding == {"left": None, "right": "banana"}
