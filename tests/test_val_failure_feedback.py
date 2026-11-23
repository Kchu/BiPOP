from pathlib import Path

from bipop.agents import bipop_agent
from bipop.agents.base import AgentRunContext
from bipop.agents.bipop_agent import BiPOPAgent
from bipop.llm import llm_pddl, planner


def _invalid_val_report(output: str):
    return {
        "available": True,
        "is_valid": False,
        "value": None,
        "output": output,
        "evidence_lines": planner._extract_val_evidence(output, is_valid=False),
        "origin_excerpt_lines": planner._extract_val_origin_excerpt(output, is_valid=False),
    }


def test_extract_val_excerpt_from_type_checking_error() -> None:
    output = "\n".join(
        [
            "Type-checking grasp_from_air",
            "...action passes type checking.",
            "Type-checking grasp_from_block",
            "...action passes type checking.",
            "Type-checking move_to",
            "...action passes type checking.",
            "Type-checking move_above",
            "...action passes type checking.",
            "Type-checking place",
            "...action passes type checking.",
            "Type-checking release",
            "...action passes type checking.",
            "Object with unknown type: overlap_point_1",
            "Error: Error in type-checking!",
            "",
            "",
            "Errors: 0, warnings: 2",
        ]
    )

    excerpt = planner._extract_val_origin_excerpt(output, is_valid=False)

    assert "L13: Object with unknown type: overlap_point_1" in excerpt
    assert "L14: Error: Error in type-checking!" in excerpt


def test_extract_val_evidence_prioritizes_real_type_checking_error() -> None:
    output = "\n".join(
        [
            "Type-checking grasp_from_air",
            "...action passes type checking.",
            "Type-checking grasp_from_block",
            "...action passes type checking.",
            "Type-checking move_to",
            "...action passes type checking.",
            "Type-checking move_above",
            "...action passes type checking.",
            "Type-checking place",
            "...action passes type checking.",
            "Type-checking release",
            "...action passes type checking.",
            "Object with unknown type: overlap_point_1",
            "Error: Error in type-checking!",
        ]
    )

    evidence = planner._extract_val_evidence(output, is_valid=False)

    assert "Object with unknown type: overlap_point_1" in evidence
    assert "Error: Error in type-checking!" in evidence
    assert "Type-checking grasp_from_air" not in evidence


def test_extract_val_excerpt_from_unsatisfied_precondition() -> None:
    output = "\n".join(
        [
            "Checking next happening (time 5.091)",
            "Plan failed because of unsatisfied precondition in:",
            "(move_to blue_block right_arm overlap_point_1 overlap_area) - start",
            "",
            "Plan failed to execute",
            "",
            "(move_to blue_block right_arm overlap_point_1 overlap_area) - start has an unsatisfied precondition at time 5.091",
            "Failed duration constraint: Set the duration to 1",
        ]
    )

    excerpt = planner._extract_val_origin_excerpt(output, is_valid=False)

    assert "L2: Plan failed because of unsatisfied precondition in:" in excerpt
    assert any("unsatisfied precondition" in line for line in excerpt)


def test_select_val_valid_plan_returns_invalid_candidate_logs(monkeypatch, tmp_path: Path) -> None:
    first_plan = tmp_path / "first_plan.txt.1"
    second_plan = tmp_path / "second_plan.txt.1"
    first_plan.write_text("0.000: (move_to a b c d) [1.000]\n", encoding="utf-8")
    second_plan.write_text("0.000: (move_to a b c d) [2.000]\n", encoding="utf-8")

    def fake_run_val(domain: str, problem: str, plan_file: str):
        return _invalid_val_report(
            "Checking next happening (time 1.000)\n"
            "Plan failed because of unsatisfied precondition in:\n"
            f"({Path(plan_file).name}) - start\n"
            "Plan failed to execute\n"
        )

    monkeypatch.setattr(planner, "run_val", fake_run_val)

    result = planner.select_val_valid_plan(
        "domain.pddl",
        "problem.pddl",
        {"first": [str(first_plan)], "second": [str(second_plan)]},
    )

    feedback = "\n".join(result["val_log_lines"])

    assert result["best_plan_path"] is None
    assert result["valid_candidates"] == 0
    assert "VAL failure details for invalid planner candidates" in feedback
    assert "preset=first" in feedback
    assert "preset=second" in feedback
    assert "VAL original output excerpt:" in feedback
    assert "Plan failed because of unsatisfied precondition in:" in feedback


def test_solve_code_string_failure_report_includes_val_excerpt(monkeypatch, tmp_path: Path) -> None:
    candidate_plan = tmp_path / "candidate_plan.txt.1"
    candidate_plan.write_text("0.000: (move_to a b c d) [1.000]\n", encoding="utf-8")

    def fake_subprocess_run(*args, **kwargs):
        class Result:
            stdout = ""
            stderr = ""
            returncode = 0

        return Result()

    def fake_run_tfd_parallel(*args, **kwargs):
        return {
            "best_preset": None,
            "best_ms": None,
            "best_plan_len": None,
            "best_plan_path": None,
            "best_plan_text": None,
            "all_plan_paths": {"fourth": [str(candidate_plan)]},
            "preset_reports": ["fourth: returncode=0 plans=1 log=fourth.log"],
        }

    def fake_run_val(domain: str, problem: str, plan_file: str):
        return _invalid_val_report(
            "Object with unknown type: overlap_point_1\n"
            "Error: Error in type-checking!\n"
            "\n"
            "Errors: 0, warnings: 2\n"
        )

    monkeypatch.setattr(planner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(planner, "run_tfd_parallel", fake_run_tfd_parallel)
    monkeypatch.setattr(planner, "run_val", fake_run_val)

    result = planner.solve_code_string(
        "problem = None",
        str(tmp_path),
        "StackBlock",
        2,
        "single",
        "action_definition",
        trial=0,
    )

    assert result["success"] is False
    assert "VAL failure details for invalid planner candidates" in result["output"]
    assert "VAL original output excerpt:" in result["output"]
    assert "Object with unknown type: overlap_point_1" in result["output"]


def test_llm_pddl_retries_with_val_failure_feedback(monkeypatch, tmp_path: Path) -> None:
    captured_messages = []
    responses = iter(
        [
            "```python\nproblem.add_goal(object_at_point(red_block, overlap_point_1))\n```",
            "```python\nproblem.add_goal(object_at_point(red_block, overlap_point_2))\n```",
        ]
    )
    failure_report = (
        "No VAL-valid plan was found.\n"
        "Checked plans: 1\n"
        "Valid plans: 0\n"
        "VAL failure details for invalid planner candidates (showing up to 4):\n"
        "preset=first plan=plan.txt.1 makespan=1.000000 plan_length=1\n"
        "  VAL error evidence:\n"
        "    - Object with unknown type: overlap_point_1\n"
        "    - Error: Error in type-checking!\n"
        "  VAL original output excerpt:\n"
        "    L13: Object with unknown type: overlap_point_1\n"
        "    L14: Error: Error in type-checking!\n"
    )
    selected_plan = tmp_path / "selected_plan.txt"
    selected_plan.write_text("0.000: (move_to a b c d) [1.000]\n", encoding="utf-8")
    solve_calls = []

    def fake_build_client(model_name: str):
        return object()

    def fake_get_llm_response(client, model_name, messages, temperature):
        captured_messages.append([dict(message) for message in messages])
        return next(responses)

    def fake_solve_code_string(*args, **kwargs):
        solve_calls.append(kwargs.get("val_bin"))
        if len(solve_calls) == 1:
            return {
                "success": False,
                "output": failure_report,
                "planner_time_cost": 0.0,
            }
        return {
            "success": True,
            "output": "New solution has been found.\n",
            "plan_path": str(selected_plan),
            "plan_text": selected_plan.read_text(encoding="utf-8"),
            "planner_time_cost": 0.0,
        }

    monkeypatch.setattr(llm_pddl, "build_client", fake_build_client)
    monkeypatch.setattr(llm_pddl, "chat", fake_get_llm_response)
    monkeypatch.setattr(llm_pddl, "solve_code_string", fake_solve_code_string)

    code_string, output, _, _ = llm_pddl.llm_pddl(
        "gpt-4o",
        "StackBlock",
        "Stack the blocks.",
        str(tmp_path),
        9,
        max_trials=2,
        divide_goal=False,
        val_bin="/tmp/custom-Validate",
    )

    retry_user_messages = [
        message["content"]
        for message in captured_messages[1]
        if message["role"] == "user"
    ]
    assert code_string is not None
    assert "New solution has been found" in output
    assert solve_calls == ["/tmp/custom-Validate", "/tmp/custom-Validate"]
    assert any("VAL original output excerpt:" in content for content in retry_user_messages)
    assert any("Object with unknown type: overlap_point_1" in content for content in retry_user_messages)


# ---------------------------------------------------------------------------
# Two-stage winner: combined plan must be validated against single-pipeline PDDL
# ---------------------------------------------------------------------------

def _make_two_stage_winner_output(task_name: str, index: int, idir: str) -> str:
    """Return a planner_output string that looks like a two-stage win."""
    return (
        "New solution has been found.\n"
        f"Plan 1: {idir}/{task_name}_plan_{index}_1.txt\n"
        f"Plan 2: {idir}/{task_name}_plan_{index}_2.txt\n"
        f"Final plan: {idir}/{task_name}_plan_{index}.txt\n"
        "Stage 1 best valid planner preset: first\n"
        "Stage 2 best valid planner preset: second\n"
        "Winner: two-stage (combined makespan 4.041000 <= single makespan 5.051000)\n"
    )


def test_two_stage_winner_validates_combined_plan_against_single_pddl(
    monkeypatch, tmp_path: Path
) -> None:
    """When two-stage wins, the combined plan is validated against the single-pipeline PDDL.

    The single-pipeline PDDL has the same initial state as stage1 and the same
    final goals as stage2, so it is the correct end-to-end validator for the
    combined stage1+stage2 plan.  The fix removed the ``not two_stage_winner``
    guard that was incorrectly skipping this check.

    Important: generate() cleans up old *_plan_{index}*.txt* files before calling
    llm_pddl(), so all plan artifacts must be written *by* fake_llm_pddl (just as
    the real llm_pddl writes them) so they exist when the post-hoc VAL check runs.
    """
    task_name = "StackBlock"
    index = 7
    pddl_plan_dir = tmp_path / "pddl_plans"
    pddl_plan_dir.mkdir(parents=True, exist_ok=True)
    idir = pddl_plan_dir / "intermediate"
    idir.mkdir(parents=True, exist_ok=True)

    combined_plan = idir / f"{task_name}_plan_{index}.txt"
    # Single-pipeline PDDL files (domain + problem, no stage suffix).
    single_domain = idir / f"{task_name}_domain_{index}.pddl"
    single_problem = idir / f"{task_name}_problem_{index}.pddl"

    planner_output = _make_two_stage_winner_output(task_name, index, str(idir))

    # Track every run_val call made by the agent.
    val_calls: list[dict] = []

    def fake_run_val(domain: Path, problem: Path, plan_file: Path, val_bin: Path, **kwargs) -> dict:
        val_calls.append(
            {
                "domain": str(domain),
                "problem": str(problem),
                "plan_file": str(plan_file),
                "val_bin": str(val_bin),
            }
        )
        return {"available": True, "is_valid": True, "value": 2.0, "output": "Plan valid"}

    def fake_llm_pddl(*args, **kwargs):
        # Simulate what the real llm_pddl writes: the combined plan + single-pipeline PDDL.
        # generate() deletes old *_plan_{index}*.txt* artifacts before this call, so
        # these files must be written here (after cleanup) for the VAL check to see them.
        combined_plan.write_text(
            "0.000000: (move_to red_block left_arm overlap_point_1 left_area) [1.000000]\n"
            "1.001000: (move_to red_block right_arm overlap_point_2 right_area) [1.000000]\n",
            encoding="utf-8",
        )
        single_domain.write_text("(define (domain test))", encoding="utf-8")
        single_problem.write_text("(define (problem test))", encoding="utf-8")
        return "code_string", planner_output, 0.5, 1.0

    monkeypatch.setattr(bipop_agent, "llm_pddl", fake_llm_pddl)
    monkeypatch.setattr(bipop_agent, "run_val", fake_run_val)

    custom_val_bin = tmp_path / "Validate"
    agent = BiPOPAgent(
        model_name="gpt-4o",
        task_name=task_name,
        max_trials=1,
        divide_goal=True,
    )
    context = AgentRunContext(
        index=index,
        task_description="Stack the blocks.",
        pddl_plan_dir=pddl_plan_dir,
        validate_with_val=True,
        val_bin=custom_val_bin,
        divide_goal=True,
    )

    result = agent.generate(context)

    # The combined plan must have been validated against the single-pipeline PDDL.
    assert val_calls, "run_val was never called - combined plan was not validated"
    call = val_calls[0]
    assert call["val_bin"] == str(custom_val_bin)
    assert call["domain"] == str(single_domain), (
        "VAL was not called with the single-pipeline domain PDDL"
    )
    assert call["problem"] == str(single_problem), (
        "VAL was not called with the single-pipeline problem PDDL"
    )
    assert call["plan_file"] == str(combined_plan), (
        "VAL was not called with the combined (stage1+stage2) plan file"
    )
    assert result.success is True
    assert result.metadata.get("val", {}).get("is_valid") is True


def test_two_stage_winner_combined_plan_invalid_marks_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """If the combined plan fails single-pipeline VAL, the result is marked failed.

    This guards against a mismatched stage1/stage2 pair where the combined
    plan does not actually satisfy the original problem's goals.
    """
    task_name = "StackBlock"
    index = 8
    pddl_plan_dir = tmp_path / "pddl_plans"
    pddl_plan_dir.mkdir(parents=True, exist_ok=True)
    idir = pddl_plan_dir / "intermediate"
    idir.mkdir(parents=True, exist_ok=True)

    combined_plan = idir / f"{task_name}_plan_{index}.txt"
    single_domain = idir / f"{task_name}_domain_{index}.pddl"
    single_problem = idir / f"{task_name}_problem_{index}.pddl"

    planner_output = _make_two_stage_winner_output(task_name, index, str(idir))

    def fake_run_val(domain, problem, plan_file, val_bin, **kwargs):
        return {
            "available": True,
            "is_valid": False,
            "value": None,
            "output": "Plan failed to execute\nGoal not satisfied",
        }

    def fake_llm_pddl(*args, **kwargs):
        combined_plan.write_text(
            "0.000000: (move_to red_block left_arm overlap_point_1 left_area) [1.000000]\n",
            encoding="utf-8",
        )
        single_domain.write_text("(define (domain test))", encoding="utf-8")
        single_problem.write_text("(define (problem test))", encoding="utf-8")
        return "code_string", planner_output, 0.5, 1.0

    monkeypatch.setattr(bipop_agent, "llm_pddl", fake_llm_pddl)
    monkeypatch.setattr(bipop_agent, "run_val", fake_run_val)

    agent = BiPOPAgent(
        model_name="gpt-4o",
        task_name=task_name,
        max_trials=1,
        divide_goal=True,
    )
    context = AgentRunContext(
        index=index,
        task_description="Stack the blocks.",
        pddl_plan_dir=pddl_plan_dir,
        validate_with_val=True,
        val_bin=tmp_path / "Validate",
        divide_goal=True,
    )

    result = agent.generate(context)

    assert result.success is False, (
        "A two-stage plan that fails single-pipeline VAL must be reported as failed"
    )
    assert result.metadata.get("val", {}).get("is_valid") is False


def test_run_val_goal_failure_is_not_valid(monkeypatch) -> None:
    """VAL prints "Plan executed successfully - checking goal" before a failed goal test."""
    from types import SimpleNamespace

    failed = (
        "Plan executed successfully - checking goal\nGoal not satisfied\nPlan invalid\n\n"
        "Plan Repair Advice:\n\nThe goal is not satisfied\n(Follow each of:\n    (Set (water_in_cup cup) to true)\n)\n"
    )
    valid = "Plan executed successfully - checking goal\nPlan valid\nFinal value: 3.003\n"
    for output, expected, value in ((failed, False, None), (valid, True, 3.003)):
        monkeypatch.setattr(planner.subprocess, "run", lambda *a, _o=output, **k: SimpleNamespace(stdout=_o))
        report = planner.run_val("d.pddl", "p.pddl", "plan.txt", val_bin="Validate")
        assert report["is_valid"] is expected
        assert report["value"] == value
    assert planner.val_output_is_valid(failed) is False


def test_trial_summary_records_every_failed_two_stage_trial(monkeypatch, tmp_path: Path) -> None:
    """With divide_goal, a trial in which both pipelines find no plan appears in llm_trial_summary_<i>.json."""
    import json

    code = "```python\nproblem.add_goal(object_at_point(red_block, serve_point_1))\n```"
    no_plan = {"success": False, "output": "No VAL-valid plan was found.\n", "planner_time_cost": 0.0}
    monkeypatch.setattr(llm_pddl, "build_client", lambda model_name: object())
    monkeypatch.setattr(llm_pddl, "chat", lambda client, model_name, messages, temperature: code)
    monkeypatch.setattr(llm_pddl, "solve_code_string", lambda *args, **kwargs: dict(no_plan))

    _, output, _, _ = llm_pddl.llm_pddl(
        "gpt-4.1", "StackBlock", "Stack the blocks.", str(tmp_path), 4, max_trials=2, divide_goal=True
    )

    summary = json.loads((tmp_path / "llm_trial_summary_4.json").read_text(encoding="utf-8"))
    assert summary["success"] is False and summary["trials_attempted"] == 2
    assert [error["trial"] for error in summary["trial_errors"]] == [0, 1]
    assert "Single pipeline also failed" in output
