# BiPOP: Partial-Order Planning for Long-Horizon Bimanual Robotic Manipulation

Kun Chu, Xufeng Zhao, Jan-Gerrit Habekost, Cornelius Weber, Stefan Wermter
[[arXiv:2611.15317]](https://arxiv.org/abs/2611.15317)

This repository contains the code of BiPOP, the baselines it is compared with, the simulated tasks and the evaluation tools of the paper.

## Contents

- [Overview](#overview)
- [How BiPOP works](#how-bipop-works)
- [Agents, models and tasks](#agents-models-and-tasks)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs and metrics](#outputs-and-metrics)
- [Configuration](#configuration)
- [Repository structure](#repository-structure)
- [Tests](#tests)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

BiPOP lets a large language model (LLM) plan long-horizon bimanual manipulation through a temporal planner instead of writing the plan itself.
The LLM translates the task description into a planning problem. Temporal Fast Downward then finds a partially ordered temporal plan in which the two arms act in parallel wherever possible, VAL validates it, and the semi-humanoid robot NICOL executes it in CoppeliaSim.
BiPOP is compared with two LLM baselines, open-loop direct planning and closed-loop ReAct, on three simulated tasks.

## How BiPOP works

In this code BiPOP is the **BiPOP-agent** (`bipop-agent`, implemented in `bipop/agents/bipop_agent.py` and `bipop/llm/llm_pddl.py`).

1. **Problem generation.** The LLM receives the task description, the planning domain and task-specific modeling rules (`bipop/llm/prompts.py`). It returns the planning problem as [Unified Planning](https://github.com/aiplan4eu/unified-planning) Python code.
2. **Planning.** The code is completed with the domain and the durative actions of the task (`bipop/llm/code_blocks.py`) and exported to PDDL. Temporal Fast Downward (TFD) runs with several search configurations in parallel (`bipop/llm/planner.py`).
3. **Validation and feedback.** VAL checks every plan TFD finds, and the valid plan with the shortest makespan is kept. If no plan is valid, the relevant VAL errors are sent back to the LLM, which tries again (`--max_trials` attempts in total, default 3).
4. **Two-stage planning (StackBlock only).** StackBlock is also planned in two stages (`bipop/llm/two_stage.py`):
   - stage 1 contains only the goals that bring the blocks to their target points;
   - the state reached by the stage-1 plan, simulated with Unified Planning, is the initial state of stage 2, which contains all goals.

   The one-stage and the two-stage pipelines run in parallel, and the plan with the shorter makespan wins. `--no-divide_goal` turns the two-stage pipeline off. ServeWater and ServeFruit are always planned in one stage.
5. **Execution.** The temporal plan is discretized into steps. Every action lasts one time unit, so step *k* holds the actions of both arms that start in time unit *k*. NICOL executes the steps one after another, both arms of a step at the same time (`NICOL/`).

## Agents, models and tasks

| Agent | Description |
|---|---|
| `bipop-agent` | **BiPOP**: the LLM writes the planning problem, TFD plans and VAL validates (aliases `bipop`, `pddl`) |
| `llm-agent` | Open-loop baseline: the LLM writes the bimanual step plan directly (aliases `llm`, `direct`) |
| `react-agent` | Closed-loop baseline: [ReAct](https://arxiv.org/abs/2210.03629); the LLM chooses one bimanual step at a time and sees the result of each step on NICOL (alias `react`) |

Models: `gpt-4.1` and `o3` (OpenAI), `deepseek-r1` (DeepSeek-R1-0528 via OpenRouter).

| Task | Goal |
|---|---|
| `ServeWater` | Pour the water from the blue cup into the yellow cup, hold the yellow cup at the serve point in front of the human and put the blue cup on the box. |
| `ServeFruit` | Put the apple and the banana into the bowl and bring the bowl to the human with both hands. |
| `StackBlock` | Rearrange six colored blocks, some of them initially stacked, into one or two goal stacks at the serve points. |

Each task has 100 scene configurations in `task_configs/<task>/task_config_<i>.json`.
A configuration contains the object positions, the table area of each object (left, overlap or right), the task description and a description of the scene.

## Installation

**1. Python environment** (Python ≥ 3.10; the simulator stack was tested with 3.10):

```bash
conda create -n bipop python=3.10 && conda activate bipop
pip install -r requirements.txt
```

**2. Planner and validator** (needed by the BiPOP-agent):

- [Temporal Fast Downward](https://github.com/neighthan/tfd/): build it and set `TFD_DIR` to its `downward` directory, the one containing `translate/`, `preprocess/` and `search/`.
- [VAL](https://github.com/KCL-Planning/VAL): build it and set `VAL_BIN` to the `Validate` binary, or put `Validate` on your `PATH`.

```bash
export TFD_DIR=/path/to/tfd/downward
export VAL_BIN=/path/to/VAL/build/linux64/bin/Validate
```

**3. LLM API keys** (read from environment variables only):

```bash
export OPENAI_API_KEY=...        # gpt-4.1, o3
export OPENAI_BASE_URL=...       # optional: an OpenAI-compatible endpoint
export OPENROUTER_API_KEY=...    # deepseek-r1
```

Check that a backend works with `python scripts/check_llm_backend.py --model_name deepseek-r1`.

**4. Simulator** (needed for executing plans, for the ReAct-agent and for sampling new configurations):

1. Install CoppeliaSim **4.5.1** as described in [nicol_coppeliasim](https://github.com/knowledgetechnologyuhh/nicol_coppeliasim). Newer versions may not work.
2. Start CoppeliaSim and open `NICOL/nicol.ttt`. The task objects are loaded automatically.
3. Choose the arm IK solver with `--ik` or the `NICOL_IK` environment variable:

| `--ik` | Solver | Setup |
|---|---|---|
| `cycleik` (default) | [CycleIK](https://github.com/jangerritha/CycleIK), a neural IK solver trained for NICOL's arms | `git clone https://github.com/jangerritha/CycleIK.git && pip install -e CycleIK` |
| `coppelia` | CoppeliaSim's built-in solver (simIK, damped least squares) | none |
| `dls` | Damped least squares on the URDF with pytorch-kinematics | a NICOL URDF: CycleIK ships one, or set `NICOL_URDF` |

## Usage

Run all scripts from the repository root; each one documents its options with `--help`.
Select task instances with `--task_nums N --start_index S` or with `--task_indices 3,7,12`.

> **One results folder per model for the BiPOP-agent.** BiPOP plan files are named by task and index only (`<task>_pddl_plan_<i>.json`), so a run with another model overwrites them. When you compare models, give each one its own folder, e.g. `--results_dir results/gpt-4.1`, and pass the same folder to `plan_validation.py`. LLM- and ReAct-agent files include the model name.

**Generate plans.** The BiPOP- and LLM-agent need no simulator; the ReAct-agent executes on NICOL while it plans.

```bash
python plan_generation.py --task_name StackBlock --model_name gpt-4.1 --agent bipop-agent --task_nums 10
python plan_generation.py --task_name StackBlock --model_name gpt-4.1 --agent bipop-agent --no-divide_goal  # one-stage planning only
python plan_generation.py --task_name ServeFruit --model_name o3 --agent llm-agent
python plan_generation.py --task_name ServeWater --model_name gpt-4.1 --agent react-agent
```

**Execute saved plans on NICOL** (CoppeliaSim running with `NICOL/nicol.ttt`):

```bash
python plan_validation.py --task_name StackBlock --model_name gpt-4.1 --agent bipop-agent --task_nums 10
python plan_validation.py --task_name ServeFruit --model_name o3 --agent llm-agent --ik coppelia
python NICOL/nicol_controller.py --plan results/pddl_plans/StackBlock/StackBlock_pddl_plan_3.json --pddl  # replay one plan, goal not checked
```

**Run generation and execution end to end** for each task instance and each selected agent:

```bash
python main.py --task_name ServeFruit --model_name gpt-4.1 --task_nums 5
python main.py --task_name ServeWater --model_name gpt-4.1 --agents bipop --use_owlv2  # scene description from OWLv2 detections
python main.py --task_name ServeWater --model_name gpt-4.1 --sample_configs            # newly sampled scenes
```

With `--use_owlv2`, OWLv2 gives the table area of the human and of every object. For StackBlock, the initial stacks and the tray area still come from the configuration, since OWLv2 can neither see which blocks are stacked nor detect the tray.

**Evaluate without the simulator:**

```bash
python scripts/gt_pddl_eval.py --tasks StackBlock --methods bipop direct react  # check saved plans with VAL against ground-truth PDDL
python scripts/replay_saved_pddl.py --task_name StackBlock --task_indices 3,7    # re-plan saved LLM code without new LLM calls
```

Add `--log_wandb` to log a run to Weights & Biases (projects `BiPOP-<task>` and `BiPOP-VAL-<task>`).
Set `BIPOP_VERBOSE=1` for a detailed log of the BiPOP pipeline: generated code, TFD configurations, VAL verdicts and timing.

## Outputs and metrics

All outputs are written to `results/` (change with `--results_dir`):

| Path | Content |
|---|---|
| `pddl_plans/<task>/<task>_pddl_plan_<i>.json` | BiPOP step plan; `_1` and `_2` are the stage plans of two-stage planning |
| `pddl_plans/<task>/<task>_llm_pddl_<i>[_stage1\|_stage2].py` | Generated planning scripts (domain, LLM code and actions) |
| `pddl_plans/<task>/intermediate/` | PDDL files, TFD runs, VAL reports, the LLM dialog, the code of every trial and the final temporal plan `<task>_plan_<i>.txt` |
| `direct_plans/<task>/direct_plan_<model>_<i>.json` | LLM-agent plan |
| `react_plans/<task>/react_{plan,result,step_log}_<model>_<i>.json` | ReAct-agent plan, full run and step log |
| `logs/<task>_<model>_{bipop,direct,react}_plan_gen.csv` | Plan generation per instance: success, plan length, LLM and planner time |
| `logs/<task>_<model>_<agent>_val.csv`, `logs/execution_logs/` | Execution on NICOL per instance: success, finished steps, execution time, console log |
| `logs/<task>_<model>_pipeline_eval.csv` | `main.py`: generation and execution per instance and agent |
| `gt_eval/` | `scripts/gt_pddl_eval.py`: ground-truth PDDL, grounded plans, VAL logs and summaries |
| `replay/`, `task_configs/`, `images/`, `wandb/` | Re-planned results, sampled configurations, camera images, W&B files |

### Plan format and execution

All agents produce step plans `{"Left": [...], "Right": [...]}`.
Step *k* of both hands is executed at the same time, and `wait()` keeps a hand still.
NICOL executes a plan with the same rules that `scripts/gt_pddl_eval.py` uses:

- if one hand has fewer steps, it waits in the remaining steps;
- a step `a || b` (two actions of one hand in one step) is executed as two consecutive steps;
- a step that cannot be parsed fails the plan at that step;
- moving both hands to the same target in one step is a failure.

### Step counts

Step counts are computed in the same way for all agents, on the plan as it is executed (`bipop.plan_format.compute_plan_stats`):

| Column | Meaning |
|---|---|
| `plan_length`, `plan_steps` | Number of steps of the plan. For the ReAct-agent, the plan is the executed sequence of steps, including failed ones. |
| `number_of_waits` | Number of times a hand waits in a step, including the steps a shorter hand waits at the end. A hand next to a `co_*` action does not count as waiting, since co-actions use both hands. |
| `finished_steps` | Number of executed steps: up to and including a failed step, or all steps of a plan that ran to the end. The ReAct-agent executes at most `--max_steps` steps. |

### ReAct-agent outcomes

The ReAct-agent executes its steps while planning, so its runs report success directly:

| Column | Meaning |
|---|---|
| `success` (`eval_success` in `main.py`) | The simulator confirms that the task goal is reached (same as `task_achieved`) |
| `llm_claimed_complete` | The LLM itself reported the task as complete |
| `stop_reason` | Why the run stopped: `llm_complete`, `llm_failed`, `parse_error`, `max_steps` or `error` |
| `failure_reason` | Empty for a successful run; otherwise one of the following |

| `failure_reason` | Meaning |
|---|---|
| `Misjudged` | The LLM reported the task as complete, but the goal is not reached |
| `Aborted` | The LLM gave up, or its answer could not be parsed even after a second request |
| `Incomplete` | The step limit `--max_steps` was reached |
| `Error` | The run crashed, e.g. because of an API error |

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI models |
| `OPENROUTER_API_KEY` | `deepseek-r1` |
| `TFD_DIR` | TFD `downward` directory (BiPOP-agent) |
| `VAL_BIN` | VAL `Validate` binary (default: `Validate` on `PATH`). Without VAL, plans are not validated and a warning is printed. |
| `VAL_TOLERANCE` | VAL time tolerance (default `0.0001`, smaller than TFD's time step of 0.001) |
| `NICOL_IK`, `NICOL_URDF` | IK solver of the simulated NICOL, and the URDF for `--ik dls` |
| `BIPOP_VERBOSE` | `1` prints a detailed log of the BiPOP pipeline |

## Repository structure

```
BiPOP/
├── plan_generation.py          # Generate plans for the task configurations
├── plan_validation.py          # Execute saved plans on NICOL and measure task success
├── main.py                     # End to end: (OWLv2 perception →) planning → execution
├── bipop/
│   ├── agents/                 # BiPOP-agent, LLM-agent, ReAct-agent and the agent factory
│   ├── llm/
│   │   ├── client.py           # OpenAI and OpenRouter chat backends
│   │   ├── prompts.py          # System prompts, modeling rules and skill descriptions
│   │   ├── code_blocks.py      # Planning domain, problem initialization and durative actions per task
│   │   ├── llm_pddl.py         # BiPOP generation loop with VAL feedback
│   │   ├── planner.py          # TFD configurations, parallel TFD runs, VAL validation, plan selection
│   │   ├── two_stage.py        # Two-stage planning of StackBlock
│   │   └── llm_direct.py       # LLM-agent
│   ├── perception/owlv2.py     # OWLv2 object detection → table areas (main.py --use_owlv2)
│   ├── plan_format.py          # Plan parsing, discretization, concatenation and step counts
│   ├── experiment.py           # Command-line options and helpers shared by the scripts
│   ├── task_selection.py       # --start_index, --task_nums and --task_indices
│   └── paths.py                # Repository paths, result layout, TFD and VAL locations
├── NICOL/
│   ├── nicol.ttt               # CoppeliaSim scene with NICOL
│   ├── ServeWater.ttm, ServeFruit.ttm, StackBlock.ttm
│   ├── nicol_sim.py            # Simulator interface of NICOL: joint control, arm IK, head, eye cameras
│   ├── nicol_executor.py       # Robot skills (grasp, move_to, place, pour, push, co_hold, ...), one process per hand
│   ├── nicol_controller.py     # Executes a step plan with both hands in parallel
│   └── tasks.py                # Task scenes: reset from configurations, sampling, success checks
├── scripts/
│   ├── gt_pddl_eval.py         # Check saved plans with VAL against rule-based ground-truth PDDL
│   ├── replay_saved_pddl.py    # Re-plan saved LLM code with TFD and VAL, without LLM calls
│   ├── generate_task_configs.py# Sample new task configurations (CoppeliaSim)
│   ├── check_llm_backend.py    # Test an LLM API key and backend
│   └── wandb_plan_monitor.py   # Download plans uploaded to W&B (to plan and execute on different machines)
├── task_configs/<task>/        # 100 scene configurations per task
├── tests/                      # Unit tests (pytest)
└── results/                    # Outputs of the scripts
```

## Tests

```bash
python -m pytest tests
```

The tests need `unified-planning` but no LLM access, TFD, VAL or CoppeliaSim; these are stubbed where needed. They cover:

| File | What it checks |
|---|---|
| `test_val_failure_feedback.py` | VAL verdicts and the error excerpts fed back to the LLM, the retry loop, the trial summary, and the final VAL check of the BiPOP-agent |
| `test_two_stage.py` | Two-stage planning: goal split, stage-1 state transfer, stage-2 time shift when the plans are concatenated, and that only StackBlock uses it |
| `test_stale_artifacts.py` | Plans and PDDL files of earlier runs are removed, and only those of the same task instance |
| `test_nicol_controller.py` | Plan execution on NICOL: hand alignment, `a \|\| b` steps, unreadable steps, same-target moves, replay without goal check |
| `test_plan_stats.py` | The step and wait counts shared by all agents |
| `test_react_agent.py` | ReAct-agent: answer parsing (also of the first answer), success checked by the simulator, failure reasons, the step limit, and the hand state passed to the robot skills |

## Citation

```bibtex
@article{chu2026bipop,
  title   = {BiPOP: Partial-Order Planning for Long-Horizon Bimanual Robotic Manipulation},
  author  = {Chu, Kun and Zhao, Xufeng and Habekost, Jan-Gerrit and Weber, Cornelius and Wermter, Stefan},
  journal = {arXiv preprint arXiv:2611.15317},
  year    = {2026}
}
```

## Acknowledgements

The robot side builds on NICOL and its software from the Knowledge Technology group at the University of Hamburg:

- **NICOL**: [NICOL: A Neuro-inspired Collaborative Semi-humanoid Robot that Bridges Social Interaction and Reliable Manipulation](https://ieeexplore.ieee.org/document/10304130) (Kerzel, Allgeuer, Strahl, Frick, Habekost, Eppe and Wermter).
- **[nicol_coppeliasim](https://github.com/knowledgetechnologyuhh/nicol_coppeliasim)**: the CoppeliaSim environment of NICOL (Lennart Clasmeier, Jan-Gerrit Habekost).
- **[CycleIK](https://github.com/jangerritha/CycleIK)**: [CycleIK: Neuro-inspired Inverse Kinematics](https://link.springer.com/chapter/10.1007/978-3-031-44207-0_38) (Habekost, Strahl, Allgeuer, Kerzel and Wermter), the default IK solver.

Planning uses [Unified Planning](https://github.com/aiplan4eu/unified-planning), [Temporal Fast Downward](http://gki.informatik.uni-freiburg.de/tools/tfd/) and [VAL](https://github.com/KCL-Planning/VAL).
The ReAct baseline follows [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
