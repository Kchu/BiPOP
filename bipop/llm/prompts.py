"""Prompts of the BiPOP-agent (PDDL code), the LLM-agent (direct plans) and the skill descriptions.

The prompt texts are kept exactly as used in the experiments.
"""

from .code_blocks import domain_definition, problem_initialization

direct_system_message = """
You are a helpful assistant that programs a **bimanual robot** on a worktable to solve the task given by a human user.

### WORKSPACE LAYOUT
Three table areas:
  - Left Area: only the left hand can reach;
  - Right Area: only the right hand can reach;
  - Overlap Area: at the middle (center) of the table, both hands can reach.
- The robot's hands are initialized vertically and open at the beginning.
- overlap_point_1 and overlap_point_2 are two points in the overlap area, which are accessible by both hands.

### INSTRUCTIONS
First, analyze the areas where the task-related objects are located.

Then follow the spatio-temporal rules below:
**Uncoordination**: for task-related objects at left and right areas, two hands manipulate objects in their own area independently.
1). In left and right areas, each hand's manipulations are sequential.
2). At the same time, left and right hand manipulations do not interfere and can run in parallel.

**Coordination**: for task-related objects in overlap area (or moved from left/right to overlap area), two hands manipulate the object in the overlap area dependently.
1). *Synchronous*: both hands act symmetrically and simultaneously for mutual supportive collaboration. Left and right hand actions must be the same and executed at the same time.
2). *Asynchronous*: one hand's action relies on another hand's action first to meet spatial pre-conditions.

Based on spatial relationships of objects, a specific object can be transferred via the overlap area for further manipulations.
"""

direct_rule_message = """
Generate the valid plan with the least number of steps for accomplishing the task goal based on the current task description.

### RULES
Format it in **one** fenced JSON array only (no prose). All parameters for the skills are chosen only from the defined task-related objects and points. Each element is a timestep.
For example:
```json
[
    {"Left": "command1(paras)", "Right": "command2(paras)"},
    ...,
    {"Left": "command3(paras)", "Right": "command4(paras)"}
]
```
"""

pddl_system_message = f"""
You are a helpful assistant that writes **Unified Planning (UP) Python code** for a bimanual robot task.
Your output must build a planning problem equivalent to a PDDL model.

WORKSPACE LAYOUT
Three table areas:
  - Left Area: only the left hand can reach.
  - Right Area: only the right hand can reach.
  - Overlap Area: center area, both hands can reach.
- Hands start vertical and open.
- overlap_point_1 and overlap_point_2 are in overlap_area and reachable by both hands.

PROVIDED DOMAIN & BASE INITIALIZATION
These are already defined before your generated code runs. Reuse them exactly.
```python
{domain_definition}
```

```python
{problem_initialization}
```

IMPORTANT: USE EXISTING SYMBOL NAMES EXACTLY
- Types: area_type, arm_type, obj_type, point_type
- Existing objects: left_area, right_area, overlap_area, left_arm, right_arm, overlap_point_1, overlap_point_2
- Existing fluents include:
  is_releasable, is_accessible, point_at,
  is_graspable, on_table,
  object_at_area, object_at_point,
  object_on_object, object_above_object,
  water_in_cup

PREDICATE MEANINGS (DO NOT REINTERPRET)
- is_releasable(point): releasing an object at this point is allowed. Default: True.
- is_accessible(point): this point is not currently occupied (by another hand or object). Default: True.
- point_at(point, area): this point is at/in front of this area.
- is_graspable(obj): this object can be grasped by a hand. Default: False — must be set explicitly.
- clear(obj): True if no other object is directly on top of obj AND obj is not held by any arm. Set False explicitly when a block is supporting another block.
- on_table(obj): True if the object is released directly on the table surface (not on another object, not held). set False explicitly when obj is on another object or being held.
- object_at_area(obj, area): this object is in this area (broad region). Always set this for every object.
- object_at_point(obj, point): this object is at this specific named point (serve_point, overlap_point). Only set when the object is at a named point. IMPORTANT: do NOT set object_at_point for the bottom block of an initial stack — it rests on the table, not at a named point. One object cannot be at two points simultaneously.
- object_on_object(obj1, obj2): obj1 is DIRECTLY resting on obj2 (contact/support). Only true for adjacent blocks; not transitive. Default: False.
- object_above_object(obj1, obj2): obj1 is above obj2 with no contact — a temporary state during manipulation (e.g., before a place or pour action). Not a resting state. Default: False.
- water_in_cup(obj): this object currently contains water. Default: False.

DO NOT REDEFINE OR REDECLARE
- Problem, UserTypes, Fluents, Actions, existing areas/arms/overlap points.
- Do not call problem.add_fluent(...).

UP SYNTAX REMINDER
```python
cup = Object("cup", obj_type)
store_point = Object("store_point", point_type)
problem.set_initial_value(predicate(...), True)   # second arg is always True or False
problem.add_goal(predicate(...))    # correct: fluent expression only, no second argument
```

OBJECTIVE OF YOUR OUTPUT
Produce one Python code block that:
1) declares task objects and task points;
2) adds them to the existing problem;
3) sets needed initial values line-by-line with problem.set_initial_value(...);
4) adds goal predicates line-by-line with problem.add_goal(...).


"""

PDDL_RULE_INDEPENDENT = """
Generate code once for the full task.
Return exactly ONE fenced ```python``` block and no other text.

REQUIRED CODE STRUCTURE
1) Define objects and points
    - Include all goal-relevant items and minimal support items for feasibility.
    - Layout:
    ```python
    obj1 = Object("...", ...)
    objs = [ ... ]
    points = [ ... ]
    problem.add_objects(objs + points)
    ```
    - ALWAYS include overlap_point_1 and overlap_point_2 in the points list.

2) Set initial facts (line-by-line)
    - For each point: set point_at(...), is_releasable(...), is_accessible(...), etc.
    - For each object: set object_at_area(...), is_graspable(...), etc.
    - See task-specific rules below for additional predicates.

3) Add goals
    - problem.add_goal(...), one per line. And, Or, or Not are not allowed to be used here.
    - Do NOT add object_at_area or is_graspable as goals.
    - See task-specific rules below for goal encoding.

CONSISTENCY RULES
1. Use only symbols already defined in the provided domain/init blocks.
2. Do not import modules, redefine types/fluents/actions, or create a new Problem.
3. Do not define new actions.
4. Ensure code is executable as-is when appended after the provided initialization.
5. object_on_object(X, Y) = direct contact only (not transitive).
"""

PDDL_RULE_STACK_BLOCK = """
TASK-SPECIFIC RULES (StackBlock)

Initial facts — block predicates:
    - clear(obj) default True; on_table(obj) default True.
    - Only deviate explicitly:
        - A block with another directly on top (Y in object_on_object(X, Y)): set clear(Y) = False.
        - A block not resting on the table (on another object): set on_table(...) = False.
    - For init_top_down [A on B on C]:
        A (top): on_table=False (set it); clear=True (default, skip).
        B (mid): on_table=False (set it); clear=False (set it).
        C (bot): on_table=True (default, skip); clear=False (set it).
    - Set object_on_object(...) for each adjacent pair in every initial stack.
    - Set object_at_point(...) only for objects at a named point (NOT the bottom block of an initial stack).
    - Set object_above_object(...) and water_in_cup(...) only when explicitly true.
    - Do not generate exhaustive pairwise false facts.

Goal encoding — for goal_top_down [A on B on C] at serve_point:

    Position | Predicates to add_goal(...)
    ---------|---------------------------------------------------------------
    top  (A) | object_on_object(A, B),  clear(A)
    mid  (B) | object_on_object(B, C)
    bot  (C) | on_table(C),  object_at_point(C, serve_point)

    No object_at_point, on_table, or clear for middle blocks.
    serve point 2 is at the same area as human.

STACK ENCODING RULE:
    object_on_object(X, Y): X sits on Y → clear(Y)=False, on_table(X)=False.
    clear(X)=True unless another block is explicitly on top of X.
    ERROR: do NOT set clear(X)=False from object_on_object(X, Y) — it is Y that loses clear.

INITIAL vs GOAL — never mix:
    - INITIAL STATE: set_initial_value(...) — stacks from init_top_down.
    - GOAL STATE: add_goal(...) — stacks from goal_top_down in ## Task Goal.
    - goal_top_down stacks must NOT appear in set_initial_value(object_on_object(...)).
    - init_top_down stacks must NOT appear in add_goal(object_on_object(...)).
"""

PDDL_RULE_SERVE_WATER = """
TASK-SPECIFIC RULES (ServeWater)
  -- The serve_point is not on the table, but in the air in front of human.
  -- Do not define clear(...) but define on_table(...) on the goal.
  -- Do not generate exhaustive pairwise false facts.
"""

PDDL_RULE_SERVE_FRUIT = """
TASK-SPECIFIC RULES (ServeFruit)
    - Do not generate exhaustive pairwise false facts.
"""

PDDL_RULE_BY_TASK = {
    "ServeWater": PDDL_RULE_SERVE_WATER,
    "ServeFruit": PDDL_RULE_SERVE_FRUIT,
    "StackBlock": PDDL_RULE_STACK_BLOCK,
}


def choose_pddl_rule_description(task_name: str) -> str:
    canonical_name = _canonical_task_name(task_name)
    task_specific = PDDL_RULE_BY_TASK.get(canonical_name, "")
    if not task_specific:
        print(f"No task-specific PDDL rules for {task_name}!")
    return PDDL_RULE_INDEPENDENT + task_specific

SERVE_WATER_SKILL_DESC = """
Use only names from the given goal-relevant objs and points. All parameters must come from those lists.

SKILLS (signatures & semantics)
- move_to(source_obj, target_point): Move the acting hand—already holding source_obj—to target_point.
  Preconditions: the hand is holding source_obj; the target_point is accessible (not occupied); the target_point is reachable by this hand.
  Effects: the hand is at target_point; still holding source_obj; the target_point is no longer accessible.

- move_above(source_obj, target_object): Position the held source_obj above target_object (no contact).
  Preconditions: the hand is holding source_obj; the target_object is reachable by this hand. there is no other object or hand (moving) above target_object at the current time.
  Effects: the source_obj is above target_object (no contact).

- grasp(source_obj): Move to and grasp source_obj.
  Preconditions: the hand is free; the source_obj is free to be grasped; the source_obj is reachable by this hand.
  Effects: the hand is holding source_obj.

- release(source_obj, target_point): Open hand to place source_obj ON THE TABLE at target_point.
  Preconditions: the hand is holding source_obj; the hand is at target_point; the target_point is releasable.
  Effects: the source_obj is at target_point; the source_obj is graspable; the hand is free; the target_point is no longer accessible.

- pour(source_obj, target_object): Pour contents from held source_obj into target_object.
  Preconditions: the hand is holding source_obj; the source_obj is above target_object (no contact).
  Effects: contents transferred from source_obj to target_object.

- wait(): No operation; the hand (and any held object) maintains its current state.
"""

SERVE_FRUIT_SKILL_DESC = """
Use only names from the given goal-relevant objs and points. All parameters must come from those lists.

SKILLS (signatures & semantics)
- move_to(source_obj, target_point): Move the acting hand—already holding source_obj—to target_point.
  Preconditions: the hand is holding source_obj; the target_point is accessible (not occupied); the target_point is reachable by this hand.
  Effects: the hand is at target_point; still holding source_obj; the target_point is no longer accessible.

- move_above(source_obj, target_object): Position the held source_obj above target_object (no contact).
  Preconditions: the hand is holding source_obj; the target_object is reachable by this hand; there is no other object or hand (moving) above target_object at the current time.
  Effects: the source_obj is above target_object (no contact).

- grasp(source_obj): Move to and grasp source_obj.
  Preconditions: the hand is free; the source_obj is free to be grasped; the source_obj is reachable by this hand.
  Effects: the hand is holding source_obj.

- release(source_obj, target_point): Open hand to place source_obj ON THE TABLE at target_point.
  Preconditions: the hand is holding source_obj; the hand is at target_point; the target_point is releasable.
  Effects: the source_obj is at target_point; the source_obj is graspable; the hand is free; the target_point is no longer accessible.

- place(source_obj, target_object): Place held source_obj onto target_object (object-to-object contact, not a table point).
  Preconditions: the hand is holding source_obj; the source_obj is above target_object; target_object is clear (nothing on top of it).
  Effects: the source_obj is on target_object; target_object is no longer clear.

- push_to(source_obj, target_area): Push the large source_obj to target area (not a point). available areas: left_area, overlap_area, right_area.
  Preconditions: the hand is free; source_obj is free to be pushed; source_obj is reachable by this hand; target_area is reachable by this hand.
  Effects: the source_obj is at target_area; the hand is free.

- co_hold(target_object): BOTH hands simultaneously hold up large target_object from the bottom side. Write as {"Left": "co_hold(target_object)", "Right": "co_hold(target_object)"} at the same timestep.
  Preconditions: both hands are free; target_object is free to be co-held; target_object is in the overlap area (reachable by both hands).
  Effects: both hands are holding target_object.

- co_move_to(target_point): BOTH hands move together to target_point while co-holding an object. Write as {"Left": "co_move_to(target_point)", "Right": "co_move_to(target_point)"} at the same timestep.
  Preconditions: both hands are co-holding an object; target_point is in the overlap area (reachable by both hands).
  Effects: both hands (and the held object) are at target_point.

- wait(): No operation; the hand (and any held object) maintains its current state.
"""

STACK_BLOCKS_SKILL_DESC = """
Use only names from the given goal-relevant objs and points. All parameters must come from those lists.

SKILLS (signatures & semantics)
- move_to(source_obj, target_point): Move the acting hand—already holding source_obj—to target_point.
  Preconditions: the hand is holding source_obj; the target_point is accessible (not occupied); the target_point is reachable by this hand.
  Effects: the hand with source_obj is at target_point; the target_point is no longer accessible.

- move_above(source_obj, target_object): Position held source_obj above target_object (no contact).
  Preconditions: the hand is holding source_obj; target_object is reachable by this hand.
  Effects: the source_obj is above target_object (no contact).

- grasp(source_obj): Move to and grasp source_obj.
  Preconditions: the hand is free; source_obj is reachable by this hand; source_obj is not held by any hand; no other object is on top of source_obj (i.e., source_obj is clear).
  Effects: the hand is holding source_obj; any object below/under the source_obj is clear;

- release(source_obj, target_point): Open hand to place source_obj ON THE TABLE at target_point (not onto another object — use place for stacking).
  Preconditions: the hand is holding source_obj; the hand is at target_point; the target_point is releasable.
  Effects: the source_obj is at target_point; the hand is free; the target_point is no longer accessible.

- place(source_obj, target_object): Place held source_obj onto target_object (object-to-object contact, not a table point).
  Preconditions: the hand is holding source_obj; source_obj is above target_object; target_object is clear (nothing on top of it).
  Effects: the source_obj is on target_object; target_object is no longer clear.

- wait(): No operation; the hand (and any held object) maintains its current state.
"""


def _canonical_task_name(task_name: str) -> str:
    if task_name in {"ServeWater", "ServeFruit"}:
        return task_name
    if "StackBlock" in task_name or "StackBlocks" in task_name:
        return "StackBlock"
    return task_name


SKILL_DESC_BY_TASK = {
    "ServeWater": SERVE_WATER_SKILL_DESC,
    "ServeFruit": SERVE_FRUIT_SKILL_DESC,
    "StackBlock": STACK_BLOCKS_SKILL_DESC,
}


def choose_skill_desc(task_name):
    canonical_name = _canonical_task_name(task_name)
    description = SKILL_DESC_BY_TASK.get(canonical_name)
    if description is None:
        print(f"The description for {task_name} is not specified!")
        return ""
    return description
