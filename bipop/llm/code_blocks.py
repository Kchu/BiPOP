"""Unified Planning code blocks of the BiPOP pipeline.

``domain_definition`` and ``problem_initialization`` are shown to the LLM in the
system prompt, and the action blocks of ``choose_actions`` are sent with it, so
their text is part of the prompt. Any change here changes the experiments.

The generated planning script is assembled as
``domain_definition + problem_initialization + <LLM code> + choose_actions(task)``.
"""

domain_definition = """## System code string:

from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from unified_planning.io import PDDLWriter

## DOMAIN DEFINITION

# UserTypes
area_type = UserType("area_type")
arm_type = UserType("arm_type")
obj_type = UserType("obj_type")
point_type = UserType("point_type")

# Universal Fluents

# point properties
is_releasable = Fluent("is_releasable", BoolType(), arg=point_type) # releasing an object at this point is allowed. Default: True.
is_accessible = Fluent("is_accessible", BoolType(), arg=point_type) # this point is not currently occupied (by another hand or object). Default: True.
# object properties
is_graspable = Fluent("is_graspable", BoolType(), arg=obj_type) # this object can be grasped by a single hand. Default: False — must be set explicitly; never True for a large object.
clear = Fluent("clear", BoolType(), arg=obj_type) # True if no other object is directly on top of obj AND obj is not held by any arm. Default: True — only set False explicitly when another object is directly on top of obj.
on_table = Fluent("on_table", BoolType(), arg=obj_type) # True if the object is resting directly on the table surface (not on another object, not held). Default: True — only set False explicitly when obj is on another object or being held.
# arm properties
# control = Fluent("control", BoolType(), arg=arm) # determine the control of the arm for different agents
available = Fluent("available", BoolType(), arg=arm_type) # determine the availability of the arm
arm_free = Fluent("arm_free", BoolType(), arg=arm_type) # determine if the arm is free

arm_at = Fluent("arm_at", BoolType(), arg1=arm_type, arg2=area_type)  # this arm is at this area.
arm_access = Fluent("arm_access", BoolType(), arg1=arm_type, arg2=area_type) # this arm can access this area.
lifting = Fluent("lifting", BoolType(), arg1=obj_type, arg2=arm_type) # this arm is lifting this object.
object_at_area = Fluent("object_at_area", BoolType(), arg1=obj_type, arg2=area_type) # this object is in this area (broad region). Default: False — always set exactly one area for every object.
object_at_point = Fluent("object_at_point", BoolType(), arg1=obj_type, arg2=point_type) # this object is at this specific named point (serve_point, overlap_point). Default: False — only set when the object is currently at a named point. IMPORTANT: do NOT set object_at_point for blocks in an initial stack (including the bottom block) — the stack rests on the table, not at a named point. One object cannot be at two points simultaneously.
point_at = Fluent("point_at", BoolType(), arg1=point_type, arg2=area_type) # this point belongs to this area. Default: False — set exactly one area for every task point.
water_in_cup = Fluent("water_in_cup", BoolType(), arg1=obj_type) # this object currently contains water. Default: False.
object_on_object = Fluent("object_on_object", BoolType(), arg1=obj_type, arg2=obj_type) # obj1 is DIRECTLY resting on obj2 (contact/support). Only true for adjacent blocks; not transitive. point_type is not applicable. Default: False.
object_above_object = Fluent("above", BoolType(), arg1=obj_type, arg2=obj_type) # obj1 is above obj2 with no contact — a temporary state during manipulation (e.g., before a place or pour action). Not a resting state. point_type is not applicable. Default: False. Always refer to it via the Python name object_above_object (the internal PDDL name 'above' is intentional).

static = Fluent("static", BoolType(), arg=obj_type) # determine if the object is static (not at moving state).

# Universal Objects
left_area = Object("left_area", area_type)
right_area = Object("right_area", area_type)
overlap_area = Object("overlap_area", area_type)
left_arm = Object("left_arm", arm_type)
right_arm = Object("right_arm", arm_type)
areas = [left_area, right_area, overlap_area]

# task_related_objects
overlap_point_1 = Object("overlap_point_1", point_type)
overlap_point_2 = Object("overlap_point_2", point_type)
"""

problem_initialization = """
## Problem code string:

## DEFINE THE PROBLEM AND SET UP THE INITIAL VALUES

problem = Problem("temp_bimanual")
problem.add_objects([left_area, right_area, overlap_area, left_arm, right_arm])
# problem.add_fluent(arm_at, default_initial_value=False)
problem.add_fluent(arm_access, default_initial_value=False)
problem.add_fluent(available, default_initial_value=False)
problem.add_fluent(lifting, default_initial_value=False)
problem.add_fluent(water_in_cup, default_initial_value=False)
problem.add_fluent(object_at_area, default_initial_value=False)
problem.add_fluent(object_at_point, default_initial_value=False)
problem.add_fluent(object_on_object, default_initial_value=False)
problem.add_fluent(object_above_object, default_initial_value=False)
problem.add_fluent(is_graspable, default_initial_value=False)
problem.add_fluent(point_at, default_initial_value=False)
problem.add_fluent(is_releasable, default_initial_value=True)
problem.add_fluent(is_accessible, default_initial_value=True)
problem.add_fluent(clear, default_initial_value=True)
problem.add_fluent(arm_free, default_initial_value=True)
problem.add_fluent(on_table, default_initial_value=True)
problem.add_fluent(static, default_initial_value=True)

# problem.set_initial_value(arm_at(left_arm, left_area), True)
# problem.set_initial_value(arm_at(left_arm, right_area), False)
# problem.set_initial_value(arm_at(left_arm, overlap_area), False)
problem.set_initial_value(arm_access(left_arm, left_area), True)
problem.set_initial_value(arm_access(left_arm, overlap_area), True)
problem.set_initial_value(arm_access(left_arm, right_area), False)
problem.set_initial_value(available(left_arm), True)
# problem.set_initial_value(arm_at(right_arm, right_area), True)
# problem.set_initial_value(arm_at(right_arm, overlap_area), False)
# problem.set_initial_value(arm_at(right_arm, left_area), False)
problem.set_initial_value(arm_access(right_arm, right_area), True)
problem.set_initial_value(arm_access(right_arm, overlap_area), True)
problem.set_initial_value(arm_access(right_arm, left_area), False)
problem.set_initial_value(available(right_arm), True)

problem.set_initial_value(point_at(overlap_point_1, overlap_area), True)
problem.set_initial_value(point_at(overlap_point_2, overlap_area), True)
problem.set_initial_value(is_releasable(overlap_point_1), True)
problem.set_initial_value(is_releasable(overlap_point_2), True)
problem.set_initial_value(is_accessible(overlap_point_1), True)
problem.set_initial_value(is_accessible(overlap_point_2), True)
##############################################################################################
"""


### action definitions: 
grasp_act = """ 

\n
grasp = DurativeAction("grasp", x=obj_type, y=arm_type, z=area_type)
x = grasp.parameter("x")
y = grasp.parameter("y")
z = grasp.parameter("z")
grasp.set_fixed_duration(1)

# grasp.add_condition(StartTiming(), control(y))
grasp.add_condition(StartTiming(), arm_access(y, z))
grasp.add_condition(StartTiming(), object_at_area(x, z))
grasp.add_condition(StartTiming(), available(y))
grasp.add_condition(StartTiming(), is_graspable(x))
grasp.add_condition(StartTiming(), clear(x))
grasp.add_condition(StartTiming(), arm_free(y))

# grasp.add_effect(StartTiming(), lifting(x, y), True)
grasp.add_effect(StartTiming(), arm_free(y), False)
grasp.add_effect(StartTiming(), clear(x), False)
grasp.add_effect(EndTiming(), on_table(x), False)
grasp.add_effect(EndTiming(), lifting(x, y), True)
grasp.add_effect(EndTiming(), arm_free(y), True)
grasp.add_effect(EndTiming(), clear(x), False)
grasp.add_effect(EndTiming(), available(y), False)
"""

stack_block_grasp_act = """ 
grasp_a = DurativeAction("grasp_from_air", x=obj_type, y=arm_type, z=area_type)
x = grasp_a.parameter("x")
y = grasp_a.parameter("y")
z = grasp_a.parameter("z")
grasp_a.set_fixed_duration(1)

grasp_a.add_condition(StartTiming(), arm_access(y, z))
grasp_a.add_condition(StartTiming(), object_at_area(x, z))
grasp_a.add_condition(StartTiming(), available(y))
grasp_a.add_condition(StartTiming(), is_graspable(x))
grasp_a.add_condition(StartTiming(), clear(x))
grasp_a.add_condition(StartTiming(), arm_free(y))
for oo in objs:
    grasp_a.add_condition(StartTiming(), Not(object_on_object(x, oo)))
    # grasp_a.add_condition(StartTiming(), Not(object_on_object(oo, x)))

grasp_a.add_effect(StartTiming(), lifting(x, y), True)
grasp_a.add_effect(StartTiming(), arm_free(y), False)
grasp_a.add_effect(StartTiming(), clear(x), False)
grasp_a.add_effect(EndTiming(), on_table(x), False)
grasp_a.add_effect(EndTiming(), lifting(x, y), True)
grasp_a.add_effect(EndTiming(), arm_free(y), True)
grasp_a.add_effect(EndTiming(), clear(x), False)
grasp_a.add_effect(EndTiming(), available(y), False)
for pp in points:
    grasp_a.add_effect(EndTiming(), object_at_point(x, pp), False)

grasp_b = DurativeAction("grasp_from_block", x=obj_type, y=arm_type, z=area_type, w=obj_type)
x = grasp_b.parameter("x")
y = grasp_b.parameter("y")
z = grasp_b.parameter("z")
w = grasp_b.parameter("w")
grasp_b.set_fixed_duration(1)

grasp_b.add_condition(StartTiming(), arm_access(y, z))
grasp_b.add_condition(StartTiming(), object_at_area(x, z))
grasp_b.add_condition(StartTiming(), available(y))
grasp_b.add_condition(StartTiming(), is_graspable(x))
grasp_b.add_condition(StartTiming(), clear(x))
grasp_b.add_condition(StartTiming(), arm_free(y))
grasp_b.add_condition(StartTiming(), object_on_object(x, w))
# for oo in objs:
#     grasp_b.add_condition(StartTiming(), Not(object_on_object(oo, x)))

grasp_b.add_effect(StartTiming(), arm_free(y), False)
grasp_b.add_effect(StartTiming(), clear(x), False)
grasp_b.add_effect(EndTiming(), on_table(x), False)
grasp_b.add_effect(EndTiming(), lifting(x, y), True)
grasp_b.add_effect(EndTiming(), arm_free(y), True)
grasp_b.add_effect(EndTiming(), clear(x), False)
grasp_b.add_effect(EndTiming(), clear(w), True)
grasp_b.add_effect(EndTiming(), available(y), False)
grasp_b.add_effect(EndTiming(), object_on_object(x, w), False)
grasp_b.add_effect(EndTiming(), object_above_object(x, w), False)
"""

move_to_act = """
# move_to = DurativeAction("move_to", x=obj_type, y=arm_type, z=point_type, w=area_type)
# x = move_to.parameter("x")
# y = move_to.parameter("y")
# z = move_to.parameter("z")
# w = move_to.parameter("w")
# move_to.set_fixed_duration(1)

# move_to.add_condition(StartTiming(), is_graspable(x))
# move_to.add_condition(StartTiming(), lifting(x, y))
# move_to.add_condition(StartTiming(), arm_access(y, w))
# move_to.add_condition(StartTiming(), point_at(z, w))
# move_to.add_condition(StartTiming(), is_accessible(z))
# move_to.add_condition(StartTiming(), arm_free(y))

# for aa in areas:
#     move_to.add_effect(StartTiming(), object_at_area(x, aa), False)
#     # move_to.add_effect(StartTiming(), arm_at(y, aa), False)
# # for pp in points:
# #     move_to.add_effect(StartTiming(), is_accessible(pp), True, condition=And(object_at_point(x, pp), Not(Equals(pp, z))))
# #     move_to.add_effect(StartTiming(), object_at_point(x, pp), False)
# move_to.add_effect(StartTiming(), static(x), False)
# # for oo in objs:
# #     move_to.add_effect(StartTiming(), object_above_object(x, oo), False, condition=Not(Equals(oo, x)))
# #     move_to.add_effect(StartTiming(), object_above_object(oo, x), False, condition=Not(Equals(oo, x)))
# #     move_to.add_effect(StartTiming(), object_on_object(x, oo), False, condition=Not(Equals(oo, x)))
# for oo in objs:
#     move_to.add_effect(StartTiming(), object_above_object(x, oo), False)
#     move_to.add_effect(StartTiming(), object_above_object(oo, x), False)
#     move_to.add_effect(StartTiming(), object_on_object(x, oo), False)
# move_to.add_effect(StartTiming(), arm_free(y), False)
# move_to.add_effect(StartTiming(), is_accessible(z), False)
# move_to.add_effect(EndTiming(), object_at_area(x, w), True)
# # move_to.add_effect(EndTiming(), arm_at(y, w), True)
# move_to.add_effect(EndTiming(), object_at_point(x, z), True)
# move_to.add_effect(EndTiming(), is_accessible(z), False)
# move_to.add_effect(EndTiming(), arm_free(y), True)
# move_to.add_effect(EndTiming(), static(x), True)
# move_to.add_effect(EndTiming(), clear(x), False)
move_to_from = DurativeAction("move_to_from", x=obj_type, y=arm_type, z=point_type, w=area_type, q=point_type)
x = move_to_from.parameter("x")
y = move_to_from.parameter("y")
z = move_to_from.parameter("z")
w = move_to_from.parameter("w")
q = move_to_from.parameter("q")
move_to_from.set_fixed_duration(1)

move_to_from.add_condition(StartTiming(), is_graspable(x))
move_to_from.add_condition(StartTiming(), lifting(x, y))
move_to_from.add_condition(StartTiming(), arm_access(y, w))
move_to_from.add_condition(StartTiming(), point_at(z, w))
move_to_from.add_condition(StartTiming(), is_accessible(z))
move_to_from.add_condition(StartTiming(), arm_free(y))
move_to_from.add_condition(StartTiming(), object_at_point(x, q))
move_to_from.add_condition(StartTiming(), static(x))

for aa in areas:
    move_to_from.add_effect(StartTiming(), object_at_area(x, aa), False)
move_to_from.add_effect(StartTiming(), is_accessible(q), True)
move_to_from.add_effect(StartTiming(), object_at_point(x, q), False)
move_to_from.add_effect(StartTiming(), static(x), False)
for oo in objs:
    move_to_from.add_effect(EndTiming(), object_above_object(x, oo), False)
    move_to_from.add_effect(EndTiming(), object_above_object(oo, x), False)
    move_to_from.add_effect(EndTiming(), object_on_object(x, oo), False)
move_to_from.add_effect(StartTiming(), arm_free(y), False)
move_to_from.add_effect(StartTiming(), is_accessible(z), False)
move_to_from.add_effect(EndTiming(), object_at_area(x, w), True)
move_to_from.add_effect(EndTiming(), object_at_point(x, z), True)
move_to_from.add_effect(EndTiming(), is_accessible(z), False)
move_to_from.add_effect(EndTiming(), arm_free(y), True)
move_to_from.add_effect(EndTiming(), static(x), True)
move_to_from.add_effect(EndTiming(), clear(x), False)
 
move_to = DurativeAction("move_to", x=obj_type, y=arm_type, z=point_type, w=area_type)
x = move_to.parameter("x")
y = move_to.parameter("y")
z = move_to.parameter("z")
w = move_to.parameter("w")
move_to.set_fixed_duration(1)

move_to.add_condition(StartTiming(), is_graspable(x))
move_to.add_condition(StartTiming(), lifting(x, y))
move_to.add_condition(StartTiming(), arm_access(y, w))
move_to.add_condition(StartTiming(), point_at(z, w))
move_to.add_condition(StartTiming(), is_accessible(z))
move_to.add_condition(StartTiming(), arm_free(y))
for pp in points:
    move_to.add_condition(StartTiming(), Not(object_at_point(x, pp)))
move_to.add_condition(StartTiming(), static(x))

for aa in areas:
    move_to.add_effect(StartTiming(), object_at_area(x, aa), False)
move_to.add_effect(StartTiming(), static(x), False)
for oo in objs:
    move_to.add_effect(EndTiming(), object_above_object(x, oo), False)
    move_to.add_effect(EndTiming(), object_above_object(oo, x), False)
    move_to.add_effect(EndTiming(), object_on_object(x, oo), False)
move_to.add_effect(StartTiming(), arm_free(y), False)
move_to.add_effect(StartTiming(), is_accessible(z), False)
move_to.add_effect(EndTiming(), object_at_area(x, w), True)
move_to.add_effect(EndTiming(), object_at_point(x, z), True)
move_to.add_effect(EndTiming(), is_accessible(z), False)
move_to.add_effect(EndTiming(), arm_free(y), True)
move_to.add_effect(EndTiming(), static(x), True)
move_to.add_effect(EndTiming(), clear(x), False)
"""

move_above_act = """\n
move_above = DurativeAction("move_above", x=obj_type, y=arm_type, z=obj_type, w=area_type)
x = move_above.parameter("x")
y = move_above.parameter("y")
z = move_above.parameter("z")
w = move_above.parameter("w")
move_above.set_fixed_duration(1)

move_above.add_condition(StartTiming(), arm_access(y, w))
move_above.add_condition(StartTiming(), object_at_area(z, w))
move_above.add_condition(StartTiming(), lifting(x, y))
move_above.add_condition(StartTiming(), Not(Equals(x, z)))
move_above.add_condition(StartTiming(), is_graspable(x))
move_above.add_condition(StartTiming(), arm_free(y))
move_above.add_condition(StartTiming(), static(z))
for oo in objs:
    move_above.add_condition(StartTiming(), Not(object_above_object(oo, z)))
    move_above.add_condition(StartTiming(), Not(object_on_object(oo, z)))

for aa in areas: 
    move_above.add_effect(StartTiming(), object_at_area(x, aa), False)
    # move_above.add_effect(StartTiming(), arm_at(y, aa), False)
# for pp in points: 
#     move_above.add_effect(StartTiming(), is_accessible(pp), True, condition=object_at_point(x, pp))
#     move_above.add_effect(EndTiming(), object_at_point(x, pp), False)
# for oo in objs:
#     move_above.add_effect(StartTiming(), object_above_object(x, oo), False, condition=Not(Equals(oo, x)))
#     move_above.add_effect(StartTiming(), object_above_object(oo, x), False, condition=Not(Equals(oo, x)))
#     move_above.add_effect(StartTiming(), object_on_object(x, oo), False, condition=Not(Equals(oo, x)))
for oo in objs:
    move_above.add_effect(StartTiming(), object_above_object(x, oo), False)
    move_above.add_effect(StartTiming(), object_above_object(oo, x), False)
    move_above.add_effect(StartTiming(), object_on_object(x, oo), False)
move_above.add_effect(StartTiming(), arm_free(y), False)
move_above.add_effect(StartTiming(), static(x), False)
move_above.add_effect(StartTiming(), static(z), False)
move_above.add_effect(EndTiming(), object_above_object(x, z), True)
move_above.add_effect(EndTiming(), object_at_area(x, w), True)
# move_above.add_effect(EndTiming(), arm_at(y, w), True)
move_above.add_effect(EndTiming(), arm_free(y), True)
move_above.add_effect(EndTiming(), static(x), True)
move_above.add_effect(EndTiming(), static(z), True)
move_above.add_effect(EndTiming(), clear(x), False)
for pp in points:
    move_above.add_effect(EndTiming(), object_at_point(x, pp), False)
"""

pour_act = """\n
pour = DurativeAction("pour", x=obj_type, y=arm_type, z=obj_type)
x = pour.parameter("x")
y = pour.parameter("y")
z = pour.parameter("z")
pour.set_fixed_duration(1)
pour.add_condition(StartTiming(), lifting(x, y))
pour.add_condition(StartTiming(), water_in_cup(x))
pour.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), object_above_object(x, z))
pour.add_condition(StartTiming(), static(x))
pour.add_condition(StartTiming(), static(z))

pour.add_effect(StartTiming(), static(x), False)
pour.add_effect(StartTiming(), static(z), False)
pour.add_effect(StartTiming(), arm_free(y), False)
pour.add_effect(EndTiming(), water_in_cup(z), True)
pour.add_effect(EndTiming(), water_in_cup(x), False)
pour.add_effect(EndTiming(), arm_free(y), True)
pour.add_effect(EndTiming(), static(x), True)
pour.add_effect(EndTiming(), static(z), True)
"""

push_act = """\n
push = DurativeAction("push", x=obj_type, y=arm_type, z=area_type, w=area_type)
x = push.parameter("x")
y = push.parameter("y")
z = push.parameter("z")
w = push.parameter("w")
push.set_fixed_duration(1)

push.add_condition(StartTiming(), object_at_area(x, z))
push.add_condition(StartTiming(), arm_access(y, w))
push.add_condition(StartTiming(), arm_access(y, z))
push.add_condition(StartTiming(), available(y))
push.add_condition(StartTiming(), Not(is_graspable(x)))
push.add_condition(StartTiming(), arm_free(y))

push.add_effect(StartTiming(), arm_free(y), False)
push.add_effect(StartTiming(), object_at_area(x, z), False)
push.add_effect(EndTiming(), object_at_area(x, w), True)
push.add_effect(EndTiming(), arm_free(y), True)
"""

release_act = """\n
release = DurativeAction("release", x=obj_type, y=arm_type, z=point_type)
x = release.parameter("x")
y = release.parameter("y")
z = release.parameter("z")
release.set_fixed_duration(1)

# release.add_condition(StartTiming(), control(y))
release.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), lifting(x, y))
release.add_condition(StartTiming(), object_at_point(x, z))
release.add_condition(StartTiming(), is_releasable(z))
release.add_condition(StartTiming(), is_graspable(x))
release.add_condition(StartTiming(), arm_free(y))

# release.add_effect(StartTiming(), available(y), False)
# release.add_effect(StartTiming(), clear(x), False)
# release.add_effect(StartTiming(), lifting(x, y), True)
release.add_effect(StartTiming(), arm_free(y), False)
release.add_effect(StartTiming(), is_accessible(z), False)
release.add_effect(EndTiming(), arm_free(y), True)
release.add_effect(EndTiming(), lifting(x, y), False)
release.add_effect(EndTiming(), clear(x), True)
release.add_effect(EndTiming(), is_accessible(z), False)
release.add_effect(EndTiming(), available(y), True)
release.add_effect(EndTiming(), on_table(x), True)
"""

place_block_act = """\n
place = DurativeAction("place", x=obj_type, y=arm_type, z=obj_type, w=area_type)
x = place.parameter("x")
y = place.parameter("y")
z = place.parameter("z")
w = place.parameter("w")
place.set_fixed_duration(1)

# place.add_condition(StartTiming(), control(y))
place.add_condition(StartTiming(), lifting(x, y))
# if y == left_arm: place.add_condition(Not(lifting(x, right_arm)))
# if y == right_arm: place.add_condition(Not(lifting(x, left_arm)))
place.add_condition(StartTiming(), Not(Equals(x, z)))
place.add_condition(StartTiming(), arm_access(y, w))
# place.add_condition(StartTiming(), object_at_area(x, w))
place.add_condition(StartTiming(), object_at_area(z, w))
place.add_condition(StartTiming(), object_above_object(x, z))
place.add_condition(StartTiming(), clear(z))
place.add_condition(StartTiming(), arm_free(y))
place.add_condition(StartTiming(), static(x))
place.add_condition(StartTiming(), static(z))
for oo in objs:
    place.add_condition(StartTiming(), Not(object_on_object(oo, z)))

place.add_effect(StartTiming(), clear(z), False)
place.add_effect(StartTiming(), arm_free(y), False)
place.add_effect(EndTiming(), lifting(x, y), False)
place.add_effect(EndTiming(), clear(x), True)
place.add_effect(EndTiming(), object_on_object(x, z), True)
place.add_effect(EndTiming(), clear(z), False)
place.add_effect(EndTiming(), arm_free(y), True)
place.add_effect(EndTiming(), available(y), True)
place.add_effect(EndTiming(), object_above_object(x, z), False)
"""

place_fruit_act = """\n
place = DurativeAction("place", x=obj_type, y=arm_type, z=obj_type, w=area_type)
x = place.parameter("x")
y = place.parameter("y")
z = place.parameter("z")
w = place.parameter("w")
place.set_fixed_duration(1)

place.add_condition(StartTiming(), lifting(x, y))
place.add_condition(StartTiming(), arm_access(y, w))
# place.add_condition(StartTiming(), object_at_area(x, w))
place.add_condition(StartTiming(), object_at_area(z, w))
place.add_condition(StartTiming(), object_above_object(x, z))
place.add_condition(StartTiming(), clear(z))
place.add_condition(StartTiming(), arm_free(y))
place.add_condition(StartTiming(), static(z))

place.add_effect(StartTiming(), clear(z), False)
place.add_effect(StartTiming(), arm_free(y), False)
place.add_effect(StartTiming(), object_above_object(x, z), False)
place.add_effect(EndTiming(), lifting(x, y), False)
place.add_effect(EndTiming(), clear(x), True)
place.add_effect(EndTiming(), object_on_object(x, z), True)
place.add_effect(EndTiming(), arm_free(y), True)
place.add_effect(EndTiming(), available(y), True)

# needed for ServeFruit: make sure the bowl can receive another object in the future.
####################################
place.add_effect(EndTiming(), clear(z), True)    
####################################
"""

co_hold_act = """\n
co_hold = DurativeAction("co_hold", x=obj_type)
x = co_hold.parameter("x")
co_hold.set_fixed_duration(1)

co_hold.add_condition(StartTiming(), available(left_arm))
co_hold.add_condition(StartTiming(), available(right_arm))
co_hold.add_condition(StartTiming(), arm_free(left_arm))
co_hold.add_condition(StartTiming(), arm_free(right_arm))
co_hold.add_condition(StartTiming(), object_at_area(x, overlap_area))
co_hold.add_condition(StartTiming(), Not(is_graspable(x)))

co_hold.add_effect(StartTiming(), arm_free(left_arm), False)
co_hold.add_effect(StartTiming(), arm_free(right_arm), False)
co_hold.add_effect(EndTiming(), available(left_arm), False)
co_hold.add_effect(EndTiming(), available(right_arm), False)
co_hold.add_effect(StartTiming(), lifting(x, left_arm), True)
co_hold.add_effect(StartTiming(), lifting(x, right_arm), True)
co_hold.add_effect(EndTiming(), arm_free(left_arm), True)
co_hold.add_effect(EndTiming(), arm_free(right_arm), True)
co_hold.add_effect(EndTiming(), on_table(x), False)

"""

co_move_to_act = """\n
co_move_to = DurativeAction("co_move_to", x=obj_type, y=point_type)
x = co_move_to.parameter("x")
y = co_move_to.parameter("y")
co_move_to.set_fixed_duration(1)

co_move_to.add_condition(StartTiming(), object_at_area(x, overlap_area))
co_move_to.add_condition(StartTiming(), point_at(y, overlap_area))
co_move_to.add_condition(StartTiming(), lifting(x, left_arm))
co_move_to.add_condition(StartTiming(), lifting(x, right_arm))
co_move_to.add_condition(StartTiming(), arm_free(left_arm))
co_move_to.add_condition(StartTiming(), arm_free(right_arm))

co_move_to.add_effect(StartTiming(), arm_free(left_arm), False)
co_move_to.add_effect(StartTiming(), arm_free(right_arm), False)
co_move_to.add_effect(StartTiming(), object_at_point(x, y), True)
co_move_to.add_effect(EndTiming(), arm_free(left_arm), True)
co_move_to.add_effect(EndTiming(), arm_free(right_arm), True)
"""

def choose_actions(task_name):
    """Action definitions (UP Python code) of a task, or None for an unknown task."""
    if "StackBlock" in task_name:
        action_chunk = f"""
{stack_block_grasp_act} \n {move_to_act} \n {move_above_act} \n {place_block_act} \n {release_act} \n """
        action_chunk += """
action_blocks = [grasp_a, grasp_b, move_to, move_above, move_to_from, place, release]
problem.add_actions(action_blocks)
problem.add_quality_metric(MinimizeMakespan())
# problem.add_quality_metric(MinimizeActionCosts(
#     costs={},
#     default=1
# ))
"""
    elif task_name == "ServeWater":
        action_chunk = f"""
{grasp_act} \n {move_to_act} \n {move_above_act} \n {pour_act} \n {release_act} \n """
        action_chunk += """
action_blocks = [grasp, move_to, move_above, move_to_from, pour, release]
problem.add_actions(action_blocks)
problem.add_quality_metric(MinimizeMakespan())
# problem.add_quality_metric(MinimizeActionCosts(
#     costs={},
#     default=1
# ))
"""
    elif task_name == "ServeFruit":
        action_chunk = f"""
{grasp_act} \n {move_to_act} \n {move_above_act} \n {place_fruit_act} \n {push_act} \n {release_act} \n {co_hold_act} \n {co_move_to_act} \n """
        action_chunk += """
action_blocks = [grasp, move_to, move_to_from, move_above, place, push, release, co_hold, co_move_to]
problem.add_actions(action_blocks)
problem.add_quality_metric(MinimizeMakespan())
# problem.add_quality_metric(MinimizeActionCosts(
#     costs={},
#     default=1
# ))
"""
    else:
        print("You choose the wrong task name!")
        return None
    return action_chunk
