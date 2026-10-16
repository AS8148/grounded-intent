"""Observed-only intent compilation; no model, gold labels, or robot involved."""
from __future__ import annotations

from itertools import permutations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from grounded_intent.compiler import MAX_OBJECTS, compile_intent
from grounded_intent.domain import Goal, ObjectObservation, Relation, SceneGraph, TaskIntent
from grounded_intent.planner import BLOCKING_CONFIDENCE, PlanViolation, verify


def observation(name, index=0, *, label="cube", movable=True):
    x = 10 + 20 * index
    return ObjectObservation(
        id=name, label=label, box=(x, 10, x + 12, 22), pixel=(x + 3, 17),
        confidence=0.95, movable=movable, graspable=True,
    )


def relation(upper, lower, predicate="on", confidence=0.9):
    return Relation(subject=upper, predicate=predicate, object=lower,
                    confidence=confidence, evidence="synthetic observation")


def scene(objects=None, relations=()):
    return SceneGraph(
        scene="synthetic", width=4096, height=100, image_sha256="synthetic-only",
        objects=tuple(objects) if objects is not None else (
            observation("a"), observation("b", 1), observation("c", 2),
            observation("tray", 3, label="tray", movable=False),
        ), relations=tuple(relations), perception_model="test-observations", perception_seconds=0,
    )


def intent(*goals, forbidden=(), reason="Grounded task rationale."):
    return TaskIntent(kind="plan", goals=tuple(Goal(target=target, destination=destination)
                                              for target, destination in goals),
                      forbidden_movement=forbidden, reason=reason)


def steps(proposal):
    return tuple((action.action, action.object) for action in proposal.actions)


def compiled(graph, task):
    before_graph, before_task = graph.model_dump(), task.model_dump()
    proposal = compile_intent(graph, task)
    decision = verify(graph, proposal, "compiler-test")
    assert all(check.passed for check in decision.checks)
    assert decision.kind == proposal.kind
    assert graph.model_dump() == before_graph
    assert task.model_dump() == before_task
    if proposal.kind == "plan":
        assert proposal.reason == task.reason
    else:
        assert not proposal.actions
        assert len(decision.actions) == 1
        assert decision.target_pixel is None
    return proposal


def assert_violation(graph, task, rule, step=None):
    before_graph, before_task = graph.model_dump(), task.model_dump()
    with pytest.raises(PlanViolation) as caught:
        compile_intent(graph, task)
    checks = caught.value.checks
    assert isinstance(checks, tuple) and len(checks) == 1
    assert checks[0].rule == rule and not checks[0].passed
    assert checks[0].step == step
    assert str(caught.value) == checks[0].detail
    assert graph.model_dump() == before_graph
    assert task.model_dump() == before_task


@pytest.mark.parametrize("destination", [None, "tray", "b"])
def test_unobstructed_goal_needs_no_staging(destination):
    graph = scene([observation("a"), observation("b", 1),
                   observation("tray", 2, label="tray")])
    result = compiled(graph, intent(("a", destination)))
    expected = (("pick", "a"),) + ((("place", destination),) if destination else ())
    assert steps(result) == expected
    assert not result.question and result.reason == "Grounded task rationale."


def test_unobstructed_pickup_without_any_surface_keeps_holding():
    result = compiled(scene([observation("a")]), intent(("a", None)))
    assert steps(result) == (("pick", "a"),)


@pytest.mark.parametrize("predicate", ["on", "blocks"])
@pytest.mark.parametrize("confidence", [BLOCKING_CONFIDENCE, 0.700001, 1.0])
def test_direct_blocker_is_staged_before_target(predicate, confidence):
    graph = scene(relations=[relation("b", "a", predicate, confidence)])
    result = compiled(graph, intent(("a", None)))
    assert steps(result) == (("pick", "b"), ("place", "tray"), ("pick", "a"))
    assert verify(graph, result, "pixel").target_pixel == graph.by_id()["b"].pixel


@pytest.mark.parametrize("predicate", ["on", "blocks"])
@pytest.mark.parametrize("confidence", [0.0, 0.699999])
def test_low_confidence_relations_do_not_require_staging(predicate, confidence):
    graph = scene(relations=[relation("b", "a", predicate, confidence)])
    assert steps(compiled(graph, intent(("a", None)))) == (("pick", "a"),)


@pytest.mark.parametrize("predicate", ["inside", "near", "left_of", "above"])
def test_other_relations_do_not_invent_blockers(predicate):
    graph = scene(relations=[relation("b", "a", predicate, 1.0)])
    assert steps(compiled(graph, intent(("a", None)))) == (("pick", "a"),)


@given(st.integers(min_value=1, max_value=MAX_OBJECTS - 1))
def test_stack_is_cleared_top_first_without_recursion(depth):
    names = [f"item_{index:03d}" for index in range(depth)]
    graph = scene([observation(name, index) for index, name in enumerate(names)] +
                  [observation("tray", depth, label="tray")],
                  [relation(names[index + 1], names[index]) for index in range(depth - 1)])
    result = compiled(graph, intent((names[0], None)))
    expected = tuple(step for name in reversed(names[1:])
                     for step in (("pick", name), ("place", "tray"))) + (("pick", names[0]),)
    assert steps(result) == expected


@pytest.mark.parametrize("order", list(permutations(("a", "b", "c"))))
def test_parallel_blockers_have_lexicographic_order_independent_of_observation_order(order):
    graph = scene([observation("target")] + [observation(name, i + 1) for i, name in enumerate(order)]
                  + [observation("tray", 4, label="tray")],
                  [relation(name, "target") for name in order])
    result = compiled(graph, intent(("target", None)))
    assert steps(result) == (("pick", "a"), ("place", "tray"), ("pick", "b"),
                             ("place", "tray"), ("pick", "c"), ("place", "tray"),
                             ("pick", "target"))


def test_diamond_and_duplicate_edges_clear_each_required_blocker_once():
    graph = scene([observation(name, i) for i, name in enumerate(("a", "b", "c", "d"))]
                  + [observation("tray", 4, label="tray")],
                  [relation("d", "b"), relation("d", "c"), relation("b", "a"),
                   relation("c", "a"), relation("c", "a", "blocks")])
    assert steps(compiled(graph, intent(("a", None)))) == (
        ("pick", "d"), ("place", "tray"), ("pick", "b"), ("place", "tray"),
        ("pick", "c"), ("place", "tray"), ("pick", "a"),
    )


@pytest.mark.parametrize("restricted", ["a", "b", "c"])
def test_forbidden_required_moves_refuse_actual_constraint(restricted):
    graph = scene(relations=[relation("c", "b"), relation("b", "a")])
    result = compiled(graph, intent(("a", None), forbidden=(restricted,)))
    assert result.kind == "refuse" and "restricted" in result.reason
    assert repr(restricted) in result.reason and "goal 'a'" in result.reason
    assert result.forbidden_movement == (restricted,) and not result.question


@pytest.mark.parametrize("label", ["plate", "Dinner Plate", "saucer", "TRAY", "blue trays"])
def test_fixed_blocker_refuses_even_if_movable_flag_is_wrong(label):
    graph = scene([observation("a"), observation("fixed", 1, label=label),
                   observation("tray", 2, label="tray")], [relation("fixed", "a")])
    result = compiled(graph, intent(("a", None)))
    assert result.kind == "refuse" and "fixed/nonmovable" in result.reason
    assert "'fixed'" in result.reason


# A model "immovable" estimate no longer refuses a cube: the organisers' class rule says
# every object but plate, saucer and tray can be picked. Fixed surfaces still refuse.
@pytest.mark.parametrize("label,movable", [("tray", True), ("plate", True), ("hand", True)])
def test_fixed_goal_target_is_not_picked(label, movable):
    graph = scene([observation("a", label=label, movable=movable)])
    assert compiled(graph, intent(("a", None))).kind == "refuse"


@pytest.mark.parametrize("forbidden", [(), ("b",)])
def test_infeasible_later_goal_refuses_the_whole_requested_task(forbidden):
    graph = scene([observation("a"), observation("b", 1),
                   observation("tray", 2, label="tray", movable=False)])
    later = "b" if forbidden else "tray"
    result = compiled(graph, intent(("a", "tray"), (later, None), forbidden=forbidden))
    assert result.kind == "refuse"
    assert not result.actions


def test_infeasible_first_goal_still_refuses():
    # No physically executable prefix can make an impossible requested goal complete.
    graph = scene([observation("a"), observation("tray", 2, label="tray", movable=False)])
    assert compiled(graph, intent(("tray", None))).kind == "refuse"


def test_unrelated_forbidden_object_and_stationary_destination_are_allowed():
    result = compiled(scene(), intent(("a", "tray"), forbidden=("b", "tray")))
    assert steps(result) == (("pick", "a"), ("place", "tray"))
    assert result.forbidden_movement == ("b", "tray")


@pytest.mark.parametrize("label", [None, "plate", "table", "portray cube"])
def test_no_fixed_staging_tray_asks_without_inventing_a_surface(label):
    objects = [observation("a"), observation("b", 1)]
    if label is not None:
        objects.append(observation("surface", 2, label=label, movable=False))
    graph = scene(objects, [relation("b", "a")])
    result = compiled(graph, intent(("a", None)))
    assert result.kind == "ask_user" and not result.reason
    assert "No observed fixed tray" in result.question
    assert "'a'" in result.question


def test_forbidden_blocker_refusal_takes_precedence_over_missing_tray():
    graph = scene([observation("a"), observation("b", 1)], [relation("b", "a")])
    result = compiled(graph, intent(("a", None), forbidden=("b",)))
    assert result.kind == "refuse" and "restricted" in result.reason


@pytest.mark.parametrize("destination,staging", [(None, "a_tray"), ("a_tray", "a_tray"),
                                                ("z_tray", "z_tray"), ("c", "a_tray")])
@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_trays_use_goal_destination_or_stable_lexical_id(destination, staging, reverse):
    objects = [observation("a"), observation("b", 1), observation("c", 2),
               observation("z_tray", 3, label="Tray"), observation("a_tray", 4, label="blue trays")]
    graph = scene(objects[::-1] if reverse else objects, [relation("b", "a")])
    result = compiled(graph, intent(("a", destination)))
    assert steps(result)[:2] == (("pick", "b"), ("place", staging))
    assert steps(result)[2:] == (("pick", "a"),) + (
        (("place", destination),) if destination is not None else ())


@pytest.mark.parametrize("goals", [(("a", None), ("b", None)),
                                   (("a", None), ("a", "tray")),
                                   (("c", "tray"), ("a", None), ("b", "tray"))])
def test_nonfinal_null_destination_stages_to_tray_so_picks_still_execute(goals):
    # A null-destination goal must still execute its pick (the scored action);
    # resting it on an observed tray frees the gripper for the next goal rather
    # than asking a question the plan already answered. ask_user only survives
    # when no tray exists at all (covered by a separate test).
    result = compiled(scene(), intent(*goals))
    assert result.kind == "plan"


def test_requested_goal_order_is_preserved_not_reordered_for_convenience():
    result = compiled(scene(relations=[relation("b", "a")]),
                      intent(("a", "tray"), ("b", "tray"), ("c", None)))
    assert steps(result) == (("pick", "b"), ("place", "tray"), ("pick", "a"),
                             ("place", "tray"), ("pick", "b"), ("place", "tray"),
                             ("pick", "c"))


def test_repeated_target_moves_are_not_deduplicated_or_reused_from_stale_state():
    result = compiled(scene(), intent(("a", "tray"), ("a", "b"), ("a", None)))
    assert steps(result) == (("pick", "a"), ("place", "tray"), ("pick", "a"),
                             ("place", "b"), ("pick", "a"))


def test_new_placement_on_movable_object_creates_required_blocker():
    result = compiled(scene(), intent(("a", "b"), ("b", None)))
    assert steps(result) == (("pick", "a"), ("place", "b"), ("pick", "a"),
                             ("place", "tray"), ("pick", "b"))


def test_dynamic_blocker_without_observed_tray_requires_clarification():
    graph = scene([observation("a"), observation("b", 1)])
    result = compiled(graph, intent(("a", "b"), ("b", None)))
    assert result.kind == "ask_user" and "'b'" in result.question


def test_successive_goals_relocate_new_blockers_using_current_state():
    result = compiled(scene(), intent(("a", "b"), ("b", "c"), ("c", "tray"), ("a", None)))
    assert steps(result) == (
        ("pick", "a"), ("place", "b"), ("pick", "a"), ("place", "tray"),
        ("pick", "b"), ("place", "c"), ("pick", "b"), ("place", "tray"),
        ("pick", "c"), ("place", "tray"), ("pick", "a"),
    )


def test_fixed_tray_placement_does_not_block_its_other_contents():
    graph = scene(relations=[relation("b", "tray", "inside", 1.0)])
    result = compiled(graph, intent(("a", "tray"), ("b", None)))
    assert steps(result) == (("pick", "a"), ("place", "tray"), ("pick", "b"))


def test_picking_removes_all_old_outgoing_blocking_relationships():
    graph = scene(relations=[relation("c", "a"), relation("c", "b", "blocks")])
    result = compiled(graph, intent(("a", "tray"), ("b", None)))
    assert steps(result) == (("pick", "c"), ("place", "tray"), ("pick", "a"),
                             ("place", "tray"), ("pick", "b"))


def test_unrelated_stack_and_existing_tray_contents_are_not_relocated():
    names = ("a", "b", "c", "d", "content")
    graph = scene([observation(name, i) for i, name in enumerate(names)] +
                  [observation("tray", 5, label="tray")],
                  [relation("b", "a"), relation("c", "d"), relation("content", "tray", "inside")])
    result = compiled(graph, intent(("a", None), forbidden=("c", "d", "content")))
    assert steps(result) == (("pick", "b"), ("place", "tray"), ("pick", "a"))


@pytest.mark.parametrize("goals", [(("unknown", None),), (("a", "unknown"),),
                                   (("a", ""),), ((" ", None),),
                                   (("a ", None),), (("a", " b"),),
                                   (("a", "tray"), ("unknown", None))])
def test_unknown_or_blank_goal_ids_produce_bounded_repair_checks(goals):
    assert_violation(scene(), intent(*goals), "known_ids", len(goals) - 1)


@pytest.mark.parametrize("unknown", ["", " ", "unknown", "a "])
@pytest.mark.parametrize("kind", ["plan", "ask_user", "refuse"])
def test_unknown_forbidden_ids_are_not_normalized_or_discarded(unknown, kind):
    task = intent(("a", None), forbidden=(unknown,)) if kind == "plan" else TaskIntent(
        kind=kind, question="Which?" if kind == "ask_user" else "",
        reason="Unavailable." if kind == "refuse" else "", forbidden_movement=(unknown,))
    assert_violation(scene(), task, "known_ids")


@given(st.text(min_size=1).filter(lambda value: value.strip() and value not in {"a", "b", "c", "tray"}))
def test_arbitrary_unknown_ids_never_become_actions(unknown):
    assert_violation(scene(), intent((unknown, None)), "known_ids", 0)


@pytest.mark.parametrize("goals", [(("a", "a"),), (("b", "tray"), ("a", "a"))])
def test_self_destinations_produce_repair_checks(goals):
    assert_violation(scene(), intent(*goals), "no_self_placement", len(goals) - 1)


@given(st.integers(min_value=2, max_value=MAX_OBJECTS))
def test_cycles_are_rejected_iteratively_with_bounded_feedback(depth):
    names = [f"item_{index}" for index in range(depth)]
    graph = scene([observation(name, i) for i, name in enumerate(names)],
                  [relation(names[i], names[(i + 1) % depth]) for i in range(depth)])
    assert_violation(graph, intent((names[0], None)), "acyclic_supports")


def test_unrelated_cycle_is_rejected_for_verifier_parity():
    graph = scene(relations=[relation("b", "c"), relation("c", "b")])
    assert_violation(graph, intent(("a", None)), "acyclic_supports")


def test_low_confidence_reverse_relation_does_not_create_cycle():
    graph = scene(relations=[relation("b", "a"), relation("a", "b", confidence=0.699999)])
    assert steps(compiled(graph, intent(("a", None)))) == (
        ("pick", "b"), ("place", "tray"), ("pick", "a"))


@pytest.mark.parametrize("kind", ["ask_user", "refuse"])
def test_nonplans_are_exclusive_and_do_not_attempt_cyclic_graph_compilation(kind):
    task = TaskIntent(kind=kind, question="Which cube?" if kind == "ask_user" else "",
                      reason="Requested constraint cannot be met." if kind == "refuse" else "",
                      forbidden_movement=("b",))
    result = compiled(scene(relations=[relation("a", "b"), relation("b", "a")]), task)
    assert result.kind == task.kind and result.question == task.question
    assert result.reason == task.reason and result.forbidden_movement == ("b",)


@pytest.mark.parametrize("payload", [{"kind": "plan", "goals": ()},
                                     {"kind": "plan", "goals": (Goal(target="a"),), "question": "Which?"},
                                     {"kind": "ask_user", "question": "Which?", "reason": "No."}])
def test_unvalidated_intent_cannot_bypass_exclusive_contract(payload):
    assert_violation(scene(), TaskIntent.model_construct(**payload), "exclusive_output")


def test_exact_object_bound_accepts_maximal_stack():
    names = [f"item_{index:03d}" for index in range(MAX_OBJECTS - 1)]
    graph = scene([observation(name, i) for i, name in enumerate(names)] +
                  [observation("tray", MAX_OBJECTS - 1, label="tray")],
                  [relation(names[i + 1], names[i]) for i in range(len(names) - 1)])
    result = compiled(graph, intent((names[0], "tray")))
    assert len(result.actions) == 2 * (MAX_OBJECTS - 1)
    assert steps(result)[0] == ("pick", names[-1])


def test_over_object_bound_rejects_before_traversing_deep_graph():
    names = [f"item_{i}" for i in range(MAX_OBJECTS + 1)]
    graph = scene([observation(name, i) for i, name in enumerate(names)],
                  [relation(names[i + 1], names[i]) for i in range(len(names) - 1)])
    assert_violation(graph, intent((names[0], None)), "object_bound")


@given(st.sets(st.tuples(st.integers(0, 7), st.integers(0, 7)).filter(lambda pair: pair[0] > pair[1])))
def test_arbitrary_acyclic_blocker_graphs_produce_verifiable_plans(edges):
    names = [f"item_{index}" for index in range(8)]
    graph = scene([observation(name, i) for i, name in enumerate(names)] +
                  [observation("tray", 8, label="tray")],
                  [relation(names[upper], names[lower], "blocks") for upper, lower in sorted(edges)])
    result = compiled(graph, intent((names[0], None)))
    picks = [action.object for action in result.actions if action.action == "pick"]
    assert len(picks) == len(set(picks)) and picks[-1] == names[0]
    assert len(result.actions) <= 2 * len(names) - 1
    reversed_graph = graph.model_copy(update={"objects": graph.objects[::-1],
                                              "relations": graph.relations[::-1]})
    assert compile_intent(reversed_graph, intent((names[0], None))) == result


@given(st.lists(st.tuples(st.integers(0, 3), st.integers(0, 4)), min_size=1, max_size=20))
def test_arbitrary_repeated_goals_and_dynamic_placements_preserve_requested_order(pairs):
    names = ("a", "b", "c", "d", "tray")
    graph = scene([observation(name, i, label="tray" if name == "tray" else "cube")
                   for i, name in enumerate(names)])
    goals = [(names[target], names[destination] if target != destination else "tray")
             for target, destination in pairs] + [("a", None)]
    result = compiled(graph, intent(*goals))
    actions = steps(result)
    cursor = 0
    for target, destination in goals:
        requested = (("pick", target),) + ((("place", destination),) if destination else ())
        while actions[cursor:cursor + len(requested)] != requested:
            cursor += 1
            assert cursor < len(actions), "Requested goal sequence was lost or reordered"
        cursor += len(requested)
    assert actions[-1] == ("pick", "a")
    assert len(actions) <= 2 * len(names) * len(goals)


def test_legacy_current_under_transfers_to_receiver_without_restoring_upper():
    graph = scene(relations=[relation("b", "a")])
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="tray", relation="under"),
    ))
    result = compiled(graph, task)
    assert result.goals == task.goals
    assert steps(result) == (
        ("pick", "b"), ("place", "tray"), ("pick", "a"), ("place", "tray"),
    )


def test_repeated_target_to_duplicate_receivers_compiles_as_one_move():
    first = observation("grey_tray_1", 3, label="tray", movable=False).model_copy(
        update={"box": (200, 10, 390, 90), "pixel": (300, 50), "receives_objects": True})
    duplicate = observation("grey_tray_2", 4, label="tray", movable=False).model_copy(
        update={"box": (200, 10, 390, 90), "pixel": (300, 50), "receives_objects": True})
    graph = scene([observation("red_cube"), first, duplicate])
    result = compiled(graph, intent(("red_cube", "grey_tray_1"),
                                    ("red_cube", "grey_tray_2")))
    assert steps(result) == (("pick", "red_cube"), ("place", "grey_tray_1"))


def test_repeated_target_to_distinct_receivers_preserves_both_requested_moves():
    first = observation("grey_tray_1", 3, label="tray", movable=False).model_copy(
        update={"box": (100, 10, 210, 80), "pixel": (150, 45), "receives_objects": True})
    second = observation("grey_tray_2", 4, label="tray", movable=False).model_copy(
        update={"box": (260, 10, 370, 80), "pixel": (310, 45), "receives_objects": True})
    graph = scene([observation("red_cube"), first, second])
    result = compiled(graph, intent(("red_cube", "grey_tray_1"),
                                    ("red_cube", "grey_tray_2")))
    assert steps(result) == (("pick", "red_cube"), ("place", "grey_tray_1"),
                             ("pick", "red_cube"), ("place", "grey_tray_2"))


def test_explicit_relation_prevents_collapsing_duplicate_receivers():
    first = observation("grey_tray_1", 3, label="tray", movable=False).model_copy(
        update={"box": (200, 10, 390, 90), "pixel": (300, 50), "receives_objects": True})
    duplicate = observation("grey_tray_2", 4, label="tray", movable=False).model_copy(
        update={"box": (200, 10, 390, 90), "pixel": (300, 50), "receives_objects": True})
    graph = scene([observation("red_cube"), first, duplicate],
                  [relation("grey_tray_1", "grey_tray_2", "on", 0.9)])
    result = compiled(graph, intent(("red_cube", "grey_tray_1"),
                                    ("red_cube", "grey_tray_2")))
    assert steps(result) == (("pick", "red_cube"), ("place", "grey_tray_1"),
                             ("pick", "red_cube"), ("place", "grey_tray_2"))


@pytest.mark.parametrize(("relation_name", "placement"), [("in", "inside"), ("on", "on")])
def test_explicit_receiver_relation_survives_compilation(relation_name, placement):
    graph = scene()
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="tray", relation=relation_name),
    ), goal_mode="final_state")
    result = compiled(graph, task)
    assert result.goals == task.goals
    assert result.goal_mode == task.goal_mode
    assert result.actions[-1].relation == placement
    assert verify(graph, result, "schema").actions == (
        {"action": "pick", "object": "a"}, {"action": "place", "object": "tray"},
    )


@pytest.mark.parametrize("base_relation", ["on", "inside"])
def test_explicit_final_under_clears_lower_places_on_base_and_restores_upper(base_relation):
    graph = scene(relations=[relation("c", "b"), relation("b", "a")])
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="tray", relation="under",
             under_object="b", base_relation=base_relation),
    ))
    result = compiled(graph, task)
    assert result.goals == task.goals
    assert steps(result) == (
        ("pick", "c"), ("place", "tray"), ("pick", "b"), ("place", "tray"),
        ("pick", "a"), ("place", "tray"), ("pick", "b"), ("place", "a"),
    )
    assert result.actions[5].relation == base_relation
    assert result.actions[-1].relation == "on"


def test_final_under_moves_explicit_upper_even_without_observed_blocking_relation():
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="tray", relation="under", under_object="b"),
    ))
    assert steps(compiled(scene(), task)) == (
        ("pick", "b"), ("place", "tray"), ("pick", "a"), ("place", "tray"),
        ("pick", "b"), ("place", "a"),
    )


@pytest.mark.parametrize("protected", ["b", "c"])
def test_final_under_refuses_protected_upper_or_its_blocker(protected):
    graph = scene(relations=[relation("c", "b"), relation("b", "a")])
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="tray", relation="under", under_object="b"),
    ), forbidden_movement=(protected,))
    assert compiled(graph, task).kind == "refuse"


def test_final_under_refuses_upper_forbidden_by_movement_class():
    graph = scene([observation("a"), observation("b", 1, label="plate"),
                   observation("tray", 2, label="tray")])
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="tray", relation="under", under_object="b"),
    ))
    assert compiled(graph, task).kind == "refuse"


def test_final_under_without_named_base_requires_clarification():
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", relation="under", under_object="b"),
    ))
    assert compiled(scene(), task).kind == "ask_user"


@pytest.mark.parametrize("upper,base,rule", [
    ("absent", "tray", "known_ids"),
    ("b", "absent", "known_ids"),
    ("b", "b", "no_self_placement"),
    ("b", "a", "no_self_placement"),
])
def test_final_under_validates_upper_and_base_roles(upper, base, rule):
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination=base, relation="under", under_object=upper),
    ))
    assert_violation(scene(), task, rule, 0)


def test_final_under_requires_stationary_staging_for_upper():
    graph = scene([observation("a"), observation("b", 1),
                   observation("base", 2, label="plate")])
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", destination="base", relation="under", under_object="b"),
    ))
    assert compiled(graph, task).kind == "ask_user"


def test_later_required_protected_blocker_does_not_leave_an_executable_prefix():
    graph = scene(relations=[relation("c", "b")])
    task = intent(("a", "tray"), ("b", "tray"), forbidden=("c",))
    assert compiled(graph, task).kind == "refuse"


def test_explicit_held_goal_is_not_silently_staged_to_free_gripper():
    task = TaskIntent(kind="plan", goals=(
        Goal(target="a", relation="held"), Goal(target="b", destination="tray"),
    ))
    assert compiled(scene(), task).kind == "ask_user"
