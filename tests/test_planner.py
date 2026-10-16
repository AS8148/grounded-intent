"""State-transition tests: no model, dataset labels, or robot is involved."""
from __future__ import annotations

from itertools import permutations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from grounded_intent.domain import Goal, ObjectObservation, Proposal, Relation, SceneGraph
from grounded_intent.planner import PlanViolation, _movable, verify


def observation(name: str, index: int = 0, *, label: str = "cube", movable: bool = True):
    x, y = 10 + index * 20, 12 + index * 15
    return ObjectObservation(
        id=name, label=label, box=(x, y, x + 12, y + 12), pixel=(x + 3, y + 7),
        confidence=0.95, movable=movable, graspable=True,
    )


def relation(upper: str, lower: str, predicate="on", confidence=0.9):
    return Relation(
        subject=upper, predicate=predicate, object=lower, confidence=confidence,
        evidence="observed support",
    )


def scene(objects=None, relations=()):
    return SceneGraph(
        scene="synthetic", width=1000, height=1000, image_sha256="synthetic-only",
        objects=tuple(objects) if objects is not None else (
            observation("a"), observation("b", 1),
            observation("tray", 2, label="tray", movable=False),
        ), relations=tuple(relations), perception_model="test-observations",
        perception_seconds=0,
    )


def plan(*steps, forbidden=(), goals=(), goal_mode="ordered"):
    return Proposal(
        kind="plan", actions=tuple({"action": action, "object": name} for action, name in steps),
        forbidden_movement=forbidden,
        goals=goals, goal_mode=goal_mode,
    )


def assert_violation(graph, proposal, rule, step=None):
    with pytest.raises(PlanViolation) as caught:
        verify(graph, proposal, "task")
    checks = caught.value.checks
    assert checks and checks[-1].rule == rule and not checks[-1].passed
    if step is not None:
        assert checks[-1].step == step
    assert str(caught.value) == checks[-1].detail
    assert all(check.passed for check in checks[:-1])


@pytest.mark.parametrize("steps", [
    (("pick", "a"),),
    (("pick", "a"), ("place", "tray")),
    (("pick", "a"), ("place", "tray"), ("pick", "b")),
    (("pick", "a"), ("place", "tray"), ("pick", "a")),
    (("pick", "a"), ("place", "b"), ("pick", "a"), ("place", "tray"), ("pick", "b")),
])
def test_valid_sequences_include_final_hold_and_repicks(steps):
    graph = scene()
    decision = verify(graph, plan(*steps), "task", "Apertus-local", 1.25, {"tokens": 42})
    assert decision.kind == "plan"
    assert decision.actions == tuple({"action": action, "object": name} for action, name in steps)
    assert decision.target_pixel == graph.by_id()[steps[0][1]].pixel
    assert all(check.passed for check in decision.checks)
    assert decision.model == "Apertus-local" and decision.reasoning_seconds == 1.25
    assert decision.usage == {"tokens": 42} and not decision.repaired


@pytest.mark.parametrize(("steps", "rule", "index"), [
    ((("place", "tray"),), "held_before_place", 0),
    ((("pick", "a"), ("pick", "b")), "held_empty_before_pick", 1),
    ((("pick", "a"), ("pick", "a")), "held_empty_before_pick", 1),
    ((("pick", "a"), ("place", "tray"), ("place", "b")), "held_before_place", 2),
    ((("pick", "a"), ("place", "a")), "no_self_placement", 1),
    ((("pick", "a"), ("place", "b"), ("pick", "b")), "unblocked_pick", 2),
    ((("pick", "a"), ("place", "tray"), ("pick", "b"), ("place", "b")),
     "no_self_placement", 3),
    ((("pick", "a"), ("place", "tray"), ("pick", "absent")), "known_ids", 2),
])
def test_entire_sequence_is_verified(steps, rule, index):
    assert_violation(scene(), plan(*steps), rule, index)


@pytest.mark.parametrize("object_id", ["", " ", "\t", "absent", "A", "a ", " a", "table"])
def test_empty_unknown_ids_rejected_not_normalized(object_id):
    if not object_id:
        with pytest.raises(ValidationError):
            plan(("pick", object_id))
    else:
        assert_violation(scene(), plan(("pick", object_id)), "known_ids", 0)


@pytest.mark.parametrize("object_id", ["", " ", "absent", "a "])
def test_forbidden_ids_are_not_invented(object_id):
    assert_violation(scene(), plan(("pick", "a"), forbidden=(object_id,)), "known_ids")


@pytest.mark.parametrize("label", ["plate", "Dinner Plate", "saucer", "saucers", "TRAY", "blue trays"])
def test_fixed_containers_never_picked_even_if_perception_marks_movable(label):
    graph = scene([observation("container", label=label, movable=True)])
    assert_violation(graph, plan(("pick", "container")), "movable_object", 0)


def test_class_rule_overrides_a_model_immovable_estimate():
    # "Only the plate and the saucer ... all other objects can be picked" (organisers).
    # Requiring the vision model's own movable estimate here would turn a pickable object
    # it called immovable into a false refusal, which also costs the safety multiplier.
    graph = scene([observation("a", label="cube", movable=False)])
    assert verify(graph, plan(("pick", "a")), "task").kind == "plan"


@pytest.mark.parametrize("label", ["hand", "left arm", "finger", "person"])
def test_a_person_is_never_graspable(label):
    graph = scene([observation("a", label=label, movable=True)])
    assert_violation(graph, plan(("pick", "a")), "movable_object", 0)


def test_container_substrings_do_not_make_unrelated_objects_immovable():
    graph = scene([observation("a", label="plated cube")])
    assert verify(graph, plan(("pick", "a")), "task").kind == "plan"


@pytest.mark.parametrize("predicate", ["on", "blocks"])
@pytest.mark.parametrize("confidence", [0.7, 0.700001, 1.0])
def test_reliable_upper_object_blocks_lower(predicate, confidence):
    graph = scene(relations=[relation("b", "a", predicate, confidence)])
    assert_violation(graph, plan(("pick", "a")), "unblocked_pick", 0)
    decision = verify(graph, plan(("pick", "b"), ("place", "tray"), ("pick", "a")), "task")
    assert decision.target_pixel == graph.by_id()["b"].pixel


@pytest.mark.parametrize("predicate", ["on", "blocks"])
@pytest.mark.parametrize("confidence", [0.0, 0.699999])
def test_below_threshold_relation_does_not_assert_blocking(predicate, confidence):
    graph = scene(relations=[relation("b", "a", predicate, confidence)])
    assert verify(graph, plan(("pick", "a")), "task").kind == "plan"


@pytest.mark.parametrize("predicate", ["inside", "near", "left_of", "above"])
def test_nonblocking_relations_are_not_promoted_to_support(predicate):
    graph = scene(relations=[relation("b", "a", predicate, 1)])
    assert verify(graph, plan(("pick", "a")), "task").kind == "plan"


def test_moving_upper_then_replacing_on_lower_keeps_lower_blocked():
    graph = scene(relations=[relation("b", "a")])
    assert_violation(graph, plan(("pick", "b"), ("place", "a"), ("pick", "a")),
                     "unblocked_pick", 2)


def test_multiple_blockers_all_need_removal():
    graph = scene(
        [observation("a"), observation("b", 1), observation("c", 2),
         observation("tray", 3, label="tray", movable=False)],
        [relation("b", "a"), relation("c", "a", "blocks")],
    )
    assert_violation(graph, plan(("pick", "b"), ("place", "tray"), ("pick", "a")),
                     "unblocked_pick", 2)
    decision = verify(graph, plan(("pick", "b"), ("place", "tray"), ("pick", "c"),
                                 ("place", "tray"), ("pick", "a")), "task")
    assert decision.kind == "plan" and len(decision.actions) == 5


def test_no_repair_destination_is_invented():
    graph = scene([observation("a"), observation("b", 1)], [relation("b", "a")])
    assert_violation(graph, plan(("pick", "a")), "unblocked_pick", 0)


def test_forbidden_blocker_is_not_silently_moved():
    graph = scene(relations=[relation("b", "a")])
    assert_violation(graph, plan(("pick", "b"), ("place", "tray"), ("pick", "a"),
                                forbidden=("b",)), "forbidden_movement", 0)
    assert_violation(graph, plan(("pick", "a"), forbidden=("b",)), "unblocked_pick", 0)


def test_forbidden_destination_can_stay_stationary_during_place():
    decision = verify(scene(), plan(("pick", "a"), ("place", "tray"), forbidden=("tray",)), "task")
    assert decision.kind == "plan"


def test_late_forbidden_pick_is_rejected():
    assert_violation(scene(), plan(("pick", "a"), ("place", "tray"), ("pick", "b"),
                                  forbidden=("b",)), "forbidden_movement", 2)


@pytest.mark.parametrize("relations", [
    (relation("a", "b"), relation("b", "a")),
    (relation("a", "b", "blocks"), relation("b", "a")),
])
def test_cyclic_supports_rejected_without_attempting_repair(relations):
    assert_violation(scene(relations=relations), plan(("pick", "a")), "acyclic_supports")


def test_duplicate_relations_do_not_make_false_cycle():
    graph = scene(relations=[relation("b", "a"), relation("b", "a", "blocks")])
    assert verify(graph, plan(("pick", "b"), ("place", "tray"), ("pick", "a")), "task").kind == "plan"


@pytest.mark.parametrize("kind", ["ask_user", "refuse"])
def test_nonmovement_decisions_are_exactly_one_official_action(kind):
    proposal = Proposal(kind=kind, question="Which cube?" if kind == "ask_user" else "",
                        reason="Requested object absent." if kind == "refuse" else "")
    graph = scene(relations=[relation("a", "b"), relation("b", "a")])
    result = verify(graph, proposal, "question-or-refusal")
    assert result.target_pixel is None and result.kind == kind
    assert len(result.actions) == 1 and result.actions[0]["action"] == kind
    assert result.official() == {
        "instruction_id": "question-or-refusal", "target_pixel": None,
        "actions": [{"action": kind, **({"question": "Which cube?"} if kind == "ask_user"
                                      else {"reason": "Requested object absent."})}],
    }


def test_nonmovable_blocker_cannot_be_cleared_by_a_pick():
    graph = scene(
        [observation("a"), observation("fixed", 1, label="plate", movable=False),
         observation("tray", 2, label="tray", movable=False)],
        [relation("fixed", "a")],
    )
    assert_violation(graph, plan(("pick", "a")), "unblocked_pick", 0)
    assert_violation(graph, plan(("pick", "fixed"), ("place", "tray"), ("pick", "a")),
                     "movable_object", 0)



@pytest.mark.parametrize("payload", [
    {"kind": "plan"},
    {"kind": "ask_user"},
    {"kind": "refuse"},
    {"kind": "ask_user", "question": "   "},
    {"kind": "refuse", "reason": "\t"},
    {"kind": "refuse", "reason": "No.", "actions": [{"action": "pick", "object": "a"}]},
    {"kind": "ask_user", "question": "Which?", "actions": [{"action": "pick", "object": "a"}]},
])
def test_invalid_status_shape_is_not_silently_converted(payload):
    with pytest.raises(ValidationError):
        Proposal.model_validate(payload)


@pytest.mark.parametrize("proposal", [
    # Constructed without validation on purpose: the contract now rejects these
    # shapes at construction, so the verifier's own guard is what is under test.
    Proposal.model_construct(kind="ask_user", question="Which?", reason="No."),
    Proposal.model_construct(kind="refuse", reason="No.", question="Which?"),
    Proposal.model_construct(kind="plan", actions=({"action": "pick", "object": "a"},),
                             question="Which?"),
])
def test_cross_status_text_rejected(proposal):
    assert_violation(scene(), proposal, "exclusive_output")


def test_unvalidated_construct_cannot_bypass_contract():
    proposal = Proposal.model_construct(kind="plan", actions=())
    assert_violation(scene(), proposal, "exclusive_output")


def test_plan_rationale_is_not_a_refusal():
    proposal = plan(("pick", "a")).model_copy(update={"reason": "Object is unblocked."})
    decision = verify(scene(), proposal, "task")
    assert decision.kind == "plan" and decision.actions == ({"action": "pick", "object": "a"},)


def test_graph_is_unchanged_after_success_and_failure():
    graph = scene(relations=[relation("b", "a")])
    before = graph.model_dump(mode="json")
    verify(graph, plan(("pick", "b"), ("place", "tray"), ("pick", "a")), "task")
    assert graph.model_dump(mode="json") == before
    assert_violation(graph, plan(("pick", "a")), "unblocked_pick")
    assert graph.model_dump(mode="json") == before


@given(st.text(min_size=1).filter(lambda value: value.strip() and value not in {"a", "b", "tray"}))
def test_arbitrary_unknown_ids_never_become_actions(unknown):
    assert_violation(scene(), plan(("pick", unknown)), "known_ids", 0)


@given(st.integers(min_value=1, max_value=12))
def test_chain_top_first_and_original_first_pixel(depth):
    names = [f"item_{index}" for index in range(depth)]
    objects = [observation(name, index) for index, name in enumerate(names)]
    objects.append(observation("tray", depth, label="tray", movable=False))
    graph = scene(objects, [relation(names[i + 1], names[i]) for i in range(depth - 1)])
    steps = []
    for name in reversed(names):
        steps.append(("pick", name))
        if name != names[0]:
            steps.append(("place", "tray"))
    before = graph.model_dump()
    decision = verify(graph, plan(*steps), "chain")
    assert decision.target_pixel == objects[depth - 1].pixel
    assert decision.official()["target_pixel"] == list(objects[depth - 1].pixel)
    assert graph.model_dump() == before
    if depth > 1:
        assert_violation(graph, plan(("pick", names[0])), "unblocked_pick", 0)


@given(st.integers(min_value=2, max_value=25))
def test_any_length_support_cycle_is_rejected(depth):
    names = [f"item_{index}" for index in range(depth)]
    graph = scene([observation(name, index) for index, name in enumerate(names)],
                  [relation(names[i], names[(i + 1) % depth]) for i in range(depth)])
    assert_violation(graph, plan(("pick", names[0])), "acyclic_supports")


@pytest.mark.parametrize("order", list(permutations(("bottom", "middle", "top"))))
def test_only_top_first_stack_order_is_valid(order):
    graph = scene(
        [observation(name, i) for i, name in enumerate(("bottom", "middle", "top"))]
        + [observation("tray", 3, label="tray", movable=False)],
        [relation("top", "middle"), relation("middle", "bottom")],
    )
    steps = [(action, name) for item in order for action, name in (("pick", item), ("place", "tray"))]
    if order == ("top", "middle", "bottom"):
        assert verify(graph, plan(*steps), "task").kind == "plan"
    else:
        with pytest.raises(PlanViolation) as caught:
            verify(graph, plan(*steps), "task")
        assert caught.value.checks[-1].rule == "unblocked_pick"


@st.composite
def state_programs(draw):
    count = draw(st.integers(min_value=1, max_value=6))
    supports = draw(st.lists(st.booleans(), min_size=count - 1, max_size=count - 1))
    forbidden = draw(st.sets(st.integers(min_value=0, max_value=count - 1)))
    steps = draw(st.lists(st.tuples(st.sampled_from(["pick", "place"]),
                                   st.integers(min_value=0, max_value=count)),
                          min_size=1, max_size=30))
    return count, supports, forbidden, steps


@given(state_programs())
def test_random_programs_match_independent_state_machine(program):
    count, supports, forbidden, steps = program
    names = [f"item_{index}" for index in range(count)] + ["tray"]
    graph = scene(
        [observation(name, i, label="tray" if i == count else "cube", movable=i != count)
         for i, name in enumerate(names)],
        [relation(names[i + 1], names[i]) for i, supported in enumerate(supports) if supported],
    )
    # Independent relation-pair oracle, not a re-use of planner helpers.
    occupied = {(i + 1, i) for i, supported in enumerate(supports) if supported}
    held = None
    expected_valid = True
    for action, index in steps:
        if action == "pick":
            if held is not None or index == count or index in forbidden or any(
                lower == index for _, lower in occupied
            ):
                expected_valid = False
                break
            occupied = {(upper, lower) for upper, lower in occupied if upper != index}
            held = index
        else:
            if held is None or held == index:
                expected_valid = False
                break
            occupied.add((held, index))
            held = None
    proposal = plan(*((action, names[index]) for action, index in steps),
                    forbidden=tuple(names[index] for index in sorted(forbidden)))
    before = graph.model_dump()
    if expected_valid:
        decision = verify(graph, proposal, "generated")
        assert decision.kind == "plan"
        assert all(check.passed for check in decision.checks)
        first_index = steps[0][1]
        assert decision.target_pixel == graph.by_id()[names[first_index]].pixel
        assert len(decision.actions) == len(steps)
    else:
        with pytest.raises(PlanViolation):
            verify(graph, proposal, "generated")
    assert graph.model_dump() == before


def test_specific_graspable_close_up_overrides_a_coarse_fixed_surface_label():
    # The whole-image inventory called a cube "tray", which the class rule would normally
    # fix; but its close-up morphology explicitly identifies "cube" and reports it graspable.
    cube_observation = ObjectObservation(
        id="cube_1", label="tray", noun="tray", color="red",
        box=(10, 10, 60, 60), pixel=(35, 35), confidence=0.95,
        movable=True, graspable=True, visual_form="A solid cube (close-up name: cube)",
    )
    assert _movable(cube_observation) is True


@pytest.mark.parametrize("close_up", ["plastic tub", "plate", "saucer", "hand"])
def test_close_up_that_is_not_a_movable_class_does_not_override_the_rule(close_up):
    observation = ObjectObservation(
        id="item_1", label="tray", noun="tray", color="grey",
        box=(10, 10, 60, 60), pixel=(35, 35), confidence=0.95,
        movable=True, graspable=True, visual_form=f"A flat surface (close-up name: {close_up})",
    )
    assert _movable(observation) is False


@pytest.mark.parametrize("requested,actual", [("in", "inside"), ("on", "on")])
def test_explicit_final_receiver_relation_is_verified_without_changing_official_schema(requested, actual):
    goal = Goal(target="a", destination="tray", relation=requested)
    proposal = Proposal(kind="plan", goals=(goal,), goal_mode="final_state", actions=(
        {"action": "pick", "object": "a"},
        {"action": "place", "object": "tray", "relation": actual},
    ))
    graph = scene()
    before = graph.model_dump()
    decision = verify(graph, proposal, "goal")
    assert decision.actions == (
        {"action": "pick", "object": "a"}, {"action": "place", "object": "tray"},
    )
    assert decision.proposal.goals == (goal,)
    assert graph.model_dump() == before


@pytest.mark.parametrize("requested,actual", [("in", "on"), ("on", "inside")])
def test_support_and_containment_are_not_interchangeable_goal_states(requested, actual):
    proposal = Proposal(kind="plan", goal_mode="final_state", goals=(
        Goal(target="a", destination="tray", relation=requested),
    ), actions=(
        {"action": "pick", "object": "a"},
        {"action": "place", "object": "tray", "relation": actual},
    ))
    assert_violation(scene(), proposal, "final_goal_state")


def test_ordered_goal_coverage_rejects_a_dropped_operation_even_when_last_goal_is_done():
    proposal = plan(("pick", "b"), ("place", "tray"), goals=(
        Goal(target="a", destination="tray"), Goal(target="b", destination="tray"),
    ))
    assert_violation(scene(), proposal, "requested_goals")


@pytest.mark.parametrize("goal_mode", ["ordered", "final_state"])
def test_executable_prefix_is_not_a_complete_requested_task(goal_mode):
    proposal = plan(("pick", "a"), ("place", "tray"), goals=(
        Goal(target="a", destination="tray"), Goal(target="b", destination="tray"),
    ), goal_mode=goal_mode)
    rule = "requested_goals" if goal_mode == "ordered" else "final_goal_state"
    assert_violation(scene(), proposal, rule)


@pytest.mark.parametrize("goal_mode", ["ordered", "final_state"])
def test_requested_receiver_cannot_be_replaced_by_an_incompatible_final_hold(goal_mode):
    proposal = plan(("pick", "a"), ("place", "tray"), ("pick", "a"),
                    goals=(Goal(target="a", destination="tray"),), goal_mode=goal_mode)
    assert_violation(scene(), proposal, "final_goal_state")


@pytest.mark.parametrize("goal_mode", ["ordered", "final_state"])
def test_explicit_hold_cannot_be_replaced_by_an_unrequested_final_placement(goal_mode):
    proposal = plan(("pick", "a"), ("place", "tray"),
                    goals=(Goal(target="a", relation="held"),), goal_mode=goal_mode)
    assert_violation(scene(), proposal, "final_goal_state")


def test_ordered_repeated_target_operations_can_supersede_intermediate_placements():
    goals = (Goal(target="a", destination="b"),
             Goal(target="a", destination="tray"), Goal(target="a", relation="held"))
    proposal = plan(("pick", "a"), ("place", "b"), ("pick", "a"), ("place", "tray"),
                    ("pick", "a"), goals=goals)
    assert verify(scene(), proposal, "ordered").kind == "plan"


def test_final_state_mode_does_not_accept_a_superseded_requested_placement():
    proposal = plan(("pick", "a"), ("place", "b"), ("pick", "a"), ("place", "tray"),
                    goals=(Goal(target="a", destination="b"),
                           Goal(target="a", destination="tray")), goal_mode="final_state")
    assert_violation(scene(), proposal, "final_goal_state")


@pytest.mark.parametrize("base_relation", ["on", "inside"])
def test_final_under_requires_both_base_placement_and_upper_restoration(base_relation):
    goal = Goal(target="a", destination="tray", relation="under",
                under_object="b", base_relation=base_relation)
    actions = (
        {"action": "pick", "object": "b"}, {"action": "place", "object": "tray"},
        {"action": "pick", "object": "a"},
        {"action": "place", "object": "tray", "relation": base_relation},
        {"action": "pick", "object": "b"}, {"action": "place", "object": "a"},
    )
    graph = scene(relations=[relation("b", "a")])
    proposal = Proposal(kind="plan", goals=(goal,), actions=actions)
    assert verify(graph, proposal, "under").kind == "plan"
    assert_violation(graph, proposal.model_copy(update={"actions": proposal.actions[:-2]}),
                     "requested_goals")
    assert_violation(graph, proposal.model_copy(update={"forbidden_movement": ("b",)}),
                     "forbidden_movement", 0)


def test_final_under_cannot_finish_with_upper_held_after_restoration():
    proposal = plan(
        ("pick", "b"), ("place", "tray"), ("pick", "a"), ("place", "tray"),
        ("pick", "b"), ("place", "a"), ("pick", "b"),
        goals=(Goal(target="a", destination="tray", relation="under", under_object="b"),),
    )
    assert_violation(scene(relations=[relation("b", "a")]), proposal, "final_goal_state")


def test_final_under_checks_explicit_base_not_merely_upper_on_lower():
    graph = scene([observation("a"), observation("b", 1),
                   observation("tray", 2, label="tray"), observation("base", 3, label="plate")])
    proposal = plan(
        ("pick", "b"), ("place", "tray"), ("pick", "a"), ("place", "tray"),
        ("pick", "b"), ("place", "a"),
        goals=(Goal(target="a", destination="base", relation="under", under_object="b"),),
        goal_mode="final_state",
    )
    assert_violation(graph, proposal, "final_goal_state")


def test_legacy_current_under_still_enforces_the_requested_final_receiver():
    goal = Goal(target="a", destination="tray", relation="under")
    graph = scene(relations=[relation("b", "a")])
    proposal = plan(("pick", "b"), ("place", "tray"), ("pick", "a"), ("place", "tray"),
                    goals=(goal,))
    assert verify(graph, proposal, "current-under").kind == "plan"
    assert_violation(graph, proposal.model_copy(update={"actions": proposal.actions[:-1]}),
                     "requested_goals")


@pytest.mark.parametrize("goal", [
    Goal(target="unknown", destination="tray"),
    Goal(target="a", destination="unknown"),
    Goal(target="a", destination="tray", relation="under", under_object="unknown"),
])
def test_retained_goal_roles_must_be_observed_ids(goal):
    assert_violation(scene(), plan(("pick", "a"), goals=(goal,)), "known_ids")


def test_final_under_requires_distinct_upper_and_base():
    goal = Goal(target="a", destination="b", relation="under", under_object="b")
    assert_violation(scene(), plan(("pick", "a"), goals=(goal,)), "no_self_placement")


def test_final_under_requires_an_explicit_base():
    goal = Goal(target="a", relation="under", under_object="b")
    assert_violation(scene(), plan(("pick", "a"), goals=(goal,)), "goal_base")


def test_explicit_goals_reject_observed_support_containment_cycles():
    graph = scene(relations=[relation("a", "b", "inside"), relation("b", "a", "inside")])
    assert_violation(graph, plan(("pick", "a"), goals=(Goal(target="a", relation="held"),)),
                     "acyclic_supports")


def test_explicit_goal_placement_cannot_create_a_containment_cycle():
    graph = scene(relations=[relation("b", "a", "inside")])
    proposal = Proposal(kind="plan", goals=(Goal(target="a", destination="b", relation="in"),),
                        actions=({"action": "pick", "object": "a"},
                                 {"action": "place", "object": "b", "relation": "inside"}))
    assert_violation(graph, proposal, "acyclic_supports", 1)


def test_staging_after_a_pickup_does_not_replace_a_later_requested_transfer():
    proposal = plan(("pick", "a"), ("place", "tray"), goals=(
        Goal(target="a"), Goal(target="a", destination="tray"),
    ))
    assert_violation(scene(), proposal, "requested_goals")
