"""Regression tests for the deterministic referent resolver + plan veto.

These pin the failure modes Apertus-8B's referent binding exhibits that the
resolver recovers honestly from scene geometry — wrong colour-noun binding,
ordinal/spatial instance selection, generic-noun + landmark binding, and the
'without moving X' overlap veto.
"""
from __future__ import annotations

from grounded_intent.domain import (
    Goal,
    ObjectObservation,
    Relation,
    SceneGraph,
    TaskIntent,
)
from grounded_intent.referents import (
    clear_blockers_first,
    clear_unrequested_destinations,
    fix_under_destinations,
    prune_incompatible_category_goals,
    rescue_ambiguous_refusal,
    rescue_clarification,
    resolve_target,
    under_placement_first_pick,
    veto_plan,
    veto_unmatched_colour,
)


def _obj(oid, noun, color, px, mov=True, box=None, label=None):
    x, y = px
    b = box or (max(0, x - 50), max(0, y - 50), x + 50, y + 50)
    return ObjectObservation(
        id=oid, label=label or noun, noun=noun, color=color,
        box=b, pixel=px,
        confidence=0.9, movable=mov, polygon=(), graspable=mov,
    )


def _scene(objs, rels=()):
    return SceneGraph(
        scene="s", width=2500, height=2500, image_sha256="x",
        objects=tuple(objs), relations=tuple(rels),
        perception_model="t", perception_seconds=0,
    )


def _plan(target, destination=None):
    return TaskIntent(kind="plan", goals=(Goal(target=target, destination=destination),),
                      reason="r", forbidden_movement=())


def _rel(sub, obj, pred="on"):
    return Relation(subject=sub, object=obj, predicate=pred, confidence=0.9, evidence="")


def _t(intent):
    return intent.goals[0].target


def test_ordinal_from_the_right_binds_correct_instance():
    g = _scene([
        _obj("red_gear_1", "gear", "red", (100, 50)),
        _obj("red_gear_2", "gear", "red", (200, 50)),
        _obj("red_gear_3", "gear", "red", (300, 50)),
    ])
    i = resolve_target(g, _plan("red_gear_3"), "pick up the second red gear from the right")
    assert _t(i) == "red_gear_2"


def test_generic_object_left_of_landmark_binds_nearest_left():
    g = _scene([
        _obj("red_cube", "cube", "red", (50, 50)),
        _obj("yellow_gear", "gear", "yellow", (80, 50)),
    ])
    i = resolve_target(g, _plan("yellow_gear"), "pick up the object to the left of the yellow gear")
    assert _t(i) == "red_cube"


def test_left_of_requires_horizontal_dominance_not_below():
    # mandarin below the mug (dx small, dy large) must NOT be 'left of' it —
    # the cube that is genuinely left-of is the referent.
    g = _scene([
        _obj("orange_orange", "orange", "orange", (2161, 923)),
        _obj("red_cube", "cube", "red", (1673, 460)),
        _obj("white_mug", "mug", "white", (2187, 352)),
    ])
    i = resolve_target(g, _plan("orange_orange"), "pick up the object to the left of the mug")
    assert _t(i) == "red_cube"


def test_color_drift_falls_back_to_unique_noun_candidate():
    g = _scene([
        _obj("orange_cube", "cube", "orange", (338, 1049)),
        _obj("red_gear", "gear", "red", (825, 503)),
    ])
    i = resolve_target(g, _plan("red_gear"), "pick up the red cube")
    assert _t(i) == "orange_cube"


def test_containment_relation_binds_insider():
    g = _scene(
        [
            _obj("yellow_gear", "gear", "yellow", (10, 50)),
            _obj("red_cube", "cube", "red", (20, 50)),
            _obj("grey_tray", "tray", "grey", (30, 50), mov=False),
        ],
        [_rel("yellow_gear", "grey_tray")],
    )
    i = resolve_target(g, _plan("red_cube"), "pick up the gear in the grey tray")
    assert _t(i) == "yellow_gear"


def test_supported_reference_can_share_its_class_with_the_target():
    graph = _scene([
        _obj("load", "part", "silver", (200, 200)),
        _obj("support", "part", "silver", (200, 260)),
        _obj("separate", "part", "silver", (700, 700)),
    ], [_rel("load", "support")])
    corrected = resolve_target(
        graph, _plan("support"),
        "Retrieve the component supported by the other component.",
    )
    assert corrected is not None
    assert corrected.goals[0].target == "load"


def test_multiple_same_class_support_pairs_remain_unresolved():
    graph = _scene([
        _obj("load_a", "part", "silver", (200, 200)),
        _obj("support_a", "part", "silver", (200, 260)),
        _obj("load_b", "part", "silver", (700, 700)),
        _obj("support_b", "part", "silver", (700, 760)),
    ], [_rel("load_a", "support_a"), _rel("load_b", "support_b")])
    assert resolve_target(
        graph, _plan("support_a"),
        "Retrieve the component supported by the other component.",
    ) is None


def test_containment_geometric_fallback_without_relation():
    # 'the object lying on the grey part' — cube overlaps part horizontally.
    cube = _obj("red_cube", "cube", "red", (931, 868), box=(747, 651, 1171, 1089))
    part = _obj("white_part", "part", "white", (813, 572), box=(727, 493, 1059, 714))
    orange = _obj("orange_orange", "orange", "orange", (626, 1361), box=(485, 1212, 767, 1503))
    g = _scene([cube, part, orange])
    i = resolve_target(g, _plan("orange_orange"), "pick up the object lying on the grey part")
    assert _t(i) == "red_cube"


def test_landmark_never_binds_itself():
    # 'the object closest to the mug' must not resolve to the mug itself.
    g = _scene([
        _obj("red_cube", "cube", "red", (338, 1049)),
        _obj("white_mug", "mug", "white", (825, 503)),
        _obj("yellow_gear", "gear", "yellow", (799, 885)),
    ])
    i = resolve_target(g, _plan("white_mug"), "pick up the object closest to the mug")
    assert _t(i) == "yellow_gear"


def test_ambiguous_same_noun_no_qualifier_not_rebound():
    # Two identical gears and no qualifier — never guess; disambiguate_plan
    # asks the user instead.
    g = _scene([
        _obj("red_gear_1", "gear", "red", (100, 50)),
        _obj("red_gear_2", "gear", "red", (200, 50)),
    ])
    assert resolve_target(g, _plan("red_gear_1"), "pick up the red gear") is None


def test_multi_goal_plan_not_rebound():
    g = _scene([_obj("a", "cube", "red", (10, 50)), _obj("b", "gear", "red", (20, 50))])
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="a", destination=None), Goal(target="b", destination=None)),
        reason="r", forbidden_movement=(),
    )
    assert resolve_target(g, intent, "pick up the red cube then the red gear") is None


def test_veto_refuses_when_pinned_object_overlaps_target():
    g = _scene([
        _obj("red_gear", "gear", "red", (425, 491), box=(282, 373, 636, 682)),
        _obj("white_mug", "mug", "white", (588, 705), box=(393, 519, 882, 898)),
    ])
    v = veto_plan(g, _plan("red_gear"), "pick up the red gear without moving the mug")
    assert v is not None and v.kind == "refuse"


def test_veto_ignores_non_overlapping_pinned_object():
    g = _scene([
        _obj("red_gear", "gear", "red", (100, 100), box=(50, 50, 150, 150)),
        _obj("white_mug", "mug", "white", (500, 500), box=(450, 450, 550, 550)),
    ])
    assert veto_plan(g, _plan("red_gear"), "pick up the red gear without moving the mug") is None


def test_veto_ignores_fixed_surface_overlap():
    g = _scene([
        _obj("red_gear", "gear", "red", (425, 491), box=(282, 373, 636, 682)),
        _obj("grey_tray", "tray", "grey", (500, 500), mov=False, box=(0, 0, 1200, 1800)),
    ])
    assert veto_plan(g, _plan("red_gear"), "pick up the red gear without moving the tray") is None


# --- clarification rescue: a qualifier the scene settles is planned, not asked ---


def test_clarification_rescued_when_qualifier_settles_it():
    g = _scene([
        _obj("white_mug", "mug", "white", (588, 705), box=(393, 519, 882, 898)),
        _obj("red_cube", "cube", "red", (700, 800), box=(650, 750, 750, 850)),
        _obj("yellow_gear", "gear", "yellow", (1500, 1600), box=(1450, 1550, 1550, 1650)),
    ])
    ask = TaskIntent(kind="ask_user", question="Which object do you mean?", forbidden_movement=())
    settled = rescue_clarification(g, ask, "pick up the object closest to the mug")
    assert settled is not None and settled.kind == "plan"
    assert _t(settled) == "red_cube"


def test_clarification_kept_when_no_qualifier_resolves():
    g = _scene([
        _obj("red_gear_1", "gear", "red", (100, 100)),
        _obj("red_gear_2", "gear", "red", (400, 100)),
    ])
    ask = TaskIntent(kind="ask_user", question="Which object do you mean?", forbidden_movement=())
    assert rescue_clarification(g, ask, "pick up the red gear") is None


def test_clarification_never_touches_a_plan():
    g = _scene([_obj("red_cube", "cube", "red", (700, 800))])
    plan = _plan("red_cube")
    assert rescue_clarification(g, plan, "pick up the object closest to the mug") is None


# --- blocker ordering: a goal whose blocker must move comes first ---


def test_blockers_first_reorders_when_a_later_goal_is_blocked():
    g = _scene(
        [_obj("red_gear_1", "gear", "red", (100, 100)),
         _obj("red_gear_2", "gear", "red", (400, 100)),
         _obj("red_cube", "cube", "red", (400, 60))],
        rels=[_rel("red_cube", "red_gear_2")],
    )
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="red_gear_1", destination="grey_tray"),
               Goal(target="red_gear_2", destination="grey_tray")),
        reason="r", forbidden_movement=())
    reordered = clear_blockers_first(g, intent, "put both red gears in the grey tray")
    assert reordered is not None
    assert [goal.target for goal in reordered.goals] == ["red_gear_2", "red_gear_1"]


def test_blockers_first_leaves_unblocked_goals_alone():
    g = _scene([_obj("red_gear_1", "gear", "red", (100, 100)),
                _obj("red_gear_2", "gear", "red", (400, 100))])
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="red_gear_1", destination="grey_tray"),
               Goal(target="red_gear_2", destination="grey_tray")),
        reason="r", forbidden_movement=())
    assert clear_blockers_first(g, intent, "put both red gears in the grey tray") is None


def test_blockers_first_ignores_a_fixed_blocker():
    # A tray can never be picked, so it cannot be cleared first.
    g = _scene(
        [_obj("red_gear_1", "gear", "red", (100, 100)),
         _obj("red_gear_2", "gear", "red", (400, 100)),
         _obj("grey_tray", "tray", "grey", (400, 60), mov=False)],
        rels=[_rel("grey_tray", "red_gear_2")],
    )
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="red_gear_1", destination="grey_tray"),
               Goal(target="red_gear_2", destination="grey_tray")),
        reason="r", forbidden_movement=())
    assert clear_blockers_first(g, intent, "put both red gears in the grey tray") is None


def test_blockers_first_leaves_under_placements_alone():
    # For "put X under Y" the model's own order already names the resting object
    # first, and under_placement_first_pick owns that correction.
    g = _scene(
        [_obj("red_gear", "gear", "red", (400, 100)),
         _obj("yellow_gear", "gear", "yellow", (400, 60))],
        rels=[_rel("yellow_gear", "red_gear")],
    )
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="yellow_gear", destination=None),
               Goal(target="red_gear", destination="grey_tray")),
        reason="r", forbidden_movement=())
    assert clear_blockers_first(
        g, intent, "put the red gear under the yellow gear in the grey tray") is None


def test_on_top_of_binds_the_observed_supporter():
    # "the object on top of the red gears" is decided by the observed support edge rather
    # than by geometry: the cube the graph shows resting on a red gear is the referent.
    g = _scene([_obj("red_gear_1", "gear", "red", (100, 300)),
                _obj("red_gear_2", "gear", "red", (400, 300)),
                _obj("red_cube", "cube", "red", (400, 200))],
               rels=[_rel("red_cube", "red_gear_2")])
    out = resolve_target(g, _plan("red_gear_1"), "pick up the object on top of the red gears")
    assert out is not None and out.goals[0].target == "red_cube"


def test_on_top_of_without_an_observed_supporter_changes_nothing():
    g = _scene([_obj("red_gear_1", "gear", "red", (100, 300)),
                _obj("red_cube", "cube", "red", (400, 200))])
    assert resolve_target(g, _plan("red_gear_1"), "pick up the object on top of the red gears") is None


# --- refusal rescue: ambiguous referents ask, absent ones refuse ---


def test_ambiguous_referent_refusal_becomes_a_question():
    # "give me that one" singles out nothing, so refusing claims an impossibility the
    # instruction never stated; the honest answer is the question it invites.
    g = _scene([_obj("red_gear_1", "gear", "red", (100, 100)),
                _obj("red_gear_2", "gear", "red", (400, 100))])
    refuse = TaskIntent(kind="refuse", reason="No red cube is present.",
                        forbidden_movement=())
    asked = rescue_ambiguous_refusal(g, refuse, "give me that one")
    assert asked is not None and asked.kind == "ask_user"
    assert "red_gear_1" in asked.question and "red_gear_2" in asked.question


def test_absent_referent_refusal_is_kept():
    # Nothing the instruction names exists: the refusal is about a real impossibility.
    g = _scene([_obj("red_cube", "cube", "red", (100, 100))])
    refuse = TaskIntent(kind="refuse", reason="There is no blue cube.",
                        forbidden_movement=())
    assert rescue_ambiguous_refusal(g, refuse, "pick up the blue cube") is None


def test_unsafe_refusal_is_kept():
    g = _scene([_obj("red_gear_1", "gear", "red", (100, 100)),
                _obj("red_gear_2", "gear", "red", (400, 100))])
    refuse = TaskIntent(kind="refuse", reason="unsafe", forbidden_movement=())
    assert rescue_ambiguous_refusal(g, refuse, "throw the red gear off the table") is None


# --- under placement: the named upper object is the first pick ---


def test_under_placement_picks_the_named_upper_object_without_a_relation():
    # "put the red gear under the mug": the mug lies on top, so the mug is the
    # first pick even when the inventory did not record the support relation.
    g = _scene([_obj("red_gear", "gear", "red", (400, 100)),
                _obj("white_and_green_mug", "mug", "white_and_green", (400, 60))])
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="red_gear", destination=None),
               Goal(target="white_and_green_mug", destination="grey_tray")),
        reason="r", forbidden_movement=())
    staged = under_placement_first_pick(g, intent, "put the red gear under the mug in the black tray")
    assert staged is not None
    assert [goal.target for goal in staged.goals] == ["white_and_green_mug", "red_gear"]
    assert staged.goals[0].destination == "grey_tray"  # the goal's own placement is kept
    assert len(staged.goals) == 2  # and it is not duplicated


def test_under_placement_leaves_a_correct_order_alone():
    g = _scene([_obj("red_gear_2", "gear", "red", (400, 100)),
                _obj("yellow_gear", "gear", "yellow", (400, 60))])
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="yellow_gear", destination=None),
               Goal(target="red_gear_2", destination="grey_tray")),
        reason="r", forbidden_movement=())
    assert under_placement_first_pick(
        g, intent, "put the red gear under the yellow gear in the grey tray") is None


def test_under_placement_ignores_an_ambiguous_upper_referent():
    g = _scene([_obj("red_gear", "gear", "red", (400, 100)),
                _obj("white_mug_1", "mug", "white", (100, 60)),
                _obj("white_mug_2", "mug", "white", (700, 60))])
    intent = TaskIntent(kind="plan",
                        goals=(Goal(target="red_gear", destination=None),),
                        reason="r", forbidden_movement=())
    assert under_placement_first_pick(
        g, intent, "put the red gear under the mug in the black tray") is None


def test_under_placement_stages_an_observed_blocker_first():
    g = _scene([_obj("red_gear", "gear", "red", (400, 100)),
                _obj("yellow_gear", "gear", "yellow", (400, 60))],
               rels=[_rel("yellow_gear", "red_gear")])
    intent = TaskIntent(kind="plan",
                        goals=(Goal(target="red_gear", destination="grey_tray"),),
                        reason="r", forbidden_movement=())
    staged = under_placement_first_pick(
        g, intent, "put the red gear under the yellow gear in the grey tray")
    assert staged is not None and [goal.target for goal in staged.goals] == ["yellow_gear", "red_gear"]


# --- distant colour mismatch: the one licence an instruction cannot give ---


def test_distant_colour_is_refused():
    g = _scene([_obj("red_cube", "cube", "red", (700, 800))])
    veto = veto_unmatched_colour(g, _plan("red_cube"), "pick up the blue cube")
    assert veto is not None and veto.kind == "refuse"


def test_adjacent_colour_is_tolerated():
    # A yellow gear inventoried over a yellow-green one must still be pickable.
    g = _scene([_obj("green_gear", "gear", "green", (700, 800))])
    assert veto_unmatched_colour(g, _plan("green_gear"), "pick up the yellow gear") is None


def test_achromatic_colour_is_never_vetoed():
    # Black, grey and silver are lighting and material, not a different object.
    g = _scene([_obj("grey_tray", "tray", "grey", (500, 500), mov=False)])
    assert veto_unmatched_colour(g, _plan("grey_tray"), "put the cube in the black tray") is None


def test_colour_veto_skips_under_placement():
    # The first noun of an under-placement names the lower object, not the pick.
    g = _scene([_obj("blue_gear", "gear", "blue", (700, 800))])
    assert veto_unmatched_colour(
        g, _plan("blue_gear"), "put the red gear under the blue gear in the tray") is None


def test_colour_veto_skips_multi_goal_plans():
    g = _scene([_obj("red_cube", "cube", "red", (700, 800)),
                _obj("red_gear", "gear", "red", (400, 100))])
    intent = TaskIntent(
        kind="plan",
        goals=(Goal(target="red_cube", destination="grey_tray"),
               Goal(target="red_gear", destination="grey_tray")),
        reason="r", forbidden_movement=())
    assert veto_unmatched_colour(g, intent, "pick up the blue cube") is None




def test_landmark_color_equivalence_keeps_exact_noun_precedence():
    from grounded_intent.referents import _landmark_objects

    widget = _obj("silver_widget", "widget", "silver", (100, 100))
    coarse_match = _obj("grey_gear", "gear", "grey", (400, 100))
    graph = _scene([widget, coarse_match])
    assert [obj.id for obj in _landmark_objects(graph, "grey widget")] == ["silver_widget"]
    assert widget.color == "silver"


def test_landmark_color_equivalence_applies_to_cream_without_reclassifying():
    from grounded_intent.referents import _landmark_objects

    widget = _obj("cream_widget", "widget", "cream", (100, 100))
    coarse_match = _obj("white_gear", "gear", "white", (400, 100))
    graph = _scene([widget, coarse_match])
    assert [obj.id for obj in _landmark_objects(graph, "white widget")] == ["cream_widget"]


def test_clear_unrequested_destinations_removes_hallucinated_tray():
    plan = TaskIntent(kind="plan", goals=(Goal(target="red_cube", destination="grey_tray"),))
    cleared = clear_unrequested_destinations(plan, "pick up the red cube")
    assert cleared is not None
    assert cleared.goals[0].destination is None


def test_clear_unrequested_destinations_preserves_explicit_prepositions():
    plan = TaskIntent(kind="plan", goals=(Goal(target="red_cube", destination="grey_tray"),))
    assert clear_unrequested_destinations(plan, "put the red cube into the grey tray") is None
    assert clear_unrequested_destinations(plan, "put the red cube on the plate") is None


def test_prune_incompatible_category_goals_removes_non_gears():
    graph = _scene([
        _obj("red_gear_1", "gear", "red", (100, 100)),
        _obj("red_gear_2", "gear", "red", (200, 100)),
        _obj("green_mug", "mug", "green", (300, 100)),
        _obj("green_watch", "watch", "green", (400, 100)),
    ])
    plan = TaskIntent(kind="plan", goals=(
        Goal(target="red_gear_1", destination="orange_plate"),
        Goal(target="red_gear_2", destination="orange_plate"),
        Goal(target="green_mug", destination="orange_plate"),
        Goal(target="green_watch", destination="orange_plate"),
    ))
    pruned = prune_incompatible_category_goals(graph, plan, "put all the gears on the plate except the yellow one")
    assert pruned is not None
    assert [g.target for g in pruned.goals] == ["red_gear_1", "red_gear_2"]


def test_clear_unrequested_destinations_handles_location_qualifiers_in_pick():
    plan = TaskIntent(kind="plan", goals=(Goal(target="red_gear_1", destination="grey_tray"),))
    cleared = clear_unrequested_destinations(plan, "pick up the red gear on the left")
    assert cleared is not None
    assert cleared.goals[0].destination is None


def test_fix_under_destinations_transfers_destination_to_under_target():
    graph = _scene([
        _obj("white_mug", "mug", "white", (100, 100)),
        _obj("red_gear", "gear", "red", (200, 100)),
        _obj("grey_tray", "tray", "grey", (300, 100), mov=False),
    ])
    plan = TaskIntent(kind="plan", goals=(
        Goal(target="white_mug", destination="grey_tray"),
        Goal(target="red_gear", destination=None, relation="under"),
    ))
    fixed = fix_under_destinations(graph, plan, "put the red gear under the mug in the grey tray")
    assert fixed is not None
    assert fixed.goals[0].destination is None
    assert fixed.goals[1].destination == "grey_tray"


def test_resolve_landmark_supports_blocking_relation():
    graph = _scene([
        _obj("yellow_gear", "gear", "yellow", (100, 100)),
        _obj("red_gear_2", "gear", "red", (100, 100)),
        _obj("red_gear_1", "gear", "red", (300, 100)),
    ], [_rel("yellow_gear", "red_gear_2", "blocks")])
    plan = TaskIntent(kind="plan", goals=(Goal(target="red_gear_2"),))
    resolved = resolve_target(graph, plan, "pick up the gear lying on the other gear")
    assert resolved is not None
    assert resolved.goals[0].target == "yellow_gear"


def test_top_ordinal_resolves_topmost_candidate():
    graph = _scene([
        _obj("red_gear_1", "gear", "red", (100, 300)),
        _obj("red_gear_2", "gear", "red", (100, 100)),
        _obj("red_gear_3", "gear", "red", (100, 200)),
    ])
    plan = TaskIntent(kind="plan", goals=(Goal(target="red_gear_1"),))
    resolved = resolve_target(graph, plan, "pick up the top red gear")
    assert resolved is not None
    assert resolved.goals[0].target == "red_gear_2"


def test_second_from_top_ordinal_resolves_second_candidate():
    graph = _scene([
        _obj("red_gear_1", "gear", "red", (100, 300)),
        _obj("red_gear_2", "gear", "red", (100, 100)),
        _obj("red_gear_3", "gear", "red", (100, 200)),
    ])
    plan = TaskIntent(kind="plan", goals=(Goal(target="red_gear_1"),))
    resolved = resolve_target(graph, plan, "pick up the second red gear from the top")
    assert resolved is not None
    assert resolved.goals[0].target == "red_gear_3"
