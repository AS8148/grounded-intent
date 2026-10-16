"""Deterministic checks of an entire pick/place proposal against observed state.

The graph is immutable; a private directed blocker graph is updated as objects are
picked and placed. Invalid proposals raise repairable errors, never fake refusals.
This verifier does not establish grasp feasibility or certified physical safety.
"""
from __future__ import annotations

import re
from collections import deque

from pydantic import ValidationError

from .canonical import _CANON_NOUN, _FIXED_SURFACE, canonical_noun
from .domain import Check, Decision, Goal, ObjectObservation, Proposal, SceneGraph

BLOCKING_CONFIDENCE = 0.7
_FIXED_CONTAINERS = frozenset({"plate", "plates", "saucer", "saucers", "tray", "trays"})
# Never graspable, by the organisers' own rules: the fixed surfaces above, plus a
# person. The class rule frees every other object, and the one object it freed that
# must not be freed was a person's arm in scene_10 (the watch beside it is pickable).
_NEVER_GRASPABLE = _FIXED_CONTAINERS | frozenset({
    "hand", "hands", "arm", "arms", "finger", "fingers", "thumb", "wrist", "person",
    "human", "head", "face",
})

# Backwards-compatible local name used by the verifier's class predicate.
_NEVER_GRASP = _NEVER_GRASPABLE


class PlanViolation(ValueError):
    """A rejected proposal and its audit trail, suitable for one constrained repair."""

    def __init__(self, message: str, checks: tuple[Check, ...]) -> None:
        super().__init__(message)
        self.checks = checks


def _movable(obj: ObjectObservation) -> bool:
    """Whether the observed physical object may be grasped under the class rule.

    The organizer's rule is class-based: plates, saucers and trays are fixed;
    other ordinary objects are movable, regardless of the coarse model's
    ``movable`` bit. A full-scene inventory can misname a small solid object as
    a tray, though. When its detailed image-only close-up names a specific
    non-fixed class and the model also marks it graspable, use that more
    specific visual class; otherwise the close-up crop would never correct the
    coarse whole-image label it was added to disambiguate.
    """
    words = set(re.findall(r"[a-z]+", obj.label.casefold()))
    base = canonical_noun(obj.label, obj.noun)
    words |= set(re.findall(r"[a-z]+", base.casefold()))
    # Human classes remain fixed regardless of what a crop resembles. Container and
    # surface classes are handled below so a specific, graspable close-up can correct
    # a coarse whole-image resemblance (e.g. a solid cube labelled "tray").
    if words.intersection(_NEVER_GRASPABLE - _FIXED_CONTAINERS):
        return False
    if words.intersection(_FIXED_CONTAINERS):
        close = re.search(r"\(close-up name:\s*([^)]*)\)", obj.visual_form, re.IGNORECASE)
        if close and obj.graspable:
            specific = canonical_noun(close.group(1))
            if specific in _CANON_NOUN.values() and specific not in _FIXED_SURFACE:
                return True
        return False
    return True


def _cycle_nodes(edges: dict[str, set[str]]) -> tuple[str, ...]:
    """Kahn traversal avoids recursion limits and terminates on cyclic observations."""
    indegree = dict.fromkeys(edges, 0)
    for destinations in edges.values():
        for destination in destinations:
            indegree[destination] += 1
    ready = deque(node for node, degree in indegree.items() if degree == 0)
    while ready:
        node = ready.popleft()
        for destination in edges[node]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    return tuple(sorted(node for node, degree in indegree.items() if degree))


def verify(
    graph: SceneGraph,
    proposal: Proposal,
    instruction_id: str,
    model: str = "Apertus",
    reasoning_seconds: float = 0,
    usage: dict[str, int] | None = None,
) -> Decision:
    """Verify all actions, preserving original first-pick pixel attribution.

    ``place.object`` is the destination, not the carried object's ID. Explicit
    forbidden movement applies to picks, not to use of stationary destinations.
    No blocker moves are inserted: feedback lets the language model reconsider the
    actual intent and choose a permitted observed destination. No instruction text
    is available here; semantic grounding/restriction extraction is Apertus's job.
    """
    checks: list[Check] = []

    def require(rule: str, condition: bool, detail: str, step: int | None = None) -> None:
        checks.append(Check(rule=rule, passed=condition, detail=detail, step=step))
        if not condition:
            raise PlanViolation(detail, tuple(checks))

    # Revalidation also guards callers that bypassed Pydantic via model_construct.
    try:
        proposal = Proposal.model_validate(proposal.model_dump())
    except ValidationError as exc:
        require("exclusive_output", False, f"Invalid proposal shape: {exc}")

    exclusive = not proposal.question.strip() if proposal.kind == "plan" else (
        not proposal.reason.strip() if proposal.kind == "ask_user"
        else not proposal.question.strip()
    )
    require("exclusive_output", exclusive, "Output must be exclusively plan, ask_user, or refuse")
    objects = graph.by_id()
    for object_id in proposal.forbidden_movement:
        require(
            "known_ids", bool(object_id.strip()) and object_id in objects,
            f"Forbidden-movement ID must be an observed object: {object_id!r}",
        )
    for index, action in enumerate(proposal.actions):
        require(
            "known_ids", bool(action.object.strip()) and action.object in objects,
            f"Action ID must be an observed object: {action.object!r}", index,
        )

    if proposal.kind != "plan":
        action = {"action": "ask_user", "question": proposal.question} if (
            proposal.kind == "ask_user"
        ) else {"action": "refuse", "reason": proposal.reason}
        return Decision(
            instruction_id=instruction_id, target_pixel=None, actions=(action,),
            kind=proposal.kind, checks=tuple(checks), proposal=proposal, model=model,
            reasoning_seconds=reasoning_seconds, usage={} if usage is None else usage,
        )

    for index, goal in enumerate(proposal.goals):
        for role in ("target", "destination", "under_object"):
            object_id = getattr(goal, role)
            if object_id is not None:
                require("known_ids", bool(object_id.strip()) and object_id in objects,
                        f"Goal {role} must be an observed object: {object_id!r}", index)
        require("no_self_placement", goal.target != goal.destination,
                f"Goal cannot place {goal.target!r} onto itself", index)
        if goal.under_object is not None:
            require("goal_base", goal.destination is not None,
                    "A final under goal requires an explicit observed base", index)
            require("no_self_placement",
                    len({goal.target, goal.under_object, goal.destination}) == 3,
                    "An under goal requires distinct lower, upper, and base objects", index)

    edges: dict[str, set[str]] = {object_id: set() for object_id in objects}
    incoming: dict[str, set[str]] = {object_id: set() for object_id in objects}
    contained_in: dict[str, set[str]] = {object_id: set() for object_id in objects}
    placements: dict[str, set[tuple[str, str]]] = {object_id: set() for object_id in objects}
    for relation in graph.relations:
        if relation.predicate in {"inside", "on"} and relation.confidence >= BLOCKING_CONFIDENCE:
            contained_in[relation.subject].add(relation.object)
            placements[relation.subject].add((relation.predicate, relation.object))
    for relation in graph.relations:
        if relation.predicate in {"on", "blocks"} and (
            relation.confidence >= BLOCKING_CONFIDENCE
        ):
            # A container cannot block an object it contains or supports: a
            # tray/plate edge "blocking" its contents is an occlusion misreport.
            if relation.subject in contained_in[relation.object] and not objects[relation.subject].movable:
                continue
            edges[relation.subject].add(relation.object)
            incoming[relation.object].add(relation.subject)
    cycle = _cycle_nodes(edges)
    require("acyclic_supports", not cycle, f"Blocking/support cycle detected: {', '.join(cycle)}"
            if cycle else "Observed blocking/support relations are acyclic")
    support_edges = {
        subject: {destination for _, destination in locations}
        for subject, locations in placements.items()
    }
    if proposal.goals:
        cycle = _cycle_nodes(support_edges)
        require("acyclic_supports", not cycle,
                f"Support/containment cycle detected: {', '.join(cycle)}"
                if cycle else "Observed support/containment relations are acyclic")

    forbidden = set(proposal.forbidden_movement)
    held: str | None = None
    first_pick: str | None = None
    completed_goals = 0
    goal_target_picked = False

    def achieved(goal: Goal) -> bool:
        if goal.relation == "held" or goal.destination is None:
            return held == goal.target
        if held == goal.target:
            return False
        if goal.relation == "under" and goal.under_object is not None:
            return (
                (goal.base_relation, goal.destination) in placements[goal.target]
                and ("on", goal.target) in placements[goal.under_object]
                and held != goal.under_object
            )
        relation = "inside" if goal.relation == "in" else "on"
        return (relation, goal.destination) in placements[goal.target]

    for index, action in enumerate(proposal.actions):
        object_id = action.object
        carried = held
        if action.action == "pick":
            require("held_empty_before_pick", held is None,
                    f"Cannot pick {object_id!r} while holding {held!r}", index)
            require("movable_object", _movable(objects[object_id]),
                    f"Cannot pick fixed/nonmovable object {object_id!r}", index)
            require("forbidden_movement", object_id not in forbidden,
                    f"Cannot move explicitly restricted object {object_id!r}", index)
            blockers = sorted(incoming[object_id])
            require("unblocked_pick", not blockers,
                    f"Cannot pick {object_id!r}; blockers: {', '.join(blockers)}"
                    if blockers else f"Object {object_id!r} is unblocked", index)
            # Its former placement/blocking relationships disappear on removal.
            for destination in edges[object_id]:
                incoming[destination].discard(object_id)
            edges[object_id].clear()
            placements[object_id].clear()
            support_edges[object_id].clear()
            held = object_id
            if first_pick is None:
                first_pick = object_id
        else:
            require("held_before_place", held is not None,
                    f"Cannot place onto {object_id!r} with an empty gripper", index)
            require("no_self_placement", held != object_id,
                    f"Cannot place {held!r} onto itself", index)
            # A placed object now obstructs its destination, even after later steps.
            # Since held was unblocked when picked and lost its outgoing edges, this
            # new edge cannot introduce a cycle in the previously checked DAG.
            edges[held].add(object_id)
            incoming[object_id].add(held)
            placements[held].add((action.relation, object_id))
            support_edges[held].add(object_id)
            if proposal.goals:
                cycle = _cycle_nodes(support_edges)
                require("acyclic_supports", not cycle,
                        f"Placement creates a support/containment cycle: {', '.join(cycle)}"
                        if cycle else "Placement preserves acyclic support/containment", index)
            held = None
        if proposal.goal_mode == "ordered" and completed_goals < len(proposal.goals):
            goal = proposal.goals[completed_goals]
            if action.action == "pick" and object_id == goal.target:
                goal_target_picked = True
            # Goals are requested operations, not incidental states of a staging
            # object. An under arrangement completes only when its upper is restored.
            actor = goal.under_object if goal.relation == "under" and goal.under_object else goal.target
            relevant = object_id == actor if action.action == "pick" else carried == actor
            if goal_target_picked and relevant and achieved(goal):
                completed_goals += 1
                goal_target_picked = False

    if proposal.goals:
        if proposal.goal_mode == "ordered":
            require("requested_goals", completed_goals == len(proposal.goals),
                    f"Completed {completed_goals} of {len(proposal.goals)} requested goals")
            final_goals = proposal.goals[-1:]
        else:
            final_goals = proposal.goals
        for goal in final_goals:
            require("final_goal_state", achieved(goal),
                    f"Final state must satisfy requested goal {goal.model_dump()}")

    # The validated plan is nonempty, and a first place was rejected above.
    assert first_pick is not None
    return Decision(
        instruction_id=instruction_id, target_pixel=objects[first_pick].pixel,
        # The official answer schema allows exactly `action` and `object` (plus an
        # optional pixel) per pick/place step, so the internal `relation` field
        # stays in the proposal and never reaches the submitted file.
        actions=tuple({"action": action.action, "object": action.object}
                      for action in proposal.actions),
        kind="plan", checks=tuple(checks), proposal=proposal, model=model,
        reasoning_seconds=reasoning_seconds, usage={} if usage is None else usage,
    )
