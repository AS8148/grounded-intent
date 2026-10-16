"""Compile grounded goals into deterministic, observed-only pick/place actions.

Only required blockers are relocated; requested goals retain their order. This
compiler models one gripper and shares the verifier's movement/support rules,
not certified physical safety or grasp feasibility.
"""
from __future__ import annotations

import heapq
import re

from pydantic import ValidationError

from .canonical import canonical_noun
from .domain import Check, Proposal, ProposedAction, SceneGraph, TaskIntent
from .geometry import prune_duplicates
from .planner import BLOCKING_CONFIDENCE, PlanViolation, _cycle_nodes, _movable

MAX_OBJECTS = 128



def _collapse_duplicate_destination_aliases(
    graph: SceneGraph, intent: TaskIntent,
) -> TaskIntent:
    """Collapse repeated moves to the same observed receiving-surface alias.

    Image inventories can report the same broad container twice (slightly different
    boxes, same class/colour, mutually-contained pick points). The perception
    duplicate-pruner already defines when two observations are aliases and preserves
    any pair tied by an explicit physical relation. Reuse that exact contract here,
    but only collapse a plan when *every* goal is the same target with otherwise
    identical fields: two actions on one object into two aliases of the same receiver
    are one move, not two requested moves.
    """
    if intent.kind != "plan" or len(intent.goals) < 2:
        return intent
    first = intent.goals[0]
    if first.destination is None:
        if all(goal.target == first.target and goal.destination is None for goal in intent.goals):
            return intent.model_copy(update={"goals": (first,)})
        return intent
    if not all(
        goal.target == first.target
        and goal.relation == first.relation
        and goal.under_object == first.under_object
        and goal.base_relation == first.base_relation
        and goal.destination is not None
        for goal in intent.goals
    ):
        return intent
    destinations = {goal.destination for goal in intent.goals}
    objects = graph.by_id()
    if any(dest not in objects or not objects[dest].receives_objects
           for dest in destinations):
        return intent
    protected = {
        frozenset((relation.subject, relation.object))
        for relation in graph.relations
        if relation.predicate in {"on", "inside", "blocks"}
        and relation.confidence >= BLOCKING_CONFIDENCE
    }
    _, aliases = prune_duplicates(
        tuple(objects[dest] for dest in sorted(destinations)), protected)
    representatives = {aliases[dest] for dest in destinations}
    if len(representatives) != 1:
        return intent
    destination = next(iter(representatives))
    if all(goal.destination == destination for goal in intent.goals):
        return intent
    return intent.model_copy(update={
        "goals": (first.model_copy(update={"destination": destination}),),
    })

def compile_intent(graph: SceneGraph, intent: TaskIntent) -> Proposal:
    """Ground all actions in ``graph`` without modifying it or inventing surfaces.

    Invalid grounding/cycles raise a single failed Check for bounded repair
    feedback. Actual movement restrictions yield refusals, whereas missing
    staging surfaces or an intermediate pickup destination need clarification.
    A destination-free final goal leaves its target in the gripper.
    """
    def require(rule: str, condition: bool, detail: str, step: int | None = None) -> None:
        if not condition:
            raise PlanViolation(detail, (Check(rule=rule, passed=False, detail=detail, step=step),))

    # Like verify, reject contracts constructed without Pydantic validation.
    try:
        intent = TaskIntent.model_validate(intent.model_dump())
    except ValidationError as exc:
        require("exclusive_output", False, f"Invalid task intent: {exc}")

    objects = graph.by_id()
    for object_id in intent.forbidden_movement:
        require("known_ids", bool(object_id.strip()) and object_id in objects,
                f"Forbidden-movement ID must be an observed object: {object_id!r}")

    if intent.kind != "plan":
        return Proposal(kind=intent.kind, question=intent.question, reason=intent.reason,
                        forbidden_movement=intent.forbidden_movement)

    require("object_bound", len(objects) <= MAX_OBJECTS,
            f"Compilation supports at most {MAX_OBJECTS} observed objects; got {len(objects)}")
    resolved: dict[str, str] = {}
    for goal in intent.goals:
        for field in ("target", "destination", "under_object"):
            ref = getattr(goal, field)
            if ref is None or not ref.strip() or ref in objects:
                continue
            # Resolve to the unique matching object by canonical noun — first a
            # direct id/label/noun hit, then the canonicalised referent (so an
            # instruction noun like 'mandarin' or 'grey part' binds the observed
            # 'orange_orange' or 'white_part' via the shared everyday ontology).
            candidates = [o for o in graph.objects if (o.movable if field == "target" else True)]
            hits = {o.id for o in candidates
                    if o.label == ref or o.noun == ref or o.id.split("_", 1)[-1] == ref}
            if not hits:
                canon = canonical_noun(ref)
                hits = {o.id for o in candidates
                        if canonical_noun(o.label, o.noun) == canon}
            if len(hits) == 1:
                resolved[ref] = hits.pop()
            elif field == "destination" and len(hits) > 1:
                # A bare-noun destination ('the plate') ambiguous among several
                # same-noun objects: every hit is an equally valid instance of
                # the asked kind and only the pick target is scored, so resolve
                # deterministically to the first rather than fail the goal.
                resolved[ref] = sorted(hits)[0]
    if resolved:
        intent = intent.model_copy(update={"goals": tuple(
            g.model_copy(update={
                "target": resolved.get(g.target, g.target),
                "destination": resolved.get(g.destination, g.destination),
                "under_object": resolved.get(g.under_object, g.under_object),
            }) for g in intent.goals)})
    # A restriction names an object the same way a goal does, so it resolves the
    # same way: an instruction word ('mug') becomes the observed ID it denotes.
    if intent.forbidden_movement:
        restriction: list[str] = []
        for ref in intent.forbidden_movement:
            if ref in objects:
                restriction.append(ref)
                continue
            canon = canonical_noun(ref)
            hits = {o.id for o in graph.objects
                    if canonical_noun(o.label, o.noun) == canon and canon}
            restriction.append(hits.pop() if len(hits) == 1 else ref)
        intent = intent.model_copy(update={"forbidden_movement": tuple(restriction)})
    # Repeated goals can arise when one receiving surface has two near-identical
    # observations. Collapse only if the same target goes to exact geometric aliases.
    intent = _collapse_duplicate_destination_aliases(graph, intent)
    for index, goal in enumerate(intent.goals):
        require("known_ids", bool(goal.target.strip()) and goal.target in objects,
                f"Goal target must be an observed object: {goal.target!r}", index)
        if goal.destination is not None:
            require("known_ids", bool(goal.destination.strip()) and goal.destination in objects,
                    f"Goal destination must be an observed object: {goal.destination!r}", index)
            require("no_self_placement", goal.target != goal.destination,
                    f"Cannot place {goal.target!r} onto itself", index)
        if goal.under_object is not None:
            require("known_ids", bool(goal.under_object.strip()) and goal.under_object in objects,
                    f"Goal upper object must be observed: {goal.under_object!r}", index)
            require("no_self_placement",
                    len({goal.target, goal.under_object, goal.destination}) == 3,
                    "An under goal requires distinct lower, upper, and base objects", index)

    edges: dict[str, set[str]] = {object_id: set() for object_id in objects}
    incoming: dict[str, set[str]] = {object_id: set() for object_id in objects}
    contained_in: dict[str, set[str]] = {object_id: set() for object_id in objects}
    for relation in graph.relations:
        if relation.predicate in {"inside", "on"} and relation.confidence >= BLOCKING_CONFIDENCE:
            contained_in[relation.subject].add(relation.object)
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
    require("acyclic_supports", not cycle,
            f"Blocking/support cycle detected: {', '.join(cycle)}")

    forbidden = set(intent.forbidden_movement)
    trays = sorted(object_id for object_id, obj in objects.items()
                   if not _movable(obj) and
                   set(re.findall(r"[a-z]+", obj.label.casefold())).intersection({"tray", "trays"}))
    actions: list[ProposedAction] = []

    def restriction(object_id: str, target: str) -> Proposal | None:
        if object_id in forbidden:
            reason = (f"Cannot move explicitly restricted object {object_id!r} "
                      f"required for goal {target!r}.")
        elif not _movable(objects[object_id]):
            reason = (f"Cannot move fixed/nonmovable object {object_id!r} "
                      f"required for goal {target!r}.")
        else:
            return None
        return Proposal(kind="refuse", reason=reason,
                        forbidden_movement=intent.forbidden_movement)

    def pick(object_id: str) -> None:
        actions.append(ProposedAction(action="pick", object=object_id))
        for destination in edges[object_id]:
            incoming[destination].discard(object_id)
        edges[object_id].clear()

    def place(carried: str, destination: str, relation: str = "on") -> None:
        actions.append(ProposedAction(action="place", object=destination, relation=relation))
        # This edge blocks the destination, not other objects inside a fixed tray.
        # It matches verify and makes later picks of movable destinations respect
        # newly placed objects rather than consulting stale observations.
        edges[carried].add(destination)
        incoming[destination].add(carried)

    # A physically executable prefix is not the requested task. Refuse the entire
    # request when any requested target cannot be moved.
    for goal in intent.goals:
        refusal = restriction(goal.target, goal.target)
        if refusal is not None:
            return refusal

    for index, goal in enumerate(intent.goals):
        if goal.relation == "under" and goal.under_object is not None and goal.destination is None:
            return Proposal(
                kind="ask_user",
                question=f"Which observed base should support {goal.target!r} under {goal.under_object!r}?",
                forbidden_movement=intent.forbidden_movement,
            )
        if goal.relation == "held" and index != len(intent.goals) - 1:
            return Proposal(
                kind="ask_user",
                question=f"How should I complete later goals while keeping {goal.target!r} held?",
                forbidden_movement=intent.forbidden_movement,
            )
        # Discover ancestors iteratively. Each observed object is expanded once;
        # the explicit object bound also bounds each per-goal topological pass.
        required: set[str] = set()
        pending = list(incoming[goal.target])
        # Only an explicit upper-object role requests a final under arrangement.
        # Legacy `under` without that role qualifies the target's current location;
        # its destination is still the requested final receiver.
        upper = goal.under_object if goal.relation == "under" else None
        if upper is not None:
            required.add(upper)
            pending.extend(incoming[upper] - required)
        while pending:
            blocker = pending.pop()
            if blocker not in required:
                required.add(blocker)
                pending.extend(incoming[blocker] - required)
        require("acyclic_supports", goal.target not in required,
                f"Blocking/support cycle reaches goal {goal.target!r}", index)
        for object_id in sorted(required):
            refusal = restriction(object_id, goal.target)
            if refusal is not None:
                return refusal

        # Prefer this goal's explicitly requested tray for staging blockers,
        # otherwise choose by ID. A goal with no named destination must still be
        # executed: the scored action is its pick. Only ask when a non-final goal
        # has no observed tray to rest on while a later goal needs the gripper.
        staging = goal.destination if goal.destination in trays else (trays[0] if trays else None)
        if required and not trays:
            return Proposal(
                kind="ask_user",
                question=(f"Where should I stage the objects obstructing {goal.target!r}? "
                          "No observed fixed tray is available."),
                forbidden_movement=intent.forbidden_movement,
            )
        if goal.destination is None and index != len(intent.goals) - 1 and staging is None:
            return Proposal(
                kind="ask_user",
                question=(f"Where should I place {goal.target!r} before proceeding to "
                          f"goal {intent.goals[index + 1].target!r}? Only one object can be held at a time."),
                forbidden_movement=intent.forbidden_movement,
            )

        indegree = {object_id: len(incoming[object_id] & required) for object_id in required}
        ready = [object_id for object_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: list[str] = []
        for _ in range(len(required)):
            if not ready:
                break
            object_id = heapq.heappop(ready)
            order.append(object_id)
            for destination in edges[object_id] & required:
                indegree[destination] -= 1
                if indegree[destination] == 0:
                    heapq.heappush(ready, destination)
        require("acyclic_supports", len(order) == len(required),
                f"Cannot topologically clear blockers for goal {goal.target!r}", index)
        for object_id in order:
            assert staging is not None  # Required blockers already demand an observed tray.
            pick(object_id)
            place(object_id, staging)
        pick(goal.target)
        if goal.destination is not None:
            placement = goal.base_relation if upper is not None else (
                "inside" if goal.relation == "in" else "on")
            place(goal.target, goal.destination, placement)
        elif index != len(intent.goals) - 1 and staging is not None:
            # Null destination on a non-final goal: rest it on the staging tray so
            # the gripper is empty for the next pick. The pick is still scored.
            place(goal.target, staging)
        if upper is not None:
            pick(upper)
            place(upper, goal.target)

    return Proposal(kind="plan", actions=tuple(actions), goals=intent.goals,
                    goal_mode=intent.goal_mode, reason=intent.reason,
                    forbidden_movement=intent.forbidden_movement)
