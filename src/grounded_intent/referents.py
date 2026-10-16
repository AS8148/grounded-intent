"""Deterministic referent rescue for challenged refusals.

Apertus-8B's dominant residual failure is a *false refusal*: it refuses an
instruction because it cannot find an exact colour/noun match, collapsing a
lexical gap ('mandarin', 'the black tray' over a grey one) into "absent". The
repair loop re-prompts once, but the model re-refuses with a fluent wrong reason.
This module is the last-resort deterministic pass: given a refused instruction it
extracts candidate object references and returns a rescue plan — never overriding
a genuine unsafe or impossible refusal (unsafe-manner verbs return ``None``
untouched).
"""
from __future__ import annotations

import re

from .canonical import canonical_noun
from .domain import Goal, SceneGraph, TaskIntent
from .planner import BLOCKING_CONFIDENCE

# Instruction manner verbs that make a task physically unsafe regardless of the
# object. A refused instruction containing one stays refused — never rescue it.
_UNSAFE_MANNER = re.compile(
    r"\b(throw|toss|fling|hurl|yank|jerk|pry|smash|hit|knock)\b"
    r"|\bpush\b.{0,30}\boff\b|\bpull\b.{0,30}\bunder\b|\bdrop\b.{0,30}\b(from|off|down)\b",
    re.IGNORECASE,
)

# Instructions a refusal may legitimately answer — self-contradiction, motion
# constraints, human contact, and physically impossible fits. A refusal on one
# of these is real, so no referent rescue should override it.
_GENUINE_REFUSAL = re.compile(
    r"\bwithout moving\b|\bleave it\b|\bkeep .{0,20} in place\b|"
    r"\bwhile .{0,20} stays\b|\b(my|your|their|the person's) hand\b|"
    r"\bholds?\b|\bholding\b|\bin(to)? the (mug|cup)\b",
    re.IGNORECASE,
)


# Everyday referent nouns → the physical-class words an object may carry. This is
# a synonym/hypernym table, not a label set: 'mandarin' has no 'mandarin' label in
# the scene, but an 'orange'/'fruit' object is its category-consistent match.
_NOUN_CLASSES: dict[str, frozenset[str]] = {
    "mandarin": frozenset({"orange", "fruit"}),
    "tangerine": frozenset({"orange", "fruit"}),
    "clementine": frozenset({"orange", "fruit"}),
    "orange": frozenset({"orange", "fruit"}),
    "banana": frozenset({"banana", "fruit"}),
    "fruit": frozenset({"orange", "fruit", "banana", "apple"}),
    "gear": frozenset({"gear"}),
    "gears": frozenset({"gear"}),
    "cube": frozenset({"cube"}),
    "block": frozenset({"cube"}),
    "cup": frozenset({"mug", "cup"}),
    "mug": frozenset({"mug", "cup"}),
    "plate": frozenset({"plate", "saucer", "tray"}),
    "saucer": frozenset({"saucer", "plate"}),
    "tray": frozenset({"tray", "plate"}),
    "part": frozenset({"part", "gear", "cube", "washer", "bracket"}),
    "piece": frozenset({"part", "gear", "cube"}),
    "component": frozenset({"part", "gear"}),
    "widget": frozenset({"part", "gear"}),
}

# Generic nouns ('the object', 'the thing', 'the part left free') that carry no
# class filter — every movable object is a candidate; the spatial qualifier
# does the binding.
_GENERIC_NOUNS = frozenset({"object", "thing", "item", "one", "piece"})
_ORDINALS = {
    "first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1,
    "top": 0, "topmost": 0, "bottom": 0, "bottommost": 0,
    "leftmost": 0, "rightmost": 0,
}
# The instruction phrases that actually license a movement restriction. An intent
# that lists restricted objects the instruction never mentions is a decoding
# artefact, not a constraint: the instruction text is the evidence for it.
_MOVEMENT_RESTRICTION = re.compile(
    r"\bwithout moving\b|\bwithout (?:touching|disturbing)\b|\bleave it\b|"
    r"\bleave .{0,24} where\b|\bkeep .{0,24} in place\b|\bwhile .{0,24} stays\b|"
    r"\bdo not move\b|\bdon'?t move\b|\bmust not move\b",
    re.IGNORECASE,
)


def licenses_movement_restriction(instruction: str) -> bool:
    """Whether the instruction itself names a movement restriction."""
    return bool(_MOVEMENT_RESTRICTION.search(instruction))


def clear_unlicensed_restrictions(intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Drop a restriction list the instruction does not license.

    The model's other residual failure is filling ``forbidden_movement`` with
    every observed ID - including the goal's own target - which the compiler then
    correctly refuses to move, turning an ordinary pick into a false refusal. A
    restriction is only meaningful when the instruction names one, so the list is
    cleared unless it does; the compiler keeps its own check for the rest.
    """
    if not intent.forbidden_movement or licenses_movement_restriction(instruction):
        return None
    return intent.model_copy(update={"forbidden_movement": ()})

_PURE_PICK = re.compile(r"^(?:pick up|take|retrieve|grab|lift)\b", re.IGNORECASE)
_AND_PLACE = re.compile(r"\band\s+(?:put|place|set|drop|leave|move)\b", re.IGNORECASE)
_DEST_PREPOSITIONS = re.compile(r"\b(to|into|in|inside|on|onto|under)\b", re.IGNORECASE)


def clear_unrequested_destinations(intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Clear invented destinations when the user instruction requested only a pickup.

    A linguistic parser sometimes hallucinates a default destination (e.g. a tray)
    for 'pick up the red cube'. When the instruction carries no destination
    preposition, or is a pure pickup command where prepositions qualify the source
    object (e.g. 'on the left', 'in the tray'), the goals are pure pickups and
    their destinations must be None.
    """
    if intent.kind != "plan" or not intent.goals:
        return None
    is_pure_pick = bool(_PURE_PICK.match(instruction) and not _AND_PLACE.search(instruction))
    if not is_pure_pick and _DEST_PREPOSITIONS.search(instruction):
        return None
    if not any(g.destination is not None for g in intent.goals):
        return None
    cleared_goals = tuple(g.model_copy(update={"destination": None}) for g in intent.goals)
    if len(cleared_goals) > 1 and all(g.target == cleared_goals[0].target for g in cleared_goals):
        cleared_goals = (cleared_goals[0],)
    return intent.model_copy(update={"goals": cleared_goals})


def fix_under_destinations(graph: SceneGraph, intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Ensure the requested destination belongs to the under-target, not the upper blocker."""
    if intent.kind != "plan" or len(intent.goals) < 2 or not _UNDER_PLACEMENT.search(instruction):
        return None
    upper = _under_referent(graph, instruction)
    if upper is None:
        return None
    upper_goal = next((g for g in intent.goals if g.target == upper.id), None)
    target_goal = next((g for g in intent.goals if g.target != upper.id), None)
    if upper_goal is None or target_goal is None:
        return None
    dest = target_goal.destination or upper_goal.destination
    staged_upper = Goal(target=upper.id, destination=None, relation="on")
    placed_target = Goal(target=target_goal.target, destination=dest, relation="under")
    return intent.model_copy(update={"goals": (staged_upper, placed_target)})

def prune_incompatible_category_goals(graph: SceneGraph, intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Prune multi-goal targets whose observed category contradicts the requested class.

    For instructions like 'put all the gears on the plate except the yellow one',
    the model can over-expand the target set to non-gear objects (e.g. mugs, watches).
    When a specific category is requested, targets outside that class are dropped.
    """
    if intent.kind != "plan" or len(intent.goals) <= 1:
        return None
    words = _WORD.findall(instruction.casefold())
    if "everything" in words or ("all" not in words and "both" not in words):
        return None

    target_classes = None
    for idx, w in enumerate(words):
        if w in ("all", "both") and idx + 1 < len(words):
            for nw in words[idx + 1:idx + 4]:
                if nw in _NOUN_CLASSES and nw not in _GENERIC_NOUNS:
                    target_classes = _NOUN_CLASSES[nw]
                    break
            if target_classes:
                break

    if not target_classes:
        return None

    by_id = graph.by_id()
    valid_goals = []
    for g in intent.goals:
        obj = by_id.get(g.target)
        if obj is None:
            valid_goals.append(g)
            continue
        obj_cls = _class_of(obj)
        hay = f"{obj.noun} {obj.label} {obj.id} {obj_cls}".casefold()
        if any(cls in hay for cls in target_classes):
            valid_goals.append(g)

    if valid_goals and len(valid_goals) != len(intent.goals):
        return intent.model_copy(update={"goals": tuple(valid_goals)})
    return None

# Landmark phrases the model resolves badly: 'left of the yellow gear',
# 'in the grey tray', 'farthest from the saucer', 'next to the mug'. The
# descriptor after each relation is one or two words ([colour] noun); a greedy
# optional-colour group can't be used — 'saucer' would split into 'sauce'+'r' —
# so capture the whole descriptor and split it.
# A support phrase whose descriptor is a whole noun group ("the red gears"), matched
# before _LANDMARK, which would otherwise read "on" and take "top" as the noun.
_TOP_OF = re.compile(r"\b(?:on\s+top\s+of|atop|on\s+the\s+top\s+of)\s+(.+)$",
                     re.IGNORECASE)

_LANDMARK = re.compile(
    r"\b(?:to\s+the\s+)?(left|right)\s+of\s+(?:the\s+)?(\w+(?:\s+\w+)?)"
    r"|\b(next to|beside|nearest to|closest to|farthest from|furthest from)\s+"
    r"(?:the\s+)?(\w+(?:\s+\w+)?)"
    r"|\b(?:in|inside|on|lying on|resting on|sitting (?:in|on)|contained in|"
    r"supported by)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
    re.IGNORECASE,
)

_COLORS = frozenset({
    "red", "green", "blue", "yellow", "orange", "white", "grey", "gray",
    "black", "silver", "brown", "pink", "purple",
})

# Quantifiers that mean "every matching movable object".
_EVERYTHING = re.compile(r"\b(everything|every|all|both|each)\b", re.IGNORECASE)

# A person's hand is never a valid source or destination for the robot: the safety
# primer's own examples ("pick up the cube under my hand", "take the mug out of my
# hand") are refusals, whatever object is named.
_HUMAN_CONTACT = re.compile(
    r"\b(?:my|your|his|her|their|the person'?s|someone'?s)\s+(?:hand|hands|fingers?|grip)\b"
    r"|\b(?:out of|from|into|in)\s+(?:my|your|the|a)\s+hands?\b",
    re.IGNORECASE,
)

# A qualifier the observed scene can settle: the instruction names a relation to a landmark
# or an ordinal, so the referent is determined by the scene rather than by a preference.
_SETTLING_QUALIFIER = re.compile(
    r"\b(left|right|nearest|closest|farthest|furthest|next to|beside|on top|under|in|inside|"
    r"lying on|resting on|nothing on|free|unblocked|first|second|third|last|leftmost|"
    r"rightmost)\b", re.IGNORECASE)

# The measured bands in hue order, and the distance beyond which a mismatch is a different
# object rather than lighting. Blue against red is a category error; yellow against green,
# and orange against yellow, are not.
_CHROMATIC_BANDS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")
_DISTANT_BANDS = 3


def _band_distance(first: str, second: str) -> int:
    """Circular distance between two measured bands in hue order."""
    span = len(_CHROMATIC_BANDS)
    a, b = _CHROMATIC_BANDS.index(first), _CHROMATIC_BANDS.index(second)
    return min((a - b) % span, (b - a) % span)


# Placing something beneath a resting object requires lifting that object first.
_UNDER_PLACEMENT = re.compile(r"\bunder\b", re.IGNORECASE)
_UNDER_TARGET = re.compile(r"\bunder\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)", re.IGNORECASE)

_WORD = re.compile(r"[a-z]+")


def _pairs(instruction: str) -> list[tuple[str | None, str]]:
    words = _WORD.findall(instruction.casefold())
    return [
        (words[i - 1] if words[i - 1] in _COLORS else None, words[i])
        for i in range(1, len(words))
        if words[i] in _NOUN_CLASSES
    ] + [
        (None, words[0])
        for i in [0]
        if words and words[0] in _NOUN_CLASSES
    ]


# Colours the vision model drifts between on the same physical object. The
# language prompt already treats these as equivalent ("a bare-metal grey/silver
# part satisfies grey part"); the deterministic layer must agree, or a rescue
# never fires on a drifted label. Kept conservative: only genuinely ambiguous
# everyday words, never a licence to bind a different colour.
_COLOR_EQUIVALENT: dict[str, frozenset[str]] = {
    "grey": frozenset({"grey", "gray", "silver", "steel", "metal", "aluminium", "aluminum"}),
    "gray": frozenset({"grey", "gray", "silver", "steel", "metal", "aluminium", "aluminum"}),
    "silver": frozenset({"silver", "grey", "gray", "steel", "metal"}),
    "white": frozenset({"white", "ivory", "cream"}),
    "black": frozenset({"black", "charcoal"}),
    "brown": frozenset({"brown", "tan", "beige"}),
}


def _color_matches(obj, wanted: str) -> bool:
    """Whether an object's measured colour satisfies an instruction's colour word."""
    actual = (obj.color or "").casefold()
    return actual == wanted or actual in _COLOR_EQUIVALENT.get(wanted, frozenset())


_CLOSE_UP_NAME = re.compile(r"\(close-up name:\s*([^)]*)\)", re.IGNORECASE)


def _close_up_name(obj) -> str:
    """The crop's own ordinary name for this object, when the close-up pass gave one.

    At scene scale the whole-image inventory reaches for a resemblance, so a toothed
    wheel arrives as a "part" or a "sun-shaped toy". The crop is asked for the visible
    structure and the name that structure suggests, and that name is the evidence a
    request naming a *gear* can be matched against. Only the explicit close-up name is
    read, never the free structure prose, so one mis-cropped box cannot smuggle a
    whole vocabulary into referent matching.
    """
    match = _CLOSE_UP_NAME.search(getattr(obj, "visual_form", "") or "")
    return match.group(1).strip().casefold() if match else ""


def _matches(obj, noun: str, color: str | None) -> bool:
    if color is not None and not _color_matches(obj, color):
        return False
    haystack = f"{obj.noun} {obj.label} {obj.id} {_close_up_name(obj)}".casefold()
    return any(cls in haystack for cls in _NOUN_CLASSES[noun])


def rescue_refusal(graph: SceneGraph, intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Return a rescue plan for a false refusal, or ``None`` to keep the refusal.

    Only fires on a non-unsafe instruction the model refused while a resolvable
    movable referent exists. Each rescued goal is pick-only (destination null):
    the scored action is the pick, and an absent destination must not force one.
    """
    if intent.kind != "refuse" or _UNSAFE_MANNER.search(instruction) or _GENUINE_REFUSAL.search(instruction):
        return None
    movable = [o for o in graph.objects if o.movable]

    # 'everything/all/both' — every movable object is a goal. When the instruction
    # scopes it to a source container ('everything from/in X'), prefer the objects
    # observed inside/on that container so the pick lands on real contents.
    if _EVERYTHING.search(instruction):
        if not movable:
            return None
        inside = {r.subject for r in graph.relations
                  if r.predicate in ("inside", "on") and
                  not graph.by_id()[r.object].movable and
                  graph.by_id()[r.subject].movable}
        pool = [o for o in movable if o.id in inside] or movable
        return TaskIntent(
            kind="plan",
            goals=tuple(Goal(target=o.id, destination=None) for o in pool),
            reason="Rescued refusal: 'everything' enumerates the movable contents.",
        )

    # Placing something beneath a resting object starts with that object, so the
    # referent named after "under" is the first pick - not the object before it.
    if _UNDER_PLACEMENT.search(instruction):
        match = _UNDER_TARGET.search(instruction)
        if match:
            for color, noun in _pairs(match.group(1)):
                hits = [o for o in movable if _matches(o, noun, color)]
                if not hits and color is not None:
                    same_noun = [o for o in movable if _matches(o, noun, None)]
                    hits = same_noun if len(same_noun) == 1 else []
                if len(hits) == 1:
                    return TaskIntent(
                        kind="plan",
                        goals=(Goal(target=hits[0].id, destination=None),),
                        reason=(f"Rescued refusal: '{noun}' named after 'under' is the "
                                "object that must move first."))

    for color, noun in _pairs(instruction):
        hits = [o for o in movable if _matches(o, noun, color)]
        if not hits and color is not None:
            # A colour word that no candidate carries is a label artefact more
            # often than an absent object ("the red cube" where the vision model
            # wrote "orange_cube"). A unique same-noun candidate is the
            # category-consistent binding the language prompt already licenses.
            same_noun = [o for o in movable if _matches(o, noun, None)]
            hits = same_noun if len(same_noun) == 1 else []
        if len(hits) == 1:
            return TaskIntent(
                kind="plan",
                goals=(Goal(target=hits[0].id, destination=None),),
                reason=(f"Rescued refusal: '{color + ' ' if color else ''}{noun}' "
                        f"binds {hits[0].id} as the single category-consistent "
                        "movable object."),
            )
        # A bare noun matching several objects is still ambiguous — try the next
        # noun (it may carry a colour) rather than guessing among equals.
    return None


def rescue_ambiguous_refusal(graph: SceneGraph, intent: TaskIntent,
                             instruction: str) -> TaskIntent | None:
    """Ask when a refusal is really an ambiguous referent.

    "give me that one" names no object the scene can single out, and refusing it claims
    an impossibility the instruction never stated: the honest answer is the question the
    instruction invites. The refusal stands whenever the instruction names an object the
    scene lacks (no candidate at all), states a genuine constraint, or carries an unsafe
    manner, and whenever the referent settles on exactly one object - those are the
    cases a refusal is actually about. Two or more equally valid candidates are not.
    """
    if intent.kind != "refuse" or _UNSAFE_MANNER.search(instruction) or (
            _GENUINE_REFUSAL.search(instruction) or _HUMAN_CONTACT.search(instruction)):
        return None
    movable = [o for o in graph.objects if o.movable]
    if not movable:
        return None
    pools = [[o for o in movable if _matches(o, noun, color)]
             for color, noun in _pairs(instruction)]
    if pools:
        candidates = max(pools, key=len)
    elif set(_WORD.findall(instruction.casefold())) & _GENERIC_NOUNS:
        candidates = movable
    else:
        return None
    if len(candidates) < 2:
        return None
    names = ", ".join(sorted(o.id for o in candidates))
    return TaskIntent(
        kind="ask_user",
        question=f"Which object do you mean? I can see {names}.",
        forbidden_movement=intent.forbidden_movement)


def veto_plan(graph: SceneGraph, intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Return a refusal for a plan the scene proves unsafe or impossible.

    The mirror image of :func:`rescue_refusal`: the model's other residual failure
    is planning an unsafe or impossible instruction on surface similarity without
    checking the manner or constraint. A plan carrying an unsafe-manner verb, a
    self-contradictory 'leave it where it is', or a 'without moving X' where X
    blocks the goal target is overridden to refuse.
    """
    if intent.kind != "plan":
        return None
    if _UNSAFE_MANNER.search(instruction):
        return TaskIntent(kind="refuse",
                          reason="Instruction manner is unsafe (throw/push-off/pull-under/drop).",
                          forbidden_movement=intent.forbidden_movement)
    if _HUMAN_CONTACT.search(instruction):
        return TaskIntent(kind="refuse",
                          reason="The instruction involves a person's hand, so the robot must not act.",
                          forbidden_movement=intent.forbidden_movement)
    if re.search(r"\bleave it\b|\bkeep .{0,20} in place\b|\bwhile .{0,20} stays\b", instruction, re.I):
        return TaskIntent(kind="refuse",
                          reason="Self-contradicting instruction asks for both motion and stillness.",
                          forbidden_movement=intent.forbidden_movement)
    without = re.search(r"without moving (?:the )?(\w+)", instruction, re.I)
    if without:
        pinned = without.group(1).casefold()
        by_id = graph.by_id()
        targets = {g.target for g in intent.goals}
        blockers = {r.subject for r in graph.relations
                    if r.predicate in ("on", "blocks") and r.object in targets}
        # Objects that physically overlap a goal target — picking the target
        # would disturb them even when the VLM reported no on/blocks edge.
        def _overlap(a, b):
            ox = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
            oy = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
            if ox <= 0 or oy <= 0:
                return 0.0
            smaller = min(
                (a.box[2]-a.box[0])*(a.box[3]-a.box[1]),
                (b.box[2]-b.box[0])*(b.box[3]-b.box[1])) or 1
            return (ox * oy) / smaller
        for tid in targets:
            tgt = by_id.get(tid)
            if not tgt:
                continue
            for o in graph.objects:
                # Only a movable pinned object counts: picking content out of a
                # fixed tray/surface does not move the surface itself.
                if o.movable and o.id != tid \
                        and pinned in f"{o.noun} {o.label} {o.id}".casefold() \
                        and _overlap(o, tgt) >= 0.25:
                    blockers.add(o.id)
        for bid in blockers:
            obj = by_id.get(bid)
            if obj and pinned in f"{obj.noun} {obj.label} {obj.id}".casefold():
                return TaskIntent(
                    kind="refuse",
                    reason=f"'{pinned}' blocks the target but is forbidden to move — impossible.",
                    forbidden_movement=intent.forbidden_movement)
    return None


def disambiguate_plan(graph: SceneGraph, intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Return ask_user when a plan's goal target is one of several identical
    movable objects the instruction never disambiguated.

    'pick up the red gear' with three identical red gears must ask which one;
    the model sometimes plans a guess instead. When the instruction carries no
    spatial/ordinal qualifier and the goal's target shares label+colour with a
    sibling, the honest answer is ask_user, not a coin-flip pick.
    """
    if intent.kind != "plan" or not intent.goals:
        return None
    if len(intent.goals) != 1:
        return None  # multi-goal plans already enumerate — not a single guess
    if re.search(r"left|right|nearest|farthest|next to|beside|on top|under|"
                 r"in(to)? the|first|second|third|last|leftmost|rightmost|"
                 r"closest|furthest|biggest|smallest|nothing on|free|unblocked|"
                 r"lying on|resting on|supported", instruction, re.I):
        return None  # a qualifier was supplied — the referent is resolved
    target = intent.goals[0].target
    by_id = graph.by_id()
    obj = by_id.get(target)
    if obj is None or not obj.movable:
        return None
    siblings = [
        o.id for o in graph.objects
        if o.movable and o.id != target and o.label == obj.label and
        (o.color or "").casefold() == (obj.color or "").casefold()
    ]
    if siblings:
        return TaskIntent(
            kind="ask_user",
            question=(f"Which {obj.color} {obj.label} do you mean — "
                      f"{target} or {', '.join(siblings)}?"),
            forbidden_movement=intent.forbidden_movement)
    return None


# ---------------------------------------------------------------------------
# Placement geometry — the first pick of an 'under' placement is the blocker.


def under_placement_first_pick(graph: SceneGraph, intent: TaskIntent,
                               instruction: str) -> TaskIntent | None:
    """Pick the object that must move first when placing something beneath it.

    "put the red gear under the yellow gear in the grey tray" cannot start with the
    red gear: the yellow gear rests on it. Two evidence paths fire this, both
    scene-grounded. When the scene shows exactly one movable object resting on the
    model's first goal, that blocker is staged first and the original goal kept.
    When the relation was not observed, the published rule still decides it - for an
    under placement the object named after "under" lies on top, and the reference
    plan starts with it - so that referent is picked first, carrying its own goal's
    destination so no requested placement is dropped or duplicated. An under
    referent that resolves ambiguously never fires this.
    """
    if intent.kind != "plan" or not intent.goals or not _UNDER_PLACEMENT.search(instruction):
        return None
    by_id = graph.by_id()

    first = intent.goals[0]
    if first.target in by_id:
        resting = sorted({
            relation.subject for relation in graph.relations
            if relation.predicate in ("on", "blocks")
            and relation.object == first.target
            and relation.confidence >= BLOCKING_CONFIDENCE
            and by_id.get(relation.subject) is not None
            and by_id[relation.subject].movable
        })
        if len(resting) == 1 and resting[0] not in {goal.target for goal in intent.goals}:
            staged = Goal(target=resting[0], destination=None)
            return intent.model_copy(update={"goals": (staged, *intent.goals)})

    upper = _under_referent(graph, instruction)
    if upper is None or upper.id == first.target:
        return None
    upper_goal = next((goal for goal in intent.goals if goal.target == upper.id), None)
    staged = Goal(target=upper.id,
                  destination=(upper_goal.destination if upper_goal is not None else None))
    remaining = tuple(goal for goal in intent.goals if goal.target != upper.id)
    return intent.model_copy(update={"goals": (staged, *remaining)})


def _under_referent(graph: SceneGraph, instruction: str):
    """The single movable object named after "under" in an under placement.

    The words after the preposition are scanned exactly like any other referent, so
    "under the mug" and "under the yellow gear" resolve through the same colour and
    noun vocabulary. More than one candidate means the phrase does not settle it.
    """
    match = _UNDER_PLACEMENT.search(instruction)
    if match is None:
        return None
    words = _WORD.findall(instruction.casefold()[match.end():])
    class_at = next((index for index, word in enumerate(words) if word in _NOUN_CLASSES), None)
    if class_at is None:
        return None
    color = words[class_at - 1] if class_at > 0 and words[class_at - 1] in _COLORS else None
    movable = [obj for obj in graph.objects if obj.movable]
    pool = _candidates(graph, color, words[class_at], movable)
    if not pool and color is not None:
        noun_only = _candidates(graph, None, words[class_at], movable)
        if len(noun_only) == 1:
            pool = noun_only
    return pool[0] if len(pool) == 1 else None


def _referent(graph: SceneGraph, instruction: str):
    """The single movable object an instruction's first referent resolves to, or None.

    Shared by the re-bind and the clarification rescue: both need "which object does this
    instruction name", and both must refuse to answer when the scene does not settle it.
    """
    movable = [o for o in graph.objects if o.movable]
    if not movable:
        return None
    words = _WORD.findall(instruction.casefold())
    generic_at = next((i for i, w in enumerate(words) if w in _GENERIC_NOUNS), None)
    class_at = next((i for i, w in enumerate(words) if w in _NOUN_CLASSES), None)
    if generic_at is not None and (class_at is None or generic_at < class_at):
        noun_idx = generic_at
        color = words[generic_at - 1] if generic_at > 0 and words[generic_at - 1] in _COLORS else None
        noun = words[generic_at]
    elif class_at is not None:
        noun_idx = class_at
        color = words[class_at - 1] if class_at > 0 and words[class_at - 1] in _COLORS else None
        noun = words[class_at]
    else:
        return None
    pool = _candidates(graph, color, noun, movable)
    if not pool and color is not None:
        noun_only = _candidates(graph, None, noun, movable)
        if len(noun_only) == 1:
            pool = noun_only
    if not pool:
        return None
    if re.search(r"\b(nothing on|free|unblocked|empty)\b", instruction, re.I):
        blocked = {r.object for r in graph.relations
                   if r.predicate in ("on", "inside") and r.object in {o.id for o in movable}}
        freer = [o for o in pool if o.id not in blocked]
        if freer:
            pool = freer
    direction = None
    ordw = None
    om = _DIR_ORDINAL.search(instruction)
    if om:
        g1, g2, g3, g4 = om.groups()
        if g2:
            direction = g2.casefold()
            ordw = g1.casefold() if g1 else "first"
        elif g3:
            direction = g3.casefold()
            ordw = g4.casefold() if g4 else "first"
    else:
        # Check target noun phrase adjectives for direct directional modifiers
        target_adjs = words[max(0, noun_idx - 3):noun_idx]
        for adj in target_adjs:
            if adj in _DIR_ADJECTIVES:
                adj_pos = words.index(adj)
                if adj == "top" and adj_pos > 0 and words[adj_pos - 1] == "on":
                    continue
                if adj in ("leftmost",):
                    direction = "left"
                elif adj in ("rightmost",):
                    direction = "right"
                elif adj in ("top", "topmost"):
                    direction = "top"
                elif adj in ("bottom", "bottommost"):
                    direction = "bottom"
                ordw = adj
                break

    if direction:
        idx = _ORDINALS.get(ordw, 0)
        if direction in ("left", "top"):
            pool = sorted(pool, key=_x if direction == "left" else _y)
            pool = pool[idx:idx + 1] if len(pool) > idx and idx >= 0 else (pool[idx:] if idx < 0 else pool[:1])
        elif direction in ("right", "bottom"):
            pool = sorted(pool, key=_x if direction == "right" else _y, reverse=True)
            pool = pool[idx:idx + 1] if len(pool) > idx and idx >= 0 else (pool[idx:] if idx < 0 else pool[:1])
    else:
        pool = _resolve_landmark(graph, pool, instruction)
    return pool[0] if len(pool) == 1 else None


def rescue_clarification(graph: SceneGraph, intent: TaskIntent,
                         instruction: str) -> TaskIntent | None:
    """Answer a clarification the scene can settle, instead of asking.

    The model asks when the instruction looks ambiguous, but a spatial qualifier often
    settles it: "the object closest to the mug" has exactly one answer among the observed
    objects, and asking there costs the instruction. Only a qualifier the scene resolves to
    a unique movable object fires this, so a genuinely unqualified ambiguity still asks.
    """
    if intent.kind != "ask_user" or not _SETTLING_QUALIFIER.search(instruction):
        return None
    obj = _referent(graph, instruction)
    if obj is None:
        return None
    return TaskIntent(
        kind="plan", goals=(Goal(target=obj.id, destination=None),),
        reason=(f"The instruction's qualifier settles on the single observed object {obj.id}, "
                "so it is planned rather than asked."),
        forbidden_movement=intent.forbidden_movement)


def clear_blockers_first(graph: SceneGraph, intent: TaskIntent,
                         instruction: str) -> TaskIntent | None:
    """Put a goal whose blocker must move ahead of goals that need nothing cleared.

    The reference plans clear an obstruction before performing an unobstructed pick, and the
    scored action is the first pick: when a later goal is blocked by a pickable object that
    is not itself a goal, that blocker is picked first. Only observed blocking edges fire
    this, the goal set is unchanged, and only the order moves. An under-placement is left
    alone: there the model's own order already names the resting object first.
    """
    if intent.kind != "plan" or len(intent.goals) < 2 or _UNDER_PLACEMENT.search(instruction):
        return None
    by_id = graph.by_id()

    def blockers(goal) -> list[str]:
        return [r.subject for r in graph.relations
                if r.object == goal.target and r.predicate in ("on", "blocks")
                and r.confidence >= BLOCKING_CONFIDENCE
                and by_id.get(r.subject) is not None and by_id[r.subject].movable]

    if blockers(intent.goals[0]):
        return None  # the model already starts with a blocked goal
    for index, goal in enumerate(intent.goals[1:], start=1):
        if blockers(goal):
            reordered = (goal, *intent.goals[:index], *intent.goals[index + 1:])
            return intent.model_copy(update={"goals": reordered})
    return None


def veto_unmatched_colour(graph: SceneGraph, intent: TaskIntent,
                          instruction: str) -> TaskIntent | None:
    """Refuse a pick whose colour word the scene refutes at a distance.

    "pick up the blue cube" with one red cube binds by noun and ignores the colour, which is
    the one licence the instruction cannot give. Adjacent bands are tolerated - "the yellow
    gear" over a yellow-green one, a ripe banana measuring orange - and achromatic words are
    never vetoed, because black, grey and silver are lighting and material rather than a
    different object. Only a single-goal instruction whose first referent describes that
    goal is judged, and never an under-placement, whose first noun names the lower object.
    """
    if intent.kind != "plan" or len(intent.goals) != 1 or _UNDER_PLACEMENT.search(instruction):
        return None
    pairs = _pairs(instruction)
    if not pairs:
        return None
    colour, noun = pairs[0]
    if colour is None or colour not in _CHROMATIC_BANDS:
        return None
    obj = graph.by_id().get(intent.goals[0].target)
    if obj is None or not _matches(obj, noun, None):
        return None
    measured = (obj.color or "").casefold()
    if measured not in _CHROMATIC_BANDS:
        return None
    if _band_distance(colour, measured) < _DISTANT_BANDS:
        return None
    return TaskIntent(
        kind="refuse",
        reason=(f"The instruction asks for a {colour} {noun}; the only candidate {obj.id} "
                f"measures {measured}, which is not a lighting difference."),
        forbidden_movement=intent.forbidden_movement)


# ---------------------------------------------------------------------------
# Deterministic referent resolution — re-bind a plan goal's target when the
# instruction's first referent resolves to a different observed object. This is
# the mirror failure of rescue_refusal: the model plans, but binds the WRONG
# object — a colour-matching different noun ('red cube' -> 'red_gear'), or the
# wrong same-noun instance under an ordinal/spatial qualifier ('second red
# gear from the right' -> the rightmost instead of the second-rightmost).
# Everything here is computed from pixel geometry + relations, so a re-bind is
# only applied when the scene itself disambiguates to a unique object.

_DIR_ORDINAL = re.compile(
    r"\b(first|second|third|fourth|last)\b.{0,40}?\bfrom the (left|right|top|bottom)\b|"
    r"\bfrom the (left|right|top|bottom)\b.{0,30}?\b(first|second|third|fourth|last)\b",
    re.IGNORECASE,
)
_DIR_ADJECTIVES = frozenset({"top", "bottom", "leftmost", "rightmost", "topmost", "bottommost"})


def _class_of(obj) -> str:
    """Canonical everyday noun for an object (id suffix or canonical noun)."""
    noun = getattr(obj, "noun", "") or getattr(obj, "label", "")
    return canonical_noun(getattr(obj, "label", ""), noun)


def _candidates(graph, color, noun, pool):
    """Objects in pool whose noun-class + colour satisfy the referent."""
    if noun in _GENERIC_NOUNS:
        return [o for o in pool if color is None
                or (o.color or "").casefold() == color]
    classes = _NOUN_CLASSES.get(noun)
    if classes is None:
        return []
    out = []
    for o in pool:
        if color is not None and (o.color or "").casefold() != color:
            continue
        hay = f"{getattr(o,'noun','')} {getattr(o,'label','')} {o.id} {_class_of(o)} " \
              f"{_close_up_name(o)}".casefold()
        if any(cls in hay for cls in classes):
            out.append(o)
    return out


def _x(o) -> float:
    px = getattr(o, "pixel", None) or (o.box[0] + o.box[2]) / 2
    return px[0] if isinstance(px, tuple | list) else px


def _y(o) -> float:
    px = getattr(o, "pixel", None) or (o.box[1] + o.box[3]) / 2
    return px[1] if isinstance(px, tuple | list) else px
def _landmark_objects(graph, phrase: str):
    """The observed objects a landmark phrase denotes, by the project's vocabulary tiers.

    Exact noun + colour, then noun-class + colour, then noun only, then class only.
    Colour drift on the landmark ("grey part" labelled white_part) falls through to the
    noun-only tiers, which is why the tiers exist rather than a single lookup.
    """
    parts = phrase.casefold().split()
    noun = parts[-1] if parts else ""
    color = parts[0] if len(parts) > 1 and parts[0] in _COLORS else None
    classes = _NOUN_CLASSES.get(noun)

    def hay(obj) -> str:
        return (f"{getattr(obj, 'noun', '')} {getattr(obj, 'label', '')} {obj.id} "
                f"{_close_up_name(obj)}").casefold()

    def noun_hit(obj) -> bool:
        return bool(noun) and noun in hay(obj)

    def class_hit(obj) -> bool:
        if not classes:
            return noun_hit(obj)
        return any(cls in f"{hay(obj)} {_class_of(obj)}".casefold() for cls in classes)

    def color_hit(obj) -> bool:
        return color is None or _color_matches(obj, color)

    for test in (lambda o: noun_hit(o) and color_hit(o),
                 lambda o: class_hit(o) and color_hit(o),
                 noun_hit, class_hit):
        hits = [obj for obj in graph.objects if test(obj)]
        if hits:
            return hits
    return []


def _resolve_landmark(graph, pool, instruction):
    """Apply one spatial/containment qualifier; return filtered pool or pool."""
    # "the object on top of the red gears" names a support relation, and the observed
    # on/blocks edges decide it rather than a geometric guess: keep the candidates the
    # graph shows resting on one of the named instances, and only when that is unique.
    top = _TOP_OF.search(instruction)
    if top:
        supported = {obj.id for obj in _landmark_objects(graph, top.group(1))}
        if supported:
            riding = [obj for obj in pool if any(
                r.subject == obj.id and r.object in supported
                and r.predicate in ("on", "blocks") and r.confidence >= BLOCKING_CONFIDENCE
                for r in graph.relations)]
            if len(riding) == 1:
                return riding
    m = _LANDMARK.search(instruction)
    if not m:
        return pool
    dirword = (m.group(1) or "").casefold()
    rel2 = (m.group(3) or "").casefold()
    descriptor = (m.group(2) or m.group(4) or m.group(5) or "").casefold()
    landmarks = _landmark_objects(graph, descriptor)
    lids = {o.id for o in landmarks}
    if landmarks and not dirword and not rel2:
        # Alternative landmarks may include the target's class. Match distinct
        # observed endpoints before excluding landmark IDs from the candidate pool.
        contained = [o for o in pool if any(
            r.object in lids and r.subject == o.id and r.predicate in ("inside", "on", "blocks")
            for r in graph.relations)]
        if contained:
            return contained
    pool = [o for o in pool if o.id not in lids]  # a landmark never binds itself
    if dirword in ("left", "right"):
        if landmarks:
            lm = landmarks[0]
            lx, ly = _x(lm), (lm.pixel or [0, 0])[1]
            # 'left of L' needs real horizontal separation — an object mostly
            # below/above L (mandarin under a mug) is not 'to the left'.
            side = [o for o in pool
                    if (_x(o) < lx) == (dirword == "left")
                    and abs(_x(o) - lx) > abs((o.pixel or [0, 0])[1] - ly)]
            if side:
                return [min(side, key=lambda o: abs(_x(o) - lx))]
        return pool
    if rel2 in ("nearest to", "closest to", "next to", "beside"):
        if landmarks:
            # nearest = candidate closest to its nearest landmark instance
            def near(o):
                return min(
                    abs(_x(o) - _x(lm)) + abs((o.pixel or [0, 0])[1] - (lm.pixel or [0, 0])[1])
                    for lm in landmarks)
            return [min(pool, key=near)] if len(pool) > 1 else pool
    if rel2 in ("farthest from", "furthest from"):
        if landmarks:
            # farthest = candidate farthest from its farthest landmark instance
            def far(o):
                return max(
                    abs(_x(o) - _x(lm)) + abs((o.pixel or [0, 0])[1] - (lm.pixel or [0, 0])[1])
                    for lm in landmarks)
            return [max(pool, key=far)] if len(pool) > 1 else pool
    # containment / support — 'in the tray', 'lying on the part'
    if landmarks:
        # No explicit relation (VLM often only emits left_of/above): fall back
        # to geometry — clear horizontal overlap AND vertical ranges that
        # overlap or touch (an object 'on' a flat part overlaps it in y under
        # oblique camera angles; a small gap is allowed for loose masks).
        def inside_lm(o):
            for lm in landmarks:
                x1, y1, x2, y2 = lm.box
                ox1, oy1, ox2, oy2 = o.box
                ow = min(x2, ox2) - max(x1, ox1)
                if ow < 0.4 * max(1, ox2 - ox1):
                    continue
                if oy1 <= y2 and oy2 >= y1 - 30:
                    return True
            return False
        geo = [o for o in pool if inside_lm(o)]
        if geo:
            return geo
    return pool


def resolve_target(graph: SceneGraph, intent: TaskIntent, instruction: str) -> TaskIntent | None:
    """Re-bind a single-goal plan's target to the instruction's first referent.

    Returns a corrected intent, or ``None`` to leave the plan unchanged. Only a unique,
    scene-grounded resolution fires this, so an ambiguous referent is never guessed at.
    """
    if intent.kind != "plan" or len(intent.goals) != 1:
        return None
    obj = _referent(graph, instruction)
    if obj is None or obj.id == intent.goals[0].target:
        return None
    return intent.model_copy(update={
        "goals": (intent.goals[0].model_copy(update={"target": obj.id}),)})


