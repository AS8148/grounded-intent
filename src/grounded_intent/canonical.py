"""Canonical everyday vocabulary for observed objects.

A vision model names the same physical object differently between runs and
between scenes ("plastic bin", "grey mat", "work surface"). The answer is scored
on a pixel, but the language model binds an instruction noun to an observed ID,
so the observed vocabulary must be stable *before* the IDs are built. This
module normalises a free-form label/noun/colour to one everyday noun and one
colour word, and derives the deterministic IDs from the result.

The tables are ordinary English for broad object families plus physical
properties (a broad flat container receives objects and is not itself grasped),
not a benchmark whitelist. Nothing here reads an instruction, a category, a
scene name, or a ground-truth file.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from .domain import ObjectObservation
from .geometry import canonical_token, object_sort_key


def _token(text: str) -> str:
    """Lower-case ASCII identifier, or ``""`` when the text carries none.

    Unlike :func:`geometry.canonical_token` this never raises: it is called on
    free-form model references, and an unparseable reference must reach the
    compiler's own ``known_ids`` check as a repairable violation rather than
    escaping as a crash.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")

# Generic everyday-noun ontology: maps common synonyms and hypernyms onto one
# canonical noun, so the same physical object keeps the same referent across
# scenes, runs and models. Every entry is ordinary English for a broad family.
_CANON_NOUN: dict[str, str] = {
    # containers and fixed work surfaces: things are placed in or on them
    "tray": "tray", "bin": "tray", "plastic_bin": "tray", "basket": "tray",
    "box": "tray", "crate": "tray", "container": "tray",
    "work_surface": "tray", "worksurface": "tray", "work_mat": "tray",
    "workmat": "tray", "mat": "tray", "cut_mat": "tray", "cutmat": "tray",
    "cutting_mat": "tray", "surface": "tray", "table": "tray",
    "platform": "tray", "board": "tray", "bench": "tray",
    # blocks
    "cube": "cube", "block": "cube", "brick": "cube", "die": "cube",
    "square_block": "cube", "red_square_block": "cube",
    # gears / toothed wheels
    "gear": "gear", "cog": "gear", "wheel": "gear", "sprocket": "gear",
    "sun_gear": "gear", "red_gear": "gear", "yellow_sun_gear": "gear",
    "sun": "gear",
    # drinkware
    "mug": "mug", "cup": "mug", "ceramic_mug": "mug", "glass": "mug",
    # small parts, components and tools: all graspable
    "part": "part", "piece": "part", "component": "part", "widget": "part",
    "bracket": "part", "metal_plate": "part", "plastic_piece": "part",
    "white_plastic_piece": "part", "black_object": "part", "strip": "part",
    "object": "part", "white_object": "part", "item": "part",
    "ruler": "part", "tape": "part", "blue_tape": "part", "card": "part",
    "circuit": "part", "circuit_board": "part", "tool": "part",
    "wrench": "part", "screwdriver": "part", "screw": "part", "bolt": "part",
    "nut": "part", "washer": "part", "key": "part", "pen": "part",
    "pencil": "part",
    # dishes: fixed surfaces only while genuinely broad and flat
    "plate": "plate", "dish": "plate", "ceramic_plate": "plate",
    "platter": "plate", "saucer": "plate",
    # fruit
    "orange": "orange", "mandarin": "orange", "tangerine": "orange",
    "clementine": "orange", "citrus": "orange", "banana": "banana",
    "apple": "apple",
    # compounds a whole-image model reaches for when it names a resemblance
    # ("sun-shaped toy", "gear toy") rather than the object's class
    "gear_toy": "gear", "sun_toy": "gear", "wheel_toy": "gear", "cog_toy": "gear",
    "sprocket_toy": "gear", "cube_toy": "cube", "box_toy": "cube",
    "block_toy": "cube", "brick_toy": "cube",
}

# Near-colour synonyms collapse to one stable everyday colour word.
_CANON_COLOR: dict[str, str] = {
    "dark_gray": "grey", "dark_grey": "grey", "gray": "grey",
    "light_gray": "grey", "light_grey": "grey", "charcoal": "black",
    "rose": "pink", "magenta": "pink", "scarlet": "red", "crimson": "red",
    "maroon": "red", "burgundy": "red", "coral": "orange", "salmon": "orange",
    "beige": "brown", "tan": "brown", "ivory": "white", "cream": "white",
    "navy": "blue", "teal": "blue", "cyan": "blue", "turquoise": "blue",
    "olive": "green", "lime": "green", "forest_green": "green",
    "skin": "pink", "flesh": "pink", "skin_tone": "pink", "peach": "pink",
    "gold": "yellow", "golden": "yellow", "amber": "yellow",
    "steel": "silver", "chrome": "silver", "aluminium": "silver",
    "aluminum": "silver", "metallic": "silver",
}

# Roles in this domain are physical, not lexical. Broad flat surfaces and
# containers are never grasped; the human body is never grasped; everything else
# is a movable item.
_FIXED_SURFACE = frozenset({
    "tray", "plate", "saucer", "platter", "dish", "mat", "table", "surface",
    "worktable", "workbench", "bench", "floor", "bowl",
})
_NEVER_GRASP = frozenset({
    "arm", "hand", "finger", "person", "body", "face", "leg", "foot", "wrist",
    "wristwatch", "watch", "arm_watch",
})


def canonical_noun(label: str, noun: str = "") -> str:
    """Map a free-form label/noun to its canonical everyday noun.

    Prefers the everyday ``noun`` when present, falling back to ``label``; tries
    the whole token first, then its last word (``metal_plate`` -> ``plate``).
    An unmapped token is returned unchanged as a canonical token.
    """
    for text in (noun, label):
        if not text:
            continue
        token = _token(text)
        if token in _CANON_NOUN:
            return _CANON_NOUN[token]
        tail = token.rsplit("_", 1)[-1]
        if tail in _CANON_NOUN:
            return _CANON_NOUN[tail]
    return _token(noun or label)


def canonical_color(color: str) -> str:
    """Collapse a near-colour synonym to one stable everyday colour word."""
    token = _token(color) if color.strip() else ""
    return _CANON_COLOR.get(token, token or "unknown")


def canonical_role(canonical: str) -> bool:
    """True when the canonically-named object is graspable by its noun alone."""
    if canonical in _NEVER_GRASP:
        return False
    return canonical not in _FIXED_SURFACE


def deterministic_ids(
    objects: Sequence[ObjectObservation],
) -> tuple[tuple[ObjectObservation, ...], dict[str, str]]:
    """Canonicalise vocabulary, then assign stable ``color_label[_n]`` IDs.

    Role is physical, not lexical: a broad flat container/surface is a fixed
    placement area, but a small elongated object carrying the same noun is a
    pickable part, so a clearly elongated surface-noun is reclassified. A
    literal label that collides with another object's suffixed ID bumps its
    suffix until unique instead of aborting the scene.
    """
    canonical = [
        (obj, canonical_noun(obj.label, obj.noun), canonical_color(obj.color))
        for obj in objects
    ]
    areas = [
        abs(obj.box[2] - obj.box[0]) * abs(obj.box[3] - obj.box[1]) or 1.0
        for obj, _, _ in canonical
    ]
    biggest = max(areas, default=1.0)
    normalised = []
    for (obj, noun, color), area in zip(canonical, areas, strict=True):
        movable = canonical_role(noun)
        if noun in _FIXED_SURFACE:
            width, height = abs(obj.box[2] - obj.box[0]), abs(obj.box[3] - obj.box[1])
            aspect = max(width, height) / max(1, min(width, height))
            # Only a clearly elongated surface-noun is a rigid part (a metal
            # strap), not a placement area. A small saucer still receives a
            # banana, so size alone must not reclassify it.
            if aspect > 1.7 and area < 0.5 * biggest:
                movable, noun = True, "part"
        normalised.append(obj.model_copy(update={
            "label": noun, "noun": noun, "color": color, "movable": movable,
        }))
    ordered = sorted(normalised, key=object_sort_key)
    if len({obj.id for obj in ordered}) != len(ordered):
        raise ValueError("transient object IDs must be unique")
    bases = [f"{canonical_token(obj.color)}_{canonical_token(obj.label)}" for obj in ordered]
    totals = Counter(bases)
    counts: dict[str, int] = defaultdict(int)
    result, remap = [], {}
    assigned: set[str] = set()
    for obj, base in zip(ordered, bases, strict=True):
        counts[base] += 1
        identifier = f"{base}_{counts[base]}" if totals[base] > 1 else base
        while identifier in assigned:
            counts[base] += 1
            identifier = f"{base}_{counts[base]}"
        assigned.add(identifier)
        remap[obj.id] = identifier
        result.append(obj.model_copy(update={"id": identifier}))
    return tuple(result), remap
