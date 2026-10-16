"""Canonical vocabulary and the deterministic IDs built from it.

The tables are ordinary English for broad object families; what is under test is
that a free-form label collapses to one stable referent, that the physical role
follows the noun rather than the wording, and that IDs stay unique and
order-independent whatever the vision model called the objects.
"""
from __future__ import annotations

import pytest

from grounded_intent.canonical import (
    canonical_color,
    canonical_noun,
    canonical_role,
    deterministic_ids,
)
from grounded_intent.domain import ObjectObservation


def observation(identifier="0", *, label="gear", color="red", box=(0, 0, 10, 10),
                pixel=(4, 4), confidence=0.9, movable=True, noun=""):
    return ObjectObservation(
        id=identifier, label=label, noun=noun, color=color, box=box, pixel=pixel,
        confidence=confidence, graspable=True, movable=movable,
    )


@pytest.mark.parametrize("label,noun,expected", [
    ("mandarin", "", "orange"),
    ("plastic_bin", "", "tray"),
    ("work_surface", "", "tray"),
    ("metal_plate", "", "part"),
    ("block", "", "cube"),
    ("sun_gear", "", "gear"),
    ("cup", "", "mug"),
    ("dish", "", "plate"),
    ("tray", "", "tray"),
    ("", "banana", "banana"),
    ("unknown_gizmo", "", "unknown_gizmo"),
    ("grey", "toothed wheel", "gear"),
])
def test_canonical_noun_maps_ordinary_english(label, noun, expected):
    assert canonical_noun(label, noun) == expected


def test_canonical_colour_collapses_near_synonyms():
    assert canonical_color("dark_gray") == "grey"
    assert canonical_color("Gray") == "grey"
    assert canonical_color("crimson") == "red"
    assert canonical_color("") == "unknown"


def test_role_follows_the_physical_noun_not_the_wording():
    assert canonical_role("tray") is False
    assert canonical_role("plate") is False
    assert canonical_role("hand") is False
    assert canonical_role("gear") is True
    assert canonical_role("part") is True


def test_ids_are_canonical_stable_and_order_independent():
    objects = [
        observation("8", box=(30, 40, 40, 50), pixel=(35, 45)),
        observation("7", box=(0, 20, 10, 30), pixel=(5, 25)),
        observation("5", box=(0, 0, 10, 10), pixel=(5, 5)),
        observation("3", label="cube", color="blue"),
    ]
    result, remap = deterministic_ids(objects)
    assert [obj.id for obj in result] == ["blue_cube", "red_gear_1", "red_gear_2", "red_gear_3"]
    assert remap == {"3": "blue_cube", "5": "red_gear_1", "7": "red_gear_2", "8": "red_gear_3"}
    reverse, reverse_map = deterministic_ids(list(reversed(objects)))
    assert result == reverse and remap == reverse_map


def test_a_free_form_label_becomes_its_canonical_referent():
    objects = [
        observation("a", label="plastic bin", color="gray"),
        observation("b", label="mandarin", color="orange"),
    ]
    result, remap = deterministic_ids(objects)
    assert remap["a"] == "grey_tray"
    assert remap["b"] == "orange_orange"
    assert [obj.label for obj in result] == ["tray", "orange"]


def test_exact_ties_have_a_repeatable_breaker():
    first, remap = deterministic_ids([observation("8"), observation("2")])
    second, mapping = deterministic_ids([observation("2"), observation("8")])
    assert first == second and remap == mapping
    assert remap["2"] == "red_gear_1"


def test_a_broad_plate_stays_fixed_but_an_elongated_one_is_a_part():
    objects = [
        observation("tray", label="tray", color="grey", box=(0, 0, 1000, 1000),
                    pixel=(500, 500)),
        observation("plate", label="plate", color="white", box=(10, 10, 510, 410),
                    pixel=(200, 200)),
        observation("strip", label="plate", color="silver", box=(0, 0, 100, 20),
                    pixel=(50, 10)),
    ]
    result, remap = deterministic_ids(objects)
    by_id = {obj.id: obj for obj in result}
    assert by_id[remap["plate"]].movable is False
    assert by_id[remap["strip"]].movable is True
    assert by_id[remap["strip"]].label == "part"


def test_literal_label_collision_bumps_the_suffix():
    # Two 'red gear' instances auto-suffix to red_gear_1/red_gear_2, while a
    # literal label 'gear 1' independently canonicalises to the base red_gear_1.
    # The builder must bump the suffix, never abort the scene.
    objects = [
        observation("a", label="gear", color="red"),
        observation("b", label="gear", color="red"),
        observation("c", label="gear 1", color="red"),
    ]
    result, remap = deterministic_ids(objects)
    ids = [obj.id for obj in result]
    assert len(ids) == len(set(ids)) == 3
    assert set(remap.values()) == set(ids)
    assert remap["a"] == "red_gear_1" and remap["b"] == "red_gear_2"
    assert remap["c"] == "red_gear_1_2"


def test_collision_disambiguation_is_order_independent():
    objects = [
        observation("a", label="gear", color="red"),
        observation("b", label="gear", color="red"),
        observation("c", label="gear 1", color="red"),
    ]
    first, remap_first = deterministic_ids(objects)
    second, remap_second = deterministic_ids(list(reversed(objects)))
    assert first == second and remap_first == remap_second


def test_duplicate_transient_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        deterministic_ids([observation(), observation()])
