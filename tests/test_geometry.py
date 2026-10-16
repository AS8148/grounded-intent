from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from grounded_intent.domain import ObjectObservation
from grounded_intent.geometry import (
    MIN_OBJECT_CONFIDENCE,
    MIN_RELATION_CONFIDENCE,
    box_iou,
    canonical_token,
    contains_pixel,
    mask_interior,
    mask_polygon,
    normalized_box,
    normalized_point,
    numeric_relations,
    prune_duplicates,
    refine_mask,
)


def observation(identifier="0", *, label="gear", color="red", box=(0, 0, 10, 10),
                pixel=(4, 4), confidence=0.9):
    return ObjectObservation(
        id=identifier, label=label, color=color, box=box, pixel=pixel,
        confidence=confidence, graspable=True,
    )


@pytest.mark.parametrize("width,height", [(1, 1), (1920, 1080), (4656, 3496), (7, 13)])
def test_normalized_full_image_half_open_edges(width, height):
    assert normalized_box([0, 0, 1000, 1000], width, height) == (0, 0, width, height)
    assert normalized_point([1000, 1000], width, height) == (width - 1, height - 1)
    assert normalized_point([0, 0], width, height) == (0, 0)


def test_normalized_point_uses_original_not_thumbnail_coordinates():
    assert normalized_point([500, 250], 4656, 3496) == (2328, 874)
    assert normalized_box([250, 250, 750, 750], 4656, 3496) == (1164, 874, 3492, 2622)


def test_clamps_box_but_rejects_out_of_frame_point():
    # Boxes may legitimately touch or slightly exceed the frame; a point outside
    # the declared 0..1000 frame must be rejected, never snapped to the edge.
    assert normalized_box([-100, -10, 1200, 1050], 100, 80) == (0, 0, 100, 80)
    with pytest.raises(ValueError, match="0..1000"):
        normalized_point([-10, 1050], 100, 80)
    with pytest.raises(ValueError, match="0..1000"):
        normalized_point([1000.5, 500], 100, 80)
    assert normalized_box([999.99, 999.99, 1000, 1000], 100, 80) == (99, 79, 100, 80)


@pytest.mark.parametrize("box", [
    [20, 20, 20, 30], [30, 20, 20, 30], [20, 30, 30, 20],
    [-20, 10, -1, 20], [1001, 10, 1010, 20], [10, 1001, 20, 1010],
])
def test_rejects_zero_reversed_or_entirely_outside_boxes(box):
    with pytest.raises(ValueError):
        normalized_box(box, 100, 100)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "2"])
def test_rejects_nonfinite_and_non_numeric_coordinates(value):
    with pytest.raises(ValueError):
        normalized_point([value, 20], 100, 100)
    with pytest.raises(ValueError):
        normalized_box([0, 0, value, 100], 100, 100)


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 20)])
def test_rejects_nonpositive_dimensions(width, height):
    with pytest.raises(ValueError):
        normalized_box([0, 0, 1000, 1000], width, height)
    with pytest.raises(ValueError):
        normalized_point([500, 500], width, height)


def test_subpixel_box_rounding_and_halfopen_membership():
    assert normalized_box([1, 1, 2, 2], 100, 100) == (0, 0, 1, 1)
    assert contains_pixel((0, 0, 10, 10), (9, 9))
    assert not contains_pixel((0, 0, 10, 10), (10, 9))
    assert not contains_pixel((0, 0, 10, 10), (9, 10))


@given(
    width=st.integers(1, 8000), height=st.integers(1, 8000),
    x0=st.integers(0, 999), y0=st.integers(0, 999),
    dx=st.integers(1, 1000), dy=st.integers(1, 1000),
)
def test_normalized_positive_boxes_remain_positive_and_bounded(width, height, x0, y0, dx, dy):
    box = normalized_box([x0, y0, min(1000, x0 + dx), min(1000, y0 + dy)], width, height)
    assert 0 <= box[0] < box[2] <= width
    assert 0 <= box[1] < box[3] <= height
    point = normalized_point([x0, y0], width, height)
    assert contains_pixel(box, point)


@given(x=st.floats(0, 1000, allow_nan=False, allow_infinity=False),
       y=st.floats(0, 1000, allow_nan=False, allow_infinity=False),
       width=st.integers(1, 10000), height=st.integers(1, 10000))
def test_normalized_point_always_in_original_frame(x, y, width, height):
    px, py = normalized_point([x, y], width, height)
    assert 0 <= px < width and 0 <= py < height


@given(x=st.one_of(
    st.floats(-10000, -0.0001, allow_nan=False, allow_infinity=False),
    st.floats(1000.0001, 10000, allow_nan=False, allow_infinity=False),
), y=st.floats(0, 1000, allow_nan=False, allow_infinity=False))
def test_out_of_frame_normalized_point_is_rejected_not_clamped(x, y):
    with pytest.raises(ValueError):
        normalized_point([x, y], 640, 480)


@given(x=st.floats(0, 1000, allow_nan=False, allow_infinity=False),
       y=st.one_of(
           st.floats(-10000, -0.0001, allow_nan=False, allow_infinity=False),
           st.floats(1000.0001, 10000, allow_nan=False, allow_infinity=False),
       ))
def test_out_of_frame_normalized_point_y_is_rejected(x, y):
    with pytest.raises(ValueError):
        normalized_point([x, y], 640, 480)


def test_iou_touching_partial_and_nested_boxes():
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 10, 10), (2, 2, 8, 8)) == pytest.approx(0.36)


@given(x=st.integers(0, 1000), y=st.integers(0, 1000),
       width=st.integers(1, 1000), height=st.integers(1, 1000))
def test_iou_symmetric_and_translation_invariant(x, y, width, height):
    first = (x, y, x + width, y + height)
    second = (x + 1, y + 2, x + width + 1, y + height + 2)
    assert 0 <= box_iou(first, second) <= 1
    assert box_iou(first, second) == box_iou(second, first)
    assert box_iou(first, second) == box_iou(
        (0, 0, width, height), (1, 2, width + 1, height + 2),
    )


def test_canonical_token_and_empty_label():
    assert canonical_token(" Grey Gear ") == "grey_gear"
    with pytest.raises(ValueError):
        canonical_token("!!!")


def test_duplicate_pruning_keeps_best_and_remaps_alias():
    objects = [observation("low", confidence=0.6), observation("high", confidence=0.9)]
    kept, aliases = prune_duplicates(objects)
    assert [obj.id for obj in kept] == ["high"]
    assert aliases == {"high": "high", "low": "high"}


def test_duplicate_pruning_does_not_remove_neighbors_different_classes_or_stacks():
    first = observation("first")
    neighbor = observation("neighbor", box=(5, 0, 15, 10), pixel=(7, 4))
    other_class = observation("cube", label="cube")
    other_color = observation("blue", color="blue")
    layer = observation("layer", confidence=0.8)
    kept, _ = prune_duplicates(
        [first, neighbor, other_class, other_color, layer],
        {frozenset(("first", "layer"))},
    )
    assert {obj.id for obj in kept} == {"first", "neighbor", "cube", "blue", "layer"}


def test_duplicate_pruning_point_membership_matters():
    first = observation("first", box=(0, 0, 100, 100), pixel=(0, 50))
    second = observation("second", box=(1, 0, 101, 100), pixel=(100, 50))
    assert len(prune_duplicates([first, second])[0]) == 2


@pytest.mark.parametrize("threshold", [0, -1, 1.1])
def test_duplicate_threshold_validated(threshold):
    with pytest.raises(ValueError):
        prune_duplicates([], threshold=threshold)


def test_numeric_relations_never_make_physical_stack_from_overlap():
    first = observation("a", box=(0, 0, 10, 10), pixel=(4, 4), confidence=0.7)
    second = observation("b", box=(5, 5, 15, 15), pixel=(9, 9))
    result = numeric_relations([first, second])
    assert {(link.subject, link.predicate, link.object) for link in result} == {
        ("a", "left_of", "b"), ("a", "above", "b"),
    }
    assert all(link.confidence == 0.7 for link in result)
    assert all("numeric" in link.evidence for link in result)
    assert numeric_relations([first, observation("c")]) == ()


def test_mask_interior_avoids_donut_hole():
    mask = np.zeros((51, 51), np.uint8)
    yy, xx = np.ogrid[:51, :51]
    radius = (xx - 25) ** 2 + (yy - 25) ** 2
    mask[(radius <= 23 ** 2) & (radius >= 11 ** 2)] = 1
    point = mask_interior(mask)
    assert mask[point[1], point[0]]
    assert point != (25, 25)


def test_mask_interior_concave_shape_not_centroid():
    mask = np.zeros((30, 30), np.uint8)
    mask[2:28, 2:8] = 1
    mask[22:28, 2:28] = 1
    ys, xs = np.nonzero(mask)
    assert not mask[int(ys.mean()), int(xs.mean())]
    point = mask_interior(mask)
    assert mask[point[1], point[0]]


def test_mask_interior_full_image_pads_edges_and_ties_deterministically():
    assert mask_interior(np.ones((5, 5), np.uint8)) == (2, 2)
    assert mask_interior(np.ones((4, 4), np.uint8)) == (1, 1)
    assert mask_interior(np.ones((1, 1), np.uint8)) == (0, 0)


@pytest.mark.parametrize("mask", [
    np.zeros((5, 5)), np.ones((2, 2, 2)),
    np.full((2, 2), np.nan), np.full((2, 2), 0.4),
])
def test_mask_interior_rejects_empty_invalid_or_unthresholded_logits(mask):
    with pytest.raises(ValueError):
        mask_interior(mask)


@given(st.lists(st.booleans(), min_size=25, max_size=25).filter(any))
def test_mask_interior_is_actual_foreground_for_arbitrary_masks(values):
    mask = np.asarray(values, np.uint8).reshape(5, 5)
    x, y = mask_interior(mask)
    assert mask[y, x] == 1


def test_polygon_rectangular_contour_and_empty_mask():
    mask = np.zeros((20, 30), np.uint8)
    mask[3:17, 5:25] = 1
    polygon = mask_polygon(mask)
    assert len(polygon) == 4
    assert set(polygon) == {(5, 3), (24, 3), (24, 16), (5, 16)}
    assert mask_polygon(np.zeros((5, 5), np.uint8)) == ()


def test_refinement_box_and_pixel_same_original_coordinates():
    mask = np.zeros((80, 120), np.uint8)
    mask[20:60, 30:90] = 1
    refined = refine_mask(mask, (25, 15, 95, 65), (40, 30))
    assert refined.box == (30, 20, 90, 60)
    assert contains_pixel(refined.box, refined.pixel)
    assert mask[refined.pixel[1], refined.pixel[0]]
    assert refined.area_pixels == 2400
    assert all(30 <= x < 90 and 20 <= y < 60 for x, y in refined.polygon)


def test_refinement_never_snaps_point_in_gear_hole():
    mask = np.ones((30, 30), np.uint8)
    mask[10:20, 10:20] = 0
    with pytest.raises(ValueError, match="original VLM point"):
        refine_mask(mask, (0, 0, 30, 30), (15, 15))
    refined = refine_mask(mask, (0, 0, 30, 30), (5, 5))
    assert mask[refined.pixel[1], refined.pixel[0]]


def test_refinement_uses_prompt_component_not_large_unrelated_neighbor():
    mask = np.zeros((100, 100), np.uint8)
    mask[5:15, 5:15] = 1
    mask[30:90, 30:90] = 1
    refined = refine_mask(mask, (3, 3, 17, 17), (7, 7))
    assert refined.box == (5, 5, 15, 15)
    assert refined.area_pixels == 100


def test_refinement_rejects_entire_table_mask_and_tiny_artifact():
    with pytest.raises(ValueError, match="inconsistent"):
        refine_mask(np.ones((100, 100), np.uint8), (20, 20, 40, 40), (30, 30))
    mask = np.zeros((10, 10), np.uint8)
    mask[4, 4] = 1
    with pytest.raises(ValueError, match="fewer than three"):
        refine_mask(mask, (2, 2, 8, 8), (4, 4))


def test_refinement_rejects_mask_mostly_outside_coarse_box():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:50, 10:70] = 1
    with pytest.raises(ValueError, match="mostly lies outside"):
        refine_mask(mask, (30, 10, 70, 50), (40, 30), min_box_coverage=0.80)


@pytest.mark.parametrize("box,prompt", [
    ((0, 0, 20, 20), (20, 4)), ((-1, 0, 20, 20), (4, 4)),
    ((0, 0, 21, 20), (4, 4)), ((0, 0, 0, 20), (0, 4)),
])
def test_refinement_rejects_out_of_frame_box_or_prompt(box, prompt):
    with pytest.raises(ValueError):
        refine_mask(np.ones((20, 20), np.uint8), box, prompt)


def test_confidence_thresholds_explicit_and_not_probabilistic_calibration():
    assert MIN_OBJECT_CONFIDENCE == 0.60
    assert MIN_RELATION_CONFIDENCE == 0.65


def test_point_within_receiver_box_cannot_establish_physical_containment():
    item = observation("item", box=(20, 20, 40, 40), pixel=(30, 30))
    receiver = observation("receiver", box=(0, 0, 100, 100), pixel=(80, 80)).model_copy(
        update={"movable": False, "graspable": False, "receives_objects": True},
    )
    relations = numeric_relations((item, receiver))
    assert relations
    assert {relation.predicate for relation in relations} <= {"left_of", "above"}
