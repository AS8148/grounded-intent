"""Image-plane geometry, never inferred metric gripper or 3-D measurements.

Boxes use half-open original-image coordinates; pixels use (x/column, y/row).
Mask polygons are display outlines only: their holes are not represented by the
single-ring domain contract. Pick pixels are checked against the actual mask.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from .domain import Box, ObjectObservation, Pixel, Relation

MIN_OBJECT_CONFIDENCE = 0.60
MIN_RELATION_CONFIDENCE = 0.65
DUPLICATE_IOU = 0.92
# A visible manipulable object spans at least this fraction of each image axis.
# Anything smaller is degenerate output, not an object.
MIN_OBJECT_SPAN_FRACTION = 0.01
GEOMETRY_REVISION = "model-reported-labels-v8"


def _numbers(values: Sequence[float], length: int) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"expected {length} coordinates")
    if any(isinstance(value, (bool, str)) for value in values):
        raise ValueError("coordinates must be finite numbers")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("coordinates must be finite numbers")
    return result


def _dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")


def normalized_box(values: Sequence[float], width: int, height: int) -> Box:
    """Map 0..1000 edges to original pixels, clamp, and reject collapsed boxes.

    Floor the lower edge and ceil the upper edge so subpixel positive boxes keep
    their coverage. Reversed/zero-area model boxes are errors, never reordered.
    """
    _dimensions(width, height)
    x0, y0, x1, y1 = _numbers(values, 4)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("normalized box must have positive area")
    x0, x1 = max(0.0, min(1000.0, x0)), max(0.0, min(1000.0, x1))
    y0, y1 = max(0.0, min(1000.0, y0)), max(0.0, min(1000.0, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("normalized box collapses outside image after clamping")
    box = (
        max(0, min(width - 1, math.floor(x0 * width / 1000))),
        max(0, min(height - 1, math.floor(y0 * height / 1000))),
        max(1, min(width, math.ceil(x1 * width / 1000))),
        max(1, min(height, math.ceil(y1 * height / 1000))),
    )
    return box


def normalized_point(values: Sequence[float], width: int, height: int) -> Pixel:
    """Map a 0..1000 point, not scaled JPEG pixels, to original integer pixels.

    Points outside the declared 0..1000 frame are rejected, never clamped: a
    degenerate model point must drop its object rather than fabricate a pixel on
    the image edge. The exact edge coordinate 1000 is still legal and maps to
    the last row/column.
    """
    _dimensions(width, height)
    x, y = _numbers(values, 2)
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        raise ValueError("normalized point must lie within the 0..1000 frame")
    return (
        max(0, min(width - 1, math.floor(x * width / 1000))),
        max(0, min(height - 1, math.floor(y * height / 1000))),
    )


def contains_pixel(box: Box, pixel: Pixel) -> bool:
    return box[0] <= pixel[0] < box[2] and box[1] <= pixel[1] < box[3]


def box_has_plausible_extent(box: Box, width: int, height: int) -> bool:
    """Is this box a plausible visible-object extent in a full-resolution frame?

    A real manipulable object spans a non-trivial fraction of the image. A box of
    a few pixels is the signature of a model that emitted near-zero coordinates
    (e.g. a 0..1 fraction instead of the required 0..1000 normalized frame), and
    accepting it would place the pick pixel on background and feed a meaningless
    support graph to the planner.
    """
    return (box[2] - box[0]) >= max(1, MIN_OBJECT_SPAN_FRACTION * width) \
        and (box[3] - box[1]) >= max(1, MIN_OBJECT_SPAN_FRACTION * height)


def box_iou(first: Box, second: Box) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    union = a + b - overlap
    return overlap / union if union > 0 else 0.0


def canonical_token(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not value:
        raise ValueError("object label/color must contain a nonempty ASCII identifier")
    return value


def object_sort_key(obj: ObjectObservation) -> tuple:
    """Spatial order is box-center x then y, with deterministic tie breakers."""
    return (
        canonical_token(obj.color), canonical_token(obj.label),
        obj.box[0] + obj.box[2], obj.box[1] + obj.box[3],
        obj.box, obj.pixel, -obj.confidence, obj.evidence, obj.id,
    )





def prune_duplicates(
    objects: Sequence[ObjectObservation], protected_pairs: set[frozenset[str]] | None = None,
    threshold: float = DUPLICATE_IOU,
) -> tuple[tuple[ObjectObservation, ...], dict[str, str]]:
    """Prune near-identical proposals, not adjacent or visibly stacked instances.

    Only same class/color boxes with very high IoU and mutually contained points
    qualify. Explicit physical relations protect distinct visible layers.
    Returned aliases let relations still reference the retained observation.
    """
    if not 0 < threshold <= 1:
        raise ValueError("duplicate IoU threshold must be in (0, 1]")
    protected_pairs = protected_pairs or set()
    retained: list[ObjectObservation] = []
    aliases: dict[str, str] = {}
    for obj in sorted(objects, key=lambda item: (-item.confidence, object_sort_key(item))):
        duplicate = next((
            kept for kept in retained
            if canonical_token(obj.label) == canonical_token(kept.label)
            and canonical_token(obj.color) == canonical_token(kept.color)
            and frozenset((obj.id, kept.id)) not in protected_pairs
            and box_iou(obj.box, kept.box) >= threshold
            and contains_pixel(obj.box, kept.pixel)
            and contains_pixel(kept.box, obj.pixel)
        ), None)
        if duplicate is None:
            retained.append(obj)
            aliases[obj.id] = obj.id
        else:
            aliases[obj.id] = duplicate.id
    return tuple(sorted(retained, key=object_sort_key)), aliases


def numeric_relations(objects: Sequence[ObjectObservation]) -> tuple[Relation, ...]:
    """Only image-plane order; overlap never creates on/inside/blocks relations."""
    result = []
    for first, second in combinations(objects, 2):
        x_first, y_first = first.box[0] + first.box[2], first.box[1] + first.box[3]
        x_second, y_second = second.box[0] + second.box[2], second.box[1] + second.box[3]
        confidence = min(first.confidence, second.confidence)
        for a, b, predicate, evidence in (
            (first if x_first < x_second else second,
             second if x_first < x_second else first, "left_of", "numeric box-center x order"),
            (first if y_first < y_second else second,
             second if y_first < y_second else first, "above", "numeric box-center y order"),
        ):
            if (predicate == "left_of" and x_first == x_second) or (
                predicate == "above" and y_first == y_second
            ):
                continue
            result.append(Relation(
                subject=a.id, predicate=predicate, object=b.id,
                confidence=confidence, evidence=evidence,
            ))
    return tuple(result)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("mask must be a finite two-dimensional array")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be binary, not raw logits")
    return np.ascontiguousarray(mask, dtype=np.uint8)


def mask_interior(mask: np.ndarray) -> Pixel:
    """Maximum-clearance visible pixel, including concave and perforated masks.

    Zero padding makes image-edge foreground behave like every other boundary.
    np.argmax resolves equal-clearance ties by row then column deterministically.
    """
    binary = _binary_mask(mask)
    if not binary.any():
        raise ValueError("cannot choose an interior pixel from an empty mask")
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(int(distance.argmax()), binary.shape)
    return int(x), int(y)


def mask_polygon(mask: np.ndarray, epsilon: float = 0.005) -> tuple[Pixel, ...]:
    """Simplified outer display contour, not a substitute for mask membership."""
    if not 0 <= epsilon <= 0.1:
        raise ValueError("polygon epsilon must be in [0, 0.1]")
    binary = _binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    return tuple((int(x), int(y)) for x, y in simplified.reshape(-1, 2))


@dataclass(frozen=True)
class MaskGeometry:
    box: Box
    pixel: Pixel
    polygon: tuple[Pixel, ...]
    area_pixels: int


def refine_mask(
    mask: np.ndarray, box: Box, prompt: Pixel, *, min_box_iou: float = 0.15,
    min_box_coverage: float = 0.30,
) -> MaskGeometry:
    """Validate an actual SAM mask before selecting a new target point.

    The original VLM point must belong to the mask — that anchors the component
    to the observed object. Reject masks that clearly engulf a container or
    unrelated surface (very low IoU or coverage); accept masks somewhat larger
    than the coarse VLM box, since a correct segmentation legitimately includes
    teeth, rims, and shadows the VLM box omitted. A separate disconnected
    region is excluded; the retained component must itself pass every check.
    """
    if not 0 <= min_box_iou <= 1 or not 0 <= min_box_coverage <= 1:
        raise ValueError("mask consistency thresholds must be in [0, 1]")
    binary = _binary_mask(mask)
    height, width = binary.shape
    _dimensions(width, height)
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("coarse box must lie within the mask image")
    if not contains_pixel(box, prompt):
        raise ValueError("VLM point does not lie inside its coarse box")
    x, y = prompt
    if not binary[y, x]:
        raise ValueError("SAM mask does not include the original VLM point")
    _, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_id = int(components[y, x])
    component = (components == component_id).astype(np.uint8)
    left, top, w, h, area = (int(value) for value in stats[component_id])
    refined_box = (left, top, left + w, top + h)
    if area < 3:
        raise ValueError("SAM mask prompt component has fewer than three visible pixels")
    if box_iou(box, refined_box) < min_box_iou:
        raise ValueError("SAM mask bounding box is inconsistent with VLM box")
    coverage = np.count_nonzero(component[box[1]:box[3], box[0]:box[2]]) / area
    if coverage < min_box_coverage:
        raise ValueError("SAM mask mostly lies outside the VLM box")
    pixel = mask_interior(component[top:top + h, left:left + w])
    pixel = (pixel[0] + left, pixel[1] + top)
    if not component[pixel[1], pixel[0]] or not contains_pixel(refined_box, pixel):
        raise ValueError("mask interior selection produced an invalid target pixel")
    polygon = mask_polygon(component[top:top + h, left:left + w])
    polygon = tuple((px + left, py + top) for px, py in polygon)
    return MaskGeometry(refined_box, pixel, polygon, area)
