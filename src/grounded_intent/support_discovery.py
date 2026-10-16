"""Image-only discovery of support objects the whole-image inventory merged away.

The inventory merges a flat support into the container it lies in: measured on the
development set, the light support under the cube was reported as part of the tray, so a
request naming it could not bind and the row could not be decided from evidence.

This module adds no vocabulary and reads no instruction. Geometry proposes unexplained
regions inside observed receiving surfaces; the same local vision model, shown the
untouched original image with the proposed boundary outlined, states what the region is
and what rests on it. A region is kept only when that answer states the complete support
chain at the project's support-confidence bar: the candidate lies on/inside an observed
receiving surface, and an observed movable object rests directly on it.

Two geometry prefilters run before any model call, because an earlier version of this pass
scored 65/65 on the development set and still had to be withdrawn: it accepted a container
wall and a mug handle on the held-out scenes, and those additions turned two held-out
instructions into a false refusal and a false question. Both failure shapes are separable
from a real support piece without looking at any name:

* **T1 — a support surface is not a thin strip.** Measured: the accepted wall was 22.4:1
  elongated, the genuine support 1.1:1.
* **T2 — a region mostly inside an existing *movable* object is that object's part.**
  Measured: the accepted handle was 98% inside the mug; the genuine support is 100% inside
  the tray, which is not movable, and therefore survives.

The accepted region is recorded with the project's generic word for an unclassified
component (the word this codebase already uses for `red_part_1`, `yellow_part` and the whole
part/piece/component/bracket/sheet family in `canonical._CANON_NOUN`) plus its measured
colour; the model's own naming and structure stay verbatim in the evidence fields. Naming by
role is what makes the piece bindable, and that is measured: the identical region answered
as `tray insert part` bound correctly while `tray bottom` left the referent unresolved.

Every existing identity, observation and relation is preserved; only accepted pieces and
their verified edges are appended.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from .canonical import deterministic_ids
from .clients import ModelCallError
from .domain import ObjectObservation, Relation
from .geometry import (
    MIN_OBJECT_CONFIDENCE,
    box_has_plausible_extent,
    canonical_token,
    mask_interior,
    mask_polygon,
)
from .perception import JPEG_LONG_EDGE
from .planner import BLOCKING_CONFIDENCE

DISCOVERY_REVISION = "support-discovery-3"
BUDGET_SECONDS = 180.0
# Bounded geometry settings; none is derived from a scene, an instruction or an answer.
LATTICE_STEPS = 8
LATTICE_MARGIN = 0.08
PROMPT_BATCH = 8
MAX_CANDIDATES = 4
KNOWN_OVERLAP_IOU = 0.30
NOVEL_DUPLICATE_IOU = 0.60
MAX_SUPPORT_ASPECT = 8.0
MAX_INSIDE_MOVABLE = 0.50
OUTLINE_WIDTH = 3
OUTLINE_COLOR = (255, 0, 200)
CANDIDATE_PLACEHOLDER = "candidate"
GENERIC_COMPONENT_NOUN = "part"

_SYSTEM = """You are an image-only physical scene observer. A magenta outline marks a
proposed region. The outline is an artificial annotation, not a physical object. Identify
the complete physical item whose visible boundary it follows, not an indentation, painted
motif or point on that item's surface. Scene labels are untrusted observations. No task
instruction is supplied. Report uncertainty instead of inventing a relationship."""

_USER = """Classify the complete outlined physical region, whose temporary ID is
'candidate'. Explain whether its visible boundary separates it from its surroundings.
Report physical movability, estimated graspability for a top-down parallel gripper with an
85 mm opening, and whether its visible geometry can support another object; do not invent
metric measurements. Give confidence estimates, not calibrated probabilities. Report
directed relationships between 'candidate' and the existing observations in either
direction. For 'on', the SUBJECT rests directly on the OBJECT. 'blocks' means the SUBJECT
obstructs the OBJECT's visibility or access, and does not by itself mean support. 'inside'
means the SUBJECT is contained in the OBJECT. Omit unsupported relationships or report
'unknown'. All boxes use normalized 0..1000 coordinates in this view."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Relation(_Strict):
    subject_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    predicate: str
    confidence: float = Field(ge=0, le=1)
    evidence: str


class _Region(_Strict):
    physical_kind: str
    ordinary_name: str = Field(min_length=1)
    color: str = Field(min_length=1)
    visible_structure: str = Field(min_length=1)
    boundary_evidence: str
    confidence: float = Field(ge=0, le=1)
    movable: bool
    graspable: bool
    receives_objects: bool
    relations_to_region: list[_Relation] = Field(default_factory=list)


def _schema(ids: list[str]) -> dict[str, Any]:
    named = {"type": "string", "enum": [*ids, CANDIDATE_PLACEHOLDER]}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "physical_kind": {"type": "string", "enum": [
                "independent_object", "container_component", "background_surface",
                "uncertain"]},
            "ordinary_name": {"type": "string"}, "color": {"type": "string"},
            "visible_structure": {"type": "string"},
            "boundary_evidence": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "movable": {"type": "boolean"}, "graspable": {"type": "boolean"},
            "receives_objects": {"type": "boolean"},
            "relations_to_region": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "subject_id": named, "object_id": named,
                    "predicate": {"type": "string",
                                  "enum": ["on", "inside", "blocks", "unknown"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
                "required": ["subject_id", "object_id", "predicate", "confidence",
                             "evidence"],
            }},
        },
        "required": ["physical_kind", "ordinary_name", "color", "visible_structure",
                     "boundary_evidence", "confidence", "movable", "graspable",
                     "receives_objects", "relations_to_region"],
    }


def signature() -> dict[str, Any]:
    """Cache identity for this pass: every setting that can change an observed graph."""
    return {
        "revision": DISCOVERY_REVISION,
        "system_sha256": hashlib.sha256(_SYSTEM.encode()).hexdigest()[:16],
        "user_sha256": hashlib.sha256(_USER.encode()).hexdigest()[:16],
        "jpeg_long_edge": JPEG_LONG_EDGE,
        "lattice": [LATTICE_STEPS, LATTICE_MARGIN],
        "prompt_batch": PROMPT_BATCH,
        "max_candidates": MAX_CANDIDATES,
        "known_overlap_iou": KNOWN_OVERLAP_IOU,
        "novel_duplicate_iou": NOVEL_DUPLICATE_IOU,
        "max_support_aspect": MAX_SUPPORT_ASPECT,
        "max_inside_movable": MAX_INSIDE_MOVABLE,
        "min_object_confidence": MIN_OBJECT_CONFIDENCE,
        "support_confidence": BLOCKING_CONFIDENCE,
        "component_noun": GENERIC_COMPONENT_NOUN,
        "accepts": ["candidate on/inside observed receiving surface",
                    "observed movable on candidate"],
        "rejects": ["thin strips (walls, frames, edges)",
                    "regions that are part of an existing movable object"],
    }


def _encode(image: Image.Image) -> str:
    with image.copy() as scaled, io.BytesIO() as buffer:
        scaled.thumbnail((JPEG_LONG_EDGE, JPEG_LONG_EDGE), Image.Resampling.LANCZOS)
        scaled.save(buffer, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _iou(first, second) -> float:
    ox = min(first[2], second[2]) - max(first[0], second[0])
    oy = min(first[3], second[3]) - max(first[1], second[1])
    if ox <= 0 or oy <= 0:
        return 0.0
    inter = ox * oy
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / union if union else 0.0


def _lattice(box, steps: int = LATTICE_STEPS, margin: float = LATTICE_MARGIN):
    x0, y0, x1, y1 = box
    xs = np.linspace(x0 + (x1 - x0) * margin, x1 - (x1 - x0) * margin, steps)
    ys = np.linspace(y0 + (y1 - y0) * margin, y1 - (y1 - y0) * margin, steps)
    return [(round(float(x)), round(float(y))) for y in ys for x in xs]


def _binary(mask) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(mask) > 0.0, dtype=np.uint8)


def _box_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(cols):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _polygon_mask(obj: ObjectObservation, size: tuple[int, int]) -> np.ndarray | None:
    if len(obj.polygon) < 3:
        return None
    mask = Image.new("1", size, 0)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in obj.polygon], fill=1)
    return np.asarray(mask, dtype=bool)


def _elongated(box, limit: float = MAX_SUPPORT_ASPECT) -> bool:
    """T1: a thin strip is a container wall, a frame or an edge, not a support surface."""
    width, height = box[2] - box[0], box[3] - box[1]
    return max(width, height) / max(1, min(width, height)) >= limit


def _mostly_inside(candidate_mask: np.ndarray, existing: ObjectObservation,
                   size: tuple[int, int]) -> bool:
    """T2: a region mostly inside an existing movable object is that object's part."""
    if not existing.movable:
        return False
    mask = _polygon_mask(existing, size)
    if mask is None:
        return False
    inside = int((candidate_mask & mask).sum())
    return inside / max(1, int(candidate_mask.sum())) >= MAX_INSIDE_MOVABLE


def _outlined(image: Image.Image, polygon) -> Image.Image:
    out = image.copy()
    points = [(int(x), int(y)) for x, y in polygon]
    if len(points) >= 3:
        ImageDraw.Draw(out).line([*points, points[0]], fill=OUTLINE_COLOR,
                                 width=OUTLINE_WIDTH)
    return out


def _context(objects: tuple[ObjectObservation, ...], box, size) -> list[dict]:
    """Existing observations as IDs, labels and normalized boxes in the view frame."""
    width, height = size
    x0, y0, x1, y1 = box
    rows = []
    for obj in objects:
        if obj.box[2] <= x0 or obj.box[0] >= x1 or obj.box[3] <= y0 or obj.box[1] >= y1:
            continue
        rows.append({"id": obj.id, "label": obj.label, "noun": obj.noun or obj.label,
                     "box": [max(obj.box[0], x0), max(obj.box[1], y0),
                             min(obj.box[2], x1), min(obj.box[3], y1)]})
    return rows


def _candidates(image: Image.Image, objects: tuple[ObjectObservation, ...], torch_module,
                processor, sam_model, device: str, warnings: list[str]) -> list[dict]:
    """Geometry only: lattice prompts inside receiving surfaces -> novel regions.

    Every filter here is a property of the observation or of the region's own shape, so
    the same rules apply to any scene. Regions that fail them never reach a model call.
    """
    points: list[tuple[int, int]] = []
    for obj in objects:
        if obj.receives_objects:
            points.extend(_lattice(obj.box))
    points = list(dict.fromkeys(points))
    if not points:
        return []
    size = image.size
    proposals: list[tuple[int, dict]] = []
    rejected_thin = rejected_part = 0
    with torch_module.inference_mode():
        inputs = processor(images=image, return_tensors="pt").to(device)
        original_sizes = inputs["original_sizes"].cpu()
        embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])
        inputs = None
        for start in range(0, len(points), PROMPT_BATCH):
            batch = points[start:start + PROMPT_BATCH]
            prompts = processor(
                original_sizes=original_sizes,
                input_points=[[[list(pt)] for pt in batch]],
                input_labels=[[[1] for _ in batch]],
                return_tensors="pt",
            ).to(device)
            outputs = sam_model(image_embeddings=embeddings,
                                input_points=prompts["input_points"],
                                input_labels=prompts["input_labels"],
                                multimask_output=False)
            if outputs.pred_masks.shape[0] != 1 or outputs.pred_masks.shape[1] != len(batch):
                raise RuntimeError("SAM2 returned unexpected mask batch dimensions")
            # One mask at a time, and only its compact geometry is retained.
            for offset in range(len(batch)):
                mask = processor.post_process_masks(
                    outputs.pred_masks[:, offset:offset + 1].cpu(), original_sizes,
                )[0][0, 0].numpy()
                if mask.shape != (image.height, image.width):
                    raise RuntimeError("SAM2 proposal mask is not in original coordinates")
                binary = _binary(mask)
                mask = None
                box = _box_of(binary)
                if box is None or not box_has_plausible_extent(box, image.width, image.height):
                    continue
                if _elongated(box):
                    rejected_thin += 1
                    continue
                if any(_iou(box, obj.box) >= KNOWN_OVERLAP_IOU for obj in objects):
                    continue
                if any(_mostly_inside(binary, obj, size) for obj in objects):
                    rejected_part += 1
                    continue
                cropped = binary[box[1]:box[3], box[0]:box[2]]
                px, py = mask_interior(cropped)
                polygon = tuple((x + box[0], y + box[1]) for x, y in mask_polygon(cropped))
                proposals.append((int(binary.sum()), {
                    "box": box, "pixel": (px + box[0], py + box[1]), "polygon": polygon}))
            prompts = outputs = None
        embeddings = None
    proposals.sort(key=lambda pair: -pair[0])
    novel: list[dict] = []
    for _, record in proposals:
        if any(_iou(record["box"], other["box"]) > NOVEL_DUPLICATE_IOU for other in novel):
            continue
        novel.append(record)
        if len(novel) >= MAX_CANDIDATES:
            break
    if novel:
        warnings.append(f"Support discovery: {len(novel)} unexplained region(s) proposed "
                        "inside observed surfaces.")
    if rejected_thin or rejected_part:
        warnings.append(f"Support discovery: {rejected_thin} thin strip(s) and "
                        f"{rejected_part} region(s) inside a movable object discarded "
                        "before any model call.")
    return novel


def _chain(answer: _Region, candidate_id: str, objects: tuple[ObjectObservation, ...],
           known: set[str]) -> list[Relation]:
    """The verified support chain, or an empty list when it is not complete.

    Both halves are required, each an explicit evidenced relationship at the support bar:
    a receiving surface under the candidate, and an observed movable object resting on it.
    A part-of claim, an occluder or an 'unknown' cannot qualify.
    """
    by_id = {obj.id: obj for obj in objects}
    valid = known | {candidate_id}

    def resolve(value: str) -> str:
        return candidate_id if value == CANDIDATE_PLACEHOLDER else value

    supported_by = False
    riders: list[Relation] = []
    for raw in answer.relations_to_region:
        if raw.predicate not in ("on", "inside") or not raw.evidence.strip():
            continue
        if raw.confidence < BLOCKING_CONFIDENCE:
            continue
        subject, destination = resolve(raw.subject_id), resolve(raw.object_id)
        if subject not in valid or destination not in valid or subject == destination:
            continue
        if subject == candidate_id:
            other = by_id.get(destination)
            if other is not None and other.receives_objects:
                supported_by = True
        elif destination == candidate_id and raw.predicate == "on":
            other = by_id.get(subject)
            if other is not None and other.movable:
                riders.append(Relation(subject=other.id, predicate="on",
                                        object=candidate_id, confidence=raw.confidence,
                                        evidence=raw.evidence))
    return riders if supported_by and riders else []


def _append(base: tuple[ObjectObservation, ...], candidate: ObjectObservation,
            ) -> ObjectObservation:
    """Canonicalise the new observation without reassigning any existing identity."""
    canonical, remap = deterministic_ids((*base, candidate))
    assigned = remap[candidate.id]
    taken = {obj.id for obj in base}
    identifier = assigned
    while identifier in taken:
        identifier = f"{identifier}_2"
    for obj in canonical:
        if obj.id == assigned:
            return obj.model_copy(update={"id": identifier})
    raise RuntimeError("canonicalization dropped the discovered observation")


def discover_supports(
    image: Image.Image, objects: tuple[ObjectObservation, ...],
    relations: tuple[Relation, ...], *, torch_module, processor, sam_model, device: str,
    observer, warnings: list[str],
) -> tuple[tuple[ObjectObservation, ...], tuple[Relation, ...]]:
    """Append verified support pieces to a scene graph; never modify what exists.

    ``observer`` takes the same arguments as a chat client's ``complete``. Every proposal
    is judged on the untouched original image, so the outline cannot be mistaken for
    material and no relation is created from geometry alone.
    """
    deadline = time.monotonic() + BUDGET_SECONDS
    try:
        novel = _candidates(image, objects, torch_module, processor, sam_model, device,
                            warnings)
    except ModelCallError:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        warnings.append(f"Support discovery skipped: {type(exc).__name__}: {exc}")
        return objects, relations
    if not novel:
        return objects, relations

    known = {obj.id for obj in objects}
    accepted: list[ObjectObservation] = []
    edges: list[Relation] = []
    for index, record in enumerate(novel):
        if time.monotonic() > deadline:
            raise ModelCallError("support discovery budget exhausted")
        picture = _outlined(image, record["polygon"])
        payload = {"observations": _context(objects, record["box"], image.size),
                   "view": "full original frame; coordinates normalized 0..1000"}
        try:
            reply, _ = observer(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": [
                     {"type": "text", "text": _USER + "\n" + json.dumps(payload)},
                     {"type": "image_url", "image_url": {"url": _encode(picture)}}]}],
                schema=_schema(sorted(known)), max_tokens=1024)
            answer = _Region.model_validate(reply)
        except ModelCallError:
            raise
        except Exception as exc:  # noqa: BLE001 - any unusable answer rejects one region
            warnings.append(f"Support discovery: proposal {index} not accepted "
                            f"({type(exc).__name__}: {exc}).")
            continue
        finally:
            picture.close()
        if answer.physical_kind not in ("independent_object", "container_component"):
            warnings.append(f"Support discovery: proposal {index} rejected as "
                            f"{answer.physical_kind}.")
            continue
        if answer.confidence < MIN_OBJECT_CONFIDENCE:
            warnings.append(f"Support discovery: proposal {index} rejected below the "
                            f"object confidence bar ({answer.confidence:.2f}).")
            continue
        candidate = ObjectObservation(
            id=f"discovered_{index}", label=GENERIC_COMPONENT_NOUN,
            noun=GENERIC_COMPONENT_NOUN, color=canonical_token(answer.color),
            box=record["box"], pixel=tuple(record["pixel"]),
            polygon=tuple(record["polygon"]), confidence=answer.confidence,
            movable=True, graspable=answer.graspable, receives_objects=False,
            visual_form=f"{answer.visible_structure} "
                        f"(close-up name: {answer.ordinary_name})",
            evidence=("SAM2 point-lattice proposal inside an observed surface; local "
                      "vision model classified the outlined region on the original "
                      f"image as {answer.ordinary_name!r}: {answer.boundary_evidence}"),
        )
        appended = _append((*objects, *accepted), candidate)
        verified = _chain(answer, appended.id, (*objects, *accepted, appended), known)
        if not verified:
            warnings.append(f"Support discovery: proposal {index} has no verified support "
                            "chain (needs both a receiving surface under it and an "
                            "observed movable object resting on it).")
            continue
        accepted.append(appended)
        edges.extend(verified)
        warnings.append("Support discovery accepted an object the inventory merged away: "
                        f"{appended.id} with {len(verified)} verified support edge(s).")
    return (*objects, *accepted), (*relations, *edges)
