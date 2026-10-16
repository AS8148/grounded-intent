"""Hybrid perception: a vision LLM names objects (label-only), then a local
open-vocabulary detector (YOLOE-26) grounds each colored label to a box + mask.

This is a *selectable backend* — the labeler is any `ChatClient` (local Qwen via
llama.cpp, or remote Apertus for comparison), so the pipeline keeps working
fully offline. YOLOE only localizes; it never invents objects the labeler did
not name, which removes the confabulation that breaks a strict-coordinate prompt.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import Field, ValidationError

from .canonical import deterministic_ids
from .domain import SceneGraph
from .geometry import (
    MIN_OBJECT_CONFIDENCE,
    MIN_RELATION_CONFIDENCE,
    box_has_plausible_extent,
    mask_interior,
    numeric_relations,
    prune_duplicates,
)
from .perception import (
    _SYSTEM,
    JPEG_LONG_EDGE,
    Perception,
    _break_support_cycles,
    _drop_impossible_support,
    _Inventory,
    _Link,
    _observations,
    _Seen,
    _StrictModel,
    _visual_links,
    resolve_overlaps_by_mask,
)

_LABEL_PROMPT = (
    "Inspect the image and list EVERY distinct physical object, including small, "
    "flat or partially occluded items and anything resting inside a container. "
    "Do not invent objects that are not actually present. For each output "
    "{\"index\",\"label\",\"color\",\"noun\",\"movable\",\"graspable\","
    "\"receives_objects\",\"confidence\"}.\n"
    "`label`: a short physical-class name for the object — the most specific "
    "everyday word a person would use. `noun`: the everyday identity a person "
    "would use pointing at it (usually the same). `color`: its dominant surface "
    "colour as a plain English word — the actual colour, not a highlight, "
    "shadow, or reflection. List each instance separately; never merge identical "
    "objects. `movable` means physically movable, not robot pick permission. "
    "`graspable` is a visual estimate of whether the robot gripper can pick the "
    "item, based on visible geometry and access rather than label rules. "
    "`receives_objects` means its visible geometry can support or contain another "
    "object, independently of mobility and graspability. Do NOT output coordinates.\n"
    "Also output relations: {\"subject\":i,\"predicate\":\"on\"|\"inside\"|"
    "\"blocks\",\"object\":j,\"confidence\":0..1,\"basis\":\"visual_stack\"|"
    "\"visual_support\"|\"visual_containment\"|\"visual_occlusion\"|\"overlap\"|"
    "\"uncertain\",\"evidence\":\"...\"} and a warnings array. Physical "
    "relations require positive visible support, containment or occlusion cues; "
    "projected overlap alone is not evidence. Return JSON "
    "{\"objects\":[...],\"relations\":[...],\"warnings\":[...]} only."
)


class _Labeled(_StrictModel):
    index: int = Field(ge=0)
    label: str = Field(min_length=1)
    color: str = Field(min_length=1)
    noun: str = Field(min_length=1)
    movable: bool
    graspable: bool
    receives_objects: bool
    confidence: float = Field(ge=0, le=1)


class _LabelInventory(_StrictModel):
    objects: list[_Labeled]
    relations: list[_Link]
    warnings: list[str]


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _inside(box, point):
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def _nms(dets):
    """Per-class NMS: drop lower-confidence boxes overlapping a kept one (IoU>0.5)."""
    keep = []
    for d in dets:
        if all(_iou(d[0], k[0]) <= 0.5 for k in keep):
            keep.append(d)
    return keep

def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)




def _detector_path(name: str) -> str:
    """Resolve detector weights without a runtime download (--network none)."""
    candidate = Path(os.environ.get("MODELS_DIR", "../.models")) / name
    if candidate.is_file():
        return str(candidate)
    local = Path(name)
    if local.is_file():
        return str(local)
    raise RuntimeError(
        f"Detector weights {name!r} not found in {Path(os.environ.get('MODELS_DIR', '../.models'))} "
        f"or {Path(name)}; run the model prepare stage first. Runtime never downloads."
    )



class HybridPerception(Perception):
    """Perception backend: labeler LLM names objects; YOLOE grounds them.

    `client` is the labeler (Qwen via llama.cpp locally, or remote Apertus).
    """

    def __init__(self, client, segment: bool = True, device: str = "cpu",
                 detector: str = "yoloe-26s-seg.pt", conf: float = 0.05):
        super().__init__(client, segment=segment, device=device)
        self._detector_name = detector
        self._conf = conf
        self._yoloe = None

    @property
    def signature(self) -> str:  # noqa: D102 - cache must differ from VLM path
        base = json.loads(super().signature)
        base["backend"] = "hybrid"
        base["detector"] = self._detector_name
        base["conf"] = self._conf
        base["label_prompt_sha256"] = hashlib.sha256(
            (_SYSTEM + _LABEL_PROMPT).encode()).hexdigest()
        return json.dumps(base, sort_keys=True, separators=(",", ":"))

    def _detector(self):
        if self._yoloe is None:
            from ultralytics import YOLOE
            self._yoloe = YOLOE(_detector_path(self._detector_name))
        return self._yoloe

    def release(self) -> None:
        super().release()
        self._yoloe = None

    # ---- label-only LLM inventory -------------------------------------------
    def _label_inventory(self, image_url: str) -> _LabelInventory:
        messages = [
            {"role": "system", "content": _SYSTEM + "\nDo not output coordinates."},
            {"role": "user", "content": [
                {"type": "text", "text": _LABEL_PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        for attempt in range(2):
            raw, _ = self.client.complete(
                messages, schema=_LabelInventory.model_json_schema(),
                max_tokens=int(os.environ.get("PERCEPTION_MAX_TOKENS", "4096")))
            try:
                return _LabelInventory.model_validate(raw)
            except (ValidationError, ValueError) as exc:
                if attempt:
                    raise RuntimeError(f"label-only inventory malformed: {exc}") from exc
                messages.append({"role": "assistant", "content": json.dumps(raw)})
                messages.append({"role": "user", "content":
                    "Return ONLY the JSON object with objects/relations/warnings."})

    # ---- YOLOE grounding ----------------------------------------------------
    def _ground(self, image: Image.Image, inventory: _LabelInventory):
        vocab = list(dict.fromkeys(
            f"{item.color.strip()} {item.label.strip()}".strip()
            for item in inventory.objects
        ))
        if not vocab:
            return {}, inventory
        det = self._detector()
        det.set_classes(vocab)
        # retina_masks keeps masks in ORIGINAL image coordinates; without it ultralytics
        # returns them in the padded model-input frame while unpadding the boxes, so a
        # naive resize misaligns mask and box on non-square images.
        res = det.predict(image, conf=self._conf, retina_masks=True, verbose=False)[0]
        W, H = image.size
        grounded = {}  # vocab_index -> list of (box,pixel,conf)
        names = res.names
        for ci, box, conf, mask in zip(
                res.boxes.cls, res.boxes.xyxy, res.boxes.conf,
                res.masks.data if res.masks is not None else [None] * len(res.boxes),
                strict=False):
            lab = names[int(ci)]
            b = box.tolist()
            px = None
            if mask is not None:
                m = mask.cpu().numpy() > 0
                # YOLOE can return a mask at the model's input resolution, not the
                # full frame; resize instead of top-left pasting truncated edges.
                if m.shape != (H, W):
                    m = cv2.resize(
                        m.astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
                x1, y1 = min(W, int(b[2])), min(H, int(b[3]))
                crop = m[y0:y1, x0:x1]
                sub = np.argwhere(crop)
                if len(sub):
                    py, px = int(sub[len(sub) // 2][0]) + y0, int(sub[len(sub) // 2][1]) + x0
                    if not m[py, px]:
                        # Median row-major index can land off a concave mask.
                        ix, iy = mask_interior(crop)
                        px, py = ix + x0, iy + y0
                    px = (float(px), float(py))
                else:
                    px = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            else:
                px = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            grounded.setdefault(lab, []).append((b, px, float(conf)))
        # per-label NMS: YOLOE returns overlapping fragments for one class.
        for lab, dets in list(grounded.items()):
            dets.sort(key=lambda d: -d[2])          # highest conf first
            grounded[lab] = _nms(dets)
        return grounded, inventory

    def observe(self, folder: Path) -> SceneGraph:  # noqa: D102
        with self._lock:
            started = time.monotonic()
            image_path = Path(folder) / "rgb.png"
            image_bytes = image_path.read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            with image:
                scaled = image.copy()
                scaled.thumbnail((JPEG_LONG_EDGE, JPEG_LONG_EDGE), Image.Resampling.LANCZOS)
                with scaled, io.BytesIO() as enc:
                    scaled.save(enc, format="JPEG", quality=92)
                    url = "data:image/jpeg;base64," + base64.b64encode(
                        enc.getvalue()).decode("ascii")
                inventory = self._label_inventory(url)
                grounded, _ = self._ground(image, inventory)

                # Detection supplies geometry only; preserve each model-reported
                # affordance and transient index so visual relations stay grounded.
                seen = []
                used = set()
                pairing_warnings = []
                named = {}
                for item in inventory.objects:
                    label = f"{item.color.strip()} {item.label.strip()}".strip()
                    named.setdefault(label, []).append(item)
                for label, items in named.items():
                    detections = sorted(grounded.get(label, ()), key=lambda det: _center(det[0]))
                    if len(items) > 1 and len(detections) > 1:
                        pairing_warnings.append(
                            f"Pairing {len(items)} '{label}' items with {len(detections)} "
                            "detections by spatial order; the labeler emits no coordinates, "
                            "so per-instance pairing is unverified.")
                    for item, (box, pixel, confidence) in zip(items, detections, strict=False):
                        seen.append(_Seen(
                            index=item.index, label=item.label, color=item.color, noun=item.noun,
                            movable=item.movable, graspable=item.graspable,
                            receives_objects=item.receives_objects,
                            box=[box[0] / image.width * 1000, box[1] / image.height * 1000,
                                 box[2] / image.width * 1000, box[3] / image.height * 1000],
                            point=[pixel[0] / image.width * 1000, pixel[1] / image.height * 1000],
                            confidence=min(item.confidence, confidence),
                        ))
                        used.add(item.index)
                inventory2 = _Inventory(objects=seen, relations=inventory.relations,
                                        warnings=inventory.warnings)

                objects, dropped = _observations(inventory2, image.width, image.height)
                warnings = [
                    "Confidence values are model estimates, not calibrated probabilities.",
                    "Image-plane observations do not establish exact 3-D geometry or gripper fit.",
                    "Detection-grounded: labels and affordances from the VLM, geometry from a detector.",
                    *pairing_warnings,
                    *[f"Dropped unlocalized object {item.index} ({item.label}); no geometry was fabricated."
                      for item in inventory.objects if item.index not in used],
                ]
                links = _visual_links(inventory2, objects, warnings)
                try:
                    warnings.extend(inventory.warnings)
                    warnings.extend(dropped)
                    protected = {frozenset((link.subject, link.object)) for link in links
                                 if link.confidence >= MIN_RELATION_CONFIDENCE}
                    objects, aliases = prune_duplicates(objects, protected)
                    if self.segment:
                        objects = self._refine(image, objects, warnings)
                    objects = self._reconcile_colours(image, objects, warnings)
                    objects, remap = deterministic_ids(objects)
                    relations = {}
                    for link in links:
                        sub, dst = remap[aliases[link.subject]], remap[aliases[link.object]]
                        if sub == dst:
                            continue
                        rel = link.model_copy(update={"subject": sub, "object": dst})
                        key = (rel.subject, rel.predicate, rel.object)
                        if key not in relations or relations[key].confidence < rel.confidence:
                            relations[key] = rel
                    for rel in numeric_relations(objects):
                        key = (rel.subject, rel.predicate, rel.object)
                        # numeric relations fill gaps only; a VLM's visually-
                        # evidenced relation for the same triple wins.
                        if key not in relations:
                            relations[key] = rel
                    relations = _break_support_cycles(relations, warnings)
                    relations = resolve_overlaps_by_mask(
                        objects, image.width, image.height, relations, warnings)
                    relations = _break_support_cycles(relations, warnings)
                    relations = _drop_impossible_support(objects, relations, warnings)
                    if not objects:
                        warnings.append("No visible objects were returned.")
                    for o in objects:
                        if o.confidence < MIN_OBJECT_CONFIDENCE:
                            warnings.append(f"Low-confidence object: {o.id} ({o.confidence:.2f}).")
                    objects = self._attach_forms(image, objects, warnings)
                    return SceneGraph(
                        scene=Path(folder).name, width=image.width, height=image.height,
                        image_sha256=image_hash, objects=objects,
                        relations=tuple(relations[k] for k in sorted(relations)),
                        warnings=tuple(dict.fromkeys(warnings)),
                        perception_model=self.client.model + "+yoloe26",
                        perception_seconds=time.monotonic() - started)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"Hybrid observation malformed: {exc}") from exc
class FusedPerception(Perception):
    """Perception backend: strict VLM inventory kept authoritative for labels and
    relations; YOLOE-26 re-grounds each observed object's box/point in original
    pixels so drifted VLM geometry cannot drop the pick point onto background.

    Objects the detector does not confirm keep their VLM geometry — the strict
    contract still rules; the detector can only sharpen, never invent or delete.
    """

    def __init__(self, client, segment: bool = True, device: str = "cpu",
                 detector: str = "yoloe-26s-seg.pt", conf: float = 0.02):
        super().__init__(client, segment=segment, device=device)
        self._detector_name = detector
        self._conf = conf
        self._yoloe = None

    @property
    def signature(self) -> str:  # noqa: D102
        base = json.loads(super().signature)
        base["backend"] = "fused"
        base["detector"] = self._detector_name
        base["conf"] = self._conf
        return json.dumps(base, sort_keys=True, separators=(",", ":"))

    def _detector(self):
        if self._yoloe is None:
            from ultralytics import YOLOE
            self._yoloe = YOLOE(_detector_path(self._detector_name))
        return self._yoloe

    def release(self) -> None:
        super().release()
        self._yoloe = None

    def _ground_objects(self, image, objects, warnings):
        if not objects:
            return objects
        det = self._detector()
        img_h, img_w = image.height, image.width
        generic = sorted({o.label for o in objects})
        det.set_classes(generic)
        # Pass the PIL image: ultralytics treats an ndarray as BGR, which would
        # channel-swap every detection. PIL preserves the true RGB channels.
        res = det.predict(image, conf=self._conf, verbose=False)[0]
        names = res.names
        per_class: dict[str, list] = {k: [] for k in generic}
        if res.boxes is not None:
            for ci, box, conf in zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf, strict=False):
                per_class[names[int(ci)]].append(
                    ([float(v) for v in box.tolist()], float(conf)))
        by_key = {k: _nms(sorted(v, key=lambda d: -d[1]))
                  for k, v in per_class.items()}

        # Match each observed object to a same-class detector proposal using
        # geometric overlap, point containment, and scale consistency.
        updated = []
        used: set[tuple[str, int]] = set()
        for obj in objects:
            best = None  # (di, dbox, dconf, strength)
            area_obj = max(1.0, (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]))
            total_img = img_h * img_w
            for di, (dbox, dconf) in enumerate(by_key.get(obj.label, [])):
                if (obj.label, di) in used:
                    continue
                iou = _iou(obj.box, dbox)
                holds_pt = _inside(dbox, obj.pixel)
                area_det = max(1.0, (dbox[2] - dbox[0]) * (dbox[3] - dbox[1]))
                area_ratio = max(area_det / area_obj, area_obj / area_det)
                if not box_has_plausible_extent(obj.box, img_w, img_h):
                    # The VLM's localisation for this object is degenerate (e.g. a
                    # near-zero patch from a 0..1 fraction). Overlap with it is
                    # meaningless, and the scale gates below would reject every real
                    # detection because the ratio against a 1-3 px box is enormous, so
                    # this branch must be evaluated FIRST. Accept the best same-class
                    # detection that is itself a plausible extent — the detector has
                    # actually seen the image.
                    if not box_has_plausible_extent(
                            tuple(round(v) for v in dbox), img_w, img_h):
                        continue
                    strength = 1.0
                else:
                    # A detector proposal cannot explode the object into an enclosing
                    # background surface/table (e.g. 45x larger area with tiny IoU):
                    if area_ratio > 3.0 and iou < 0.25:
                        continue
                    if obj.movable and area_det > 0.20 * total_img and area_obj < 0.10 * total_img:
                        continue
                    if iou < 0.15 and not holds_pt and not _inside(obj.box, _center(dbox)):
                        continue
                    strength = iou + (0.5 if holds_pt else 0.0)
                if best is None or strength > best[3]:
                    best = (di, dbox, dconf, strength)
            if best is not None:
                di, dbox, dconf, strength = best
                used.add((obj.label, di))
                ibox = tuple(round(v) for v in dbox)  # SAM2 masks index by ints
                if ibox[2] <= ibox[0] or ibox[3] <= ibox[1]:
                    # A degenerate detection cannot hold a pixel; leave the object's
                    # VLM geometry alone rather than emitting an invalid observation.
                    updated.append(obj)
                    continue
                # A detector box centre can sit on background for concave or
                # annular shapes, and rounding a narrow float box can push the
                # computed centre outside the integer box. Keep the VLM pixel while
                # the detector box still contains it; otherwise hand a clamped
                # in-box point to SAM2, whose box-only fallback selects an actual
                # visible-mask interior.
                if _inside(ibox, obj.pixel):
                    pixel = obj.pixel
                else:
                    cx, cy = _center(dbox)
                    pixel = (min(ibox[2] - 1, max(ibox[0], round(cx))),
                             min(ibox[3] - 1, max(ibox[1], round(cy))))
                updated.append(obj.model_copy(update={
                    "box": ibox,
                    "pixel": pixel,
                    "confidence": max(obj.confidence, min(dconf, 1.0)),
                    "evidence": obj.evidence
                        + f"; YOLOE re-grounded '{obj.label}' "
                          f"(IoU {_iou(obj.box, dbox):.2f})",
                }))
            else:
                updated.append(obj)
        # Name the leftovers rather than only counting them: the audit then records what
        # the detector saw that the inventory did not, with the class it was queried
        # under and its share of the frame, which is what a reviewer needs to judge
        # whether the strict-label contract is discarding real instances.
        leftover_detail = []
        for key, proposals in by_key.items():
            for di, (dbox, dconf) in enumerate(proposals):
                if (key, di) in used:
                    continue
                area = ((dbox[2] - dbox[0]) * (dbox[3] - dbox[1])) / max(1.0, img_h * img_w)
                leftover_detail.append(
                    f"{key} {[round(v) for v in dbox]} ({area:.1%} of frame, conf {dconf:.2f})")
        if leftover_detail:
            warnings.append(
                f"Detector saw {len(leftover_detail)} extra object(s) not in the VLM "
                "inventory; ignored per strict-label contract: " + "; ".join(leftover_detail[:6]))
        return tuple(updated)




