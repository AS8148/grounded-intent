"""Behavioural coverage for image-only support discovery.

No local models, no benchmark scenes: synthetic masks, injected observations and a
scripted observer. The rules under test are the ones that decide whether a proposed region
may enter the scene at all, how the accepted one is named, and the two geometry prefilters
that exist because an earlier version of this pass accepted a container wall and a mug
handle on the held-out scenes.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from grounded_intent.clients import ModelCallError
from grounded_intent.domain import ObjectObservation, Relation
from grounded_intent.support_discovery import (
    GENERIC_COMPONENT_NOUN,
    MAX_CANDIDATES,
    MAX_SUPPORT_ASPECT,
    _candidates,
    _chain,
    _elongated,
    _Region,
    discover_supports,
    signature,
)

SIZE = (400, 300)


def observation(oid, box, *, movable=True, receives=False, polygon=(), confidence=0.9):
    x0, y0, x1, y1 = box
    return ObjectObservation(
        id=oid, label=oid.rsplit("_", 1)[0], noun=oid.rsplit("_", 1)[0], color="grey",
        box=box, pixel=(x0 + 4, y0 + 4), polygon=polygon, confidence=confidence,
        movable=movable, graspable=movable, receives_objects=receives,
    )


def region(**overrides):
    base = {
        "physical_kind": "independent_object", "ordinary_name": "tray insert part",
        "color": "silver", "visible_structure": "flat plate with two cut holes",
        "boundary_evidence": "outline follows a physical edge",
        "confidence": 0.95, "movable": True, "graspable": True,
        "receives_objects": False, "relations_to_region": [],
    }
    base.update(overrides)
    return _Region.model_validate(base)


def relation(subject, obj, predicate="on", confidence=0.95, evidence="visible contact"):
    return {"subject_id": subject, "object_id": obj, "predicate": predicate,
            "confidence": confidence, "evidence": evidence}


def chain(candidate="candidate", rider="red_cube", surface="grey_tray"):
    return region(relations_to_region=[relation(candidate, surface),
                                       relation(rider, candidate)])


def scene():
    tray = observation("grey_tray", (0, 0, 400, 300), movable=False, receives=True)
    cube = observation("red_cube", (40, 40, 140, 140))
    piece = observation("grey_part", (120, 180, 260, 220))
    return tray, cube, piece


# --- the acceptance rule ------------------------------------------------------


def test_complete_support_chain_is_accepted():
    tray, cube, piece = scene()
    verified = _chain(chain(), piece.id, (tray, cube, piece), {tray.id, cube.id})
    assert [(r.subject, r.predicate, r.object) for r in verified] == [
        ("red_cube", "on", "grey_part")]


def test_container_wall_answer_is_rejected():
    # The recorded held-out control: the model calls the region a wall, claims the
    # container rests on it (inverted) and marks the candidate's own relation "unknown".
    tray = observation("grey_tray", (0, 0, 400, 300), movable=False, receives=True)
    piece = observation("grey_part", (300, 20, 340, 280))
    wall = region(
        physical_kind="container_component", ordinary_name="tray wall or side panel",
        visible_structure="vertical side panel", relations_to_region=[
            relation("grey_tray", "candidate"),
            relation("candidate", "grey_tray", predicate="unknown")])
    assert _chain(wall, piece.id, (tray, piece), {tray.id}) == []


def test_occluder_only_is_not_support():
    tray, cube, piece = scene()
    occluding = region(relations_to_region=[
        relation("candidate", "grey_tray", predicate="inside"),
        relation("red_cube", "candidate", predicate="blocks")])
    assert _chain(occluding, piece.id, (tray, cube, piece), {tray.id, cube.id}) == []


def test_missing_receiving_surface_is_rejected():
    cube, piece = scene()[1:]
    only_rider = region(relations_to_region=[relation("candidate", "red_cube"),
                                             relation("red_cube", "candidate")])
    assert _chain(only_rider, piece.id, (cube, piece), {cube.id}) == []


def test_below_confidence_bar_is_rejected():
    tray, cube, piece = scene()
    weak = region(relations_to_region=[
        relation("candidate", "grey_tray", confidence=0.69),
        relation("red_cube", "candidate", confidence=0.95)])
    assert _chain(weak, piece.id, (tray, cube, piece), {tray.id, cube.id}) == []


def test_unrelated_and_unknown_ids_are_ignored():
    tray, cube, piece = scene()
    noisy = region(relations_to_region=[
        relation("candidate", "grey_tray"),
        relation("red_cube", "candidate"),
        relation("red_cube", "grey_tray"),      # unrelated pair
        relation("ghost_part", "candidate"),    # id that does not exist
        relation("candidate", "candidate")])    # self relation
    verified = _chain(noisy, piece.id, (tray, cube, piece), {tray.id, cube.id})
    assert [(r.subject, r.object) for r in verified] == [("red_cube", "grey_part")]
    assert all(r.predicate == "on" for r in verified)


def test_immovable_rider_is_not_a_holder():
    tray = observation("grey_tray", (0, 0, 400, 300), movable=False, receives=True)
    plate = observation("white_plate", (150, 150, 250, 250), movable=False)
    piece = observation("grey_part", (120, 180, 260, 220))
    answer = region(relations_to_region=[relation("candidate", "grey_tray"),
                                         relation("white_plate", "candidate")])
    assert _chain(answer, piece.id, (tray, plate, piece), {tray.id, plate.id}) == []


def test_signature_names_the_prefilters_it_applies():
    first = signature()
    assert first["accepts"] and first["max_candidates"] == MAX_CANDIDATES
    assert first["component_noun"] == GENERIC_COMPONENT_NOUN
    assert first["max_support_aspect"] == MAX_SUPPORT_ASPECT
    assert any("thin strips" in entry for entry in first["rejects"])


# --- the two geometry prefilters ---------------------------------------------


def test_thin_strips_are_recognised_as_container_edges():
    # Measured: the accepted held-out wall was 22.4:1, the genuine support 1.1:1.
    assert _elongated((1147, 170, 1218, 1758))
    assert not _elongated((361, 721, 813, 924))
    assert not _elongated((975, 927, 1149, 1118))


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    @property
    def shape(self):
        return self.value.shape

    def __getitem__(self, item):
        return _Tensor(self.value[item])


class _Batch(dict):
    def to(self, _device):
        return self


class _Processor:
    def __call__(self, images=None, original_sizes=None, **kwargs):
        if images is not None:
            return _Batch({"pixel_values": _Tensor(np.zeros((1, 3, 8, 8))),
                           "original_sizes": _Tensor([[SIZE[1], SIZE[0]]])})
        return _Batch({key: _Tensor(kwargs[key])
                       for key in ("input_points", "input_labels")})

    def post_process_masks(self, masks, original_sizes):
        return [masks[0]]


class _SAM:
    """Point-sensitive double: each prompt returns the rectangle containing it."""

    def __init__(self, rectangles):
        self.rectangles = rectangles

    def get_image_embeddings(self, pixel_values):
        return object()

    def __call__(self, image_embeddings, input_points, input_labels, multimask_output=False):
        masks = []
        for x, y in np.asarray(input_points.numpy()).reshape(-1, 2):
            mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
            for x0, y0, x1, y1 in self.rectangles:
                if x0 <= float(x) < x1 and y0 <= float(y) < y1:
                    mask[y0:y1, x0:x1] = 1
            masks.append(mask[None])
        return type("Out", (), {"pred_masks": _Tensor(np.stack(masks, axis=0)[None])})()


class _Torch:
    class inference_mode:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False


def test_a_wall_shaped_proposal_never_reaches_the_model():
    # 30 x 252 px on 400 x 300: 8.4:1, wide enough that a lattice point lands on it and
    # tall enough to pass the extent rule, so only the elongation prefilter can reject it.
    tray = observation("grey_tray", (0, 0, 400, 300), movable=False, receives=True)
    warnings: list[str] = []
    novel = _candidates(Image.new("RGB", SIZE, "black"), (tray,), _Torch, _Processor(),
                        _SAM([(224, 24, 254, 276)]), "cpu", warnings)
    assert novel == []
    assert any("thin strip" in w for w in warnings)


def test_a_region_inside_a_movable_object_never_reaches_the_model():
    # The mug's box is far larger than the region, so the coarse IoU filter cannot catch
    # it; the region's own mask lies inside the mug's outline, which is what T2 tests.
    tray = observation("grey_tray", (0, 0, 400, 300), movable=False, receives=True)
    mug = observation("green_mug", (100, 20, 390, 290),
                      polygon=((100, 20), (390, 20), (390, 290), (100, 290)))
    warnings: list[str] = []
    novel = _candidates(Image.new("RGB", SIZE, "black"), (tray, mug), _Torch, _Processor(),
                        _SAM([(224, 96, 272, 144)]), "cpu", warnings)
    assert novel == []
    assert any("movable object" in w for w in warnings)


def test_a_region_inside_a_non_movable_container_survives():
    # The genuine case: the support lies inside the tray, which is not movable.
    tray = observation("grey_tray", (0, 0, 400, 300), movable=False, receives=True)
    cube = observation("red_cube", (40, 40, 140, 140))
    novel = _candidates(Image.new("RGB", SIZE, "black"), (tray, cube), _Torch,
                        _Processor(), _SAM([(160, 200, 300, 250)]), "cpu", [])
    assert len(novel) == 1 and novel[0]["box"][0] >= 150


# --- the end-to-end pass ------------------------------------------------------


class ScriptedObserver:
    def __init__(self, reply, *, fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    def __call__(self, messages, schema=None, max_tokens=1024, seed=42):
        self.calls.append((messages, schema))
        if self.fail:
            raise ModelCallError("endpoint refused the request")
        return self.reply, {"prompt_tokens": 1, "completion_tokens": 1}


def _run(rectangles, observer, objects, relations=()):
    warnings: list[str] = []
    result = discover_supports(
        Image.new("RGB", SIZE, "black"), objects, relations, torch_module=_Torch,
        processor=_Processor(), sam_model=_SAM(rectangles), device="cpu",
        observer=observer, warnings=warnings)
    return result, warnings


PIECE_REPLY = {
    "physical_kind": "container_component", "ordinary_name": "tray insert part",
    "color": "silver", "visible_structure": "flat plate with holes",
    "boundary_evidence": "outline follows a physical edge", "confidence": 0.95,
    "movable": True, "graspable": True, "receives_objects": False,
    "relations_to_region": [relation("candidate", "grey_tray"),
                            relation("red_cube", "candidate")],
}


def test_accepted_piece_preserves_existing_identities_and_adds_the_verified_edge():
    tray, cube, _ = scene()
    (objects, relations), warnings = _run([(160, 200, 300, 250)],
                                          ScriptedObserver(PIECE_REPLY), (tray, cube))
    assert [obj.id for obj in objects[:2]] == ["grey_tray", "red_cube"]
    assert len(objects) == 3
    assert [(r.subject, r.predicate, r.object) for r in relations] == [
        ("red_cube", "on", objects[2].id)]
    assert objects[2].polygon and objects[2].box[0] >= 150
    assert any("merged away" in warning for warning in warnings)


def test_accepted_piece_is_named_by_its_role_not_by_the_model_wording():
    tray, cube, _ = scene()
    for wording in ("tray insert part", "tray bottom", "metal plate"):
        (objects, _), _ = _run([(160, 200, 300, 250)],
                               ScriptedObserver({**PIECE_REPLY, "ordinary_name": wording}),
                               (tray, cube))
        added = objects[-1]
        assert added.noun == GENERIC_COMPONENT_NOUN and added.label == GENERIC_COMPONENT_NOUN
        assert added.id.startswith("silver_") and added.movable is True
        assert wording in added.visual_form and wording in added.evidence


def test_rejected_proposal_leaves_the_graph_exactly_as_it_was():
    tray, cube, _ = scene()
    existing = Relation(subject="red_cube", predicate="on", object="grey_tray",
                        confidence=0.9, evidence="observed")
    wall = {**PIECE_REPLY, "ordinary_name": "tray wall",
            "relations_to_region": [relation("grey_tray", "candidate"),
                                    relation("candidate", "grey_tray", predicate="unknown")]}
    (objects, relations), warnings = _run([(160, 200, 300, 250)],
                                          ScriptedObserver(wall), (tray, cube), (existing,))
    assert objects == (tray, cube) and relations == (existing,)
    assert any("no verified support chain" in warning for warning in warnings)


def test_endpoint_failure_is_reported_and_propagated():
    tray, cube, _ = scene()
    with pytest.raises(ModelCallError):
        _run([(160, 200, 300, 250)], ScriptedObserver({}, fail=True), (tray, cube))


def test_no_receiving_surface_means_no_proposals_and_no_calls():
    cube = scene()[1]
    observer = ScriptedObserver({})
    (objects, relations), warnings = _run([(160, 200, 300, 250)], observer, (cube,))
    assert (objects, relations) == ((cube,), ())
    assert observer.calls == [] and warnings == []


def test_duplicate_proposals_collapse_to_one_candidate():
    tray, cube, _ = scene()
    observer = ScriptedObserver({**PIECE_REPLY, "ordinary_name": "insert"})
    (objects, _), _ = _run([(160, 200, 300, 250)], observer, (tray, cube))
    assert len(objects) == 3
    assert len(observer.calls) == 1  # one outline question per region, not per point
