from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from grounded_intent.domain import ObjectObservation, Relation
from grounded_intent.perception import (
    _FORM_SYSTEM,
    JPEG_LONG_EDGE,
    SAM2_MODEL,
    SAM2_REVISION,
    Perception,
    resolve_overlaps_by_mask,
)


class FakeClient:
    model = "local-test-vision"

    def __init__(self, response):
        self.response = response
        self.requests = []

    def complete(self, messages, schema=None, max_tokens=1024, seed=42):
        self.requests.append((messages, schema, max_tokens))
        return self.response, {"prompt_tokens": 20, "completion_tokens": 40}


def item(index=0, label="gear", color="red", box=None, point=None, confidence=0.9,
         movable=True, receives_objects=False, graspable=None):
    return {
        "index": index, "label": label, "color": color, "noun": label,
        "movable": movable, "graspable": movable if graspable is None else graspable,
        "receives_objects": receives_objects,
        "box": box if box is not None else [100, 100, 300, 300],
        "point": point if point is not None else [200, 200], "confidence": confidence,
    }


def inventory(objects=None, relations=None):
    return {
        "objects": objects if objects is not None else [item()],
        "relations": relations if relations is not None else [], "warnings": [],
    }


def link(subject, destination, predicate="on", basis="visual_stack", confidence=0.9):
    return {
        "subject": subject, "object": destination, "predicate": predicate,
        "basis": basis, "confidence": confidence,
        "evidence": "visible elevated layer and occluding contact edge",
    }


@pytest.fixture
def folder(tmp_path):
    scene = tmp_path / "scene_private_id"
    scene.mkdir()
    Image.new("RGB", (200, 100), (150, 150, 150)).save(scene / "rgb.png")
    return scene


def test_observe_image_hash_original_dimensions_and_normalized_geometry(folder):
    client = FakeClient(inventory())
    graph = Perception(client, segment=False).observe(folder)
    assert (graph.width, graph.height) == (200, 100)
    assert graph.image_sha256 == hashlib.sha256((folder / "rgb.png").read_bytes()).hexdigest()
    assert graph.scene == folder.name
    assert graph.perception_model == client.model
    assert graph.perception_seconds >= 0
    assert graph.objects[0].box == (20, 10, 60, 30)
    assert graph.objects[0].pixel == (40, 20)
    assert graph.objects[0].id == "red_gear"
    assert graph.objects[0].polygon == ()
    assert any("not calibrated" in warning for warning in graph.warnings)
    assert any("disabled explicitly" in warning for warning in graph.warnings)


def test_request_contains_no_scene_name_category_task_id_or_gold(folder, monkeypatch):
    for filename in ("ground_truth.json", "instructions.csv", "category.json", "calibration.json"):
        (folder / filename).write_text("DO_NOT_LEAK_PRIVATE_SECRET", encoding="utf-8")
    from pathlib import Path
    original = Path.read_bytes
    opened = []

    def guarded(path):
        opened.append(path.name)
        assert path.name == "rgb.png"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    client = FakeClient(inventory())
    Perception(client, segment=False).observe(folder)
    serialized = json.dumps(client.requests[0][0])
    assert "DO_NOT_LEAK_PRIVATE_SECRET" not in serialized
    assert folder.name not in serialized
    assert "instruction_id" not in serialized
    assert "ground_truth" not in serialized
    assert opened == ["rgb.png"]
    assert client.requests[0][1]["additionalProperties"] is False
    assert client.requests[0][2] >= 2300


def test_scaled_jpeg_preserves_original_frame_and_long_edge(folder):
    Image.new("RGB", (4000, 3000), "white").save(folder / "rgb.png")
    client = FakeClient(inventory())
    graph = Perception(client, segment=False).observe(folder)
    data_url = client.requests[0][0][1]["content"][1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as thumbnail:
        assert thumbnail.format == "JPEG"
        assert thumbnail.size == (JPEG_LONG_EDGE, JPEG_LONG_EDGE * 3 // 4)
    assert (graph.width, graph.height) == (4000, 3000)
    assert graph.objects[0].pixel == (800, 600)
    assert graph.objects[0].box == (400, 300, 1200, 900)


def test_small_image_not_upscaled(folder):
    client = FakeClient(inventory())
    Perception(client, segment=False).observe(folder)
    url = client.requests[0][0][1]["content"][1]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as thumbnail:
        assert thumbnail.size == (200, 100)


def test_model_reported_movability_cannot_override_the_canonical_noun(folder):
    # A broad work surface is a fixed placement area by its canonical noun, so the
    # model's own movable estimate is overridden rather than preserved: the surface
    # stays immovable and a movable part stays movable, whatever the model said.
    response = inventory([
        item(0, "composite_work_surface", "blue", movable=True, receives_objects=True),
        item(1, "spring_clamp", "orange", movable=False),
    ])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    surface = graph.by_id()["blue_tray"]
    assert surface.label == "tray" and surface.noun == "tray"
    assert surface.movable is False
    assert surface.receives_objects is True
    assert graph.by_id()["orange_spring_clamp"].movable is True


def test_stable_ids_do_not_depend_on_model_inventory_order_and_links_remap(folder):
    left = item(19, label="spring_clamp", color="cobalt", box=[100, 100, 300, 300],
                point=[200, 200])
    right = item(7, label="spring_clamp", color="cobalt", box=[600, 100, 800, 300],
                 point=[700, 200])
    surface = item(3, "composite_work_surface", "tan", [0, 50, 1000, 900],
                   [500, 600], movable=False, receives_objects=True)
    response = inventory([right, surface, left], [link(19, 3, basis="visual_support")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    reverse = Perception(FakeClient(inventory(
        [left, surface, right], [link(19, 3, basis="visual_support")],
    )), segment=False).observe(folder)
    assert graph.objects == reverse.objects and graph.relations == reverse.relations
    assert graph.by_id()["cobalt_spring_clamp_1"].pixel == (40, 20)
    assert graph.by_id()["cobalt_spring_clamp_2"].pixel == (140, 20)
    assert any(r.subject == "cobalt_spring_clamp_1"
               and r.object == "brown_tray" and r.predicate == "on"
               for r in graph.relations)


def test_overlap_alone_does_not_prove_stack_or_containment(folder):
    response = inventory([
        item(0, box=[100, 100, 600, 600], point=[200, 200]),
        item(1, color="yellow", box=[200, 200, 700, 700], point=[600, 600]),
    ], [link(0, 1, basis="overlap"), link(0, 1, predicate="inside", basis="overlap")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert {relation.predicate for relation in graph.relations} == {"left_of", "above"}
    assert any("not positive physical evidence" in warning for warning in graph.warnings)

def test_visible_support_does_not_require_an_immovable_support(folder):
    response = inventory([
        item(0),
        item(1, "cube", "silver", movable=True, graspable=True,
             receives_objects=True),
    ], [link(0, 1, basis="visual_support")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert any(relation.predicate == "on" for relation in graph.relations)
    assert graph.by_id()["silver_cube"].movable is True


def test_visibly_stacked_same_class_objects_are_not_pruned(folder):
    response = inventory([item(5), item(9)], [link(5, 9), link(5, 9, predicate="blocks")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert len(graph.objects) == 2
    assert {r.predicate for r in graph.relations} == {"on", "blocks"}


def test_support_link_with_a_sliver_footprint_is_dropped(folder):
    # A subject keeping only a corner of itself inside the supporter is adjacent or
    # occluded, not resting on it; keeping the link moves the scored first pick onto
    # an object that was never in the way.
    response = inventory([
        item(0, "gear", "red", [100, 500, 500, 900], [300, 700]),
        item(1, "fruit", "orange", [490, 480, 640, 630], [560, 550]),
    ], [link(1, 0, basis="visual_support")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert not any(r.predicate == "on" for r in graph.relations)
    assert any("sliver support" in warning for warning in graph.warnings)


def test_support_link_with_a_real_footprint_is_kept(folder):
    response = inventory([
        item(0, "gear", "red", [100, 500, 500, 900], [300, 700]),
        item(1, "cube", "red", [200, 550, 350, 700], [275, 625]),
    ], [link(1, 0, basis="visual_support")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert any(r.predicate == "on" for r in graph.relations)


def test_positive_containment_is_separate_from_stack_evidence(folder):
    response = inventory([
        item(0), item(1, "tray", "grey", [0, 0, 1000, 1000], [600, 600]),
    ], [link(0, 1, predicate="inside", basis="visual_containment")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    inside = [relation for relation in graph.relations if relation.predicate == "inside"]
    assert len(inside) == 1 and "visual_containment" in inside[0].evidence
    assert not any(relation.predicate == "on" for relation in graph.relations)


def test_duplicates_pruned_and_relation_aliases_reference_kept_object(folder):
    response = inventory([
        item(0, confidence=0.5), item(1, confidence=0.9),
        item(2, "tray", "grey", [0, 400, 1000, 1000], [500, 700]),
    ], [link(0, 2, basis="visual_support", confidence=0.5)])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert len(graph.objects) == 2
    assert graph.by_id()["red_gear"].confidence == 0.9
    assert any(relation.subject == "red_gear" and relation.object == "grey_tray"
               for relation in graph.relations)
    assert any("duplicates" in warning for warning in graph.warnings)


def test_low_confidence_not_hidden_and_physical_confidence_capped(folder):
    response = inventory([
        item(0, confidence=0.4), item(1, "receiver", "grey", [0, 0, 1000, 1000],
                                      [600, 600], movable=False, receives_objects=True),
    ], [link(0, 1, basis="visual_support", confidence=0.95)])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert graph.by_id()["red_gear"].confidence == 0.4
    assert next(r for r in graph.relations if r.predicate == "on").confidence == 0.4
    assert any("Low-confidence object" in warning for warning in graph.warnings)
    assert any("Low-confidence physical relation" in warning for warning in graph.warnings)


def test_empty_scene_valid_with_warning_no_fabricated_object_or_sam_load(folder, monkeypatch):
    perception = Perception(FakeClient(inventory([])), segment=True)
    monkeypatch.setattr(perception, "_load_sam", lambda: pytest.fail("empty scene must not load SAM"))
    graph = perception.observe(folder)
    assert graph.objects == () and graph.relations == ()
    assert any("No visible objects" in warning for warning in graph.warnings)


@pytest.mark.parametrize("bad", [
    {}, {"objects": [], "relations": [], "warnings": [], "extra": True},
    inventory([item(confidence=1.1)]), inventory([item(confidence=float("nan"))]),
    inventory([item(box=[True, 100, 300, 300])]),
])
def test_malformed_observation_raises_clear_error_not_fake_graph(folder, bad):
    with pytest.raises(RuntimeError, match="Malformed image-only observation"):
        Perception(FakeClient(bad), segment=False).observe(folder)


@pytest.mark.parametrize("bad", [
    inventory([item(box=[100, 100, 100, 300])]),
    inventory([item(box=[1200, 100, 1300, 300])]),
    inventory([item(point=[600, 600])]),
    inventory([item(point=[-5, 600])]),
    inventory([item(label="!!!")]),
    inventory([item(color="???")]),
    inventory([item()], [link(0, 8)]), inventory([item()], [link(0, 0)]),
    inventory([item(), item()]),
])
def test_degenerate_items_dropped_with_warning_not_fake_object(folder, bad):
    graph = Perception(FakeClient(bad), segment=False).observe(folder)
    assert len(graph.objects) <= 1
    assert any("Dropped" in warning or "dropped" in warning for warning in graph.warnings)


def test_slight_out_of_frame_coordinates_clamped_consistently(folder):
    graph = Perception(FakeClient(inventory([
        item(box=[-20, -20, 1020, 1020], point=[1000, 1000]),
    ])), segment=False).observe(folder)
    assert graph.objects[0].box == (0, 0, 200, 100)
    assert graph.objects[0].pixel == (199, 99)


def test_missing_or_corrupt_image_is_not_empty_scene(folder):
    (folder / "rgb.png").unlink()
    with pytest.raises(RuntimeError, match="Cannot read perception image"):
        Perception(FakeClient(inventory()), segment=False).observe(folder)
    (folder / "rgb.png").write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="Cannot read perception image"):
        Perception(FakeClient(inventory()), segment=False).observe(folder)


def test_model_server_failure_propagates_without_inventory_fallback(folder):
    class BrokenClient(FakeClient):
        def complete(self, *args, **kwargs):
            raise RuntimeError("local vision server unavailable")

    with pytest.raises(RuntimeError, match="local vision server unavailable"):
        Perception(BrokenClient({}), segment=False).observe(folder)


def test_cache_signature_tracks_model_segmentation_and_device():
    client = FakeClient(inventory())
    base = Perception(client).signature
    assert base == Perception(client).signature
    assert base != Perception(client, segment=False).signature
    assert base != Perception(client, device="cuda").signature
    client.model = "other-local-vision"
    assert base != Perception(client).signature
    assert SAM2_MODEL in base


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __getitem__(self, index):
        return FakeTensor(self.array[index])


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.image_calls = 0
        self.prompt_calls = []
        self.postprocess_shapes = []

    def __call__(self, images=None, original_sizes=None, **kwargs):
        if images is not None:
            self.image_calls += 1
            return FakeBatch(
                pixel_values=FakeTensor(np.zeros((1, 3, 16, 16))),
                original_sizes=FakeTensor([[images.height, images.width]]),
            )
        self.prompt_calls.append(kwargs)
        return FakeBatch({
            key: FakeTensor(kwargs[key])
            for key in ("input_boxes", "input_points", "input_labels")
            if key in kwargs
        }, original_sizes=original_sizes)

    def post_process_masks(self, masks, original_sizes):
        self.postprocess_shapes.append(masks.shape)
        return [masks[0]]


class FakeSAM:
    def __init__(self, mask, *, corrupt_dimensions=False):
        self.mask = mask
        self.encode_calls = 0
        self.decode_calls = 0
        self.corrupt_dimensions = corrupt_dimensions

    def get_image_embeddings(self, pixel_values):
        self.encode_calls += 1
        return object()

    def __call__(self, image_embeddings, input_boxes, input_points=None,
                 input_labels=None, multimask_output=False):
        self.decode_calls += 1
        assert multimask_output is False
        if input_points is not None:
            assert input_points.shape == (1, input_boxes.shape[1], 1, 2)
            assert input_labels.shape == (1, input_boxes.shape[1], 1)
        masks = np.repeat(self.mask[None, None, None], input_boxes.shape[1], axis=1)
        if self.corrupt_dimensions:
            masks = masks[:, :, :0]
        return SimpleNamespace(pred_masks=FakeTensor(masks))


def fake_segmenter(client, mask, **kwargs):
    perception = Perception(client, segment=True)
    perception._torch = SimpleNamespace(inference_mode=nullcontext)
    perception._processor = FakeProcessor()
    perception._sam = FakeSAM(mask, **kwargs)
    return perception


def test_real_mask_refinement_validates_original_point_and_avoids_hole(folder):
    mask = np.zeros((100, 200), np.uint8)
    mask[10:30, 20:60] = 1
    mask[15:25, 30:50] = 0
    perception = fake_segmenter(FakeClient(inventory([item(point=[125, 150])])), mask)
    graph = perception.observe(folder)
    obj = graph.objects[0]
    assert obj.box == (20, 10, 60, 30)
    assert mask[obj.pixel[1], obj.pixel[0]] == 1
    assert obj.pixel != (40, 20)
    assert obj.polygon
    assert obj.confidence == 0.9
    assert "distance-transform" in obj.evidence
    assert perception._processor.image_calls == perception._sam.encode_calls == 1
    assert perception._processor.postprocess_shapes == [(1, 1, 1, 100, 200)]


def test_box_only_fallback_rescues_off_object_vlm_point(folder):
    # VLM point sits in a mask hole: point-prompt SAM rejects, box-only SAM
    # recovers the object and selects an interior pixel.
    mask = np.zeros((100, 200), np.uint8)
    mask[10:30, 20:60] = 1
    mask[20, 40] = 0
    perception = fake_segmenter(FakeClient(inventory()), mask)
    graph = perception.observe(folder)
    obj = graph.objects[0]
    assert mask[obj.pixel[1], obj.pixel[0]] == 1          # pixel on the object
    assert obj.box == (20, 10, 60, 30)
    assert obj.polygon and obj.confidence == 0.9
    assert "box-only" in obj.evidence
    assert perception._sam.decode_calls == 2              # point, then box-only


def test_total_refusal_still_lowers_confidence_and_never_fakes_polygon(folder):
    # Both prompts fail (empty/degenerate mask): no snap, confidence capped.
    mask = np.zeros((100, 200), np.uint8)
    perception = fake_segmenter(FakeClient(inventory()), mask)
    graph = perception.observe(folder)
    obj = graph.objects[0]
    assert obj.pixel == (40, 20)                           # untouched VLM point
    assert obj.polygon == () and obj.confidence == 0.49
    assert "rejected" in obj.evidence


def test_mask_refinement_rejects_table_sized_mask_without_fake_polygon(folder):
    perception = fake_segmenter(FakeClient(inventory()), np.ones((100, 200), np.uint8))
    obj = perception.observe(folder).objects[0]
    assert obj.polygon == () and obj.confidence == 0.49


def test_sam_embeds_image_once_and_batches_prompts_without_full_mask_batch_resize(folder):
    # Different class labels intentionally avoid duplicate pruning of these prompts.
    response = inventory([item(index, label=f"part_{index}") for index in range(9)])
    mask = np.zeros((100, 200), np.uint8)
    mask[10:30, 20:60] = 1
    perception = fake_segmenter(FakeClient(response), mask)
    assert len(perception.observe(folder).objects) == 9
    assert perception._sam.encode_calls == 1
    assert perception._sam.decode_calls == 3
    assert perception._processor.image_calls == 1
    assert [len(call["input_boxes"][0]) for call in perception._processor.prompt_calls] == [4, 4, 1]
    assert all(shape[:3] == (1, 1, 1) for shape in perception._processor.postprocess_shapes)


def test_unexpected_sam_mask_shape_is_error_not_silent_ablation(folder):
    perception = fake_segmenter(
        FakeClient(inventory()), np.ones((100, 200), np.uint8), corrupt_dimensions=True,
    )
    with pytest.raises(RuntimeError, match="unexpected mask batch dimensions"):
        perception.observe(folder)


def test_sam_original_resolution_mismatch_is_error(folder):
    perception = fake_segmenter(FakeClient(inventory()), np.ones((20, 20), np.uint8))
    with pytest.raises(RuntimeError, match="original-image coordinates"):
        perception.observe(folder)


def test_runtime_loads_sam_locally_and_reports_missing_prepare_prerequisite(monkeypatch):
    calls = []

    class Unprepared:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            raise OSError("weights not downloaded")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Sam2Model=Unprepared, Sam2Processor=Unprepared,
    ))
    perception = Perception(FakeClient(inventory()))
    with pytest.raises(RuntimeError, match="prepare stage"):
        perception._load_sam()
    assert calls == [(SAM2_MODEL, {"local_files_only": True, "revision": SAM2_REVISION})]
    assert perception._sam is None


def test_missing_vision_dependencies_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="prerequisites missing"):
        Perception(FakeClient(inventory()))._load_sam()


def test_cuda_unavailable_never_silently_changes_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Sam2Model=object, Sam2Processor=object,
    ))
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        Perception(FakeClient(inventory()), device="cuda")._load_sam()


def test_segment_false_does_not_load_sam(folder, monkeypatch):
    perception = Perception(FakeClient(inventory()), segment=False)
    monkeypatch.setattr(perception, "_load_sam", lambda: pytest.fail("SAM was loaded"))
    assert perception.observe(folder).objects


def test_observe_requests_are_serialized_even_when_called_concurrently(folder):
    class TrackingClient(FakeClient):
        def __init__(self):
            super().__init__(inventory())
            self.active = self.maximum = 0
            self.lock = threading.Lock()

        def complete(self, *args, **kwargs):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.01)
            result = super().complete(*args, **kwargs)
            with self.lock:
                self.active -= 1
            return result

    client = TrackingClient()
    perception = Perception(client, segment=False)
    results, errors = [], []

    def observe():
        try:
            results.append(perception.observe(folder))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=observe) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(results) == 3
    assert client.maximum == 1


def test_image_inventory_preserves_independent_mobility_graspability_and_receiving(folder):
    graph = Perception(FakeClient(inventory([
        item(label="workpiece", movable=True, graspable=False, receives_objects=True),
    ])), segment=False).observe(folder)
    observed = graph.objects[0]
    assert observed.movable is True
    assert observed.graspable is False
    assert observed.receives_objects is True


def test_inventory_rejects_missing_graspability_instead_of_assuming_pickable(folder):
    observed = item()
    del observed["graspable"]
    with pytest.raises(RuntimeError, match="Malformed image-only observation"):
        Perception(FakeClient(inventory([observed])), segment=False).observe(folder)


def test_observer_does_not_promote_point_within_receiver_box_to_containment(folder):
    graph = Perception(FakeClient(inventory([
        item(0, "component", "blue", [200, 200, 400, 400], [300, 300]),
        item(1, "receiver", "tan", [0, 0, 900, 900], [800, 800],
             movable=False, receives_objects=True),
    ])), segment=False).observe(folder)
    assert len(graph.objects) == 2
    assert {relation.predicate for relation in graph.relations} <= {"left_of", "above"}


def test_hybrid_label_only_schema_and_rebuilt_affordances_preserve_relation_ids(folder, monkeypatch):
    from grounded_intent.perception_hybrid import HybridPerception

    def labeled(**kwargs):
        return {key: value for key, value in item(**kwargs).items()
                if key not in {"box", "point"}}

    response = inventory([
        labeled(index=7, label="receiver", color="tan", movable=True,
                graspable=False, receives_objects=True),
        labeled(index=19, label="component", color="blue"),
        labeled(index=33, label="unlocalized", color="grey"),
    ], [link(19, 7, predicate="inside", basis="visual_containment")])
    client = FakeClient(response)
    perception = HybridPerception(client, segment=False)
    monkeypatch.setattr(perception, "_ground", lambda image, observed: ({
        "blue component": [([20, 20, 40, 40], (30, 30), 0.85)],
        "tan receiver": [([0, 0, 150, 90], (130, 80), 0.95)],
    }, observed))
    graph = perception.observe(folder)
    assert len(graph.objects) == 2
    receiver = graph.by_id()["brown_receiver"]
    assert receiver.movable is True and receiver.graspable is False
    assert receiver.receives_objects is True
    assert graph.by_id()["blue_part"].graspable is True
    assert any(relation.subject == "blue_part" and relation.predicate == "inside"
               and relation.object == "brown_receiver" for relation in graph.relations)
    assert any("unlocalized" in warning for warning in graph.warnings)
    definition = client.requests[0][1]["$defs"]["_Labeled"]
    assert {"movable", "graspable", "receives_objects"} <= set(definition["required"])
    assert not {"box", "point"} & set(definition["properties"])


class FakeYoloDetector:
    """Records the predict() source so tests can assert channel handling."""

    def __init__(self, boxes):
        self._boxes = [np.asarray(box, dtype=float) for box in boxes]
        self.calls = []
        self.classes = None

    def set_classes(self, classes):
        self.classes = list(classes)

    def predict(self, source, conf=0.25, verbose=False):
        self.calls.append(source)
        return [SimpleNamespace(
            names={0: "gear"},
            boxes=SimpleNamespace(
                cls=np.zeros(len(self._boxes)),
                xyxy=self._boxes,
                conf=np.full(len(self._boxes), 0.9),
            ),
        )]


def fused_perception(detector):
    from grounded_intent.perception_hybrid import FusedPerception
    perception = FusedPerception(FakeClient(inventory()), segment=False)
    perception._detector = lambda: detector
    return perception


def fused_observation(*, box=(20, 20, 40, 40), pixel=(30, 35)):
    from grounded_intent.domain import ObjectObservation
    return ObjectObservation(
        id="0", label="gear", noun="gear", color="red", box=box, pixel=pixel,
        confidence=0.9, movable=True, graspable=True, receives_objects=False,
        evidence="VLM inventory",
    )


def test_fused_detector_receives_pil_image_not_bgr_ndarray():
    det = FakeYoloDetector([(10, 10, 50, 50)])
    perception = fused_perception(det)
    image = Image.new("RGB", (100, 80), (200, 30, 30))
    perception._ground_objects(image, (fused_observation(),), [])
    assert det.calls == [image]
    assert not isinstance(det.calls[0], np.ndarray), (
        "ultralytics interprets ndarray input as BGR; pass the PIL image"
    )


def test_fused_regrounding_preserves_vlm_pixel_inside_detector_box():
    det = FakeYoloDetector([(10, 10, 50, 50)])
    perception = fused_perception(det)
    image = Image.new("RGB", (100, 80), "white")
    updated = perception._ground_objects(image, (fused_observation(pixel=(30, 35)),), [])
    assert updated[0].box == (10, 10, 50, 50)
    assert updated[0].pixel == (30, 35)  # kept VLM pixel, not box centre (30, 30)


def test_fused_regrounding_falls_back_to_box_centre_when_vlm_pixel_outside_box():
    det = FakeYoloDetector([(10, 10, 50, 50)])
    perception = fused_perception(det)
    image = Image.new("RGB", (100, 80), "white")
    observation_ = fused_observation(box=(20, 20, 99, 79), pixel=(95, 75))
    updated = perception._ground_objects(image, (observation_,), [])
    assert updated[0].box == (10, 10, 50, 50)
    assert updated[0].pixel == (30, 30)


def test_fused_regrounding_rescues_a_degenerate_vlm_box_with_the_detector():
    # A near-zero VLM box (0..1 fraction instead of 0..1000) must be rescued by the
    # detector. The scale gates would reject every real detection here, so the
    # rescue has to be evaluated before them.
    det = FakeYoloDetector([(10, 10, 50, 50)])
    perception = fused_perception(det)
    image = Image.new("RGB", (400, 400), "white")
    degenerate = fused_observation(box=(0, 0, 2, 2), pixel=(0, 0))
    updated = perception._ground_objects(image, (degenerate,), [])
    assert updated[0].box == (10, 10, 50, 50)
    assert updated[0].pixel == (30, 30)


def test_fused_regrounding_ignores_a_degenerate_detection_for_a_degenerate_box():
    det = FakeYoloDetector([(0.0, 0.0, 1.0, 1.0)])
    perception = fused_perception(det)
    image = Image.new("RGB", (400, 400), "white")
    degenerate = fused_observation(box=(0, 0, 2, 2), pixel=(0, 0))
    updated = perception._ground_objects(image, (degenerate,), [])
    assert updated[0].box == (0, 0, 2, 2)  # unchanged: no plausible proposal exists


class SequencedPerceptionClient:
    model = "local-test-vision"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, schema=None, max_tokens=1024, seed=42):
        self.requests.append((messages, schema, max_tokens))
        return self.responses.pop(0), {"prompt_tokens": 1, "completion_tokens": 1}


def _degenerate_item(index=0, label="gear", color="red"):
    return item(index=index, label=label, color=color, box=[100, 100, 101, 101],
                point=[100, 100])


def test_all_degenerate_inventory_is_repaired_before_it_is_accepted(folder, monkeypatch):
    monkeypatch.setenv("PERCEPTION_FORMS", "0")  # this test counts inventory requests
    client = SequencedPerceptionClient(
        [inventory(objects=[_degenerate_item()]), inventory()])
    graph = Perception(client, segment=False).observe(folder)
    assert len(client.requests) == 2
    assert any("normalized 0..1000 frame" in str(request[0])
               for request in client.requests[1:])
    assert [obj.box for obj in graph.objects] == [(20, 10, 60, 30)]


def test_persistently_degenerate_inventory_drops_the_observations(folder):
    degenerate = inventory(objects=[_degenerate_item()])
    client = SequencedPerceptionClient([degenerate, degenerate])
    graph = Perception(client, segment=False).observe(folder)
    assert graph.objects == ()
    assert any("no plausible extent" in warning for warning in graph.warnings)


def test_dropping_a_degenerate_observation_also_drops_its_relations(folder, monkeypatch):
    monkeypatch.setenv("PERCEPTION_FORMS", "0")  # this test counts inventory requests
    # A surviving relation to a dropped object has no endpoint: keeping it raised
    # KeyError and turned a partial scene into a total observation failure.
    mixed = inventory(
        objects=[item(), _degenerate_item(index=1, label="cog", color="blue")],
        relations=[link(0, 1)],
    )
    degenerate = inventory(objects=[_degenerate_item(index=1, label="cog", color="blue")])
    client = SequencedPerceptionClient([degenerate, mixed])
    graph = Perception(client, segment=False).observe(folder)
    assert [obj.id for obj in graph.objects] == ["red_gear"]
    assert graph.relations == ()
    assert any("no plausible extent" in warning for warning in graph.warnings)


def test_close_up_morphology_reports_structure_and_its_own_schema():
    # The close-up pass is the evidence that turns "sun-shaped toy" into "central hub
    # with radial teeth". It must request its OWN schema and return the visible structure
    # as a sentence; the request must never carry an instruction or an expected identity.
    class FormClient:
        model = "form-stub"

        def __init__(self):
            self.messages = None
            self.schema = None

        def complete(self, messages, schema=None, max_tokens=1024, seed=42):
            self.messages, self.schema = messages, schema
            return {"structure": "central hub with radial teeth",
                    "ordinary_name": "gear"}, {"completion_tokens": 4}

    client = FormClient()
    perception = Perception(client, segment=False)
    image = Image.new("RGB", (200, 200), "white")
    obj = ObjectObservation(
        id="g", label="toy", box=(40, 40, 120, 120), pixel=(80, 80),
        confidence=.9, graspable=True,
    )
    form, usage = perception.observe_form(image, obj)

    assert form == "central hub with radial teeth (close-up name: gear)"
    assert usage["completion_tokens"] == 4
    assert set(client.schema["properties"]) == {"structure", "ordinary_name"}
    assert client.schema["additionalProperties"] is False
    request = json.dumps(client.messages)
    assert "toy" not in request and "gear" not in request


class FormAnsweringClient:
    """Inventory answers for the whole image, structure answers for every crop."""

    model = "local-test-vision"

    def __init__(self, response):
        self.response = response
        self.requests = []

    def complete(self, messages, schema=None, max_tokens=1024, seed=42):
        self.requests.append((messages, schema, max_tokens))
        if schema and "structure" in schema.get("properties", {}):
            return ({"structure": "central hub with radial teeth", "ordinary_name": "gear"},
                    {"prompt_tokens": 5, "completion_tokens": 5})
        return self.response, {"prompt_tokens": 20, "completion_tokens": 40}


def test_close_up_morphology_is_attached_to_every_object(folder, monkeypatch):
    monkeypatch.delenv("PERCEPTION_FORMS", raising=False)
    client = FormAnsweringClient(inventory(objects=[item(0, "sun_toy", "yellow"),
                                                  item(1, "plate", "white")]))
    graph = Perception(client, segment=False).observe(folder)
    assert len(graph.objects) == 2
    assert all(obj.visual_form.startswith("central hub") for obj in graph.objects)
    crops = [request for request in client.requests
             if request[1] and "structure" in request[1].get("properties", {})]
    assert len(crops) == 2
    # Task-blind: the crop request is exactly the fixed morphology prompt plus the
    # crop image, so no instruction, scene name or identity can enter it.
    messages = crops[0][0]
    assert len(messages) == 2
    assert messages[0]["content"] == _FORM_SYSTEM
    assert messages[1]["content"][0]["text"] == (
        "Describe the visible structure of the single object in this crop.")
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_close_up_morphology_pass_is_bounded(folder, monkeypatch):
    monkeypatch.delenv("PERCEPTION_FORMS", raising=False)
    monkeypatch.setenv("PERCEPTION_FORM_MAX_OBJECTS", "1")
    client = FormAnsweringClient(inventory(objects=[item(0, "gear", "red"),
                                                  item(1, "cube", "red")]))
    graph = Perception(client, segment=False).observe(folder)
    crops = [request for request in client.requests
             if request[1] and "structure" in request[1].get("properties", {})]
    assert len(crops) == 1
    assert sum(1 for obj in graph.objects if obj.visual_form) == 1


def test_impossible_support_link_is_dropped(folder):
    # A gear whose centre sits below a comparable-size cube cannot be resting on it;
    # keeping the link would make the compiler clear the gear before picking the cube.
    cube = item(0, "cube", "red", box=[300, 200, 400, 300], point=[350, 250])
    gear = item(1, "gear", "red", box=[320, 340, 420, 440], point=[370, 390])
    response = inventory([cube, gear], [link(1, 0, basis="visual_support")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert not any(relation.predicate == "on" for relation in graph.relations)
    assert any("impossible support" in warning for warning in graph.warnings)


def test_frame_spanning_surface_is_exempt_from_the_support_rule(folder):
    # A tray holds objects whose centres are naturally lower than its own. The plate
    # is taller than wide in image pixels, so it is a surface rather than a part.
    tray = item(0, "tray", "grey", box=[0, 0, 1000, 1000], point=[500, 500], movable=False)
    plate = item(1, "plate", "white", box=[300, 400, 500, 900], point=[400, 650])
    response = inventory([tray, plate], [link(1, 0, basis="visual_support")])
    graph = Perception(FakeClient(response), segment=False).observe(folder)
    assert any(relation.predicate == "on" and relation.subject == "white_plate"
               for relation in graph.relations)


def _polygon_object(oid, box, polygon, *, movable=True, label="gear"):
    return ObjectObservation(
        id=oid, label=label, noun=label, color="red", box=box,
        pixel=(box[0] + 4, box[1] + 4), confidence=0.9, graspable=True,
        movable=movable, polygon=polygon,
    )


def test_mask_coverage_decides_the_stacking_direction():
    # The big object's visible mask stops where the small one begins; only the small
    # object's mask covers the overlap, so the small one is on top.
    big = _polygon_object("red_gear_1", (100, 100, 300, 300),
                          ((100, 100), (200, 100), (200, 300), (100, 300)))
    small = _polygon_object("red_gear_2", (200, 200, 350, 350),
                            ((200, 200), (350, 200), (350, 350), (200, 350)))
    relations = {("red_gear_1", "blocks", "red_gear_2"): Relation(
        subject="red_gear_1", predicate="blocks", object="red_gear_2",
        confidence=0.8, evidence="inventory claim, direction to be checked")}
    warnings: list[str] = []
    resolved = resolve_overlaps_by_mask((big, small), 500, 500, relations, warnings)
    assert ("red_gear_2", "blocks", "red_gear_1") in resolved
    assert not any(subject == "red_gear_1" for subject, _, _ in resolved)
    assert any("masks" in warning for warning in warnings)


def test_ambiguous_mask_coverage_leaves_the_pair_alone():
    # Both masks cover about half of the overlap: the pair is not decided.
    first = _polygon_object("red_gear_1", (100, 100, 300, 300),
                            ((100, 100), (300, 100), (300, 300), (100, 300)))
    second = _polygon_object("red_gear_2", (200, 200, 350, 350),
                             ((200, 200), (350, 200), (350, 350), (200, 350)))
    relations = {("red_gear_1", "on", "red_gear_2"): Relation(
        subject="red_gear_1", predicate="on", object="red_gear_2",
        confidence=0.8, evidence="inventory claim")}
    resolve_overlaps_by_mask((first, second), 500, 500, relations, [])
    assert ("red_gear_1", "on", "red_gear_2") in relations
