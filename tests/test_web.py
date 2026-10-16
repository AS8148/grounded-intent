"""Exercise the HTTP adapter with synthetic observations and the real verifier."""
from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grounded_intent.domain import (
    ObjectObservation,
    Proposal,
    ProposedAction,
    SceneGraph,
)
from grounded_intent.planner import verify
from grounded_intent.web import create_app


def _verified_decision(graph, proposal, instruction_id):
    """Run the real verifier over a scripted proposal, exactly as the pipeline does."""
    return verify(graph, proposal, instruction_id, model="synthetic-apertus",
                  reasoning_seconds=0.5,
                  usage={"prompt_tokens": 40, "completion_tokens": 12})


class SyntheticPipeline:
    """A deterministic observed-scene backend, not a transport mock or prompt echo."""

    def __init__(self, data: Path):
        self.data = data
        self.calls = []
        self.failure = None

    def scene(self, name):
        self._assert_worker()
        self.calls.append(("scene", name))
        if self.failure == "vision":
            raise RuntimeError("/private/weights/model.gguf: api_key=do-not-expose")
        image = self.data / name / "rgb.png"
        with Image.open(image) as rgb:
            width, height = rgb.size
        return SceneGraph(
            scene=name, width=width, height=height,
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            objects=(
                ObjectObservation(id="cube_1", label="cube", color="red", box=(4, 5, 16, 19),
                                  pixel=(10, 12), confidence=0.98, graspable=True,
                                  polygon=((4, 5), (16, 5), (16, 19), (4, 19))),
                ObjectObservation(id="plate_1", label="plate", color="white", box=(30, 10, 55, 35),
                                  pixel=(40, 20), confidence=0.97, movable=False, graspable=False,
                                  receives_objects=True),
            ),
            warnings=("Synthetic observations; human review required.",),
            perception_model="synthetic-vision", perception_seconds=0.25,
        )

    def decide(self, scene_name, instruction, instruction_id):
        self._assert_worker()
        self.calls.append(("decide", scene_name, instruction, instruction_id))
        if self.failure == "reasoning":
            raise RuntimeError("http://private-host:8082/v1 secret-token")
        graph = self.scene(scene_name)
        if "which" in instruction.casefold():
            proposal = Proposal(kind="ask_user", question="Which observed destination do you mean?")
        elif "unobserved" in instruction.casefold():
            proposal = Proposal(kind="refuse", reason="The requested object is not observed.")
        else:
            proposal = Proposal(kind="plan", actions=(
                ProposedAction(action="pick", object="cube_1"),
                ProposedAction(action="place", object="plate_1"),
            ))
        return _verified_decision(graph, proposal, instruction_id)

    @staticmethod
    def _assert_worker():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise AssertionError("Synchronous inference must run outside the server event loop")


@pytest.fixture
def backend(tmp_path):
    data = tmp_path / "scenes"
    data.mkdir()
    for name in ("scene_b", "scene_a"):
        directory = data / name
        directory.mkdir()
        Image.new("RGB", (64, 48), color=(40, 50, 60)).save(directory / "rgb.png")
        # Official labels are deliberately unusable; the web adapter must not inspect them.
        (directory / "ground_truth.json").write_text("not valid JSON")
        (directory / "instructions.csv").write_text("not a usable instruction source")
    (data / "empty").mkdir()
    (data / "corrupt").mkdir()
    (data / "corrupt" / "rgb.png").write_bytes(b"not an image")
    (data / "wrong_format").mkdir()
    Image.new("RGB", (64, 48)).save(data / "wrong_format" / "rgb.png", format="JPEG")
    return SyntheticPipeline(data)


@pytest.fixture
def client(backend):
    with TestClient(create_app(backend)) as connection:
        yield connection


def test_lists_only_valid_rgb_scene_directories(client, backend):
    response = client.get("/api/scenes")
    assert response.status_code == 200
    assert response.json() == ["scene_a", "scene_b"]
    assert backend.calls == []


def test_graph_and_original_image_are_accessible(client):
    response = client.get("/api/scenes/scene_a")
    assert response.status_code == 200
    graph = SceneGraph.model_validate(response.json())
    assert graph.scene == "scene_a"
    assert graph.objects[0].pixel == (10, 12)
    image = client.get("/api/scenes/scene_a/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.headers["cache-control"] == "no-store"
    with Image.open(BytesIO(image.content)) as decoded:
        assert decoded.size == (graph.width, graph.height)
        assert decoded.format == "PNG"


def test_actual_verified_decision_includes_audit_and_official_output(client, backend):
    response = client.post("/api/decide", json={
        "scene": "scene_a", "instruction": "  Put the red cube on the white plate.  ",
    })
    assert response.status_code == 200
    result = response.json()
    assert result["kind"] == "plan"
    assert result["target_pixel"] == [10, 12]
    assert result["actions"] == [
        {"action": "pick", "object": "cube_1"},
        {"action": "place", "object": "plate_1"},
    ]
    assert [
        {"action": action["action"], "object": action["object"]}
        for action in result["proposal"]["actions"]
    ] == result["actions"]
    assert result["checks"] and all(check["passed"] for check in result["checks"])
    assert result["official"] == {
        "instruction_id": "scene_a_interactive", "target_pixel": [10, 12],
        "actions": result["actions"],
    }
    assert result["usage"] == {"prompt_tokens": 40, "completion_tokens": 12}
    assert result["warnings"]
    assert result["timings"]["perception_seconds"] == 0.25
    assert result["timings"]["reasoning_seconds"] == 0.5
    assert result["timings"]["request_seconds"] >= 0
    assert ("decide", "scene_a", "Put the red cube on the white plate.", "scene_a_interactive") in backend.calls


@pytest.mark.parametrize(("instruction", "kind"), [
    ("Which destination should be used?", "ask_user"),
    ("Pick the unobserved object.", "refuse"),
])
def test_nonmovement_decisions_remain_exclusive(client, instruction, kind):
    response = client.post("/api/decide", json={"scene": "scene_a", "instruction": instruction})
    assert response.status_code == 200
    result = response.json()
    assert result["kind"] == kind
    assert result["proposal"]["actions"] == []
    assert len(result["actions"]) == 1
    assert result["actions"][0]["action"] == kind
    assert result["official"]["actions"] == result["actions"]
    assert result["official"]["target_pixel"] is None
    assert result["proposal"]["question" if kind == "ask_user" else "reason"]


@pytest.mark.parametrize("instruction", ["", " \t\n", "\u2003", "x" * 2001, 123, None])
def test_rejects_blank_oversized_or_nonstrings_before_inference(client, backend, instruction):
    response = client.post("/api/decide", json={"scene": "scene_a", "instruction": instruction})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert backend.calls == []


def test_accepts_exact_instruction_length_limit(client):
    response = client.post("/api/decide", json={"scene": "scene_a", "instruction": "x" * 2000})
    assert response.status_code == 200


@pytest.mark.parametrize("scene", ["../outside", "..", "/etc", "scene_a/../scene_b", "scene_a\\rgb.png", "", "a\x00b"])
def test_rejects_traversal_before_model_access(client, backend, scene):
    response = client.post("/api/decide", json={"scene": scene, "instruction": "Pick the cube."})
    assert response.status_code == 400
    assert backend.calls == []


@pytest.mark.parametrize("suffix", ["", "/image"])
def test_encoded_traversal_get_is_rejected(client, backend, suffix):
    response = client.get("/api/scenes/%2E%2E%2Foutside" + suffix)
    assert response.status_code == 400
    assert backend.calls == []


@pytest.mark.parametrize("name", ["missing", "empty", "corrupt", "wrong_format"])
def test_unknown_or_invalid_scene_is_not_analyzed(client, backend, name):
    assert client.get(f"/api/scenes/{name}").status_code == 404
    assert client.get(f"/api/scenes/{name}/image").status_code == 404
    assert client.post("/api/decide", json={"scene": name, "instruction": "Pick the cube."}).status_code == 404
    assert backend.calls == []


def test_symlinked_scene_and_image_are_not_exposed(client, backend, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    Image.new("RGB", (64, 48)).save(outside / "rgb.png")
    (backend.data / "linked_scene").symlink_to(outside, target_is_directory=True)
    linked_image = backend.data / "linked_image"
    linked_image.mkdir()
    (linked_image / "rgb.png").symlink_to(outside / "rgb.png")
    assert client.get("/api/scenes").json() == ["scene_a", "scene_b"]
    for name in ("linked_scene", "linked_image"):
        assert client.get(f"/api/scenes/{name}/image").status_code == 404
        assert client.get(f"/api/scenes/{name}").status_code == 404
    assert backend.calls == []


@pytest.mark.parametrize("failure", ["vision", "reasoning"])
def test_unavailable_models_return_readable_503_without_fake_plan_or_secrets(client, backend, failure):
    backend.failure = failure
    response = client.post("/api/decide", json={"scene": "scene_a", "instruction": "Pick the cube."})
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
    assert isinstance(response.json()["detail"], str)
    assert "api_key" not in response.text
    assert "private" not in response.text
    assert "secret-token" not in response.text
    if failure == "vision":
        assert client.get("/api/scenes/scene_a").status_code == 503


def test_unavailable_dataset_returns_readable_503(tmp_path):
    with TestClient(create_app(SyntheticPipeline(tmp_path / "missing"))) as client:
        response = client.get("/api/scenes")
        assert response.status_code == 503
        assert str(tmp_path) not in response.text


def test_missing_fields_and_extra_request_fields_are_rejected(client, backend):
    for payload in ({"scene": "scene_a"}, {"instruction": "Pick the cube."},
                    {"scene": "scene_a", "instruction": "Pick the cube.", "api_key": "secret"}):
        assert client.post("/api/decide", json=payload).status_code == 422
    assert backend.calls == []


def test_packaged_interface_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    for asset, media_type in (("app.js", "javascript"), ("styles.css", "text/css")):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200
        assert media_type in response.headers["content-type"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert client.get("/static/../domain.py").status_code == 404
