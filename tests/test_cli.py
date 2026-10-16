"""CLI coverage and expensive-failure handling with real plan verification."""
import json

import pytest
from PIL import Image

from grounded_intent import cli
from grounded_intent.clients import ModelError
from grounded_intent.compiler import compile_intent
from grounded_intent.domain import Goal, ObjectObservation, SceneGraph, TaskIntent
from grounded_intent.planner import verify


@pytest.mark.parametrize("healthy", [False, True])
def test_failed_scene_is_acquired_once_without_fabricating_answers(tmp_path, monkeypatch, healthy):
    data = tmp_path / "data"
    names = ["broken", "healthy"] if healthy else ["broken"]
    for name in names:
        folder = data / name
        folder.mkdir(parents=True)
        Image.new("RGB", (64, 48), "white").save(folder / "rgb.png")
        (folder / "instructions.txt").write_text("Retrieve the token.\nRetrieve the token again.\n")
    acquired, decided, closed = [], [], []

    class Backend:
        evaluation_mode = "synthetic_verifier_smoke"
        code_revision = "synthetic"

        def __init__(self, data, cache, **kwargs):
            self.data = data
            self.graphs = {}

        def folder(self, name):
            return self.data / name

        def scene(self, name):
            acquired.append(name)
            if name == "broken":
                raise ModelError("RGB observation audit failed strict validation")
            self.graphs[name] = SceneGraph(
                scene=name, width=64, height=48, image_sha256="synthetic",
                objects=(ObjectObservation(
                    id="token", label="token", box=(5, 5, 20, 20), pixel=(12, 12),
                    confidence=.95, movable=True, graspable=True, receives_objects=False,
                ),), perception_model="synthetic", perception_seconds=0,
            )
            return self.graphs[name]

        def decide(self, name, text, instruction_id):
            decided.append(name)
            graph = self.graphs[name]
            intent = TaskIntent(kind="plan", goals=(Goal(target="token", relation="held"),))
            return verify(graph, compile_intent(graph, intent), instruction_id)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "Pipeline", Backend)
    out, audit_path = tmp_path / "results.json", tmp_path / "audit.json"
    assert cli.main(["run", "--data", str(data), "--out", str(out),
                     "--audit", str(audit_path)]) == 0
    results = json.loads(out.read_text())["results"]
    audit = json.loads(audit_path.read_text())
    assert acquired.count("broken") == 1
    assert "broken" not in decided
    assert {item["instruction_id"] for item in results} == (
        {"healthy_i1", "healthy_i2"} if healthy else set()
    )
    assert all(item["target_pixel"] == [12, 12] for item in results)
    failures = [item for item in audit["decisions"] if item.get("status") == "failed"]
    assert {item["instruction_id"] for item in failures} == {"broken_i1", "broken_i2"}
    assert all(item["scene"] == "broken" and item["stage"] == "scene"
               and "strict validation" in item["error"] for item in failures)
    assert audit["scene_failures"]["broken"]["error_type"] == "ModelError"
    assert closed == [True]


def test_results_survive_a_later_scene_without_instructions(tmp_path, monkeypatch):
    data = tmp_path / "data"
    healthy = data / "healthy"
    healthy.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(healthy / "rgb.png")
    (healthy / "instructions.txt").write_text("Retrieve the token.\n")
    broken = data / "zz_broken"
    broken.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(broken / "rgb.png")
    # Intentionally no instructions.txt: the second scene cannot be planned, but
    # the first scene's answer must already be on disk.

    class Backend:
        evaluation_mode = "synthetic_verifier_smoke"
        code_revision = "synthetic"

        def __init__(self, data, cache, **kwargs):
            self.data = data

        def folder(self, name):
            return self.data / name

        def scene(self, name):
            return SceneGraph(
                scene=name, width=64, height=48, image_sha256="synthetic",
                objects=(ObjectObservation(
                    id="token", label="token", box=(5, 5, 20, 20), pixel=(12, 12),
                    confidence=.95, movable=True, graspable=True, receives_objects=False,
                ),), perception_model="synthetic", perception_seconds=0,
            )

        def decide(self, name, text, instruction_id):
            graph = self.scene(name)
            intent = TaskIntent(kind="plan", goals=(Goal(target="token", relation="held"),))
            return verify(graph, compile_intent(graph, intent), instruction_id)

        def close(self):
            pass

    monkeypatch.setattr(cli, "Pipeline", Backend)
    out, audit_path = tmp_path / "results.json", tmp_path / "audit.json"
    assert cli.main(["run", "--data", str(data), "--out", str(out),
                     "--audit", str(audit_path)]) == 0
    results = json.loads(out.read_text())["results"]
    assert [item["instruction_id"] for item in results] == ["healthy_i1"]
    audit = json.loads(audit_path.read_text())
    assert audit["scene_failures"]["zz_broken"]["error_type"] == "ValueError"
    assert any(item.get("stage") == "instructions" for item in audit["decisions"])
