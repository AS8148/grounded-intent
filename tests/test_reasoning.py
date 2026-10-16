"""Observable request/response boundary tests, without pinning prompt wording."""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from grounded_intent.compiler import compile_intent
from grounded_intent.domain import ObjectObservation, SceneGraph
from grounded_intent.planner import PlanViolation
from grounded_intent.reasoning import interpret


def graph(*, label="cube", evidence="observed"):
    return SceneGraph(
        scene="synthetic", width=80, height=60, image_sha256="synthetic-only",
        objects=(ObjectObservation(
            id="item", label=label, box=(8, 12, 24, 31), pixel=(13, 21),
            confidence=0.9, evidence=evidence, graspable=True,
        ),), perception_model="test-observations", perception_seconds=0,
    )


class ResponseClient:
    """A transport-boundary fixture returning a response, never echoing a prompt."""

    def __init__(self, response, usage=None, error=None):
        self.response = response
        self.usage = {} if usage is None else usage
        self.error = error
        self.requests = []

    def complete(self, messages, schema=None, max_tokens=1024):
        self.requests.append({"messages": messages, "schema": schema, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.response), self.usage


@pytest.mark.parametrize("response", [
    {},
    {"kind": "dance"},
    {"kind": "plan"},
    {"kind": "plan", "goals": []},
    {"kind": "plan", "goals": [{"target": "item", "destination": None}], "schema": "changed"},
    {"kind": "ask_user", "question": ""},
    {"kind": "ask_user", "question": "  "},
    {"kind": "refuse", "reason": ""},
    {"kind": "refuse", "reason": "  "},
])
def test_schema_errors_propagate_without_fake_refusal(response):
    client = ResponseClient(response)
    with pytest.raises(ValidationError):
        interpret(client, graph(), "Lift the cube.")
    assert len(client.requests) == 3  # bounded correction attempts, then propagate


@pytest.mark.parametrize("response,kind", [
    ({"kind": "plan", "goals": [{"target": "item", "destination": None}],
      "question": "Which?", "reason": ""}, "plan"),
    ({"kind": "ask_user", "question": "Which?",
      "goals": [{"target": "item", "destination": None}]}, "ask_user"),
    ({"kind": "refuse", "reason": "No.",
      "goals": [{"target": "item", "destination": None}]}, "refuse"),
])
def test_exclusive_branch_violation_is_repaired_to_the_declared_kind(response, kind):
    """A model that sets both sides of the exclusive contract is read by the kind it
    declared, with the other side cleared. Failing the instruction instead would lose
    an answer the model had already decided."""
    intent, _ = interpret(ResponseClient(response), graph(), "Lift the cube.")
    assert intent.kind == kind
    if kind == "plan":
        assert intent.goals and intent.question == ""
    else:
        assert intent.goals == ()


@pytest.mark.parametrize("message", ["Endpoint unavailable", "Malformed model JSON", "Timeout"])
def test_transport_errors_propagate_without_hidden_retry(message):
    error = RuntimeError(message)
    client = ResponseClient(None, error=error)
    with pytest.raises(RuntimeError) as caught:
        interpret(client, graph(), "Lift the cube.")
    assert caught.value is error
    assert len(client.requests) == 1


@pytest.mark.parametrize("object_id", ["missing", " ", "item ", "Item"])
def test_unknown_response_ids_reach_compiler_as_repairable_error(object_id):
    observed = graph()
    client = ResponseClient(
        {"kind": "plan", "goals": [{"target": object_id, "destination": None}],
         "question": "", "reason": "", "forbidden_movement": []})
    intent, _ = interpret(client, observed, "Lift the cube.")
    assert intent.kind == "plan"
    with pytest.raises(PlanViolation) as caught:
        compile_intent(observed, intent)
    assert caught.value.checks[-1].rule == "known_ids"


def test_verifier_feedback_reaches_the_reasoning_request():
    client = ResponseClient({"kind": "refuse", "reason": "Not present.",
                             "question": "", "goals": [], "forbidden_movement": []})
    interpret(client, graph(), "Lift the cube.", "Goal target must be observed: 'other'")
    payload = client.requests[0]["messages"][-1]["content"]
    assert "Goal target must be observed" in payload
    assert "Lift the cube." in payload
