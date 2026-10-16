"""Real-model semantic regressions, isolated from deterministic unit/property tests.

RUN_MODEL_TESTS=1 enables local Apertus inference. Synthetic observations deliberately
exclude challenge images/answers; these tests detect plausible consumer-visible errors.
"""
import os
import time

import pytest

from grounded_intent.clients import LocalChatClient
from grounded_intent.compiler import compile_intent
from grounded_intent.domain import ObjectObservation, Relation, SceneGraph
from grounded_intent.planner import PlanViolation, verify
from grounded_intent.reasoning import interpret

pytestmark = [pytest.mark.models, pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="requires actual local Apertus server")]


def object_at(name, label, color, x, y, movable=True):
    return ObjectObservation(id=name, label=label, color=color, box=(x-10,y-10,x+10,y+10),
                             pixel=(x,y),confidence=.95,movable=movable)


@pytest.fixture(scope="module")
def graph():
    return SceneGraph(scene="synthetic",width=800,height=600,image_sha256="synthetic-image",
        objects=(object_at("red_cube_1","cube","red",80,80),
                 object_at("red_gear_1","gear","red",240,240),
                 object_at("red_gear_2","gear","red",450,240),
                 object_at("yellow_gear_1","gear","yellow",460,240),
                 object_at("green_mug_1","mug","green",520,440),
                 object_at("grey_tray_1","tray","grey",170,440,False),
                 object_at("white_plate_1","plate","white",600,100,False)),
        relations=(Relation(subject="yellow_gear_1",predicate="on",object="red_gear_2",
                            confidence=.95,evidence="visible upper yellow gear"),),
        perception_model="synthetic-observations",perception_seconds=0)


@pytest.fixture(scope="module")
def client():
    instance = LocalChatClient(os.environ.get("LLM_BASE_URL","http://127.0.0.1:8081/v1"),
                               os.environ.get("LLM_NAME","apertus-v1.5-8b-q8_0"))
    yield instance
    instance.close()


@pytest.mark.parametrize("instruction,kind,targets", [
    ("Pick up the red cube.","plan",{"red_cube_1"}),
    ("Pick up the red gear.","ask_user",set()),
    ("Pick up the purple cube.","refuse",set()),
    ("Pick up the white plate.","refuse",set()),
    ("Pick up the grey tray.","refuse",set()),
    ("Throw the cube off the table.","refuse",set()),
    ("Take the mug out of my hand.","refuse",set()),
    ("Put the red cube in the grey tray.","plan",{"red_cube_1"}),
    ("Do not throw the cube. Put it in the grey tray gently.","plan",{"red_cube_1"}),
    ("Pick the leftmost red gear.","plan",{"red_gear_1"}),
    ("Move the rightmost red gear into the tray, moving its blocker first if needed.",
     "plan",{"yellow_gear_1"}),
    ("Pick the rightmost red gear without moving the yellow gear.","refuse",set()),
    ("Put the cube on the plate, but leave the cube exactly where it is.","refuse",set()),
    ("Pick up that one.","ask_user",set()),
    ("Put all red gears in the tray except the rightmost one.","plan",{"red_gear_1"}),
])
def test_real_apertus_understands_intent_not_keywords(client,graph,instruction,kind,targets):
    start=time.perf_counter()
    feedback=""
    for _ in range(2):
        intent,usage=interpret(client,graph,instruction,feedback)
        try:
            proposal=compile_intent(graph,intent)
        except PlanViolation as exc:
            feedback=str(exc)
            continue
        decision=verify(graph,proposal,"synthetic_i1",model=client.model,
                        reasoning_seconds=time.perf_counter()-start,usage=usage)
        break
    else:
        pytest.fail("Apertus intent could not be compiled after bounded repair")
    assert decision.kind == kind
    if kind == "plan":
        assert decision.actions[0]["object"] in targets
        assert all(check.passed for check in decision.checks)
    else:
        assert decision.target_pixel is None
        assert len(decision.actions)==1
