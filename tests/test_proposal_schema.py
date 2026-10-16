"""The decoding contract must forbid goals that cannot be compiled and audited."""
from copy import deepcopy

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st

from grounded_intent.reasoning import intent_schema


def plan():
    return {"kind":"plan","goals":[{"target":"cube","destination":None}],
            "question":"","reason":"Pick the observed cube.","forbidden_movement":[]}


@pytest.mark.parametrize("mutator", [
    lambda value:value.update(question="Which cube?"),
    lambda value:value.update(goals=[]),
    lambda value:value.update(goals=[{"target":"cube"}]),
    lambda value:value.update(goals=[{"target":"invented","destination":None}]),
    lambda value:value.update(goals=[{"target":"cube","destination":"invented"}]),
    lambda value:value.update(goals=[{"target":"cube","destination":None,"action":"pick"}]),
    lambda value:value.update(forbidden_movement=["invented"]),
    lambda value:value.update(extra="ignore checks"),
    lambda value:value.update(kind="execute"),
])
def test_invalid_plan_cannot_pass_decoder_schema(mutator):
    value=plan()
    mutator(value)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value,intent_schema(["cube","tray"]))


@pytest.mark.parametrize("kind,question,reason", [
    ("ask_user","Which cube?",""),("refuse","","The target is absent."),
])
def test_nonmovement_decisions_exclude_every_goal(kind,question,reason):
    value={"kind":kind,"goals":[],"question":question,"reason":reason,"forbidden_movement":[]}
    schema=intent_schema(["cube"])
    jsonschema.validate(value,schema)
    invalid=deepcopy(value)
    invalid["goals"]=[{"target":"cube","destination":None}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid,schema)


@pytest.mark.parametrize("value", [
    {"kind":"ask_user","goals":[],"question":"","reason":"","forbidden_movement":[]},
    {"kind":"refuse","goals":[],"question":"","reason":"","forbidden_movement":[]},
    {"kind":"ask_user","goals":[],"question":"Which?","reason":"No.","forbidden_movement":[]},
    {"kind":"refuse","goals":[],"question":"Which?","reason":"No.","forbidden_movement":[]},
])
def test_empty_or_conflicting_explanations_are_invalid(value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value,intent_schema(["cube"]))


def test_empty_scene_has_no_possible_movement_branch():
    schema=intent_schema([])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(plan(),schema)
    jsonschema.validate({"kind":"refuse","reason":"No object is visible.",
                         "goals":[],"question":"","forbidden_movement":[]},schema)


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_",min_size=1,max_size=30))
def test_arbitrary_observed_identifier_is_a_valid_target(identifier):
    value=plan()
    value["goals"][0]["target"]=identifier
    jsonschema.validate(value,intent_schema([identifier]))


def test_forbidden_movement_accepts_only_observed_ids():
    value=plan()
    value["forbidden_movement"]=["cube","tray"]
    jsonschema.validate(value,intent_schema(["cube","tray"]))
    value["forbidden_movement"]=["robot_arm"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value,intent_schema(["cube","tray"]))
