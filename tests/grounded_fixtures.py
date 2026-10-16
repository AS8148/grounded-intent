"""Synthetic grounded-request builders for compiler and verifier behavior tests.

These fixtures exercise deterministic execution contracts only. They do not stand in
for language parsing or reference binding accuracy; live Pipeline acceptance remains a
separate API-backed check.
"""
from __future__ import annotations

import hashlib
import json

from grounded_intent.domain import (
    CandidateJudgement,
    GroundedOperation,
    GroundedProtection,
    GroundedTask,
    ObservationPointer,
    ProtectionConstraint,
    QuantitySpec,
    ReferenceBinding,
    ReferenceExpression,
    Requirement,
    RequirementJudgement,
    SceneGraph,
    SourceOccurrence,
    TaskClause,
    TaskIntent,
    TaskProgram,
)


def _occurrence(instruction: str, phrase: str) -> SourceOccurrence:
    start = instruction.index(phrase)
    return SourceOccurrence(excerpt=phrase, start=start, end=start + len(phrase))


def grounded_from_intent(graph: SceneGraph, intent: TaskIntent) -> GroundedTask:
    """Create an explicit synthetic TaskProgram and complete bindings for low-level tests."""
    identifiers = sorted({
        value for goal in intent.goals
        for value in (goal.target, goal.destination, goal.under_object)
        if value is not None
    } | set(intent.forbidden_movement))
    role_phrases = []
    for operation_index, goal in enumerate(intent.goals):
        role_phrases.append(f"operation_{operation_index}")
        role_phrases.append(f"target_{operation_index}")
        if goal.destination is not None:
            role_phrases.append(f"destination_{operation_index}")
        if goal.under_object is not None:
            role_phrases.append(f"upper_{operation_index}")
        if goal.relation == "under" and goal.destination is not None:
            role_phrases.append(f"base_{operation_index}")
    role_phrases.extend(f"protected_{index}" for index, _ in enumerate(intent.forbidden_movement))
    if intent.kind != "plan" or not role_phrases:
        role_phrases.append("request")
    instruction = "Synthetic task: " + " ".join(role_phrases)
    references: list[ReferenceExpression] = []
    bindings: list[ReferenceBinding] = []
    object_ids = tuple(identifiers)

    def reference(reference_id: str, phrase: str, object_id: str) -> None:
        occurrence = _occurrence(instruction, phrase)
        requirement_id = f"requirement_{reference_id}"
        requirement = Requirement(
            requirement_id=requirement_id, aspect="synthetic_identity", value=phrase,
            source_occurrence=occurrence,
        )
        references.append(ReferenceExpression(
            reference_id=reference_id, source_occurrence=occurrence,
            requirements=(requirement,), current_condition=None,
            antecedent_reference_id=None, antecedent_occurrence=None,
        ))
        keys = tuple(f"key_{index}" for index, _ in enumerate(object_ids))
        key_to_id = dict(zip(keys, object_ids, strict=True))
        rows = []
        definite = []
        possible = []
        labels = {obj.id: obj.label for obj in graph.objects}
        for key, candidate_id in key_to_id.items():
            supported = candidate_id == object_id
            verdict = "supported" if supported else "contradicted"
            if supported:
                definite.append(key)
                possible.append(key)
            rows.append(RequirementJudgement(
                requirement_id=requirement_id, instance_key=key, verdict=verdict,
                observations=(ObservationPointer(
                    field="label", excerpt=labels.get(candidate_id, "synthetic object")),
                ),
                explanation="Independent synthetic fixture identity.",
                call_provenance="test-fixture",
            ))
        bindings.append(ReferenceBinding(
            reference_id=reference_id, requirement_ids=(requirement_id,),
            instance_keys=keys, judgements=tuple(rows),
            definite_instance_keys=tuple(definite), possible_instance_keys=tuple(possible),
            status="complete", parse_version=1, evidence_hash="a" * 64,
            reason="Synthetic exact identity evidence for deterministic execution tests.",
        ))

    clauses = []
    grounded_operations = []
    protections = []
    grounded_protections = []
    for index, goal in enumerate(intent.goals):
        operation_id = f"operation_{index}"
        target_ref = f"target_ref_{index}"
        reference(target_ref, f"target_{index}", goal.target)
        destination_ref = upper_ref = base_ref = None
        if goal.relation in {"on", "in"} and goal.destination is not None:
            destination_ref = f"destination_ref_{index}"
            reference(destination_ref, f"destination_{index}", goal.destination)
        if goal.relation == "under":
            if goal.under_object is not None:
                upper_ref = f"upper_ref_{index}"
                reference(upper_ref, f"upper_{index}", goal.under_object)
            if goal.destination is not None:
                base_ref = f"base_ref_{index}"
                reference(base_ref, f"base_{index}", goal.destination)
        phrase = f"operation_{index}"
        final_relation = (
            "held" if goal.relation == "held"
            or (goal.relation != "under" and goal.destination is None)
            else goal.relation
        )
        clause = TaskClause(
            operation_id=operation_id,
            action="pick_only" if final_relation == "held" else "pick_and_place",
            target_reference_id=target_ref,
            quantity=QuantitySpec(
                mode="one", exact_count=None, member_reference_ids=(),
                choice_policy="unique", choice_permission=None,
            ),
            final_relation=final_relation,
            destination_reference_id=destination_ref,
            upper_reference_id=upper_ref,
            base_reference_id=base_ref,
            base_relation=(goal.base_relation if base_ref is not None else None),
            source_reference_id=None, exclusion_reference_ids=(),
            order=(index + 1 if intent.goal_mode == "ordered" else None),
            order_occurrence=(
                _occurrence(instruction, phrase) if intent.goal_mode == "ordered" else None),
            source_occurrence=_occurrence(instruction, phrase),
        )
        clauses.append(clause)
        grounded_operations.append(GroundedOperation(
            clause=clause, definite_target_ids=(goal.target,),
            possible_target_ids=(goal.target,), selected_target_ids=(goal.target,),
            goals=(goal,), resolution_status="resolved", reason="",
        ))

    for index, object_id in enumerate(intent.forbidden_movement):
        reference_id = f"protected_ref_{index}"
        phrase = f"protected_{index}"
        reference(reference_id, phrase, object_id)
        constraint = ProtectionConstraint(
            constraint_id=f"protection_{index}", reference_id=reference_id,
            mode="do_not_move", operation_ids=(),
            source_occurrence=_occurrence(instruction, phrase),
            antecedent_reference_id=None, antecedent_occurrence=None,
        )
        protections.append(constraint)
        grounded_protections.append(GroundedProtection(
            constraint=constraint, object_ids=(object_id,),
        ))

    if intent.kind == "plan":
        judgement = CandidateJudgement(kind="none", reason="", source_occurrence=None)
        status = "grounded"
        question = reason = ""
        program_instruction = instruction
    else:
        phrase = "request"
        source = _occurrence(instruction, phrase)
        judgement = CandidateJudgement(
            kind="clarify" if intent.kind == "ask_user" else "refuse",
            reason=intent.question if intent.kind == "ask_user" else intent.reason,
            source_occurrence=source,
        )
        status = "unresolved" if intent.kind == "ask_user" else "infeasible"
        question, reason = intent.question, intent.reason
        program_instruction = instruction

    program = TaskProgram(
        instruction=program_instruction, operations=tuple(clauses),
        references=tuple(references), protection_constraints=tuple(protections),
        method_facts=(), unresolved_requirements=(), candidate_judgement=judgement,
        parse_version=1, repair_history=(),
    )
    evidence_hash = hashlib.sha256(json.dumps(
        graph.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return GroundedTask(
        instruction=instruction, program=program, parse_version=1,
        evidence_hash=evidence_hash, bindings=tuple(bindings),
        operations=tuple(grounded_operations),
        protected_ids=tuple(intent.forbidden_movement),
        protection_scopes=tuple(grounded_protections), status=status,
        unresolved_requirements=(), question=question, reason=reason,
    )
