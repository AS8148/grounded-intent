"""Validated contracts shared by perception, reasoning, verification and the demo.

Coordinates always refer to the original image: x/column first, y/row second.
These are observations and task-level checks, not certified robot safety functions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Pixel = tuple[int, int]
Box = tuple[int, int, int, int]

# Single authoritative description of the robot's gripper. Shared by the perception
# inventory and the reasoning-side graspability questions so the two can never
# describe different hardware.
GRIPPER_CONTEXT = """The robot has one parallel gripper, holds one object at a time,
and opens at most 85 mm. Estimate graspability from visible grasping geometry,
accessible opposing contact surfaces, and whether the whole item can plausibly be
lifted with this gripper. Physical movability alone does not imply graspability.
Do not turn pixel widths into millimetres or claim measured gripper fit from this
uncalibrated image. Report uncertainty in warnings; this is a visual estimate,
not certified grasp feasibility or robot safety."""


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObjectObservation(Contract):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)   # image-reported physical identity, not a policy
    noun: str = ""                     # everyday identity: 'banana', 'washer', …
    color: str = "unknown"
    box: Box
    pixel: Pixel
    confidence: float = Field(ge=0, le=1)
    movable: bool = True              # physical mobility, not robot pick permission
    graspable: bool                   # model estimate for the robot's gripper
    receives_objects: bool = False    # independent support/container affordance
    visual_description: str = ""      # image-reported cues, separate from provenance
    # Task-blind morphology observed at detail scale: the object's visible parts and how
    # they are arranged ("central cylindrical hub with radial teeth"), plus the everyday
    # name that crop suggests when the model offers one. No instruction, scene metadata
    # or expected identity enters the crop request, and no affordance or provenance is
    # mixed in. Kept separate from `label`/`noun` because a one-name whole-image
    # inventory compresses away exactly the structure a request may name.
    visual_form: str = ""
    polygon: tuple[Pixel, ...] = ()
    # Measured, not estimated: the object's real base width in millimetres, mapped onto
    # the table plane through the dataset's own camera calibration. None when the scene
    # has no calibration or the box cannot be mapped. Kept separate from `graspable`
    # because an estimate is not a measurement.
    base_width_mm: float | None = Field(default=None, gt=0)
    evidence: str = "vision model"

    @model_validator(mode="after")
    def validate_geometry(self) -> ObjectObservation:
        x0, y0, x1, y1 = self.box
        x, y = self.pixel
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("box must have positive area and nonnegative origin")
        if not x0 <= x < x1 or not y0 <= y < y1:
            raise ValueError("target pixel must be inside its observation box")
        return self


class Relation(Contract):
    subject: str
    predicate: Literal["on", "inside", "blocks", "left_of", "above", "near"]
    object: str
    confidence: float = Field(ge=0, le=1)
    evidence: str


class SceneGraph(Contract):
    scene: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_sha256: str
    objects: tuple[ObjectObservation, ...]
    relations: tuple[Relation, ...] = ()
    warnings: tuple[str, ...] = ()
    perception_model: str
    perception_seconds: float = Field(ge=0)
    # Declared by the dataset calibration, not inferred: the gripper's maximum opening.
    gripper_max_opening_mm: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_scene(self) -> SceneGraph:
        ids = {obj.id for obj in self.objects}
        if len(ids) != len(self.objects):
            raise ValueError("scene object IDs must be unique")
        for obj in self.objects:
            if obj.box[2] > self.width or obj.box[3] > self.height:
                raise ValueError("object box extends outside original image")
        for relation in self.relations:
            if relation.subject not in ids or relation.object not in ids:
                raise ValueError("relation must reference observed objects")
            if relation.subject == relation.object:
                raise ValueError("self relations are invalid")
        return self

    def by_id(self) -> dict[str, ObjectObservation]:
        return {obj.id: obj for obj in self.objects}


class SourceOccurrence(Contract):
    """An exact instruction excerpt and its half-open character range."""
    excerpt: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_range(self) -> SourceOccurrence:
        if self.end <= self.start or self.end - self.start != len(self.excerpt):
            raise ValueError("source occurrence bounds must exactly cover its excerpt")
        return self


class Requirement(Contract):
    """One intrinsic meaning requirement, with its supporting instruction phrase."""
    requirement_id: str = Field(min_length=1)
    aspect: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source_occurrence: SourceOccurrence


class IntrinsicObservationField(Contract):
    """One reported intrinsic attribute; its field name identifies the source."""
    name: Literal["label", "noun", "color", "visual_description", "visual_form"]
    value: str = Field(min_length=1)


class IntrinsicObservation(Contract):
    """Per-call opaque-key view; contains no observed IDs or geometric properties."""
    instance_key: str = Field(min_length=1)
    fields: tuple[IntrinsicObservationField, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_fields(self) -> IntrinsicObservation:
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("intrinsic observation fields must be unique")
        if "label" not in names:
            raise ValueError("intrinsic observations require the reported label")
        return self


class ObservationPointer(Contract):
    """Exact excerpt from one supplied intrinsic-observation field."""
    field: Literal["label", "noun", "color", "visual_description", "visual_form"]
    excerpt: str = Field(min_length=1)


class RequirementJudgementDraft(Contract):
    requirement_id: str = Field(min_length=1)
    instance_key: str = Field(min_length=1)
    verdict: Literal["supported", "possible", "contradicted"]
    observations: tuple[ObservationPointer, ...] = Field(max_length=8)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> RequirementJudgementDraft:
        if self.verdict in {"supported", "contradicted"} and not self.observations:
            raise ValueError("established judgements require an observation pointer")
        return self


class RequirementJudgement(RequirementJudgementDraft):
    """One complete intrinsic requirement/instance result and its call identity."""
    call_provenance: str = Field(min_length=1)


class RequirementJudgementBatch(Contract):
    """Untrusted complete response to one requirement-scoped binding call."""
    judgements: tuple[RequirementJudgementDraft, ...] = Field(max_length=128)


class ReferenceBinding(Contract):
    """Complete requirement matrix plus conservative definite/possible instance sets."""
    reference_id: str = Field(min_length=1)
    requirement_ids: tuple[str, ...] = Field(max_length=32)
    instance_keys: tuple[str, ...] = Field(max_length=128)
    judgements: tuple[RequirementJudgement, ...] = Field(max_length=4096)
    definite_instance_keys: tuple[str, ...] = Field(max_length=128)
    possible_instance_keys: tuple[str, ...] = Field(max_length=128)
    status: Literal["complete", "unavailable"]
    parse_version: int = Field(ge=1)
    evidence_hash: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binding(self) -> ReferenceBinding:
        for values, name in (
            (self.requirement_ids, "requirement"),
            (self.instance_keys, "instance key"),
            (self.definite_instance_keys, "definite instance"),
            (self.possible_instance_keys, "possible instance"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} keys must be unique")
        definite = set(self.definite_instance_keys)
        possible = set(self.possible_instance_keys)
        instances = set(self.instance_keys)
        if not definite <= possible <= instances:
            raise ValueError("definite instances must be a subset of possible instances")
        if self.status == "unavailable":
            if self.judgements or definite or possible != instances:
                raise ValueError(
                    "unavailable binding retains every instance as possible and none as definite")
            return self
        expected = {
            (requirement_id, instance_key)
            for requirement_id in self.requirement_ids
            for instance_key in self.instance_keys
        }
        actual = {
            (judgement.requirement_id, judgement.instance_key)
            for judgement in self.judgements
        }
        if len(actual) != len(self.judgements) or actual != expected:
            raise ValueError(
                "complete binding requires exactly one judgement per requirement/instance pair")
        if not self.requirement_ids:
            if definite or possible != instances:
                raise ValueError(
                    "a reference with no intrinsic requirements remains possible, not definite")
        else:
            verdicts: dict[str, list[str]] = {key: [] for key in self.instance_keys}
            for judgement in self.judgements:
                verdicts[judgement.instance_key].append(judgement.verdict)
            expected_definite = {
                key for key, values in verdicts.items()
                if all(value == "supported" for value in values)
            }
            expected_possible = {
                key for key, values in verdicts.items()
                if all(value != "contradicted" for value in values)
            }
            if definite != expected_definite or possible != expected_possible:
                raise ValueError(
                    "binding sets must be the three-valued conjunction of complete judgements")
        return self


class SpatialCondition(Contract):
    """A typed current-state or geometric condition over language references."""
    operator: Literal[
        "identity", "left_of", "right_of", "above", "below", "nearest",
        "farthest", "on", "inside", "on_or_inside", "under", "unobstructed",
        "ordinal",
    ]
    reference_ids: tuple[str, ...] = Field(max_length=32)
    direction: Literal[
        "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top",
    ] | None
    rank: int | None
    anchor_semantics: Literal[
        "alternative_single", "any", "all", "collective", "not_applicable",
    ]
    source_occurrence: SourceOccurrence

    @model_validator(mode="after")
    def validate_condition(self) -> SpatialCondition:
        if self.operator == "ordinal":
            if self.direction is None or self.rank is None or self.rank <= 0:
                raise ValueError("ordinal conditions require a direction and positive rank")
            if self.reference_ids:
                raise ValueError("ordinal conditions do not use a reference object")
        else:
            if self.direction is not None or self.rank is not None:
                raise ValueError("only ordinal conditions use direction or rank")
            if self.operator in {"identity", "unobstructed"}:
                if self.reference_ids:
                    raise ValueError(
                        f"{self.operator} conditions do not use a reference object")
            elif not self.reference_ids:
                raise ValueError(
                    f"{self.operator} conditions require a language reference")
        if any(not value for value in self.reference_ids):
            raise ValueError("condition reference IDs cannot be blank")
        has_anchor = bool(self.reference_ids)
        if has_anchor == (self.anchor_semantics == "not_applicable"):
            raise ValueError("anchor semantics must match the condition's anchor structure")
        return self


class ConditionExpression(Contract):
    """Explicit conjunction/disjunction plus set exclusions for a reference."""
    combination: Literal["all", "any"]
    conditions: tuple[SpatialCondition, ...] = Field(min_length=1, max_length=32)
    excluded_reference_ids: tuple[str, ...] = Field(max_length=32)


class ReferenceExpression(Contract):
    """An instruction-grounded entity phrase, not an observed scene object."""
    reference_id: str = Field(min_length=1)
    source_occurrence: SourceOccurrence
    requirements: tuple[Requirement, ...] = Field(max_length=32)
    current_condition: ConditionExpression | None
    antecedent_reference_id: str | None
    antecedent_occurrence: SourceOccurrence | None

    @model_validator(mode="after")
    def validate_antecedent(self) -> ReferenceExpression:
        if (self.antecedent_reference_id is None) != (self.antecedent_occurrence is None):
            raise ValueError(
                "a textual antecedent reference and its source occurrence must be paired")
        return self


class QuantitySpec(Contract):
    """Requested object coverage, independent of operation repetition."""
    mode: Literal["one", "all", "exact_count", "explicit_set"]
    exact_count: int | None
    member_reference_ids: tuple[str, ...] = Field(max_length=32)
    choice_policy: Literal["unique", "any_compatible", "named_members"] | None
    choice_permission: SourceOccurrence | None

    @model_validator(mode="after")
    def validate_quantity(self) -> QuantitySpec:
        if self.mode == "one":
            if self.exact_count is not None or self.member_reference_ids:
                raise ValueError("one quantity cannot carry a count or named members")
            if self.choice_policy not in {"unique", "any_compatible"}:
                raise ValueError("one quantity requires an explicit choice policy")
            if ((self.choice_policy == "any_compatible")
                    != (self.choice_permission is not None)):
                raise ValueError(
                    "any_compatible requires source evidence of explicit permission")
        elif self.mode == "all":
            if (self.exact_count is not None or self.member_reference_ids
                    or self.choice_policy is not None
                    or self.choice_permission is not None):
                raise ValueError("all quantity covers the full reference set without choice")
        elif self.mode == "exact_count":
            if self.exact_count is None or self.exact_count <= 0:
                raise ValueError("exact_count quantity requires a positive count")
            if self.member_reference_ids:
                raise ValueError("exact_count quantity cannot also name an explicit set")
            if self.choice_policy not in {"unique", "any_compatible"}:
                raise ValueError("exact_count quantity requires an explicit choice policy")
            if ((self.choice_policy == "any_compatible")
                    != (self.choice_permission is not None)):
                raise ValueError(
                    "any_compatible requires source evidence of explicit permission")
        elif self.mode == "explicit_set":
            if self.exact_count is not None or not self.member_reference_ids:
                raise ValueError("explicit_set quantity requires named members only")
            if self.choice_policy != "named_members" or self.choice_permission is not None:
                raise ValueError("explicit_set quantity requires named_members choice")
        if any(not value for value in self.member_reference_ids):
            raise ValueError("quantity member reference IDs cannot be blank")
        if len(self.member_reference_ids) != len(set(self.member_reference_ids)):
            raise ValueError("quantity member reference IDs must be unique")
        return self


class TaskClause(Contract):
    """One requested manipulation with reference IDs instead of role descriptions."""
    operation_id: str = Field(min_length=1)
    action: Literal["pick_and_place", "pick_only"]
    target_reference_id: str = Field(min_length=1)
    quantity: QuantitySpec
    final_relation: Literal["held", "on", "in", "under", "uncertain"]
    destination_reference_id: str | None
    upper_reference_id: str | None
    base_relation: Literal["on", "inside"] | None = None
    base_reference_id: str | None
    source_reference_id: str | None
    exclusion_reference_ids: tuple[str, ...] = Field(max_length=32)
    order: int | None
    order_occurrence: SourceOccurrence | None
    source_occurrence: SourceOccurrence

    @model_validator(mode="after")
    def validate_roles(self) -> TaskClause:
        if (self.order is None) != (self.order_occurrence is None):
            raise ValueError("explicit order and its source occurrence must be paired")
        if self.order is not None and self.order <= 0:
            raise ValueError("explicit operation order must be positive")
        if self.final_relation == "held":
            if (self.action != "pick_only" or self.destination_reference_id is not None
                    or self.upper_reference_id is not None
                    or self.base_reference_id is not None or self.base_relation is not None):
                raise ValueError("held operations cannot carry placement references")
        elif self.final_relation in {"on", "in"}:
            if (self.action != "pick_and_place" or self.destination_reference_id is None
                    or self.upper_reference_id is not None
                    or self.base_reference_id is not None or self.base_relation is not None):
                raise ValueError(
                    f"{self.final_relation} operations require only a destination reference")
        elif self.final_relation == "under":
            if (self.action != "pick_and_place" or self.upper_reference_id is None
                    or self.destination_reference_id is not None):
                raise ValueError(
                    "under operations require a distinct upper and a separate optional base")
            if (self.base_reference_id is None) != (self.base_relation is None):
                raise ValueError("an explicit under base requires its on/inside relation")
        elif (self.action != "pick_and_place" or self.destination_reference_id is not None
              or self.upper_reference_id is not None or self.base_reference_id is not None
              or self.base_relation is not None):
            raise ValueError("an unresolved placement relation cannot carry placement roles")
            raise ValueError("an unresolved placement relation must remain a placement action")
        if (self.final_relation == "under" and self.base_reference_id is not None
                and self.base_reference_id == self.upper_reference_id):
            raise ValueError("under upper and base references must be distinct")
        if self.target_reference_id in {
            self.destination_reference_id, self.upper_reference_id, self.base_reference_id,
        }:
            raise ValueError("target, destination, upper, and base references must be distinct")
        if (self.source_reference_id is not None
                and self.source_reference_id == self.target_reference_id):
            raise ValueError("source and target references must be distinct")
        return self


class ProtectionConstraint(Contract):
    """An explicit displacement prohibition or preserve-current-state requirement."""
    constraint_id: str = Field(min_length=1)
    reference_id: str = Field(min_length=1)
    mode: Literal["do_not_move", "preserve_current_state"]
    operation_ids: tuple[str, ...] = Field(max_length=32)
    source_occurrence: SourceOccurrence
    antecedent_reference_id: str | None
    antecedent_occurrence: SourceOccurrence | None

    @model_validator(mode="after")
    def validate_antecedent(self) -> ProtectionConstraint:
        if (self.antecedent_reference_id is None) != (self.antecedent_occurrence is None):
            raise ValueError("protection antecedent IDs and excerpts must be paired")
        return self

class MethodFact(Contract):
    """A physical method the instruction explicitly requests, not a safety verdict."""
    method_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    operation_id: str | None
    source_occurrence: SourceOccurrence


class UnresolvedRequirement(Contract):
    """A source-backed ambiguity retained until evidence or clarification resolves it."""
    source_occurrence: SourceOccurrence
    description: str = Field(min_length=1)
    alternatives: tuple[str, ...] = Field(min_length=2, max_length=8)
    reference_ids: tuple[str, ...] = Field(max_length=32)


class CandidateJudgement(Contract):
    """A model's requested-output opinion, separate from movement operations."""
    kind: Literal["none", "refuse", "clarify"]
    reason: str
    source_occurrence: SourceOccurrence | None
    @model_validator(mode="after")
    def validate_judgement(self) -> CandidateJudgement:
        if self.kind == "none":
            if self.reason or self.source_occurrence is not None:
                raise ValueError("a none candidate judgement carries no reason or excerpt")
        elif not self.reason.strip() or self.source_occurrence is None:
            raise ValueError("a refusal/clarification judgement needs a reason and source excerpt")
        return self



def _compositional_target_reference_ids(operations) -> set[str]:
    """Return explicit-set group targets whose meaning is their named members.

    Member references still need their own typed semantics. Do not exempt a reference
    that is itself listed as a member of another explicit set.
    """
    explicit_targets = set()
    member_ids = set()
    for operation in operations:
        if operation.quantity.mode == "explicit_set":
            explicit_targets.add(operation.target_reference_id)
            member_ids.update(operation.quantity.member_reference_ids)
    return explicit_targets - member_ids


def _validate_reference_semantics(
    references: tuple[ReferenceExpression | ReferenceExpressionDraft, ...],
    unresolved_requirements: tuple[
        UnresolvedRequirement | UnresolvedRequirementDraft, ...
    ],
    *,
    compositional_target_ids: set[str] | None = None,
) -> None:
    """Reject references with no machine-usable meaning or unresolved marker."""
    unresolved_ids = {
        reference_id
        for item in unresolved_requirements
        for reference_id in item.reference_ids
    }
    compositional_targets = compositional_target_ids or set()
    for reference in references:
        if (reference.requirements or reference.current_condition is not None
                or reference.antecedent_reference_id is not None
                or reference.reference_id in unresolved_ids
                or reference.reference_id in compositional_targets):
            continue
        raise ValueError(
            "a reference needs typed requirements, a current condition, an antecedent, "
            "an explicit unresolved entry, or membership in an explicit-set target")


class TaskProgram(Contract):
    """Complete instruction-only program; no scene IDs or geometric facts."""
    instruction: str = Field(min_length=1)
    operations: tuple[TaskClause, ...] = Field(max_length=32)
    references: tuple[ReferenceExpression, ...] = Field(max_length=64)
    protection_constraints: tuple[ProtectionConstraint, ...] = Field(max_length=32)
    method_facts: tuple[MethodFact, ...] = Field(max_length=32)
    unresolved_requirements: tuple[UnresolvedRequirement, ...] = Field(max_length=32)
    candidate_judgement: CandidateJudgement
    parse_version: int = Field(ge=1)
    repair_history: tuple[str, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_program(self) -> TaskProgram:
        reference_ids = [reference.reference_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference IDs must be unique")
        _validate_reference_semantics(
            self.references, self.unresolved_requirements,
            compositional_target_ids=_compositional_target_reference_ids(self.operations),
        )
        reference_set = set(reference_ids)
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("operation IDs must be unique")
        operation_set = set(operation_ids)
        method_ids = [fact.method_id for fact in self.method_facts]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("method fact IDs must be unique")
        requirement_ids = [
            requirement.requirement_id
            for reference in self.references
            for requirement in reference.requirements
        ]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement IDs must be unique")
        ordered = [operation.order for operation in self.operations
                   if operation.order is not None]
        if len(ordered) != len(set(ordered)):
            raise ValueError("explicit operation order values must be unique")
        if ordered and len(ordered) != len(self.operations):
            raise ValueError("explicit operation order must cover every operation")
        if ordered and sorted(ordered) != list(range(1, len(self.operations) + 1)):
            raise ValueError("explicit operation order must be contiguous from one")
            raise ValueError("explicit operation order values must be unique")
        for reference in self.references:
            if (reference.antecedent_reference_id is not None
                    and reference.antecedent_reference_id not in reference_set):
                raise ValueError("antecedent reference is not in the reference table")
            if reference.antecedent_reference_id == reference.reference_id:
                raise ValueError("a reference cannot be its own antecedent")
            if reference.current_condition is not None:
                condition_ids = {
                    reference_id
                    for condition in reference.current_condition.conditions
                    for reference_id in condition.reference_ids
                } | set(reference.current_condition.excluded_reference_ids)
                if not condition_ids <= reference_set:
                    raise ValueError("condition references must exist in the reference table")
        antecedents = {
            reference.reference_id: reference.antecedent_reference_id
            for reference in self.references
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(reference_id: str) -> None:
            if reference_id in visiting:
                raise ValueError("textual antecedent dependencies must be acyclic")
            if reference_id in visited:
                return
            visiting.add(reference_id)
            parent = antecedents[reference_id]
            if parent is not None:
                visit(parent)
            visiting.remove(reference_id)
            visited.add(reference_id)

        for reference_id in antecedents:
            visit(reference_id)
        for operation in self.operations:
            operation_reference_ids = {
                operation.target_reference_id,
                *operation.quantity.member_reference_ids,
                *operation.exclusion_reference_ids,
            }
            operation_reference_ids.update(
                reference_id for reference_id in (
                    operation.destination_reference_id,
                    operation.upper_reference_id,
                    operation.base_reference_id,
                    operation.source_reference_id,
                ) if reference_id is not None)
            if not operation_reference_ids <= reference_set:
                raise ValueError("operation references must exist in the reference table")
        for constraint in self.protection_constraints:
            if constraint.reference_id not in reference_set:
                raise ValueError("protection reference must exist in the reference table")
            if (constraint.antecedent_reference_id is not None
                    and constraint.antecedent_reference_id not in reference_set):
                raise ValueError("protection antecedent must exist in the reference table")
            if not set(constraint.operation_ids) <= operation_set:
                raise ValueError("protection scope must name existing operations")
        for fact in self.method_facts:
            if fact.operation_id is not None and fact.operation_id not in operation_set:
                raise ValueError("method fact must reference an existing operation")
        for unresolved in self.unresolved_requirements:
            if not set(unresolved.reference_ids) <= reference_set:
                raise ValueError("unresolved requirement references must exist")
        if not any((self.operations, self.protection_constraints, self.method_facts,
                    self.unresolved_requirements, self.candidate_judgement.kind != "none")):
            raise ValueError("task program must preserve at least one stated task fact")
        for occurrence in self._source_occurrences():
            if self.instruction[occurrence.start:occurrence.end] != occurrence.excerpt:
                raise ValueError("source occurrence does not match the original instruction")
        return self

    def _source_occurrences(self) -> tuple[SourceOccurrence, ...]:
        occurrences = []
        for reference in self.references:
            occurrences.append(reference.source_occurrence)
            occurrences.extend(item.source_occurrence for item in reference.requirements)
            if reference.antecedent_occurrence is not None:
                occurrences.append(reference.antecedent_occurrence)
            if reference.current_condition is not None:
                occurrences.extend(
                    condition.source_occurrence
                    for condition in reference.current_condition.conditions)
        for operation in self.operations:
            occurrences.append(operation.source_occurrence)
            if operation.order_occurrence is not None:
                occurrences.append(operation.order_occurrence)
            if operation.quantity.choice_permission is not None:
                occurrences.append(operation.quantity.choice_permission)
        for constraint in self.protection_constraints:
            occurrences.append(constraint.source_occurrence)
            if constraint.antecedent_occurrence is not None:
                occurrences.append(constraint.antecedent_occurrence)
        occurrences.extend(item.source_occurrence for item in self.method_facts)
        occurrences.extend(item.source_occurrence for item in self.unresolved_requirements)
        if self.candidate_judgement.source_occurrence is not None:
            occurrences.append(self.candidate_judgement.source_occurrence)
        return tuple(occurrences)


class RequirementDraft(Contract):


    requirement_id: str = Field(min_length=1)
    aspect: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source_excerpt: str = Field(min_length=1)


class SpatialConditionDraft(Contract):
    operator: Literal[
        "identity", "left_of", "right_of", "above", "below", "nearest",
        "farthest", "on", "inside", "on_or_inside", "under", "unobstructed",
        "ordinal",
    ]
    reference_ids: tuple[str, ...] = Field(max_length=32)
    direction: Literal[
        "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top",
    ] | None
    rank: int | None
    anchor_semantics: Literal[
        "alternative_single", "any", "all", "collective", "not_applicable",
    ]
    source_excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_condition(self) -> SpatialConditionDraft:
        if self.operator == "ordinal":
            if self.direction is None or self.rank is None or self.rank <= 0:
                raise ValueError("ordinal requires a direction and positive rank")
            if self.reference_ids:
                raise ValueError("ordinal cannot use a reference object")
        else:
            if self.direction is not None or self.rank is not None:
                raise ValueError("only ordinal conditions use direction or rank")
            if self.operator in {"identity", "unobstructed"}:
                if self.reference_ids:
                    raise ValueError(f"{self.operator} does not use a reference object")
            elif not self.reference_ids:
                raise ValueError(f"{self.operator} requires a reference object")
        if any(not value for value in self.reference_ids):
            raise ValueError("condition reference IDs cannot be blank")
        if bool(self.reference_ids) == (self.anchor_semantics == "not_applicable"):
            raise ValueError("anchor semantics must match the condition's anchor structure")
        return self


class ConditionExpressionDraft(Contract):
    combination: Literal["all", "any"]
    conditions: tuple[SpatialConditionDraft, ...] = Field(min_length=1, max_length=32)
    excluded_reference_ids: tuple[str, ...] = Field(max_length=32)


class ReferenceExpressionDraft(Contract):
    reference_id: str = Field(min_length=1)
    source_excerpt: str = Field(min_length=1)
    requirements: tuple[RequirementDraft, ...] = Field(max_length=32)
    current_condition: ConditionExpressionDraft | None
    antecedent_reference_id: str | None
    antecedent_excerpt: str | None

    @model_validator(mode="after")
    def validate_antecedent(self) -> ReferenceExpressionDraft:
        if (self.antecedent_reference_id is None) != (self.antecedent_excerpt is None):
            raise ValueError("antecedent IDs and excerpts must be paired")
        return self


class QuantitySpecDraft(Contract):
    mode: Literal["one", "all", "exact_count", "explicit_set"]
    exact_count: int | None
    member_reference_ids: tuple[str, ...] = Field(max_length=32)
    choice_policy: Literal["unique", "any_compatible", "named_members"] | None
    choice_permission_excerpt: str | None

    @model_validator(mode="after")
    def validate_quantity(self) -> QuantitySpecDraft:
        if self.mode == "one":
            if self.exact_count is not None or self.member_reference_ids:
                raise ValueError("one quantity cannot carry a count or named members")
            if self.choice_policy not in {"unique", "any_compatible"}:
                raise ValueError("one quantity requires an explicit choice policy")
        elif self.mode == "all":
            if (self.exact_count is not None or self.member_reference_ids
                    or self.choice_policy is not None
                    or self.choice_permission_excerpt is not None):
                raise ValueError("all quantity covers the full set without choice")
        elif self.mode == "exact_count":
            if self.exact_count is None or self.exact_count <= 0:
                raise ValueError("exact_count requires a positive count")
            if self.member_reference_ids:
                raise ValueError("exact_count cannot carry explicit members")
            if self.choice_policy not in {"unique", "any_compatible"}:
                raise ValueError("exact_count requires an explicit choice policy")
        elif self.mode == "explicit_set":
            if self.exact_count is not None or not self.member_reference_ids:
                raise ValueError("explicit_set requires named members")
            if self.choice_policy != "named_members":
                raise ValueError("explicit_set requires named_members choice")
        if ((self.choice_policy == "any_compatible")
                != (self.choice_permission_excerpt is not None)):
            raise ValueError("any_compatible requires a source permission excerpt")
        if len(self.member_reference_ids) != len(set(self.member_reference_ids)):
            raise ValueError("quantity member reference IDs must be unique")
        return self


class TaskClauseDraft(Contract):
    operation_id: str = Field(min_length=1)
    action: Literal["pick_and_place", "pick_only"]
    target_reference_id: str = Field(min_length=1)
    quantity: QuantitySpecDraft
    final_relation: Literal["held", "on", "in", "under", "uncertain"]
    destination_reference_id: str | None
    upper_reference_id: str | None
    base_relation: Literal["on", "inside"] | None
    base_reference_id: str | None
    source_reference_id: str | None
    exclusion_reference_ids: tuple[str, ...] = Field(max_length=32)
    order: int | None
    order_excerpt: str | None
    source_excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_clause(self) -> TaskClauseDraft:
        if (self.order is None) != (self.order_excerpt is None):
            raise ValueError("explicit order requires its exact source excerpt")
        if self.order is not None and self.order <= 0:
            raise ValueError("explicit order must be positive")
        if self.final_relation == "held":
            if (self.action != "pick_only" or self.destination_reference_id is not None
                    or self.upper_reference_id is not None
                    or self.base_reference_id is not None or self.base_relation is not None):
                raise ValueError("held operations cannot carry placement roles")
        elif self.final_relation in {"on", "in"}:
            if (self.action != "pick_and_place"
                    or self.destination_reference_id is None
                    or self.upper_reference_id is not None
                    or self.base_reference_id is not None or self.base_relation is not None):
                raise ValueError("on/in requires a destination and no under roles")
        elif self.final_relation == "under":
            if (self.action != "pick_and_place" or self.upper_reference_id is None
                    or self.destination_reference_id is not None):
                raise ValueError("under requires a distinct upper and optional base")
            if (self.base_reference_id is None) != (self.base_relation is None):
                raise ValueError("an explicit under base requires its on/inside relation")
            if (self.base_reference_id is not None
                    and self.base_reference_id == self.upper_reference_id):
                raise ValueError("under upper and base references must be distinct")
        elif (self.action != "pick_and_place" or self.destination_reference_id is not None
              or self.upper_reference_id is not None or self.base_reference_id is not None
              or self.base_relation is not None):
            raise ValueError("uncertain final relation cannot carry placement roles")
        if (self.source_reference_id is not None
                and self.source_reference_id == self.target_reference_id):
            raise ValueError("source and target references must be distinct")
        return self
        if (self.source_reference_id is not None
                and self.source_reference_id == self.target_reference_id):
            raise ValueError("source and target references must be distinct")
        return self


class ProtectionConstraintDraft(Contract):
    constraint_id: str = Field(min_length=1)
    reference_id: str = Field(min_length=1)
    mode: Literal["do_not_move", "preserve_current_state"]
    operation_ids: tuple[str, ...] = Field(max_length=32)
    source_excerpt: str = Field(min_length=1)
    antecedent_reference_id: str | None
    antecedent_excerpt: str | None

    @model_validator(mode="after")
    def validate_antecedent(self) -> ProtectionConstraintDraft:
        if (self.antecedent_reference_id is None) != (self.antecedent_excerpt is None):
            raise ValueError("protection antecedent IDs and excerpts must be paired")
        return self


class UnresolvedRequirementDraft(Contract):
    source_excerpt: str = Field(min_length=1)
    description: str = Field(min_length=1)
    alternatives: tuple[str, ...] = Field(min_length=2, max_length=8)
    reference_ids: tuple[str, ...] = Field(max_length=32)


class CandidateJudgementDraft(Contract):
    kind: Literal["none", "refuse", "clarify"]
    reason: str
    source_excerpt: str | None

    @model_validator(mode="after")
    def validate_judgement(self) -> CandidateJudgementDraft:
        if self.kind == "none":
            if self.reason or self.source_excerpt is not None:
                raise ValueError("a none candidate judgement carries no reason or excerpt")
        elif not self.reason.strip() or self.source_excerpt is None:
            raise ValueError(
                "a refusal/clarification judgement needs a reason and source excerpt")
        return self


class MethodFactDraft(Contract):
    method_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    operation_id: str | None
    source_excerpt: str = Field(min_length=1)


class TaskProgramDraft(Contract):
    """Untrusted model reply before exact source spans are attached and validated."""
    operations: tuple[TaskClauseDraft, ...] = Field(max_length=32)
    references: tuple[ReferenceExpressionDraft, ...] = Field(max_length=64)
    protection_constraints: tuple[ProtectionConstraintDraft, ...] = Field(max_length=32)
    unresolved_requirements: tuple[UnresolvedRequirementDraft, ...] = Field(max_length=32)
    candidate_judgement: CandidateJudgementDraft
    method_facts: tuple[MethodFactDraft, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def validate_links(self) -> TaskProgramDraft:
        references = {reference.reference_id for reference in self.references}
        if len(references) != len(self.references):
            raise ValueError("reference IDs must be unique")
        operations = {operation.operation_id for operation in self.operations}
        if len(operations) != len(self.operations):
            raise ValueError("operation IDs must be unique")
        method_ids = [fact.method_id for fact in self.method_facts]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("method fact IDs must be unique")
        _validate_reference_semantics(
            self.references, self.unresolved_requirements,
            compositional_target_ids=_compositional_target_reference_ids(self.operations),
        )
        stated_orders = [operation.order for operation in self.operations
                         if operation.order is not None]
        if len(stated_orders) != len(set(stated_orders)):
            raise ValueError("explicit operation order values must be unique")
        for reference in self.references:
            if (reference.antecedent_reference_id is None
                    and reference.antecedent_excerpt is not None):
                raise ValueError("antecedent excerpt requires an antecedent reference")
            if (reference.antecedent_reference_id is not None
                    and reference.antecedent_excerpt is None):
                raise ValueError("antecedent reference requires its source excerpt")
            if (reference.antecedent_reference_id is not None
                    and reference.antecedent_reference_id not in references):
                raise ValueError("antecedent reference must exist in the reference table")
            if reference.current_condition is not None:
                condition_ids = {
                    ref_id for condition in reference.current_condition.conditions
                    for ref_id in condition.reference_ids
                } | set(reference.current_condition.excluded_reference_ids)
                if not condition_ids <= references:
                    raise ValueError("condition references must exist in the reference table")
        for operation in self.operations:
            ids = {
                operation.target_reference_id,
                *operation.quantity.member_reference_ids,
                *operation.exclusion_reference_ids,
            }
            ids.update(ref_id for ref_id in (
                operation.destination_reference_id, operation.upper_reference_id,
                operation.base_reference_id, operation.source_reference_id,
            ) if ref_id is not None)
            if not ids <= references:
                raise ValueError("operation references must exist in the reference table")
            if (operation.order is None) != (operation.order_excerpt is None):
                raise ValueError("explicit order requires its source excerpt")
            if operation.quantity.choice_policy == "any_compatible":
                if operation.quantity.choice_permission_excerpt is None:
                    raise ValueError(
                        "any_compatible requires an explicit permission source excerpt")
            elif operation.quantity.choice_permission_excerpt is not None:
                raise ValueError(
                    "choice permission excerpt is only valid for any_compatible")
        for fact in self.method_facts:
            if fact.operation_id is not None and fact.operation_id not in operations:
                raise ValueError("method fact must reference an existing operation")
        for constraint in self.protection_constraints:
            if constraint.reference_id not in references:
                raise ValueError("protection reference must exist in the reference table")
            if not set(constraint.operation_ids) <= operations:
                raise ValueError("protection scope must name existing operations")
            if (constraint.antecedent_reference_id is not None
                    and constraint.antecedent_reference_id not in references):
                raise ValueError("protection antecedent must exist in the reference table")
        for unresolved in self.unresolved_requirements:
            if not set(unresolved.reference_ids) <= references:
                raise ValueError("unresolved requirement references must exist")
        return self




class Goal(Contract):
    target: str = Field(min_length=1)
    # For on/in goals destination is support/container. For under goals,
    # under_object is the upper object and destination is the base surface.
    destination: str | None = None
    relation: Literal["on", "under", "in", "held"] = "on"
    under_object: str | None = None
    base_relation: Literal["on", "inside"] = "on"

    @model_validator(mode="after")
    def validate_goal(self) -> Goal:
        # An under goal names the upper object either in `under_object` or, in the
        # two-stage intent, as the destination the compiler stages first. Both
        # spellings reach the same compiler path, so neither is required alone.
        if self.relation == "under" and self.under_object == self.target:
            raise ValueError("an under goal's target and upper object must differ")
        if self.relation != "under" and self.under_object is not None:
            raise ValueError("under_object is only valid for an under goal")
        if self.relation == "held" and self.destination is not None:
            raise ValueError("a held goal cannot have a destination")
        return self
class GroundedOperation(Contract):
    """One original manipulation with its resolved target set and goal instances."""
    clause: TaskClause
    definite_target_ids: tuple[str, ...] = Field(max_length=128)
    possible_target_ids: tuple[str, ...] = Field(max_length=128)
    selected_target_ids: tuple[str, ...] = Field(max_length=128)
    goals: tuple[Goal, ...] = Field(max_length=128)
    resolution_status: Literal["resolved", "unresolved", "absent", "unavailable"]
    reason: str = ""

    @model_validator(mode="after")
    def validate_operation(self) -> GroundedOperation:
        definite = set(self.definite_target_ids)
        possible = set(self.possible_target_ids)
        selected = set(self.selected_target_ids)
        if any(len(values) != len(set(values)) for values in (
            self.definite_target_ids, self.possible_target_ids, self.selected_target_ids,
        )):
            raise ValueError("grounded target ID sets must be unique")
        if not definite <= possible:
            raise ValueError("definite targets must be a subset of possible targets")
        if not selected <= definite:
            raise ValueError("selected targets must be definitely grounded")
        goal_targets = tuple(goal.target for goal in self.goals)
        if goal_targets != self.selected_target_ids:
            raise ValueError("grounded goals must retain selected-target order")
        goal_targets = tuple(goal.target for goal in self.goals)
        if len(goal_targets) != len(set(goal_targets)) or set(goal_targets) != selected:
            raise ValueError("grounded goals must cover exactly the selected targets")
        if self.resolution_status == "resolved":
            if not selected or self.reason:
                raise ValueError("a resolved operation needs selected targets and no error")
        elif selected or self.goals or not self.reason.strip():
            raise ValueError("an unresolved operation needs a reason and no selected goals")
        return self
class GroundedProtection(Contract):
    constraint: ProtectionConstraint
    object_ids: tuple[str, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def validate_protection(self) -> GroundedProtection:
        if not self.object_ids or len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("grounded protection needs unique observed object IDs")
        return self




class GroundedTask(Contract):
    """Immutable grounded request; its operations remain the authoritative obligations."""
    instruction: str = Field(min_length=1)
    program: TaskProgram | None
    parse_version: int = Field(ge=0)
    protection_scopes: tuple[GroundedProtection, ...] = Field(max_length=32)
    evidence_hash: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")
    bindings: tuple[ReferenceBinding, ...] = Field(max_length=64)
    operations: tuple[GroundedOperation, ...] = Field(max_length=32)
    protected_ids: tuple[str, ...] = Field(max_length=128)
    status: Literal["grounded", "unresolved", "infeasible", "unavailable"]
    unresolved_requirements: tuple[str, ...] = Field(max_length=64)
    question: str = ""
    reason: str = ""

    @model_validator(mode="after")
    def validate_grounded_task(self) -> GroundedTask:
        if len(self.protected_ids) != len(set(self.protected_ids)):
            raise ValueError("grounded protection IDs must be unique")
        scoped_ids = {
            object_id for scope in self.protection_scopes for object_id in scope.object_ids
        }
        if self.protection_scopes and scoped_ids != set(self.protected_ids):
            raise ValueError("protected_ids must be the union of grounded protection scopes")
        binding_ids = [binding.reference_id for binding in self.bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("grounded reference bindings must be unique")
        if self.program is None:
            if (self.status != "unavailable" or self.parse_version != 0
                    or self.bindings or self.operations or not self.unresolved_requirements
                    or not self.question.strip()):
                raise ValueError(
                    "a missing TaskProgram is an unavailable task with an explicit question")
            return self
        reference_ids = {reference.reference_id for reference in self.program.references}
        if set(binding_ids) != reference_ids:
            raise ValueError("grounded task must bind every TaskProgram reference exactly once")
            raise ValueError("grounded task parse version must match its TaskProgram")
        reference_ids = {reference.reference_id for reference in self.program.references}
        if not set(binding_ids) <= reference_ids:
            raise ValueError("grounded bindings must reference the TaskProgram table")
        if any(binding.parse_version != self.parse_version for binding in self.bindings):
            raise ValueError("grounded bindings must use the current parse version")
        operation_ids = tuple(item.clause.operation_id for item in self.operations)
        if operation_ids != tuple(item.operation_id for item in self.program.operations):
            raise ValueError("grounded task must preserve every operation in source order")
        program_protection = {
            constraint.constraint_id: constraint
            for constraint in self.program.protection_constraints
        }
        if any(
            scope.constraint.constraint_id not in program_protection
            or scope.constraint != program_protection[scope.constraint.constraint_id]
            for scope in self.protection_scopes
        ):
            raise ValueError("grounded protection must preserve its TaskProgram constraint")
            raise ValueError("grounded task must preserve every operation in source order")
            if (self.unresolved_requirements or self.question or self.reason
                    or any(binding.status != "complete" for binding in self.bindings)
                    or any(item.resolution_status != "resolved" for item in self.operations)):
                raise ValueError("a grounded task cannot hide unresolved operations")
        if self.status == "grounded":
            if (self.unresolved_requirements or self.question or self.reason
                    or any(item.resolution_status != "resolved" for item in self.operations)):
                raise ValueError("a grounded task cannot hide unresolved operations")
        elif self.status == "unresolved" and not self.question.strip():
            raise ValueError("an unresolved task needs a clarification question")
        elif self.status == "infeasible" and not self.reason.strip():
            raise ValueError("an infeasible task needs supporting evidence")
        elif self.status == "unavailable" and (
            not self.unresolved_requirements or not self.question.strip()
        ):
            raise ValueError("an unavailable task needs a diagnosed dependency question")
        return self

    @property
    def requested_goals(self) -> tuple[Goal, ...]:
        """All concrete goals, preserving explicit order or source order."""
        operations = list(self.operations)
        if self.goal_mode == "ordered":
            operations.sort(key=lambda operation: operation.clause.order)
        return tuple(goal for operation in operations for goal in operation.goals)

    @property
    def goal_mode(self) -> Literal["ordered", "final_state"]:
        """Only explicitly ordered operations impose temporal goal order."""
        return ("ordered" if any(operation.clause.order is not None
                                 for operation in self.operations)
                else "final_state")

    def protected_ids_for(self, operation_id: str) -> tuple[str, ...]:
        if not self.protection_scopes:
            return self.protected_ids
        active = {
            object_id
            for scope in self.protection_scopes
            if (not scope.constraint.operation_ids
                or operation_id in scope.constraint.operation_ids)
            for object_id in scope.object_ids
        }
        return tuple(object_id for object_id in self.protected_ids if object_id in active)
        """Only explicitly ordered operations impose temporal goal order."""
        return ("ordered" if any(operation.clause.order is not None
                                 for operation in self.operations)
                else "final_state")


class OperationCoverage(Contract):
    operation_id: str = Field(min_length=1)
    status: Literal["complete", "unresolved", "infeasible", "unavailable"]
    detail: str = Field(min_length=1)


def grounded_task_fingerprint(task: GroundedTask) -> str:
    """Stable identity for the exact immutable request passed to compilation."""
    payload = json.dumps(
        task.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def exclusive_output_error(
    kind: str, *, question: str, reason: str, has_goals: bool, has_actions: bool,
) -> str | None:
    """Single source of truth for plan/ask_user/refuse exclusivity.
    ``GroundedTask``, ``Proposal`` and the verifier share this contract so the
    public answer remains exactly one plan, ask_user, or refuse.
    refusal explains and never asks.
    """
    if kind == "plan":
        if not has_goals and not has_actions:
            return "a plan must contain goals or actions"
        if question.strip():
            return "a plan cannot also ask a question"
        return None
    if has_goals or has_actions:
        return "questions/refusals cannot contain movement goals or actions"
    if kind == "ask_user":
        if not question.strip():
            return "clarification requires a question"
        if reason.strip():
            return "clarification must not carry a refusal reason"
        return None
    if kind == "refuse":
        if not reason.strip():
            return "refusal requires a reason"
        if question.strip():
            return "refusal must not carry a clarification question"
        return None
    return "output must be exclusively plan, ask_user, or refuse"


class TaskIntent(Contract):
    kind: Literal["plan", "ask_user", "refuse"]
    goals: tuple[Goal, ...] = ()
    goal_mode: Literal["ordered", "final_state"] = "ordered"
    question: str = ""
    reason: str = ""
    forbidden_movement: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_intent(self) -> TaskIntent:
        error = exclusive_output_error(
            self.kind, question=self.question, reason=self.reason,
            has_goals=bool(self.goals), has_actions=False,
        )
        if error is not None:
            raise ValueError(error)
        return self


class ProposedAction(Contract):
    action: Literal["pick", "place"]
    object: str = Field(min_length=1)
    relation: Literal["on", "inside"] = "on"


class Proposal(Contract):
    kind: Literal["plan", "ask_user", "refuse"]
    actions: tuple[ProposedAction, ...] = ()
    goals: tuple[Goal, ...] = ()
    # Requested goals the compiler proved infeasible and did NOT attempt. They are kept
    # so the verifier and the audit can report an incomplete safe prefix instead of
    # presenting the surviving goals as the whole request (a plan must never silently
    # redefine the task by dropping what it cannot do).
    skipped_goals: tuple[Goal, ...] = ()
    goal_mode: Literal["ordered", "final_state"] = "ordered"
    question: str = ""
    reason: str = ""
    forbidden_movement: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_decision(self) -> Proposal:
        error = exclusive_output_error(
            self.kind, question=self.question, reason=self.reason,
            has_goals=bool(self.goals), has_actions=bool(self.actions),
        )
        if error is not None:
            raise ValueError(error)
        if self.kind == "plan" and not self.actions:
            raise ValueError("a plan must contain actions")
        return self


class Check(Contract):
    rule: str
    passed: bool
    detail: str
    step: int | None = None
class ExecutionOutcome(Contract):
    """Compiler result with explicit coverage of the original grounded operations."""
    status: Literal["complete", "unresolved", "infeasible", "unavailable"]
    proposal: Proposal
    task_sha256: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")
    parse_version: int = Field(ge=0)
    action_operation_ids: tuple[str, ...] = Field(max_length=256)
    safe_prefix_operation_ids: tuple[str, ...] = Field(max_length=256)
    evidence_hash: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")
    requested_operation_ids: tuple[str, ...] = Field(max_length=32)
    completed_operation_ids: tuple[str, ...] = Field(max_length=32)
    unsatisfied_operation_ids: tuple[str, ...] = Field(max_length=32)
    safe_prefix: tuple[ProposedAction, ...] = Field(max_length=256)
    diagnostics: tuple[str, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_coverage(self) -> ExecutionOutcome:
        requested = set(self.requested_operation_ids)
        completed = set(self.completed_operation_ids)
        unsatisfied = set(self.unsatisfied_operation_ids)
        if any(len(values) != len(set(values)) for values in (
            self.requested_operation_ids, self.completed_operation_ids,
            self.unsatisfied_operation_ids,
        )):
            raise ValueError("execution coverage operation IDs must be unique")
        if completed & unsatisfied or completed | unsatisfied != requested:
            raise ValueError("execution outcome must account for every requested operation")
        if (len(self.action_operation_ids) != len(self.proposal.actions)
                or len(self.safe_prefix_operation_ids) != len(self.safe_prefix)):
            raise ValueError("execution action ownership must align with emitted actions")
        if not set(self.action_operation_ids) <= requested or not set(
            self.safe_prefix_operation_ids
        ) <= requested:
            raise ValueError("execution actions must belong to requested operations")
        if completed & unsatisfied or completed | unsatisfied != requested:
            raise ValueError("execution outcome must account for every requested operation")
        expected_kind = {
            "complete": "plan",
            "unresolved": "ask_user",
            "unavailable": "ask_user",
            "infeasible": "refuse",
        }[self.status]
        if self.proposal.kind != expected_kind:
            raise ValueError("execution status and official proposal kind disagree")
        if self.status != "unavailable" and self.parse_version == 0:
            raise ValueError("a parsed execution outcome needs a positive parse version")
        if self.status == "complete":
            if completed != requested or self.safe_prefix or self.unsatisfied_operation_ids:
                raise ValueError("a complete outcome must satisfy every operation")
        elif self.proposal.actions:
            raise ValueError("incomplete outcomes cannot expose prefix actions as a plan")
        return self




class Decision(Contract):
    instruction_id: str
    target_pixel: Pixel | None
    actions: tuple[dict[str, str], ...]
    kind: Literal["plan", "ask_user", "refuse"]
    checks: tuple[Check, ...]
    proposal: Proposal
    # Coverage of the original operations. Optional because a goal-ordered
    # intent has no typed operation set to cover; the verifier then reports the
    # plan alone.
    execution_outcome: ExecutionOutcome | None = None
    model: str
    reasoning_seconds: float = Field(ge=0)
    usage: dict[str, int] = Field(default_factory=dict)
    repaired: bool = False
    intent: TaskIntent | None = None
    semantic_interpretation: dict = Field(default_factory=dict)

    def official(self) -> dict:
        """Project the richer internal plan to the official answer schema."""
        return {
            "instruction_id": self.instruction_id,
            "target_pixel": list(self.target_pixel) if self.target_pixel is not None else None,
            "actions": list(self.actions),
        }
