"""Cache image observations; interpret grounded goals; compile and verify actions.

One instruction is decided in four stages, each owned by a different component:

1. **Linguistic semantics** (``reasoning.extract_semantics``) reads the
   instruction alone — no scene, no IDs — and returns its structured meaning:
   target, destination, under-object, exclusions, protected objects, unsafe
   manner.
2. **Spatial grounding** (``reasoning.interpret``) binds that meaning to observed
   IDs with a schema that constrains ``target``/``destination`` to real IDs at
   decode time, so an invented object cannot be emitted.
3. **Deterministic guardrails** (``referents``) correct only what the scene itself
   proves: a refusal whose referent is uniquely present, a question a qualifier
   settles, a plan the instruction's own manner or constraint forbids, a colour the
   scene refutes at a distance, a restriction the instruction does not license, and
   the order of goals that a blocker must be cleared before. Every correction needs
   a unique, provable referent — an ambiguous one is left to the model.
4. **Compile and verify** (``compiler`` + ``planner``) sequence the plan and
   replay it against the observed state, so a surviving answer is one whose every
   action is physically executable. A rejected plan gets one bounded repair and
   then the run fails honestly rather than emitting an unverified answer.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from pydantic import ValidationError

from .clients import Deadline, LocalChatClient, ModelError
from .compiler import compile_intent
from .domain import Decision, SceneGraph
from .perception import Perception
from .planner import PlanViolation, verify
from .reasoning import extract_semantics, interpret
from .referents import (
    clear_blockers_first,
    clear_unlicensed_restrictions,
    disambiguate_plan,
    rescue_ambiguous_refusal,
    rescue_clarification,
    rescue_refusal,
    resolve_target,
    under_placement_first_pick,
    veto_plan,
    veto_unmatched_colour,
)
from .runtime import ManagedChatClient, ModelRuntime
from .trace import TraceRecorder

# Every module whose bytes can change an observed graph. The cache key hashes
# them, so editing any of them forces a re-perception instead of silently mixing
# two perception revisions in one results file.
_PERCEPTION_SOURCES = (
    "perception.py", "perception_hybrid.py", "geometry.py", "canonical.py", "colour.py",
    "domain.py", "support_discovery.py",
)


class Pipeline:
    """Own the perception cache and decide one instruction at a time.

    Vision and reasoning are separate clients: the vision model is local (it must
    see the image), and the reasoning model is either a remote Apertus endpoint
    during development or the locally owned llama-server at the submission gate.
    """

    def __init__(
        self, data: Path, cache: Path,
        vision_url: str = "http://127.0.0.1:8081/v1",
        reasoning_url: str = "http://127.0.0.1:8082/v1",
        vision_model: str = "qwen3.5-9b-mtp-q4",
        reasoning_model: str = "apertus-v1.5-8b-q8_0",
        segment: bool = True, vision_device: str = "cpu", timeout: float = 240,
        runtime: ModelRuntime | None = None, reasoning_api_key: str | None = None,
        vision_api_key: str | None = None,
        vision_backend: str = "fused",
        decision_timeout: float | None = None,
        trace_path: Path | None = None,
    ):
        if "apertus" not in reasoning_model.casefold():
            raise ValueError("Apertus is required as the language reasoning model")
        self.data = Path(data).resolve()
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.evaluation_mode = "standard_pipeline"
        self.decision_timeout = decision_timeout
        self.vision = ManagedChatClient(runtime, "vision", timeout) if runtime else (
            LocalChatClient(vision_url, vision_model, timeout, temperature=.7, top_p=.8,
                            api_key=vision_api_key))
        self.reasoner = (
            LocalChatClient(reasoning_url, reasoning_model, timeout, api_key=reasoning_api_key)
            if reasoning_api_key or runtime is None else
            ManagedChatClient(runtime, "reasoning", timeout)
        )
        if vision_backend == "hybrid":
            from .perception_hybrid import HybridPerception
            self.perception = HybridPerception(self.vision, segment=segment,
                                               device=vision_device)
        elif vision_backend == "fused":
            from .perception_hybrid import FusedPerception
            self.perception = FusedPerception(self.vision, segment=segment,
                                              device=vision_device)
        else:
            self.perception = Perception(self.vision, segment=segment, device=vision_device)
        self.trace = TraceRecorder(Path(trace_path)) if trace_path else None
        # The trace covers perception too: a raw inventory reply is the evidence for
        # what the vision model actually reported, before any filter or normalisation.
        for client in (self.vision, self.reasoner):
            client.trace = self.trace
        self._lock = threading.RLock()
        self._graphs: dict[str, SceneGraph] = {}
        self._source_hash = hashlib.sha256(b"".join(
            (Path(__file__).parent / name).read_bytes() for name in _PERCEPTION_SOURCES
        )).hexdigest()

    @property
    def code_revision(self) -> str:
        """Identity of the perception stack, used to keep resumed runs comparable."""
        return self._source_hash

    def folder(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
            raise ValueError("invalid scene name")
        folder = (self.data / name).resolve()
        if folder.parent != self.data or not folder.is_dir() or not (folder / "rgb.png").is_file():
            raise ValueError("unknown scene")
        return folder

    def scene(self, name: str) -> SceneGraph:
        """Observed graph for one scene, from the cache when the revision matches.

        The cache key covers the image bytes, the perception source revision and
        the perception configuration, so a graph is only reused for the exact
        image and the exact code that produced it.
        """
        folder = self.folder(name)
        image_hash = hashlib.sha256((folder / "rgb.png").read_bytes()).hexdigest()
        key = hashlib.sha256(
            (image_hash + self._source_hash + self.perception.signature).encode()
        ).hexdigest()
        cache_path = self.cache / (key + ".scene.json")
        with self._lock:
            if key in self._graphs:
                return self._graphs[key].model_copy(update={"scene": name})
            if cache_path.exists():
                graph = SceneGraph.model_validate_json(cache_path.read_text())
                if graph.image_sha256 != image_hash:
                    raise RuntimeError("scene cache image identity mismatch")
                graph = graph.model_copy(update={"scene": name})
            else:
                graph = self.perception.observe(folder)
                if graph.image_sha256 != image_hash:
                    raise RuntimeError("image changed during perception; retry with a stable scene")
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(graph.model_dump_json(indent=2))
                temporary.replace(cache_path)
            self._graphs[key] = graph
            return graph

    def decide(self, scene_name: str, instruction: str, instruction_id: str) -> Decision:
        """One official answer, verified against the observed state."""
        instruction = instruction.strip()
        if not instruction or len(instruction) > 2000:
            raise ValueError("instruction must contain 1-2000 characters")
        graph = self.scene(scene_name)
        deadline = Deadline(self.decision_timeout)
        # One wall-clock budget per decision; the trace is set once at construction
        # so perception requests are recorded too.
        for client in (self.reasoner, self.vision):
            client.deadline = deadline
        feedback = ""
        usage: dict[str, int] = {}
        start = time.perf_counter()
        with self._lock:
            self.perception.release()
            semantics, stage_one_usage = extract_semantics(self.reasoner, instruction)
            for key, value in stage_one_usage.items():
                usage[key] = usage.get(key, 0) + value
            for attempt in range(3):
                try:
                    intent, tokens = interpret(
                        self.reasoner, graph, instruction, feedback, semantics=semantics)
                    for key, value in tokens.items():
                        usage[key] = usage.get(key, 0) + value
                    # A first-pass refusal is challenged once: the model's most
                    # common false refusal collapses "no exact colour/noun match"
                    # into "absent". Re-listing the observed movables forces it to
                    # justify the refusal against real candidates, not a word.
                    if intent.kind == "refuse" and attempt == 0:
                        movable = [obj.id for obj in graph.objects if obj.movable]
                        feedback = (
                            f"You refused: {intent.reason} Observed movable objects: "
                            f"{', '.join(movable) or 'none'}. Reconsider - refuse only if "
                            "the target noun/category has ZERO candidates or the manner "
                            "is unsafe. A colour/specifier mismatch is not absence: bind "
                            "the nearest same-noun object and plan. If any pick is "
                            "possible, emit kind=plan.")
                        continue
                    # A refusal that survives the challenge may still be a false
                    # "no match = absent" collapse; the deterministic referent
                    # rescue runs before it is accepted as a real refusal.
                    if intent.kind == "refuse":
                        rescued = rescue_refusal(graph, intent, instruction)
                        if rescued is not None:
                            intent = rescued
                        else:
                            # A refusal that is really an ambiguous referent is the
                            # question the instruction invites, not an impossibility.
                            asked = rescue_ambiguous_refusal(graph, intent, instruction)
                            if asked is not None:
                                intent = asked
                    if intent.kind == "ask_user":
                        # A qualifier the scene settles is planned, not asked: the
                        # observed objects prove which one the instruction means.
                        settled = rescue_clarification(graph, intent, instruction)
                        if settled is not None:
                            intent = settled
                    if intent.kind == "plan":
                        # A restriction list the instruction does not license is a
                        # decoding artefact, not a constraint.
                        unlicensed = clear_unlicensed_restrictions(intent, instruction)
                        if unlicensed is not None:
                            intent = unlicensed
                        # A plan the instruction's own manner or constraints
                        # forbid is overridden to a refusal by the same layer.
                        vetoed = veto_plan(graph, intent, instruction)
                        if vetoed is not None:
                            intent = vetoed
                        else:
                            # Re-bind the pick target when the instruction's first
                            # referent resolves deterministically to a different
                            # observed object (wrong noun, or wrong ordinal/spatial
                            # instance) - never a guess, only a unique resolution.
                            resolved = resolve_target(graph, intent, instruction)
                            if resolved is not None:
                                intent = resolved
                            # An under placement starts with the object resting on
                            # the target, not with the target itself.
                            staged = under_placement_first_pick(graph, intent, instruction)
                            if staged is not None:
                                intent = staged
                            # A goal whose blocker must move comes first, so the
                            # scored first pick is the obstruction the reference
                            # plan clears first as well.
                            unblocked = clear_blockers_first(graph, intent, instruction)
                            if unblocked is not None:
                                intent = unblocked
                            disambiguated = disambiguate_plan(graph, intent, instruction)
                            if disambiguated is not None:
                                intent = disambiguated
                            # Judge the settled binding last: once the target is
                            # the object the instruction's own referent resolves
                            # to, a colour word the scene refutes at a distance is
                            # the one licence the instruction cannot give.
                            if intent.kind == "plan":
                                mismatched = veto_unmatched_colour(graph, intent, instruction)
                                if mismatched is not None:
                                    intent = mismatched
                    proposal = compile_intent(graph, intent)
                    return verify(
                        graph, proposal, instruction_id, model=self.reasoner.model,
                        reasoning_seconds=time.perf_counter() - start, usage=usage,
                    ).model_copy(update={
                        "repaired": bool(attempt), "intent": intent,
                        "semantic_interpretation": semantics,
                    })
                except (PlanViolation, ValidationError) as exc:
                    feedback = str(exc)
                    if attempt == 2:
                        raise ModelError(
                            "Apertus grounded intent still failed verification after repair: "
                            + feedback) from exc
            raise AssertionError("unreachable")

    def close(self) -> None:
        self.vision.close()
        self.reasoner.close()
        self.perception.release()
        if self.runtime is not None:
            self.runtime.close()


def save_json(path: Path, value: dict | list) -> None:
    """Never leave a partially-written results file that resembles a valid submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
