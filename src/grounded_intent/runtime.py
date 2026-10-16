"""Own exactly one llama.cpp model process at a time, within a real memory limit."""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .assets import APERTUS, VISION, VISION_ENCODER, VISION_Q6, VISION_Q8
from .clients import Deadline, LocalChatClient
from .resources import (
    MEMORY_BUDGET,
    MEMORY_RESERVE,
    acquire_run_lock,
    effective_memory_limit,
    effective_swap_limit,
    memory_available,
    process_exited,
    require_headroom,
    stop_owned_process,
)
from .trace import TraceRecorder


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    files: tuple[str, ...]
    gpu_layers: int | None    # None → let --fit decide within --fit-target margin
    context: int
    thinking: bool = False
    mtp_draft_max: int | None = None   # set → enable --spec-type draft-mtp
    # VRAM margin left free by --fit. None → the upstream default (1024 MiB). The Q6
    # needs a smaller one so its MTP draft context also fits on the card: measured
    # 2026-10-14, the Q6 at the default margin generated 33 t/s with the GPU cold and
    # idle (the draft was on the CPU), while at 256 MiB it generated 102-104 t/s with
    # the GPU at 80-93 %.
    fit_target: int | None = None

class ModelRuntime:
    """Sequential activation and bounded startup; never adopt unrelated services."""

    def __init__(self, server: Path, models: Path, logs: Path, port: int = 8081,
                 aperture_layers: int = 24, threads: int = 4):
        limit = effective_memory_limit()
        if limit is None or limit > MEMORY_BUDGET or effective_swap_limit() != 0:
            raise RuntimeError("local model runtime requires a <=12 GiB, no-swap cgroup; use scripts/local.py")
        if not 1 <= port <= 65535 or not 1 <= threads <= 8 or not 0 <= aperture_layers <= 32:
            raise ValueError("invalid bounded inference configuration")
        self.server, self.models, self.logs = Path(server).resolve(), Path(models).resolve(), Path(logs)
        self.port, self.threads = port, threads
        # One vision stack: Qwen3.5-9B MTP UD-Q4_K_XL + F16 mmproj, with MTP
        # speculative decoding (the vendor's own recommended command for this
        # model). The Q8_0 and 4B vision variants are retired.
        # One vision stack, three published quants of it. The alias carries the choice, so
        # the perception cache can never mix two quantisations of the same scene.
        quant = os.environ.get("VISION_QUANT", "ud-q4-k-xl").casefold()
        published = {
            "q8_0": ("qwen3.5-9b-mtp-q8_0", VISION_Q8.filename, None),
            "ud-q6-k-xl": ("qwen3.5-9b-mtp-q6", VISION_Q6.filename, 256),
        }
        alias, weights, margin = published.get(
            quant, ("qwen3.5-9b-mtp-q4", VISION.filename, None))
        # MTP is on by default; VISION_MTP_MAX=0 turns speculative decoding off so the
        # cost of drafting can be measured against the same model.
        draft_max = int(os.environ.get("VISION_MTP_MAX", "2")) or None
        vision = ModelSpec(alias, (weights, VISION_ENCODER.filename), None, 8192,
                           mtp_draft_max=draft_max, fit_target=margin)
        self.specs = {
            "vision": vision,
            # 12288 (was 8192): prompts carry the full scene map (ordered instances,
            # nearest/farthest, resting_on, morphology) plus doctrine, and they have
            # GROWN — 4096 overflowed at ~4650 tokens on crowded scenes (request 400
            # instead of a plan), and a local gate smoke on 2026-10-06 measured single
            # reasoning requests at 5,981-7,574 tokens against 8192. With under 1k of
            # completion room left, the model's reply hit `finish_reason=length`, the
            # client relaxed the response schema to plain JSON, and the intent came
            # back malformed (`goals.0.target` missing) on 4/4 plan instructions of
            # scene_10. 12288 restores ~4.8k of completion room. The remote CSCS
            # endpoint is unaffected: this spec only starts the LOCAL reasoning server.
            #
            # thinking=False: measured on the same benchmark, the thinking variant
            # used ~5.7x the completion tokens (median 1855 vs 323), lost 7 answers
            # to finish_reason=length/HTTP 500-504, and scored 22.25/65 against
            # 32.275/65 -> 39.9/65 for the non-thinking variant. The submission gate
            # must run the configuration that was actually measured.
            "reasoning": ModelSpec(
                "apertus-v1.5-8b-q8_0", (APERTUS.filename,), aperture_layers, 12288,
                thinking=False,
            ),
        }
        if not self.server.is_file():
            raise RuntimeError("missing llama-server; build the pinned inference runtime")
        self.logs.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._process_lock = threading.RLock()
        self._lease: int | None = None
        self.active: str | None = None
        self.process: subprocess.Popen | None = None
        self._log = None
        self.generation = 0
        self.abort_reason = ""
        self._shutdown = threading.Event()
        self._watcher = threading.Thread(target=self._watch_memory, daemon=True)
        self._lease = acquire_run_lock()
        try:
            self._watcher.start()
        except BaseException:
            os.close(self._lease)
            self._lease = None
            raise

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _watch_memory(self) -> None:
        while not self._shutdown.wait(1):
            try:
                reason = (
                    "memory reserve breached; owned inference stopped"
                    if memory_available() < MEMORY_RESERVE else ""
                )
            except (OSError, RuntimeError):
                reason = "memory monitor became unavailable; inference stopped"
            if reason:
                # Independent of the request lock: an HTTP request may hold that
                # lock while inference is running. Also excludes launch races.
                with self._process_lock:
                    self.abort_reason = reason
                    self._stop_process()
                return

    def _stop_process(self) -> None:
        """Stop only the process group created by our Popen, including children."""
        if self.process is None:
            return
        stop_owned_process(self.process)
        self.process = None

    def deactivate(self) -> None:
        with self.lock, self._process_lock:
            self._stop_process()
            if self._log is not None:
                self._log.close()
                self._log = None
            self.active = None

    def activate(self, key: str) -> None:
        with self.lock:
            with self._process_lock:
                if self._shutdown.is_set():
                    raise RuntimeError("local model runtime is closed")
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                if self.active == key and self.process is not None and not process_exited(self.process):
                    return
            spec = self.specs[key]
            paths = [self.models / name for name in spec.files]
            for path in paths:
                if not path.is_file():
                    raise RuntimeError(f"missing pinned Q8 prerequisite {path.name}; run scripts/prepare.py")
            self.deactivate()
            require_headroom(sum(path.stat().st_size for path in paths))
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=.2):
                    raise RuntimeError(f"port {self.port} already occupied; refusing to adopt another service")
            except (ConnectionRefusedError, TimeoutError):
                pass
            command = [
                str(self.server), "-m", str(paths[0]), "--alias", spec.alias,
                "--host", "127.0.0.1", "--port", str(self.port),
                "-t", str(self.threads),
                "--jinja", "--reasoning", "on" if spec.thinking else "off",
                "--no-context-shift", "--no-warmup",
                "-c", str(spec.context), "-np", "1",
                # fit only adjusts unset params; an explicit -ngl disables the
                # margin logic. Pass it only when the spec pins a layer count.
                *(["-ngl", str(spec.gpu_layers)] if spec.gpu_layers is not None else []),
            ]
            if key == "vision":
                # image-max-tokens is additionally capped at n_ubatch since PR
                # #29773 — ubatch below it silently downscales images to ~64
                # tokens. Keep -ub >= --image-max-tokens.
                # Fit is an estimated VRAM margin, not a GPU allocation cap or
                # a guarantee about WSL host/driver memory and stability. The margin is
                # 3072 MiB, not 1536: measured 2026-10-07, with the card filled to its
                # 1536 MiB edge (10.9 of 12.3 GB used) the server's image prompt
                # processing collapsed from ~2000 tokens/s to 14 tokens/s and every
                # close-up request sat out its 240 s timeout — the WSL2 driver pages an
                # oversubscribed allocation instead of failing, so the symptom is a 150x
                # slowdown rather than an error. Real headroom costs a few offloaded
                # layers and buys stability.
                command.extend([
                    "--mmproj", str(paths[1]), "--image-min-tokens", "1024",
                    "--image-max-tokens", "2048", "-b", "2048", "-ub", "2048",
                    "--flash-attn", "on", "--fit", "on",
                    # No --fit-target by default: the upstream margin (1024 MiB) applies.
                    # The 3072 MiB value this replaced was raised on 2026-10-07, when one
                    # long-lived server accumulated buffers to 10.9 of 12.3 GB and image
                    # prompt processing collapsed to 14 tok/s; that incident predates the
                    # current policy of a fresh server per scene, and the mitigations
                    # measured at the time were the morphology deadline, the
                    # abandon-on-failure rule and the 448 px crop cap, not the margin.
                    # VISION_FIT_TARGET sets it explicitly for a controlled comparison,
                    # because a margin that is too small for the model puts layers on the
                    # CPU and the slowdown looks like a slow model rather than an offload.
                    *(["--fit-target", os.environ.get("VISION_FIT_TARGET")
                       or str(spec.fit_target)] if (os.environ.get("VISION_FIT_TARGET")
                                                     or spec.fit_target is not None) else []),
                    # The server's prompt cache is disabled deliberately.  It defaults to an
                    # 8192 MiB host-RAM ceiling and saves each finished request's KV state so a
                    # later request can restore a shared prefix.  This workload shares almost
                    # none: every request carries a different image or a different object crop,
                    # so the only reusable prefix is the ~1-1.5k-token system prompt.  Measured
                    # 2026-10-12: with it enabled the run's cgroup reached 10.97 of 12.88 GB and
                    # entered direct reclaim (pgscan_direct 6.86e9, workingset_refault_file
                    # 3.99e8), wedging perception on the 253 MiB MobileCLIP load.  The cache cost
                    # host RAM and bought no measurable reuse.
                    "--cache-ram", "0",
                ])
                if spec.mtp_draft_max is not None:
                    # The model's trained MTP head drafts tokens the main head
                    # verifies in one pass — lossless by construction, strongest
                    # on structured JSON output. Requires -np 1 (already set).
                    command.extend([
                        "--spec-type", "draft-mtp",
                        "--spec-draft-n-max", str(spec.mtp_draft_max),
                    ])
            else:
                command.extend(["--flash-attn", "on", "-b", "2048", "-ub", "2048"])
            self._log = (self.logs / (key + ".server.log")).open("w")
            try:
                with self._process_lock:
                    if self.abort_reason:
                        raise RuntimeError(self.abort_reason)
                    self.process = subprocess.Popen(
                        command, stdout=self._log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.active = key
                    process = self.process
                deadline = time.monotonic() + 120
                with httpx.Client(trust_env=False, timeout=2) as client:
                    while time.monotonic() < deadline:
                        with self._process_lock:
                            if self.abort_reason:
                                raise RuntimeError(self.abort_reason)
                            if process_exited(process):
                                raise RuntimeError(f"{spec.alias} exited during startup; inspect {key}.server.log")
                        try:
                            response = client.get(self.base_url.removesuffix("/v1") + "/health")
                            if response.status_code == 200:
                                with self._process_lock:
                                    if self.abort_reason:
                                        raise RuntimeError(self.abort_reason)
                                    self.generation += 1
                                    return
                        except httpx.RequestError:
                            pass
                        time.sleep(.25)
                raise RuntimeError(f"{spec.alias} startup timed out")
            except BaseException:
                self.deactivate()
                raise

    def close(self) -> None:
        with self.lock:
            self._shutdown.set()
            self.deactivate()
            self._watcher.join(timeout=12)
            if self._watcher.is_alive():
                raise RuntimeError("owned memory watcher did not stop; retaining runtime lease")
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None


class ManagedChatClient:
    """Clients share one owned runtime; pooled connections reset on model switches."""

    def __init__(self, runtime: ModelRuntime, key: str, timeout: float = 240):
        self.runtime, self.key, self.timeout = runtime, key, timeout
        self.model = runtime.specs[key].alias
        self._client: LocalChatClient | None = None
        self._generation = -1
        # Set by the pipeline for the duration of one decision; forwarded to the
        # pooled client so a stalled local server cannot consume the whole run.
        self.deadline: Deadline | None = None
        # Set by the pipeline for a whole run; forwarded so the request/response
        # exchange is recorded whichever model the runtime activates.
        self.trace: TraceRecorder | None = None

    def complete(self, messages: list, schema: dict | None = None,
                 max_tokens: int = 1024, seed: int = 42) -> tuple[dict, dict[str, int]]:
        with self.runtime.lock:
            self.runtime.activate(self.key)
            if self._client is None or self._generation != self.runtime.generation:
                self.close()
                self._client = LocalChatClient(
                    self.runtime.base_url, self.model, self.timeout,
                    temperature=.7 if self.key == "vision" else 0,
                    top_p=.8 if self.key == "vision" else 1,
                )
                self._generation = self.runtime.generation
            self._client.deadline = self.deadline
            self._client.trace = self.trace
            # Never ask for more than the local model's context: REASONING_MAX_TOKENS
            # defaults to 12288 while the reasoning spec declares 8192, and a request
            # above the context is clipped by the server.
            spec = self.runtime.specs[self.key]
            return self._client.complete(
                messages, schema, min(max_tokens, spec.context), seed=seed)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
