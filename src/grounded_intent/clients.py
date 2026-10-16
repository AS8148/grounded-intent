"""Strict OpenAI-compatible client restricted to loopback inference servers."""
from __future__ import annotations

import ipaddress
import json
import logging
import math
import time
from urllib.parse import urlsplit

import httpx

from .trace import TraceRecorder

logger = logging.getLogger(__name__)

# Bounded, escalated recovery policy for one logical completion. Only 5xx and the
# transient 408/409/429 responses are retried; every other 4xx signals a request
# or schema error that a retry cannot fix.
_MAX_ATTEMPTS = 4
_RETRYABLE_STATUS = frozenset({408, 409, 429})
_RETRY_AFTER_CAP_SECONDS = 30.0
_BACKOFF_SECONDS = 0.4
_TOKEN_LIMIT_CEILING = 16384


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return a numeric ``Retry-After`` delay clamped to a sane ceiling, if valid."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    # A non-finite value ("nan"/"inf") passes a bare comparison and would reach
    # time.sleep as nan, raising an uncaught ValueError outside every handler.
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return min(seconds, _RETRY_AFTER_CAP_SECONDS)


class ModelError(RuntimeError):
    """A real inference dependency failed; never substitute a synthetic answer."""


class ModelCallError(ModelError):
    """A transport/generation call failed; outer semantic repairs must not retry it."""


class Deadline:
    """One wall-clock budget shared by every call of a single decision.

    A decision is the bounded unit of work: the pipeline sets the budget once, and
    every transport call clamps its own timeout to what remains, so a stalled
    endpoint can exhaust its own slice but never the run. ``None`` means unbounded.
    """

    __slots__ = ("expires",)

    def __init__(self, seconds: float | None):
        self.expires = (time.monotonic() + seconds
                        if seconds is not None and seconds > 0 else None)

    def remaining(self) -> float | None:
        if self.expires is None:
            return None
        return self.expires - time.monotonic()

    def check(self) -> None:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise ModelCallError("decision deadline exhausted")


def host_is_loopback(host: str) -> bool:
    """Whether a URL host addresses this machine, including non-canonical spellings.

    ``ipaddress`` rejects shorthand numeric forms like ``127.1`` or ``2130706433`` and
    ``localhost.`` carries a trailing dot, yet all of them resolve to loopback. A host
    that looks purely numeric but is not a valid IP literal, or a dotted form of
    localhost, is therefore treated as local so the guard fails closed.
    """
    name = host.strip().strip("[]").rstrip(".").casefold()
    if not name:
        return False
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        pass
    if name == "localhost":
        return True
    shorthand = _shorthand_ipv4(name)
    if shorthand is not None:
        # inet_aton shorthand (127.1, 2130706433) is loopback only when the top octet
        # is 127. A digits-and-dots host that does NOT parse is NOT local: treating it
        # as local would let a remote address skip both HTTPS and the API key.
        return shorthand >> 24 == 127
    return False


def _shorthand_ipv4(value: str) -> int | None:
    """Parse inet_aton shorthand forms into a 32-bit address; ``None`` if not one."""
    parts = value.split(".")
    if not 1 <= len(parts) <= 4 or any(not part.isdigit() for part in parts):
        return None
    if any(len(part) > 1 and part.startswith("0") for part in parts):
        return None  # leading zeros are octal in inet_aton; treat as unsupported
    numbers = [int(part) for part in parts]
    if any(number > 255 for number in numbers[:-1]):
        return None
    if len(parts) == 1:
        return numbers[0] if numbers[0] < 2 ** 32 else None
    # The leading parts are the top octets; the LAST part fills every remaining byte
    # ("127.1" is 127.0.0.1, not 0.0.127.1).
    if numbers[-1] >= 256 ** (5 - len(parts)):
        return None
    address = 0
    for number in numbers[:-1]:
        address = (address << 8) | number
    return (address << (8 * (5 - len(parts)))) | numbers[-1]


def require_chat_url(base_url: str, allow_remote: bool) -> str:
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    local = host_is_loopback(host)
    if parsed.scheme not in {"http", "https"} or parsed.query or parsed.fragment:
        raise ValueError("inference URL must be an http(s) endpoint without query/fragment")
    if local:
        if parsed.username or parsed.password:
            raise ValueError("loopback endpoints must not embed credentials")
    else:
        if not allow_remote or parsed.scheme != "https":
            raise ValueError("remote inference requires an API key and https")
        if parsed.username or parsed.password:
            raise ValueError("credentials belong in the API key header, not the URL")
    return base_url.rstrip("/")


def _extract_json_object(content: str):
    """Return the first complete top-level JSON object inside model prose.

    Remote backends sometimes wrap schema output in markdown fences or
    explanatory text. Salvage only a balanced ``{...}`` span; nested braces are
    counted, strings are skipped so braces inside quoted text do not unbalance
    the scan. Never fabricates content — raises ValueError when absent.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(content):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(content[start:index + 1])
    raise ValueError("model response contains no complete JSON object")


class LocalChatClient:
    """Persistent connection pool with bounded, seeded JSON generation.

    Loopback-only unless ``api_key`` is given, in which case https hosts are
    permitted (e.g. the CSCS Apertus inference service for development).
    """

    def __init__(self, base_url: str, model: str, timeout: float = 180,
                 temperature: float = 0, top_p: float = 1, api_key: str | None = None):
        if not model.strip():
            raise ValueError("model identifier is required")
        self.base_url = require_chat_url(base_url, allow_remote=bool(api_key))
        self.model = model
        if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
            raise ValueError("invalid sampling configuration")
        self.temperature, self.top_p = temperature, top_p
        self._timeout = timeout
        # Set by the pipeline for the duration of one decision; None = unbounded.
        self.deadline: Deadline | None = None
        # Set by the pipeline for a whole run; None = no tracing.
        self.trace: TraceRecorder | None = None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False,
                                  headers=headers)

    def close(self) -> None:
        self._http.close()

    def _http_failure(self, response: httpx.Response) -> ModelCallError:
        detail = response.text
        authorization = self._http.headers.get("Authorization", "")
        if authorization:
            detail = detail.replace(authorization, "[redacted]")
            detail = detail.replace(authorization.removeprefix("Bearer "), "[redacted]")
        detail = detail.strip()[:1024]
        return ModelCallError(
            f"{self.model} server returned HTTP {response.status_code}"
            + (f": {detail}" if detail else "")
        )

    def complete(
        self, messages: list, schema: dict | None = None, max_tokens: int = 1024,
        seed: int = 42,
    ) -> tuple[dict, dict[str, int]]:
        if schema is not None:
            # The schema is echoed into the system prompt as well as sent as
            # ``response_format``. Under strict decoding that copy is redundant, but
            # after relaxation to plain ``json_object`` (see below) it is the only
            # contract the model still sees — keep it.
            contract = (
                "\nFinal-response JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
            if (
                messages and messages[0].get("role") == "system"
                and isinstance(messages[0].get("content"), str)
            ):
                messages = [
                    {**messages[0], "content": messages[0]["content"] + contract},
                    *messages[1:],
                ]
            else:
                messages = [
                    {"role": "system", "content": "Return only schema-conforming JSON." + contract},
                    *messages,
                ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "seed": seed,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "grounded_intent", "strict": True, "schema": schema},
            }
        # Recovery is bounded and escalated. Constrained decoding can trip a
        # nondeterministic llama.cpp grammar bug ('Unexpected empty grammar stack',
        # or a degenerate whitespace loop that exhausts max_tokens); a different
        # seed escapes the crash, plain JSON mode escapes the loop, and a larger
        # budget escapes a genuinely short cap. Only 5xx and the transient
        # 408/409/429 responses are retried — other 4xx are request/schema errors.
        relaxed = False
        retry_after = None
        for attempt in range(_MAX_ATTEMPTS):
            request_timeout = self._timeout
            if self.deadline is not None:
                # A stalled endpoint may consume only what remains of this decision's
                # budget; once it is gone, no further attempt is started.
                self.deadline.check()
                remaining = self.deadline.remaining()
                if remaining is not None:
                    request_timeout = max(0.05, min(self._timeout, remaining))
            if attempt:
                delay = retry_after if retry_after is not None else _BACKOFF_SECONDS * attempt
                if self.deadline is not None:
                    remaining = self.deadline.remaining()
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining))
                time.sleep(delay)
                retry_after = None
            body["seed"] = seed + attempt
            if self.trace is not None:
                self.trace.request(self.model, messages, schema, attempt=attempt,
                                   seed=seed + attempt, max_tokens=body["max_tokens"],
                                   timeout=request_timeout)
                self.trace.record(
                    "attempt_settings", model=self.model, attempt=attempt,
                    temperature=self.temperature, top_p=self.top_p,
                    response_format=body.get("response_format", {}).get("type"),
                    relaxed=relaxed,
                )
            started = time.monotonic()
            finish_reason = None
            try:
                response = self._http.post(
                    self.base_url + "/chat/completions", json=body,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    if self.trace is not None:
                        raw_usage = payload.get("usage")
                        self.trace.record(
                            "incomplete_response", model=self.model, attempt=attempt,
                            latency=round(time.monotonic() - started, 3),
                            finish_reason=finish_reason, relaxed=relaxed,
                            usage={key: value for key, value in raw_usage.items()
                                   if isinstance(value, int) and not isinstance(value, bool)
                                   and value >= 0} if isinstance(raw_usage, dict) else {},
                        )
                    if attempt == _MAX_ATTEMPTS - 1:
                        raise ModelCallError(
                            "model output exhausted its token limit; no partial plan accepted")
                    if schema is not None and attempt == 1 and not relaxed:
                        # Whitespace/loop bugs only reproduce under strict grammars.
                        relaxed = True
                        body["response_format"] = {"type": "json_object"}
                    elif attempt == _MAX_ATTEMPTS - 2:
                        # A genuinely short cap is the other cause; grant headroom
                        # for the final attempt.
                        body["max_tokens"] = min(max_tokens * 2, _TOKEN_LIMIT_CEILING)
                    continue
                if finish_reason not in (None, "stop"):
                    raise ModelCallError(
                        f"model generation stopped early with finish_reason={finish_reason!r}")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ModelError("model returned an empty completion")
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    result = _extract_json_object(content)
                if not isinstance(result, dict):
                    raise ModelError("model returned JSON that is not an object")
                usage = {
                    key: value for key, value in payload.get("usage", {}).items()
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                }
                if self.trace is not None:
                    self.trace.response(self.model, attempt=attempt,
                                        latency=time.monotonic() - started,
                                        payload=result, usage=usage, relaxed=relaxed)
                return result, usage
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if self.trace is not None:
                    self.trace.failure(self.model, attempt=attempt,
                                       latency=time.monotonic() - started,
                                       error=f"HTTP {status}")
                if (status >= 500 or status in _RETRYABLE_STATUS) and attempt < _MAX_ATTEMPTS - 1:
                    if schema is not None and not relaxed:
                        relaxed = True
                        body["response_format"] = {"type": "json_object"}
                        logger.warning(
                            "%s returned HTTP %d; relaxing the response schema to plain "
                            "JSON for attempt %d", self.model, status, attempt + 1)
                    retry_after = _retry_after_seconds(exc.response)
                    continue
                raise self._http_failure(exc.response) from exc
            except httpx.RequestError as exc:
                if self.trace is not None:
                    self.trace.failure(self.model, attempt=attempt,
                                       latency=time.monotonic() - started,
                                       error=f"transport {type(exc).__name__}")
                # No response arrived (connect/read/pool failure): transient by
                # nature, so retry on the same bounded schedule.
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                raise ModelCallError(
                    f"cannot reach {self.model} inference server: {type(exc).__name__}") from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                if self.trace is not None:
                    self.trace.failure(self.model, attempt=attempt,
                                       latency=time.monotonic() - started,
                                       error=f"invalid structured output: {type(exc).__name__}")
                raise ModelError(f"{self.model} server returned invalid structured output") from exc
            except ModelError as exc:
                if self.trace is not None and finish_reason != "length":
                    self.trace.failure(
                        self.model, attempt=attempt,
                        latency=time.monotonic() - started,
                        error=f"generation {type(exc).__name__}",
                    )
                raise
        raise AssertionError("unreachable")
