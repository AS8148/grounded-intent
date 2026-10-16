"""Append-only request/response traces, sufficient to replay one decision later.

The audit keeps the *parsed* result of each model call; it does not keep what was
actually sent and returned. That gap is what makes "why did the model answer that"
unanswerable after the fact — a failure can be observed but not replayed, diffed, or
attributed to a prompt revision. This module records the exchange itself.

Two rules keep the file safe to keep and safe to share:
- Credentials never enter it. The client holds them in headers, and headers are not
  recorded; the API key is redacted from any error text that is.
- Image bytes never enter it. A ``data:`` URL is replaced by the digest of its decoded
  bytes, so the same crop is recognisable across runs without the file growing by
  megabytes per call.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path

_DATA_URL_PREFIX = "data:"


def image_digest(url: str) -> dict:
    """Replace an embedded image with a content-addressed reference.

    Returns the digest, the byte length and the media type, so a trace can be diffed
    across runs without carrying the pixels.
    """
    header, _, payload = url.partition(",")
    media = header[len(_DATA_URL_PREFIX):].split(";")[0] or "application/octet-stream"
    try:
        raw = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        raw = payload.encode("ascii", "replace")
    return {"image_sha256": hashlib.sha256(raw).hexdigest(), "image_bytes": len(raw),
            "media_type": media}


def summarize_messages(messages: list) -> list:
    """Describe a request without its pixels or its credentials."""
    summarized = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            summarized.append({"role": message.get("role"), "text": content})
            continue
        parts = []
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = str((part.get("image_url") or {}).get("url") or "")
                parts.append({"type": "image_url", **(
                    image_digest(url) if url.startswith(_DATA_URL_PREFIX)
                    else {"url": url})})
            else:
                parts.append({key: value for key, value in part.items()})
        summarized.append({"role": message.get("role"), "parts": parts})
    return summarized


class TraceRecorder:
    """One JSON object per line; appended under a lock so a crash keeps what it wrote."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: str, **fields) -> None:
        row = {"event": event, "at": round(time.time(), 3), **fields}
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def request(self, model: str, messages: list, schema: dict | None, *,
                attempt: int, seed: int, max_tokens: int, timeout: float) -> None:
        self.record(
            "request", model=model, attempt=attempt, seed=seed, max_tokens=max_tokens,
            timeout=round(timeout, 3),
            schema_sha256=(hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]
                if schema is not None else None),
            messages=summarize_messages(messages),
        )

    def response(self, model: str, *, attempt: int, latency: float,
                 payload: dict | None = None, usage: dict | None = None,
                 relaxed: bool = False) -> None:
        self.record("response", model=model, attempt=attempt,
                    latency=round(latency, 3), relaxed=relaxed,
                    usage=usage or {}, payload=payload)

    def failure(self, model: str, *, attempt: int, latency: float, error: str) -> None:
        self.record("failure", model=model, attempt=attempt,
                    latency=round(latency, 3), error=error[:2000])
