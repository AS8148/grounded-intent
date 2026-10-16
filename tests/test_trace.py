"""Request/response tracing: replayable, and safe to keep.

The trace exists so a decision can be replayed and a failure attributed after the fact.
Two properties are non-negotiable and are pinned here: no credentials, and no image
bytes — an embedded image becomes a content digest so the same crop is recognisable
across runs without the file carrying pixels.
"""
from __future__ import annotations

import base64
import json

from grounded_intent.trace import TraceRecorder, image_digest, summarize_messages


def test_an_embedded_image_becomes_a_content_digest():
    raw = b"\x89PNG\r\n\x1a\nfake-pixels"
    url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

    digest = image_digest(url)
    assert digest["media_type"] == "image/png"
    assert digest["image_bytes"] == len(raw)
    assert len(digest["image_sha256"]) == 64

    summary = summarize_messages([{
        "role": "user",
        "content": [
            {"type": "text", "text": "inventory this"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }])
    serialized = json.dumps(summary)
    assert "base64" not in serialized
    assert base64.b64encode(raw).decode("ascii") not in serialized
    assert summary[0]["parts"][1]["image_bytes"] == len(raw)


def test_a_trace_records_the_exchange_and_no_credentials(tmp_path):
    path = tmp_path / "traces.jsonl"
    recorder = TraceRecorder(path)

    recorder.request("model-x", [{"role": "system", "content": "prompt"}],
                     {"type": "object", "properties": {}},
                     attempt=0, seed=7, max_tokens=64, timeout=5.0)
    recorder.response("model-x", attempt=0, latency=0.5, payload={"kind": "plan"},
                      usage={"completion_tokens": 3})
    recorder.failure("model-x", attempt=1, latency=1.5, error="HTTP 500")

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["request", "response", "failure"]
    assert rows[0]["seed"] == 7 and rows[0]["max_tokens"] == 64
    assert len(rows[0]["schema_sha256"]) == 16
    assert rows[0]["messages"][0]["text"] == "prompt"
    assert rows[1]["payload"] == {"kind": "plan"}
    assert rows[1]["usage"] == {"completion_tokens": 3}
    assert rows[2]["error"] == "HTTP 500"
    assert "Authorization" not in path.read_text()
    assert all("at" in row for row in rows)


def test_the_trace_appends_so_a_crashed_run_keeps_what_it_wrote(tmp_path):
    path = tmp_path / "traces.jsonl"
    TraceRecorder(path).record("request", model="a")
    TraceRecorder(path).record("request", model="b")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["model"] for row in rows] == ["a", "b"]
