"""Consumer-visible transport and sovereign-runtime boundary behavior."""
import json

import httpx
import pytest

from grounded_intent.clients import LocalChatClient, ModelCallError, ModelError, require_chat_url


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8082/v1", "http://localhost:8082/v1/", "http://[::1]:8082/v1",
    "https://127.0.0.2:8443/v1", "http://127.10.20.30/v1",
    # Non-canonical spellings that also resolve to loopback: treating them as remote
    # would let a local endpoint masquerade as a remote one (and vice versa).
    "http://2130706433/v1", "http://127.1/v1", "http://localhost./v1",
])
def test_loopback_endpoints_are_accepted(url):
    assert require_chat_url(url, allow_remote=False) == url.rstrip("/")


def test_a_numeric_host_is_local_only_when_it_resolves_to_loopback():
    # inet_aton shorthand that IS loopback is accepted without a key...
    assert require_chat_url("http://127.1/v1", allow_remote=False) == "http://127.1/v1"
    assert require_chat_url("http://2130706433/v1", allow_remote=False) == "http://2130706433/v1"
    # ...but 134744072 is 8.8.8.8: a numeric host must NOT inherit local privileges,
    # which would let a remote address skip HTTPS and the API key.
    with pytest.raises(ValueError):
        require_chat_url("http://134744072/v1", allow_remote=False)
    with pytest.raises(ValueError):
        require_chat_url("http://8.8.8.8/v1", allow_remote=False)




@pytest.mark.parametrize("url", [
    "https://example.com/v1", "http://192.168.1.2/v1", "http://0.0.0.0/v1",
    "http://localhost.example.com/v1", "ftp://127.0.0.1/v1", "file:///tmp/model",
    "http://user:secret@localhost/v1", "http://localhost/v1?key=secret",
    "http://localhost/v1#fragment", "http://[::]/v1", "http://[::ffff:8.8.8.8]/v1",
    "http://127.0.0.1.example.com/v1", "", "localhost:8082",
])
def test_remote_or_credential_bearing_endpoints_are_rejected(url):
    with pytest.raises(ValueError):
        require_chat_url(url, allow_remote=False)


def test_remote_https_requires_api_key_and_https_scheme():
    with pytest.raises(ValueError):
        require_chat_url("https://api.inference.cscs.ch/v1", allow_remote=False)
    with pytest.raises(ValueError):
        require_chat_url("http://api.inference.cscs.ch/v1", allow_remote=True)
    assert require_chat_url("https://api.inference.cscs.ch/v1", allow_remote=True) == (
        "https://api.inference.cscs.ch/v1")


def client_with_response(response):
    client = LocalChatClient("http://127.0.0.1:8082/v1", "apertus-v1.5-8b")
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(lambda request: response))
    return client


@pytest.mark.parametrize("status", [301, 302, 307, 400, 401, 404, 429, 500, 503])
def test_redirects_and_server_failures_do_not_become_plans(monkeypatch, status):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    client = client_with_response(httpx.Response(status, headers={"location": "https://example.com"}))
    try:
        with pytest.raises(ModelError):
            client.complete([])
    finally:
        client.close()


@pytest.mark.parametrize("payload", [
    {}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": "not JSON"}}]},
    {"choices": [{"message": {"content": "[]"}}]},
    {"choices": [{"message": {"content": "null"}}]},
    {"choices": [{"message": {"content": "42"}}]},
    {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]},
])
def test_malformed_or_truncated_model_responses_fail_explicitly(payload):
    client = client_with_response(httpx.Response(200, json=payload))
    try:
        with pytest.raises(ModelError):
            client.complete([])
    finally:
        client.close()


def test_non_json_http_body_is_not_silently_repaired():
    client = client_with_response(httpx.Response(200, content=b"<html>error</html>"))
    try:
        with pytest.raises(ModelError):
            client.complete([])
    finally:
        client.close()


def test_untrusted_usage_values_do_not_corrupt_token_statistics():
    response = {"choices": [{"message": {"content": json.dumps({"kind": "refuse"})}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "bad": True,
                          "negative": -3, "string": "100"}}
    client = client_with_response(httpx.Response(200, json=response))
    try:
        _, usage = client.complete([])
        assert usage == {"prompt_tokens": 9, "completion_tokens": 4}
    finally:
        client.close()


def test_transport_failure_is_not_a_semantic_refusal(monkeypatch):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)

    def disconnect(request):
        raise httpx.ConnectError("unavailable", request=request)
    client = LocalChatClient("http://127.0.0.1:8082/v1", "apertus-v1.5-8b")
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(disconnect))
    try:
        with pytest.raises(ModelError, match="cannot reach apertus"):
            client.complete([])
    finally:
        client.close()


@pytest.mark.parametrize("failure,expected_calls", [
    ("server", 4), ("length", 4), ("connection", 4),
])
def test_failed_call_budget_is_not_restarted_by_semantic_repair(
    monkeypatch, failure, expected_calls,
):
    from grounded_intent import clients
    from grounded_intent.domain import ObjectObservation, SceneGraph
    from grounded_intent.reasoning import interpret

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    calls = []
    secret = "fixture-secret"

    def backend(request):
        calls.append(request)
        if failure == "connection":
            raise httpx.ConnectError("unavailable", request=request)
        if failure == "length":
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "length", "message": {"content": '{"kind":'},
            }]})
        return httpx.Response(500, json={"error": {
            "message": "decoder backend failed; credential=" + secret,
        }})

    client = LocalChatClient("https://provider.invalid/v1", "api-test", api_key=secret)
    headers = client._http.headers
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(backend), headers=headers)
    graph = SceneGraph(
        scene="synthetic", width=80, height=60, image_sha256="synthetic",
        objects=(ObjectObservation(
            id="item", label="token", box=(8, 12, 24, 31), pixel=(13, 21),
            confidence=.95, movable=True, graspable=True,
        ),), perception_model="synthetic", perception_seconds=0,
    )
    try:
        with pytest.raises(ModelCallError) as caught:
            interpret(client, graph, "Retrieve the token.",
                      semantics={"action": "pick_only", "target_quantifier": "single"})
    finally:
        client.close()
    assert len(calls) == expected_calls
    assert secret not in str(caught.value)
def test_decoder_failure_recovers_with_checked_json_object_output(monkeypatch):
    from grounded_intent import clients
    from grounded_intent.reasoning import extract_semantics

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    instruction = "Retrieve the token."
    semantics = {
        "action": "pick_only", "target_description": "the token",
        "destination_description": "", "under_object": "",
        "exclusions": [], "protected_objects": [], "unsafe_manner": "",
        "reason": "A single pick with no destination.",
    }

    def backend(request):
        body = json.loads(request.content)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(semantics)},
        }]})

    client = LocalChatClient("http://127.0.0.1:8082/v1", "api-test")
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(backend))
    try:
        parsed, _ = extract_semantics(client, instruction)
    finally:
        client.close()
    assert parsed["action"] == "pick_only"
    assert parsed["target_description"] == "the token"
    assert parsed["exclusions"] == [] and parsed["protected_objects"] == []


def json_choice(content, finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{
        "finish_reason": finish_reason, "message": {"content": json.dumps(content)},
    }]})


def mock_client(backend):
    client = LocalChatClient("http://127.0.0.1:8082/v1", "api-test")
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(backend))
    return client
def test_transient_connect_error_is_retried_then_succeeds(monkeypatch):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    attempts = []

    def backend(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("blip", request=request)
        return json_choice({"kind": "refuse"})

    client = mock_client(backend)
    try:
        result, _ = client.complete([])
    finally:
        client.close()
    assert result == {"kind": "refuse"}
    assert len(attempts) == 2


@pytest.mark.parametrize("status", [408, 409, 429])
def test_transient_http_statuses_are_retried_then_succeed(monkeypatch, status):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    attempts = []

    def backend(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(status)
        return json_choice({"kind": "refuse"})

    client = mock_client(backend)
    try:
        result, _ = client.complete([])
    finally:
        client.close()
    assert result == {"kind": "refuse"}
    assert len(attempts) == 2


def test_rate_limit_honours_retry_after(monkeypatch):
    from grounded_intent import clients

    sleeps = []
    monkeypatch.setattr(clients.time, "sleep", lambda seconds: sleeps.append(seconds))
    attempts = []

    def backend(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "2.5"})
        return json_choice({"kind": "refuse"})

    client = mock_client(backend)
    try:
        client.complete([])
    finally:
        client.close()
    assert len(attempts) == 2
    assert 2.5 in sleeps


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_transient_client_errors_are_not_retried(monkeypatch, status):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    attempts = []

    def backend(request):
        attempts.append(request)
        return httpx.Response(status, json={"error": "bad request"})

    client = mock_client(backend)
    try:
        with pytest.raises(ModelCallError):
            client.complete([])
    finally:
        client.close()
    assert len(attempts) == 1


def test_truncated_completion_is_retried_then_succeeds(monkeypatch):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    seeds = []

    def backend(request):
        seeds.append(json.loads(request.content)["seed"])
        if len(seeds) == 1:
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "length", "message": {"content": '{"kind":'},
            }]})
        return json_choice({"kind": "plan"})

    client = mock_client(backend)
    try:
        result, _ = client.complete([], schema={"type": "object"})
    finally:
        client.close()
    assert result == {"kind": "plan"}
    assert seeds == [42, 43]
def test_trace_records_each_retry_failure_and_truncated_generation(monkeypatch, tmp_path):
    """A later success must not erase the failures that made the request different.

    The seed and response format are effective treatment settings, not incidental metadata:
    HTTP 503 relaxes the schema, then ``finish_reason=length`` leaves an incomplete attempt,
    then the third request returns the parsed answer. The trace must preserve every branch
    while leaving decision policy and the accepted result unchanged.
    """
    from grounded_intent import clients
    from grounded_intent.trace import TraceRecorder

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    attempts = []

    def backend(request):
        body = json.loads(request.content)
        attempts.append(body)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "temporary"})
        if len(attempts) == 2:
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "length",
                             "message": {"content": "{\"kind\":"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            })
        return json_choice({"kind": "ask_user"})

    trace_path = tmp_path / "retry-trace.jsonl"
    client = LocalChatClient("http://127.0.0.1:8082/v1", "trace-test")
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(backend))
    client.trace = TraceRecorder(trace_path)
    try:
        result, _ = client.complete(
            [{"role": "user", "content": "Return an answer."}],
            schema={"type": "object", "properties": {"kind": {"const": "ask_user"}},
                    "required": ["kind"]},
            max_tokens=64,
        )
    finally:
        client.close()

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert result == {"kind": "ask_user"}
    assert [body["seed"] for body in attempts] == [42, 43, 44]
    assert [row["event"] for row in rows].count("request") == 3
    failures = [row for row in rows if row["event"] == "failure"]
    incomplete = [row for row in rows if row["event"] == "incomplete_response"]
    assert [(row["attempt"], row["error"]) for row in failures] == [(0, "HTTP 503")]
    assert [(row["attempt"], row["finish_reason"], row["usage"])
            for row in incomplete] == [(1, "length", {
                "prompt_tokens": 11, "completion_tokens": 7})]
    settings = [row for row in rows if row["event"] == "attempt_settings"]
    assert [(row["attempt"], row["response_format"], row["relaxed"])
            for row in settings] == [
                (0, "json_schema", False), (1, "json_object", True),
                (2, "json_object", True)]
    assert not ({"content", "partial_content"} & incomplete[0].keys())




def test_persistently_truncated_completion_escalates_and_then_raises(monkeypatch):
    from grounded_intent import clients

    monkeypatch.setattr(clients.time, "sleep", lambda _seconds: None)
    bodies = []

    def backend(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "length", "message": {"content": '{"kind":'},
        }]})

    client = mock_client(backend)
    try:
        with pytest.raises(ModelCallError, match="token limit"):
            client.complete([], schema={"type": "object"}, max_tokens=256)
    finally:
        client.close()
    assert len(bodies) == 4
    # Relaxation lands on the second retry, extra budget on the final attempt.
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[2]["response_format"] == {"type": "json_object"}
    assert bodies[3]["max_tokens"] == 512


@pytest.mark.parametrize("finish_reason", ["content_filter", "tool_calls"])
def test_generation_abort_is_reported_as_a_call_failure(finish_reason):
    client = client_with_response(httpx.Response(200, json={"choices": [{
        "finish_reason": finish_reason, "message": {"content": "{}"},
    }]}))
    try:
        with pytest.raises(ModelCallError, match="finish_reason"):
            client.complete([])
    finally:
        client.close()


def test_empty_completion_content_is_not_silently_repaired():
    client = client_with_response(httpx.Response(200, json={"choices": [{
        "finish_reason": "stop", "message": {"content": "   "},
    }]}))
    try:
        with pytest.raises(ModelError):
            client.complete([])
    finally:
        client.close()
