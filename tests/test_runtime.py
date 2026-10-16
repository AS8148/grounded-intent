"""Owned runtime lifetime and race coverage without launching any model."""
import importlib.util
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from grounded_intent import resources, runtime


@pytest.fixture
def configured(tmp_path, monkeypatch):
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir(mode=0o700)
    monkeypatch.setattr(resources, "_run_lock_directory", lambda: lease_dir)
    monkeypatch.setattr(runtime, "effective_memory_limit", lambda: resources.MEMORY_BUDGET)
    monkeypatch.setattr(runtime, "effective_swap_limit", lambda: 0)
    monkeypatch.setattr(runtime.ModelRuntime, "_watch_memory", lambda self: None)
    monkeypatch.setattr(runtime, "require_headroom", lambda estimate: None)
    server = tmp_path / "llama-server"
    server.touch()
    models = tmp_path / "models"
    models.mkdir()
    return server, models, tmp_path / "logs"


@pytest.fixture
def owned(configured, monkeypatch):
    model = runtime.ModelRuntime(*configured)
    for spec in model.specs.values():
        for name in spec.files:
            (model.models / name).touch()
    events, launches = [], []
    process = SimpleNamespace(pid=4321, exited=False, returncode=None)

    def wait(timeout):
        events.append(("reap", timeout))
        process.returncode = 0

    def send(group, signum):
        events.append((group, signum))
        process.exited = True

    def launch(command, **kwargs):
        process.exited = False
        launches.append((command, kwargs))
        return process

    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError

    process.wait = wait
    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(resources.os, "killpg", send)
    monkeypatch.setattr(runtime, "process_exited", lambda proc: proc.exited)
    monkeypatch.setattr(resources, "process_exited", lambda proc: proc.exited)
    monkeypatch.setattr(runtime.socket, "create_connection", unavailable)

    class Health:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(runtime.httpx, "Client", Health)
    try:
        yield model, process, events, launches
    finally:
        model.close()


def test_two_owners_racing_before_any_launch_are_excluded(configured):
    barrier = threading.Barrier(2)
    owners, errors = [], []

    def create():
        barrier.wait(timeout=2)
        try:
            owners.append(runtime.ModelRuntime(*configured))
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=create) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    try:
        assert not any(thread.is_alive() for thread in threads)
        assert len(owners) == 1
        assert len(errors) == 1 and "duplicate launch" in errors[0]
    finally:
        for owner in owners:
            owner.close()
    replacement = runtime.ModelRuntime(*configured)
    replacement.close()


def test_activation_owns_session_and_same_model_is_not_relaunched(owned):
    model, process, events, launches = owned
    model.activate("vision")
    model.activate("vision")
    assert len(launches) == 1
    assert launches[0][1]["start_new_session"] is True
    assert "--fit" in launches[0][0]
    # No --fit-target: the upstream default margin applies, because the 3072 MiB value it
    # replaced was raised for a long-lived server that accumulated buffers to 10.9 of
    # 12.3 GB (2026-10-07), which the per-scene reload policy has since made impossible.
    assert "--fit-target" not in launches[0][0]
    assert "--spec-type" in launches[0][0] and "draft-mtp" in launches[0][0]
    assert "--mmproj" in launches[0][0]
    assert model.generation == 1
    model.deactivate()
    assert events == [(4321, signal.SIGTERM), (4321, signal.SIGKILL), ("reap", 10)]
    assert model.process is None and model.active is None
    assert launches[0][1]["stdout"].closed


def test_closed_runtime_cannot_restart_and_close_is_idempotent(owned):
    model, process, events, launches = owned
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.activate("vision")
    assert launches == []


def test_abort_between_headroom_check_and_spawn_prevents_launch(owned, monkeypatch):
    model, process, events, launches = owned

    def abort(estimate):
        with model._process_lock:
            model.abort_reason = "memory reserve breached"

    monkeypatch.setattr(runtime, "require_headroom", abort)
    with pytest.raises(RuntimeError, match="memory reserve breached"):
        model.activate("vision")
    assert launches == []
    assert model._log is None


@pytest.mark.parametrize("exception", [KeyboardInterrupt, RuntimeError])
def test_startup_exception_cleans_owned_group_only(owned, monkeypatch, exception):
    model, process, events, launches = owned

    def fail(self, url):
        raise exception("fixture startup failure")

    monkeypatch.setattr(runtime.httpx.Client, "get", fail)
    with pytest.raises(exception):
        model.activate("vision")
    assert events == [(4321, signal.SIGTERM), (4321, signal.SIGKILL), ("reap", 10)]
    assert model.process is None and model._log is None


def test_occupied_port_is_never_adopted_or_terminated(owned, monkeypatch):
    model, process, events, launches = owned

    class Occupied:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: Occupied())
    with pytest.raises(RuntimeError, match="refusing to adopt"):
        model.activate("vision")
    assert events == [] and launches == []


@pytest.mark.parametrize("available", [0, RuntimeError("missing MemAvailable")])
def test_monitor_failure_escalates_owned_cleanup_without_request_lock(
    owned, monkeypatch, available,
):
    model, process, events, launches = owned
    model.activate("vision")
    original_shutdown = model._shutdown
    # Drive one monitor sample, without a background wait or a real low-memory host.
    model._shutdown = SimpleNamespace(wait=lambda delay: False)

    def memory():
        if isinstance(available, Exception):
            raise available
        return available

    monkeypatch.setattr(runtime, "memory_available", memory)
    monitor = threading.Thread(target=ORIGINAL_WATCH_MEMORY, args=(model,))
    try:
        with model.lock:
            monitor.start()
            monitor.join(timeout=3)
            assert not monitor.is_alive()
    finally:
        monitor.join(timeout=3)
        model._shutdown = original_shutdown
    assert model.abort_reason
    assert model.process is None
    assert events == [(4321, signal.SIGTERM), (4321, signal.SIGKILL), ("reap", 10)]
    with pytest.raises(RuntimeError):
        model.activate("vision")


ORIGINAL_WATCH_MEMORY = runtime.ModelRuntime._watch_memory


def test_local_entry_closes_owner_on_sigterm(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "local.py"
    spec = importlib.util.spec_from_file_location("grounded_intent_local_lifecycle", source)
    local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local)
    closed = []
    owner = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(local, "ModelRuntime", lambda *args, **kwargs: owner)

    def terminate(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(local, "project_main", terminate)
    with pytest.raises(SystemExit) as stopped:
        local.entry(["run"])
    assert stopped.value.code == 143
    assert closed == [True]


def test_abort_during_health_response_cannot_commit_success(owned, monkeypatch):
    model, process, events, launches = owned

    def abort(self, url):
        with model._process_lock:
            model.abort_reason = "memory reserve breached"
            model._stop_process()
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(runtime.httpx.Client, "get", abort)
    with pytest.raises(RuntimeError, match="memory reserve breached"):
        model.activate("vision")
    assert model.generation == 0
    assert model.process is None
    assert events == [(4321, signal.SIGTERM), (4321, signal.SIGKILL), ("reap", 10)]
