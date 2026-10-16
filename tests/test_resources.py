"""Resource isolation and single-owner guards; no inference or transient units."""
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from grounded_intent import resources


def test_effective_limits_include_ancestors(monkeypatch):
    monkeypatch.setattr(resources, "cgroup_directory", lambda: Path("/sys/fs/cgroup/outer/inner"))
    values = {
        "/sys/fs/cgroup/outer/inner/memory.max": "max",
        "/sys/fs/cgroup/outer/memory.max": str(resources.MEMORY_BUDGET),
        "/sys/fs/cgroup/outer/inner/memory.swap.max": "max",
        "/sys/fs/cgroup/outer/memory.swap.max": "0",
    }

    def read(path):
        if str(path) not in values:
            raise FileNotFoundError(str(path))
        return values[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    assert resources.effective_memory_limit() == resources.MEMORY_BUDGET
    assert resources.effective_swap_limit() == 0


def test_owner_lock_blocks_duplicates_and_releases_without_unlinking(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "_run_lock_directory", lambda: tmp_path)
    owner = resources.acquire_run_lock()
    try:
        with pytest.raises(RuntimeError, match="duplicate launch"):
            resources.acquire_run_lock()
    finally:
        os.close(owner)
    replacement = resources.acquire_run_lock()
    os.close(replacement)
    assert (tmp_path / "grounded-intent.lock").is_file()


def test_owner_lock_refuses_symlink_without_touching_target(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "_run_lock_directory", lambda: tmp_path)
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    (tmp_path / "grounded-intent.lock").symlink_to(target)
    with pytest.raises(OSError):
        resources.acquire_run_lock()
    assert target.read_text() == "preserve"


def test_owner_lock_refuses_nonprivate_file(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "_run_lock_directory", lambda: tmp_path)
    lease = tmp_path / "grounded-intent.lock"
    lease.touch(mode=0o666)
    lease.chmod(0o666)
    with pytest.raises(RuntimeError, match="unsafe local inference lease"):
        resources.acquire_run_lock()


@pytest.mark.parametrize("swap_limit", [None, 0])
def test_guard_requires_swap_bound_and_preserves_environment(monkeypatch, swap_limit):
    monkeypatch.setattr(resources, "effective_memory_limit", lambda: resources.MEMORY_BUDGET)
    monkeypatch.setattr(resources, "effective_swap_limit", lambda: swap_limit)
    monkeypatch.setattr(resources, "require_headroom", lambda budget: None)
    monkeypatch.setenv("CSCS_API_KEY", "fixture-secret")
    monkeypatch.setenv("VISION_MODEL", "qwen3.5-9b-mtp-q4")
    monkeypatch.setenv("PERCEPTION_DUMP_INVENTORY", "1")
    calls = []
    monkeypatch.setattr(resources.subprocess, "call", lambda command: calls.append(command) or 7)
    monkeypatch.setattr(resources, "_run_owned", lambda command: calls.append(command) or 7)
    assert resources.run_guarded(["fixture-command"]) == 7
    if swap_limit == 0:
        assert calls == [["fixture-command"]]
    else:
        command = calls[0]
        assert command[0] == "systemd-run"
        assert "--property=MemorySwapMax=0" in command
        assert "--property=KillMode=control-group" in command
        assert "--setenv=CSCS_API_KEY=fixture-secret" in command
        assert "--setenv=VISION_MODEL=qwen3.5-9b-mtp-q4" in command
        assert "--setenv=PERCEPTION_DUMP_INVENTORY=1" in command
        assert command[-1] == "fixture-command"


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_interrupted_wrapper_stops_only_its_unique_unit(monkeypatch, exception):
    monkeypatch.setattr(resources, "effective_memory_limit", lambda: None)
    monkeypatch.setattr(resources, "require_headroom", lambda budget: None)
    monkeypatch.setattr(resources.uuid, "uuid4", lambda: SimpleNamespace(hex="abcdef123456"))
    stopped = []

    def interrupt(command):
        raise exception()

    monkeypatch.setattr(resources.subprocess, "call", interrupt)
    monkeypatch.setattr(resources.subprocess, "run", lambda command, **kwargs: stopped.append((command, kwargs)))
    with pytest.raises(exception):
        resources.run_guarded(["fixture-command"])
    assert stopped == [(["systemctl", "--user", "stop", "grounded-intent-abcdef1234.service"],
                        {"check": True, "timeout": 15})]


def test_sigterm_unwinds_and_restores_previous_handler():
    original = signal.getsignal(signal.SIGTERM)
    unwound = []
    with pytest.raises(SystemExit) as stopped:
        with resources.termination_cleanup():
            try:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            finally:
                unwound.append(True)
    assert stopped.value.code == 143
    assert unwound == [True]
    assert signal.getsignal(signal.SIGTERM) == original


@pytest.mark.parametrize("already_exited", [False, True])
def test_group_cleanup_kills_children_before_reaping_leader(monkeypatch, already_exited):
    events = []
    process = SimpleNamespace(pid=4321, exited=already_exited)

    def send(group, signum):
        events.append((group, signum))
        process.exited = True

    def wait(timeout):
        events.append(("reap", timeout))

    process.wait = wait
    monkeypatch.setattr(resources.os, "killpg", send)
    monkeypatch.setattr(resources, "process_exited", lambda owned: owned.exited)
    resources.stop_owned_process(process)
    assert events == [(4321, signal.SIGTERM), (4321, signal.SIGKILL), ("reap", 10)]


def test_nonresponsive_owned_group_has_bounded_escalation(monkeypatch):
    events = []
    times = iter([0, 11])
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(resources, "process_exited", lambda owned: False)
    monkeypatch.setattr(resources.os, "killpg", lambda group, sig: events.append((group, sig)))
    process = SimpleNamespace(pid=4321, wait=lambda timeout: events.append(("reap", timeout)))
    resources.stop_owned_process(process)
    assert events == [(4321, signal.SIGTERM), (4321, signal.SIGKILL), ("reap", 10)]


def test_exit_probe_does_not_reap(monkeypatch):
    calls = []
    monkeypatch.setattr(resources.os, "waitid", lambda *args: calls.append(args) or object())
    assert resources.process_exited(SimpleNamespace(pid=4321))
    assert calls == [(os.P_PID, 4321, os.WEXITED | os.WNOHANG | os.WNOWAIT)]


def test_already_bounded_command_is_owned_on_interruption(monkeypatch):
    process = SimpleNamespace(pid=4321)
    launches, cleaned = [], []
    monkeypatch.setattr(resources.subprocess, "Popen",
                        lambda command, **kwargs: launches.append((command, kwargs)) or process)

    def interrupt(owned):
        raise KeyboardInterrupt

    monkeypatch.setattr(resources, "process_exited", interrupt)
    monkeypatch.setattr(resources, "stop_owned_process", lambda owned: cleaned.append(owned))
    with pytest.raises(KeyboardInterrupt):
        resources._run_owned(["fixture-command"])
    assert launches == [(["fixture-command"], {"start_new_session": True})]
    assert cleaned == [process]


def test_container_lease_fallback_is_private_and_account_owned(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    runtime_dir = Path("/run/user") / str(os.getuid())
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda path: False if path == runtime_dir else original_exists(path))
    monkeypatch.setattr(resources.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))
    directory = resources._run_lock_directory()
    assert directory == home / ".grounded-intent-runtime"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert directory.stat().st_uid == os.getuid()


def test_container_lease_fallback_rejects_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (home / ".grounded-intent-runtime").symlink_to(unrelated, target_is_directory=True)
    runtime_dir = Path("/run/user") / str(os.getuid())
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda path: False if path == runtime_dir else original_exists(path))
    monkeypatch.setattr(resources.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))
    with pytest.raises(RuntimeError, match="secure user runtime directory"):
        resources._run_lock_directory()
    assert list(unrelated.iterdir()) == []
