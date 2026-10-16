"""Resource safety for WSL: real cgroup limits, reserved headroom, no swap dependence."""
from __future__ import annotations

import fcntl
import os
import pwd
import signal
import stat
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

GIB = 1024 ** 3
MEMORY_BUDGET = 12 * GIB
MEMORY_RESERVE = 4 * GIB


def meminfo(text: str) -> dict[str, int]:
    values = {}
    for row in text.splitlines():
        key, separator, rest = row.partition(":")
        if separator:
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return values


def memory_available() -> int:
    values = meminfo(Path("/proc/meminfo").read_text())
    if "MemAvailable" not in values:
        raise RuntimeError("cannot establish memory headroom; refusing model launch")
    return values["MemAvailable"]


def cgroup_directory() -> Path | None:
    try:
        for row in Path("/proc/self/cgroup").read_text().splitlines():
            if row.startswith("0::"):
                return Path("/sys/fs/cgroup") / row[3:].lstrip("/")
    except OSError:
        return None
    return None


def _effective_limit(filename: str) -> int | None:
    group = cgroup_directory()
    if group is None:
        return None
    limits = []
    root = Path("/sys/fs/cgroup")
    while group == root or root in group.parents:
        try:
            value = (group / filename).read_text().strip()
            if value.isdigit():
                limits.append(int(value))
        except OSError:
            pass
        if group == root:
            break
        group = group.parent
    return min(limits) if limits else None


def effective_memory_limit() -> int | None:
    return _effective_limit("memory.max")


def effective_swap_limit() -> int | None:
    return _effective_limit("memory.swap.max")


def _run_lock_directory() -> Path:
    directory = Path("/run/user") / str(os.getuid())
    if not directory.exists():
        # Containers need not have a systemd runtime directory. The account's
        # own non-writable-by-others home is a private fallback, never /tmp.
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        info = home.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError("secure account home unavailable; refusing local inference")
        directory = home / ".grounded-intent-runtime"
        directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RuntimeError("secure user runtime directory unavailable; refusing local inference")
    return directory


def acquire_run_lock() -> int:
    """Hold a per-user lease; never unlink it or use a shared /tmp file."""
    directory = _run_lock_directory()
    descriptor = os.open(
        directory / "grounded-intent.lock",
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("unsafe local inference lease; refusing model launch")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another local inference run owns the runtime; refusing duplicate launch") from exc
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def termination_cleanup():
    """Make SIGTERM unwind owned lifetimes just like Python exceptions/SIGINT."""
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def require_headroom(estimated_bytes: int = 0) -> None:
    available = memory_available()
    if available < MEMORY_RESERVE + estimated_bytes:
        raise RuntimeError(
            f"model launch blocked: {available / GIB:.1f} GiB available; "
            f"need {estimated_bytes / GIB:.1f} GiB plus {MEMORY_RESERVE / GIB:.0f} GiB reserve"
        )

def process_exited(process: subprocess.Popen) -> bool:
    """Check without reaping: the leader's PID pins its owned process-group ID."""
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def stop_owned_process(process: subprocess.Popen) -> None:
    """Bounded cleanup for a child launched with start_new_session=True."""
    deadline = time.monotonic() + 10
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    while not process_exited(process) and time.monotonic() < deadline:
        time.sleep(.05)
    # Kill children before wait() releases the leader's PID for reuse.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _run_owned(command: list[str]) -> int:
    with termination_cleanup():
        process = subprocess.Popen(command, start_new_session=True)
        try:
            while not process_exited(process):
                time.sleep(.05)
        finally:
            stop_owned_process(process)
        return process.returncode



def run_guarded(command: list[str], budget: int = MEMORY_BUDGET) -> int:
    """Run the entire app and its model children in a bounded cgroup.

    Docker's explicit memory limit is accepted. An unrestricted machine must provide
    systemd user services; unavailable isolation is an error, not an unsafe fallback.
    """
    limit = effective_memory_limit()
    if limit is not None and limit <= budget and effective_swap_limit() == 0:
        return _run_owned(command)
    require_headroom(budget)
    unit = "grounded-intent-" + uuid.uuid4().hex[:10]
    environment = [f"--setenv={key}={os.environ[key]}" for key in (
        "HF_HOME", "DATA_DIR", "CACHE_DIR", "MODELS_DIR", "LLAMA_SERVER", "WEB_HOST",
        "LLM_NAME", "LLM_BASE_URL", "LLM_API_KEY", "CSCS_API_KEY",
        "VISION_BASE_URL", "VISION_MODEL", "VISION_API_KEY", "VISION_QUANT", "VISION_SIZE", "VISION_MTP_MAX", "VISION_FIT_TARGET",
        # Bounded tuning knobs the pipeline actually reads; without these the
        # wrapper silently pins every run to the compiled-in default.
        "STAGE1_MAX_TOKENS", "REASONING_MAX_TOKENS", "PERCEPTION_MAX_TOKENS",
        "PERCEPTION_DUMP_INVENTORY", "PERCEPTION_FORMS", "PERCEPTION_FORM_MAX_OBJECTS",
        "PERCEPTION_FORM_BUDGET_SECONDS",
    ) if key in os.environ]
    wrapped = [
        "systemd-run", "--user", "--wait", "--pipe", "--collect", "--unit=" + unit,
        "--working-directory=" + str(Path.cwd()),
        f"--property=MemoryHigh={int(budget * .85)}", f"--property=MemoryMax={budget}",
        "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
        "--property=KillMode=control-group", "--property=SendSIGKILL=yes",
        "--property=TimeoutStopSec=10s", *environment, *command,
    ]
    try:
        with termination_cleanup():
            try:
                return subprocess.call(wrapped)
            except BaseException:
                # Stopping the client alone need not stop the transient service.
                # Name only the unique unit created by this invocation.
                subprocess.run(
                    ["systemctl", "--user", "stop", unit + ".service"],
                    check=True, timeout=15,
                )
                raise
    except FileNotFoundError as exc:
        raise RuntimeError("memory isolation unavailable; use Docker with --memory=12g --memory-swap=12g") from exc
