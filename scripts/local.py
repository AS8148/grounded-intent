"""Run the complete local project inside a bounded cgroup.

One model is loaded at a time by :class:`grounded_intent.runtime.ModelRuntime`;
all memory accounting and shutdown happen in that owner, never in unrelated services.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Resolve this checkout's own package first: an editable install of another
# checkout must never shadow it, and a fresh clone has no install at all.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grounded_intent.cli import main as project_main
from grounded_intent.pipeline import save_json
from grounded_intent.resources import (
    MEMORY_BUDGET,
    effective_memory_limit,
    effective_swap_limit,
    run_guarded,
    termination_cleanup,
)
from grounded_intent.runtime import ModelRuntime


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path(os.environ.get("MODELS_DIR", "../.models")))
    parser.add_argument("--server", type=Path, default=Path(os.environ.get(
        "LLAMA_SERVER", "../.tools/llama.cpp/build/bin/llama-server")))
    parser.add_argument("--inference-port", type=int, default=8081)
    # All layers, always. Apertus has 32; the split at 24 existed because all-on-GPU
    # was measured on 2026-10-14 to drive scene_04 into the documented grammar fault
    # (six rows lost to "Unexpected empty grammar stack"). That measurement predates the
    # per-scene fresh child for the reasoning path, and does not reproduce on this tree:
    # re-measured 2026-10-14, the dev set scores 65.0/65 at 32 layers with 50/50 answers,
    # zero grammar-fault lines, and 238 s for the fifty decisions against ~600 s at 24.
    # One held-out answer differs between the two settings (a question becomes a plan),
    # and the held-out set carries no answers here, so neither setting can be preferred
    # on that evidence - the default is therefore the fast, fully offloaded one.
    parser.add_argument("--apertus-gpu-layers", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    return parser


PORT_RELEASE_SECONDS = 60.0


def _scene_folder(project_args: list[str]) -> Path | None:
    """The folder a run was *told* to use, or None when it was not told.

    Deliberately no default: the per-scene fork must fire only for a run that names its
    scenes. A fallback to ../dataset would make any bare entry() call fork the real dev
    set - which is exactly what happened inside the test suite, spawning live inference
    children and breaking the entry-lifecycle tests.
    """
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    for index, arg in enumerate(project_args):
        if arg == "--data" and index + 1 < len(project_args):
            return Path(project_args[index + 1])
        if arg.startswith("--data="):
            return Path(arg.split("=", 1)[1])
    return None


def _run_command(project_args: list[str]) -> bool:
    """Whether these project arguments mean ``run``, which is also the CLI's default.

    A bare value (the folder after ``--data``) must not be mistaken for a command word,
    so only the known command names are inspected and anything else falls through to the
    default.
    """
    for arg in project_args:
        if arg in {"perceive", "demo", "inspect"}:
            return False
        if arg == "run":
            return True
    return True


def _option(project_args: list[str], name: str, default: str) -> str:
    for index, arg in enumerate(project_args):
        if arg == name and index + 1 < len(project_args):
            return project_args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


def _without(project_args: list[str], names: set[str]) -> list[str]:
    """The parent's project arguments minus the options a child must set for itself."""
    kept: list[str] = []
    skip = False
    for arg in project_args:
        if skip:
            skip = False
            continue
        if arg in names:
            skip = True
            continue
        if any(arg.startswith(name + "=") for name in names):
            continue
        kept.append(arg)
    return kept


def _port_busy(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _grammar_faults(log: Path) -> int:
    try:
        return log.read_text(errors="ignore").count("empty grammar stack")
    except OSError:
        return 0


def _child_command(args: argparse.Namespace, project_args: list[str], name: str,
                   out: Path, audit: Path) -> list[str]:
    """The full command line for one scene child, this script's own options included.

    ``project_args`` holds what the project CLI sees, so this script's own options are
    never in it: argparse consumed them. A child that only received ``project_args``
    would silently fall back to their defaults — measured, a run launched with
    ``--apertus-gpu-layers 32`` was decided at 24 in every child, so the flag looked
    accepted and changed nothing. The parent's effective values are therefore forwarded.
    """
    # The command word is bare, so it is filtered before _without (which assumes an
    # option is followed by its value) - otherwise a child would carry "run" twice and
    # argparse would reject it, which is exactly what broke the container target.
    base = _without([arg for arg in project_args
                     if arg not in {"run", "perceive", "demo", "inspect"}],
                    {"--scene", "--out", "--audit"})
    return [sys.executable, str(Path(__file__).resolve()), "run", "--scene", name,
            "--out", str(out), "--audit", str(audit),
            "--models", str(args.models), "--server", str(args.server),
            "--inference-port", str(args.inference_port),
            "--apertus-gpu-layers", str(args.apertus_gpu_layers),
            "--threads", str(args.threads), *base]


def _decide_by_scene(args: argparse.Namespace, project_args: list[str],
                     scenes: list[str]) -> int:
    """Decide each scene in its own guarded child process, then merge the answers.

    The pinned llama-server enters its grammar fault as a *state* (llama.cpp #30419), and
    the project measured that restarting only the server inside one long-lived process is
    insufficient: a pool stopped at image 12 and 15 of 56, while one guarded child per image
    collected 65 of 65. Perception already has a per-scene driver (``scripts/scenes.py``);
    this gives a local reasoning run the same protection, and only a local one, since a
    remote endpoint owns no server that can fault this way. Each child is this same script,
    so it gets its own cgroup, its own ``ModelRuntime`` and its own llama-server.
    """
    root = Path(__file__).resolve().parents[1]
    log = root / "artifacts" / "reasoning.server.log"
    # The command word is bare, so it is filtered before _without (which assumes an
    # option is followed by its value) - otherwise a child would carry "run" twice and
    # argparse would reject it, which is exactly what broke the container target.
    results: list[dict] = []
    decisions: list[dict] = []
    graphs: dict = {}
    failures: dict = {}
    manifest_of_children: dict = {}
    for name in scenes:
        out = root / "artifacts" / f"decide_{name}.json"
        audit = root / "artifacts" / f"audit_decide_{name}.json"
        child = _child_command(args, project_args, name, out, audit)
        for attempt in range(3):
            if attempt:
                # A child owns the fixed port for its whole life; a slow release must
                # never be mistaken for the fault.
                deadline = time.monotonic() + PORT_RELEASE_SECONDS
                time.sleep(2.0)
                while time.monotonic() < deadline and _port_busy(port_of(project_args)):
                    time.sleep(1.0)
            before = _grammar_faults(log)
            completed = subprocess.run(
                child, cwd=root, check=False,
                env={**os.environ, "GROUNDED_INTENT_GUARDED": "1"})
            after = _grammar_faults(log)
            if completed.returncode != 0 or not out.is_file():
                print(f"  {name}: child exited {completed.returncode}; re-running in a "
                      f"fresh child (attempt {attempt + 1} of 3)", flush=True)
                continue
            if after > before:
                print(f"  {name}: survived {after - before} grammar fault(s); re-running "
                      f"in a fresh child (attempt {attempt + 1} of 3)", flush=True)
                continue
            payload = json.loads(out.read_text())
            results.extend(payload.get("results", []))
            if audit.is_file():
                recorded = json.loads(audit.read_text())
                decisions.extend(recorded.get("decisions", []))
                graphs.update(recorded.get("scenes", {}))
                failures.update(recorded.get("scene_failures", {}))
                # The children's own manifest, so the merged audit names the models and
                # revisions that actually ran rather than a default.
                manifest_of_children.update(recorded.get("manifest", {}))
            print(f"  {name}: {len(payload.get('results', []))} answers", flush=True)
            break
    save_json(_option(project_args, "--out", str(root / "artifacts" / "results.json")),
              {"team": "Grounded Intent", "results": results})
    save_json(_option(project_args, "--audit", str(root / "artifacts" / "audit.json")),
              {"evaluation_mode": "standard_pipeline",
               "manifest": {**manifest_of_children,
                            "per_scene_children": len(scenes)},
               "scenes": graphs, "scene_failures": failures, "decisions": decisions})
    print(f"decided {len(results)} answers across {len(scenes)} fresh child processes",
          flush=True)
    return 0


def port_of(project_args: list[str]) -> int:
    try:
        return int(_option(project_args, "--port", "8082"))
    except ValueError:
        return 8082


def entry(argv: list[str] | None = None) -> int:
    """Run the project with one sequentially-owned model runtime."""
    args, project_args = parser().parse_known_args(argv)
    if not project_args:
        project_args = ["demo"]
    # A local run over several scenes is decided one fresh child per scene, before this
    # process claims the runtime: the runtime lock is single-owner, and the documented
    # escape from the server's grammar fault is a fresh process, not a fresh server.
    folder = _scene_folder(project_args)
    scenes = (sorted(p.name for p in folder.iterdir() if (p / "rgb.png").is_file())
              if folder is not None and folder.is_dir() else [])
    if (_run_command(project_args) and len(scenes) > 1
            and "--scene" not in project_args
            and not _option(project_args, "--reasoning-url", "")):
        return _decide_by_scene(args, project_args, scenes)
    reasoning_url = _option(project_args, "--reasoning-url", "")
    runtime = None
    if not reasoning_url:
        runtime = ModelRuntime(
            args.server, args.models, Path("artifacts"), port=args.inference_port,
            aperture_layers=args.apertus_gpu_layers, threads=args.threads,
        )
    with termination_cleanup():
        try:
            return project_main(project_args, runtime=runtime)
        finally:
            if runtime is not None:
                runtime.close()


def main() -> int:
    """Re-invoke this script inside a 12 GiB cgroup, then run the project."""
    try:
        if os.environ.get("GROUNDED_INTENT_GUARDED") == "1":
            # Already inside the parent's guarded unit: a child that re-invoked the wrapper
            # would nest transient units, which is what hung the direct `local.py run` form.
            return entry()
        limit = effective_memory_limit()
        if limit is not None and limit <= MEMORY_BUDGET and effective_swap_limit() == 0:
            return entry()
        return run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
