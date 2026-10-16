#!/usr/bin/env python3
"""Perceive every scene of a folder, one fresh guarded child process per scene.

The pinned llama-server enters an upstream grammar fault after a few minutes of
grammar-constrained vision requests (llama.cpp #30419). It is a *state* the server
enters rather than a per-request coin flip: once it starts, almost every later
request fails, and restarting only the server inside one long-lived process was
measured insufficient. Each scene is therefore perceived by its own child process
with its own cgroup, its own ``ModelRuntime`` and its own llama-server.

A child that survives the fault can still come back degraded, so a scene whose
server log shows the fault is perceived again in another fresh child, bounded by
``--attempts``. Results land in the perception cache, so the scored run itself
never needs the vision model again::

    python scripts/scenes.py --data ../dataset --cache .cache/scenes
    python run.py --data ../dataset --out results.json   # cache hit: reasoning only

The child is ``scripts/local.py perceive --scene <name>``, so the perception code
path is exactly the one a single scene uses; this driver only sequences it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "scripts" / "local.py"
# The fault's own signature, from the server log the runtime owns.
GRAMMAR_FAULT = "empty grammar stack"
PORT_RELEASE_SECONDS = 60.0
PORT_COLLISION = "already occupied"


def scenes(data: Path) -> list[str]:
    """Every scene folder in ``data``, in a stable order."""
    return sorted(
        folder.name for folder in data.iterdir()
        if folder.is_dir() and (folder / "rgb.png").is_file()
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_DIR", "../dataset")))
    root.add_argument("--cache", type=Path, default=Path(os.environ.get("CACHE_DIR", ".cache/scenes")))
    root.add_argument("--scene", action="append", help="scene(s) to perceive; default all")
    root.add_argument("--attempts", type=int, default=2,
                      help="fresh-child attempts per scene, for a child that failed or "
                           "survived the grammar fault in a degraded state")
    root.add_argument("--models", type=Path,
                      default=Path(os.environ.get("MODELS_DIR", "../.models")))
    root.add_argument("--server", type=Path, default=Path(os.environ.get(
        "LLAMA_SERVER", "../.tools/llama.cpp/build/bin/llama-server")))
    root.add_argument("--vision-model", default=os.environ.get("VISION_MODEL", "qwen3.5-9b-mtp-q4"))
    root.add_argument("--vision-backend", default=os.environ.get("VISION_BACKEND", "fused"))
    return root


def fault_count(log: Path) -> int:
    """Occurrences of the grammar fault in the server log, or 0 when unreadable."""
    try:
        return log.read_text(errors="replace").count(GRAMMAR_FAULT)
    except OSError:
        return 0


def perceive_one(args: argparse.Namespace, name: str, log: Path) -> tuple[bool, str]:
    """Perceive one scene in a fresh child. Returns (clean, note)."""
    before = fault_count(log)
    command = [
        sys.executable, str(LAUNCHER), "perceive",
        "--scene", name, "--data", str(args.data), "--cache", str(args.cache),
        "--models", str(args.models), "--server", str(args.server),
        "--vision-model", args.vision_model, "--vision-backend", args.vision_backend,
        "--out", str(ROOT / "artifacts" / f"perceive_{name}.json"),
        "--audit", str(ROOT / "artifacts" / f"audit_{name}.json"),
    ]
    for attempt in range(args.attempts):
        if attempt:
            # The fixed port must be free before a child can own it; a slow
            # release must never be mistaken for the fault.
            deadline = time.monotonic() + PORT_RELEASE_SECONDS
            time.sleep(2.0)
            while time.monotonic() < deadline and _port_busy(args.server):
                time.sleep(1.0)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        after = fault_count(log)
        if completed.returncode != 0:
            note = f"child exited {completed.returncode}"
        elif after > before:
            note = f"survived the grammar fault ({after - before} faults); re-running"
        else:
            return True, "clean"
        print(f"    {name}: {note}", flush=True)
    return False, "unclean after every attempt"


def _port_busy(server: Path) -> bool:
    """True while a llama-server from this tree still holds the fixed port."""
    try:
        out = subprocess.run(["pgrep", "-f", str(server)], capture_output=True, text=True,
                             check=False)
    except OSError:
        return False
    return bool(out.stdout.strip())


def main() -> int:
    args = parser().parse_args()
    data, cache = args.data.resolve(), args.cache.resolve()
    if not data.is_dir():
        print(f"Grounded Intent: no such data folder {data}", file=sys.stderr)
        return 1
    names = args.scene or scenes(data)
    if not names:
        print(f"Grounded Intent: no RGB scenes in {data}", file=sys.stderr)
        return 1
    log = ROOT / "artifacts" / "vision.server.log"
    clean, unclean = [], []
    start = time.perf_counter()
    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        ok, note = perceive_one(args, name, log)
        (clean if ok else unclean).append(name)
        print(f"    {name}: {note}", flush=True)
    elapsed = round(time.perf_counter() - start, 1)
    print(f"\nperceived {len(clean)} of {len(names)} scenes cleanly in {elapsed}s; cache {cache}")
    if unclean:
        print("unclean scenes (re-run before scoring): " + ", ".join(unclean), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
