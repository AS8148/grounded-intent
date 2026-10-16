"""Behavioural cover for the per-scene launcher's child command.

The launcher decides a multi-scene run in one fresh child process per scene. Its own
options do not reach the children through ``project_args`` (argparse consumes them), so
the child command has to carry them explicitly. Measured failure this guards: a dev run
launched with ``--apertus-gpu-layers 32`` was decided at the default 24 in every child,
because the flag was parsed by the parent and never forwarded — the option looked accepted
and changed nothing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_path = Path(__file__).resolve().parents[1] / "scripts" / "local.py"
_spec = importlib.util.spec_from_file_location("local_launcher", _path)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)


def _options(**overrides) -> SimpleNamespace:
    base = {"models": Path("../.models"), "server": Path("../server"),
            "inference_port": 8081, "apertus_gpu_layers": 24, "threads": 8}
    base.update(overrides)
    return SimpleNamespace(**base)


def _children(options, project_args, scenes):
    """The command this launcher would run for each scene, built by the real code."""
    return [launcher._child_command(options, project_args, name,
                                    Path(f"artifacts/decide_{name}.json"),
                                    Path(f"artifacts/audit_decide_{name}.json"))
            for name in scenes]


def test_child_command_carries_the_parents_layer_count():
    # The regression: without forwarding, the children silently decide at the default.
    captured = _children(_options(apertus_gpu_layers=32),
                         ["--data", "data", "--out", "results.json"], ["scene_01", "scene_02"])
    assert len(captured) == 2
    for command, scene in zip(captured, ["scene_01", "scene_02"], strict=True):
        assert command[command.index("--scene") + 1] == scene
        assert command[command.index("--apertus-gpu-layers") + 1] == "32"


def test_child_command_carries_every_launcher_option_once():
    captured = _children(_options(threads=6, inference_port=8090),
                         ["--data", "data"], ["scene_01", "scene_02"])
    command = captured[0]
    for flag, value in (("--models", str(_options().models)), ("--server", str(_options().server)),
                        ("--inference-port", "8090"), ("--apertus-gpu-layers", "24"),
                        ("--threads", "6")):
        assert command.count(flag) == 1, flag
        assert command[command.index(flag) + 1] == value, flag


def test_child_command_keeps_project_arguments_and_drops_replaced_outputs():
    # The project's own flags pass through, while per-scene outputs are set by the parent.
    captured = _children(_options(),
                         ["--data", "dataset", "--cache", "cache", "--out", "all.json",
                          "--audit", "all-audit.json", "run"], ["scene_01", "scene_02"])
    command = captured[0]
    assert command[command.index("--data") + 1] == "dataset"
    assert command[command.index("--cache") + 1] == "cache"
    assert command.count("--out") == 1 and command.count("--audit") == 1
    # Exactly one command word: the parent supplies it, and a "run" arriving in
    # project_args is filtered so argparse cannot see it twice and reject the child.
    assert command.count("run") == 1 and command[command.index("run") + 1] == "--scene"
