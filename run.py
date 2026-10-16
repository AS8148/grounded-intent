#!/usr/bin/env python3
"""One command: run the pipeline on any folder of scenes.

    python run.py --data heldout --out results_heldout.json

A scene is a folder holding ``rgb.png`` and ``instructions.txt``. Nothing else is read:
no answer file, no ground truth, no scene-specific metadata, no category labels — the
same command runs the development scenes and a released held-out folder unchanged, which
is what lets the organisers re-run the pipeline themselves.

Everything runs locally. The guarded runtime owns the model processes inside a bounded
cgroup, activates one model at a time, and refuses to adopt an unrelated service.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.local import main  # noqa: E402 - the path is set above


if __name__ == "__main__":
    raise SystemExit(main())
