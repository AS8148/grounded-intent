"""Fetch only RGB images, calibration and task texts from the pinned public starter.

Answer labels and masks are intentionally not downloaded into runtime data.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

COMMIT = "7d89e599816b2cd26f0102e74f2a9141985c40ba"
BASE = f"https://raw.githubusercontent.com/Fatemeh-MA/see-it-say-it-pick-it/{COMMIT}/dataset/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    paths = ["calibration.json", *[f"scene_{index:02d}/{filename}"
             for index in range(1, 11) for filename in ("rgb.png", "instructions.txt")]]
    for relative in paths:
        destination = args.data / relative
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(BASE + relative, headers={"User-Agent": "GroundedIntent/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(response.read())
            temporary.replace(destination)
        print(destination)


if __name__ == "__main__":
    main()
