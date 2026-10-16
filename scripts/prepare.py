"""Download the pinned open model artifacts; runtime itself never downloads weights."""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from huggingface_hub import hf_hub_download, snapshot_download

from grounded_intent.assets import (
    APERTUS,
    MOBILECLIP_MODEL,
    MOBILECLIP_URL,
    SAM_REPOSITORY,
    SAM_REVISION,
    VISION,
    VISION_ENCODER,
    YOLOE_MODEL,
    YOLOE_URL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("../.models"))
    args = parser.parse_args()
    for artifact in (VISION, VISION_ENCODER, APERTUS):
        # Stage per-repository: two repos can publish identically named files
        # (both call the vision encoder "mmproj-F16.gguf"), so a shared local_dir
        # would silently overwrite one with the other.
        staging = args.models / ".staging" / artifact.repository.split("/")[-1]
        path = hf_hub_download(
            artifact.repository, artifact.remote_name,
            revision=artifact.revision, local_dir=staging,
        )
        target = args.models / artifact.filename
        shutil.copy2(path, target)
        print(target)
    # Staging exists only to keep same-named files from different repositories apart;
    # the copies above are the artifacts. Leaving it behind duplicated every weight
    # (~15 GB on this project), so it is removed once the copies are made.
    shutil.rmtree(args.models / ".staging", ignore_errors=True)
    print(snapshot_download(
        SAM_REPOSITORY, revision=SAM_REVISION,
        allow_patterns=["*.json", "model.safetensors"],
    ))
    # YOLOE-26 detector for the fused/hybrid perception backends. Plain HTTP
    # asset (not HF), fetched once here so `--network none` runtime never tries.
    target = args.models / YOLOE_MODEL
    with urllib.request.urlopen(YOLOE_URL, timeout=120) as src:
        target.write_bytes(src.read())
    print(target)
    # Ultralytics fetches this text encoder into the working directory on first
    # use; preparing it here is what keeps the offline run offline.
    encoder = Path.cwd() / MOBILECLIP_MODEL
    if not encoder.exists():
        with urllib.request.urlopen(MOBILECLIP_URL, timeout=600) as src:
            encoder.write_bytes(src.read())
    print(encoder)


if __name__ == "__main__":
    main()
