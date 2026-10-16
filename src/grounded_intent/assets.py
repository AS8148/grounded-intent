"""One authoritative set of pinned model artifacts; no runtime downloads."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Artifact:
    repository: str
    revision: str
    filename: str          # name under the local models dir
    remote: str = ""       # name inside the repository (defaults to filename)

    @property
    def remote_name(self) -> str:
        return self.remote or self.filename


# Vision: Qwen3.5-9B with its trained MTP head, UD-Q4_K_XL quant + F16 mmproj.
# The MTP tensors enable llama.cpp speculative decoding (--spec-type draft-mtp);
# UD-Q4_K_XL is the quant the vendor's own MTP command for this model uses, and
# community runs measure ~85-110 tok/s on a 12 GB RTX 4070 (the retired Q8_0
# baseline measured 42 tok/s on our card). The 4B and Q8_0 variants were
# removed 2026-10-06.
VISION = Artifact(
    "unsloth/Qwen3.5-9B-MTP-GGUF", "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
    "Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf", remote="Qwen3.5-9B-UD-Q4_K_XL.gguf",
)
VISION_ENCODER = Artifact(VISION.repository, VISION.revision,
                          "mmproj-9b-mtp-F16.gguf", remote="mmproj-F16.gguf")
# The same model at 8 bits: the plain Q8_0 conversion of the MTP repository, 3.5 GB
# larger than the UD-Q4_K_XL default (its header carries the same nextn prediction
# head). Selected with VISION_QUANT=q8_0; it does not fit the card whole, so llama.cpp
# leaves part of it on the CPU to respect the VRAM margin.
VISION_Q8 = Artifact(
    "unsloth/Qwen3.5-9B-MTP-GGUF", "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
    "Qwen3.5-9B-MTP-Q8_0.gguf", remote="Qwen3.5-9B-Q8_0.gguf",
)
# Dynamic 6-bit of the same model: the vendor's own benchmarks put its KL divergence at
# 0.144 against the plain Q8_0's uniform recipe and the UD-Q8_K_XL's 0.103, at 8.99 GB
# rather than 13.25 GB, so it is the better fidelity-per-gigabyte bet on this card.
VISION_Q6 = Artifact(
    "unsloth/Qwen3.5-9B-MTP-GGUF", "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
    "Qwen3.5-9B-MTP-UD-Q6_K_XL.gguf", remote="Qwen3.5-9B-UD-Q6_K_XL.gguf",
)
APERTUS = Artifact(
    "andreasmartin/apertus-v1.5-8b-text-Q8_0-GGUF", "2028c3184c0fa060b4259b33ee25e836da6896e5",
    "apertus-v1.5-8b-text-q8_0.gguf",
)
SAM_REPOSITORY = "facebook/sam2.1-hiera-small"
SAM_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
# Open-vocabulary detector used by the fused/hybrid perception backends. Not a
# Hugging Face artifact — pinned to the Ultralytics release asset.
YOLOE_MODEL = "yoloe-26s-seg.pt"
YOLOE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt"
)
# YOLOE's text encoder, which ultralytics otherwise fetches into the working
# directory on first use. Pinned here so preparation can do it once, offline runs
# never reach the network, and the file is git-ignored like the other weights.
MOBILECLIP_MODEL = "mobileclip2_b.ts"
MOBILECLIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts"
)
LLAMA_COMMIT = "436f6f89e1e581249900b37a5b8a12a36a6d0912"
