"""Local, read-only inspection of grounded observations and proposed decisions."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .domain import Decision, SceneGraph

_SCENE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_STATIC = Path(__file__).with_name("static")


class InteractivePipeline(Protocol):
    data: Path

    def scene(self, name: str) -> SceneGraph: ...

    def decide(self, scene_name: str, instruction: str, instruction_id: str) -> Decision: ...


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: str
    instruction: str = Field(min_length=1, max_length=2000)

    @field_validator("instruction")
    @classmethod
    def meaningful_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("An instruction cannot be blank.")
        return value


def _scene_image(data: Path, name: str) -> Path:
    """Validate names before touching disk; never follow scene or image symlinks."""
    if not _SCENE_NAME.fullmatch(name) or ".." in name:
        raise HTTPException(400, "Invalid scene name.")
    directory = data / name
    image = directory / "rgb.png"
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise HTTPException(404, "Scene not found.")
        if image.is_symlink() or not image.is_file():
            raise HTTPException(404, "Scene image not found.")
        with Image.open(image) as rgb:
            if rgb.format != "PNG" or min(rgb.size) <= 0:
                raise HTTPException(404, "Scene image is not a valid PNG.")
            rgb.verify()
    except (OSError, UnidentifiedImageError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(404, "Scene image is unavailable or invalid.") from None
    return image


def create_app(pipeline: InteractivePipeline) -> FastAPI:
    """Adapt the real synchronous pipeline without blocking the server event loop."""
    app = FastAPI(title="Grounded Intent", docs_url=None, redoc_url=None)
    data = Path(pipeline.data).resolve()

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _error):
        return JSONResponse(
            status_code=422,
            content={"detail": "Provide a scene and a nonblank instruction of at most 2000 characters."},
        )

    def scenes() -> list[str]:
        try:
            entries = list(data.iterdir())
        except OSError:
            raise HTTPException(503, "Local scenes are unavailable. Check the dataset location.") from None
        names = []
        for entry in entries:
            try:
                _scene_image(data, entry.name)
            except HTTPException:
                continue
            names.append(entry.name)
        return sorted(names)

    @app.get("/api/scenes")
    async def list_scenes():
        return await run_in_threadpool(scenes)

    @app.get("/api/scenes/{scene:path}/image")
    async def scene_image(scene: str):
        image = await run_in_threadpool(_scene_image, data, scene)
        return FileResponse(image, media_type="image/png")

    def analyze(scene: str) -> SceneGraph:
        _scene_image(data, scene)
        try:
            return pipeline.scene(scene)
        except Exception:
            raise HTTPException(
                503, "Scene analysis is unavailable. Check the local vision model and try again."
            ) from None

    @app.get("/api/scenes/{scene:path}")
    async def scene_graph(scene: str):
        return await run_in_threadpool(analyze, scene)

    def decide(request: DecisionRequest) -> dict:
        started = time.perf_counter()
        graph = analyze(request.scene)
        try:
            decision = pipeline.decide(
                request.scene, request.instruction, request.scene + "_interactive"
            )
        except Exception:
            raise HTTPException(
                503, "Decision reasoning is unavailable. Check the local model servers and try again."
            ) from None
        result = decision.model_dump(mode="json")
        result["official"] = decision.official()
        result["warnings"] = list(graph.warnings)
        result["timings"] = {
            "perception_seconds": graph.perception_seconds,
            "reasoning_seconds": decision.reasoning_seconds,
            "request_seconds": time.perf_counter() - started,
        }
        return result

    @app.post("/api/decide")
    async def make_decision(request: DecisionRequest):
        return await run_in_threadpool(decide, request)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_STATIC / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    return app
