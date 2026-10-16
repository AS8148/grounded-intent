"""Reproducible CLI: image-only perception, inference, diagnostics, and local demo."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .clients import ModelCallError
from .pipeline import Pipeline, save_json
from .reasoning import prompt_revisions
from .runtime import ModelRuntime


def instructions(folder: Path) -> list[tuple[str, str]]:
    """Provider category labels are deliberately discarded, never model inputs."""
    source = folder / "instructions.txt"
    if not source.is_file():
        raise ValueError(f"missing instructions.txt in {folder.name}")
    jobs = []
    for raw in source.read_text().splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if "\t" in text:
            _, text = text.split("\t", 1)
        text = text.strip()
        if text:
            jobs.append((f"{folder.name}_i{len(jobs) + 1}", text))
    if not jobs:
        raise ValueError(f"no instructions in {folder.name}")
    return jobs


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Grounded Intent — local verified Apertus planning")
    root.add_argument("command", nargs="?", default="run",
                      choices=["run", "perceive", "demo", "inspect"],
                      help="default: run (one command on any folder of scenes)")
    root.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_DIR", "../dataset")))
    root.add_argument("--cache", type=Path, default=Path(os.environ.get("CACHE_DIR", ".cache/scenes")))
    root.add_argument("--out", type=Path, default=Path("results.json"))
    root.add_argument("--audit", type=Path, default=Path("artifacts/audit.json"))
    root.add_argument("--scene", action="append", help="select scene(s), otherwise all")
    root.add_argument("--instruction", help="single interactive instruction instead of file")
    root.add_argument("--vision-url", default=os.environ.get("VISION_BASE_URL", "http://127.0.0.1:8081/v1"))
    root.add_argument("--reasoning-url", default=os.environ.get(
        "LLM_BASE_URL", "http://127.0.0.1:8082/v1"))
    root.add_argument("--vision-model", default=os.environ.get("VISION_MODEL", "qwen3.5-9b-mtp-q4"))
    root.add_argument("--reasoning-model", default=os.environ.get("LLM_NAME", "apertus-v1.5-8b-q8_0"))
    root.add_argument("--reasoning-api-key", default=os.environ.get(
        "LLM_API_KEY", os.environ.get("CSCS_API_KEY") or None),
        help="Bearer key for a remote https reasoning endpoint (e.g. CSCS); "
             "omit for the default sovereign loopback runtime")
    root.add_argument("--vision-api-key", default=os.environ.get(
        "VISION_API_KEY", os.environ.get("CSCS_API_KEY") or None),
        help="Bearer key for a remote https vision endpoint (e.g. CSCS multimodal Apertus); "
             "omit for the default sovereign loopback runtime")
    root.add_argument("--vision-backend", choices=["vlm", "hybrid", "fused"], default="fused",
        help="fused = strict VLM inventory + YOLOE geometry re-grounding (default, "
             "highest measured score); vlm = strict coordinate inventory only; "
             "hybrid = label-only VLM + YOLOE grounding")
    root.add_argument("--no-segment", action="store_true", help="explicit VLM-only perception ablation")
    root.add_argument("--vision-device", choices=["cpu", "cuda"], default="cpu")
    root.add_argument("--resume", action="store_true",
                      help="continue a previous run whose manifest matches this tree")
    root.add_argument("--decision-timeout", type=float, default=float(os.environ.get(
        "DECISION_TIMEOUT", "900")),
        help="wall-clock seconds allowed per instruction decision")
    root.add_argument("--trace", type=Path, default=Path(
        os.environ.get("TRACE_PATH", "artifacts/traces.jsonl")),
        help="append-only request/response trace; credentials and image bytes are "
             "never written, images become content digests")
    root.add_argument("--port", type=int, default=8090)
    root.add_argument("--limit", type=int, default=0, help="bounded diagnostic run; 0 = all")
    return root



def main(argv: list[str] | None = None, runtime: ModelRuntime | None = None) -> int:
    args = parser().parse_args(argv)
    pipeline_type = Pipeline
    segment, vision_backend = not args.no_segment, args.vision_backend
    pipeline = None
    try:
        pipeline = pipeline_type(
            args.data, args.cache, vision_url=args.vision_url, reasoning_url=args.reasoning_url,
            vision_model=args.vision_model, reasoning_model=args.reasoning_model,
            segment=segment, vision_device=args.vision_device, runtime=runtime,
            reasoning_api_key=args.reasoning_api_key, vision_api_key=args.vision_api_key,
            vision_backend=vision_backend, decision_timeout=args.decision_timeout,
            trace_path=args.trace,
        )
        if args.command == "demo":
            import uvicorn

            from .web import create_app
            uvicorn.run(create_app(pipeline), host=os.environ.get("WEB_HOST", "127.0.0.1"),
                        port=args.port, log_level="info")
            return 0
        names = args.scene or sorted(
            folder.name for folder in pipeline.data.iterdir()
            if folder.is_dir() and (folder / "rgb.png").is_file()
        )
        if not names:
            raise ValueError("no RGB scenes found")
        if args.instruction and args.command in {"perceive", "inspect"}:
            raise ValueError("--instruction is only supported by the run command")
        if args.instruction and len(names) != 1:
            raise ValueError("--instruction requires exactly one --scene")
        # The manifest is what makes a run comparable: the code revision and prompt
        # revisions pin the tree, and the models/settings pin the environment. A
        # resumed run must match all of them, or the answers would mix two trees.
        manifest = {
            "code_revision": pipeline.code_revision,
            "prompt_revisions": prompt_revisions(),
            "vision_model": args.vision_model,
            "reasoning_model": args.reasoning_model,
            "vision_backend": vision_backend,
            "segment": segment,
            "decision_timeout": args.decision_timeout,
            "trace": str(args.trace),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results, graphs, audit, scene_failures = [], {}, [], {}
        evaluation_mode = getattr(pipeline, "evaluation_mode", "standard_pipeline")
        answered: set[str] = set()
        if args.resume and args.out.is_file() and args.command not in {"perceive", "inspect"}:
            previous = json.loads(args.out.read_text())
            previous_manifest = {}
            if args.audit.is_file():
                try:
                    previous_manifest = json.loads(
                        args.audit.read_text()).get("manifest", {})
                except (OSError, ValueError):
                    previous_manifest = {}
            for key in ("code_revision", "prompt_revisions"):
                if previous_manifest.get(key) != manifest[key]:
                    raise ValueError(
                        "cannot resume: the previous run used a different "
                        f"{key.replace('_', ' ')}")
            for entry in previous.get("results", []):
                if isinstance(entry, dict) and entry.get("instruction_id"):
                    results.append(entry)
                    answered.add(entry["instruction_id"])
            if args.audit.is_file():
                try:
                    audit.extend(json.loads(args.audit.read_text()).get("decisions", []))
                except (OSError, ValueError):
                    pass
            print(f"resuming: {len(answered)} answers already recorded", flush=True)
        audit_report = {"evaluation_mode": evaluation_mode, "manifest": manifest,
                        "scenes": graphs, "scene_failures": scene_failures,
                        "decisions": audit, "prompt_revisions": prompt_revisions()}

        def publish() -> None:
            """Write the current answers so a later failure cannot erase them."""
            if args.command in {"perceive", "inspect"}:
                save_json(args.out, {"scenes": graphs})
            else:
                save_json(args.out, {"team": "Grounded Intent", "results": results})

        def scene_with_recycle(name: str):
            """One scene's observations, retried once on a fresh server after a failure.

            Perception runs inside this command when the cache misses, and the pinned
            llama-server enters its documented grammar fault as a *state* (llama.cpp
            #30419): one failed request takes the rest of that process with it, so a
            single fault would lose a whole scene. The batch driver re-runs such scenes;
            this gives the one command the same resilience.
            """
            try:
                return pipeline.scene(name)
            except Exception as first:  # noqa: BLE001 - retried once, then reported
                if runtime is None:
                    raise
                print(f"{name}: {type(first).__name__} while perceiving; recycling the "
                      f"server and retrying once", flush=True)
                runtime.deactivate()
                return pipeline.scene(name)

        def decide_with_recycle(scene: str, text: str, instruction_id: str):
            """One decision, retried once on a fresh server after a transport failure.

            The pinned llama-server enters the documented grammar fault as a *state*
            (llama.cpp #30419): once one constrained request fails, every later request on
            that process fails too, so without a recycle a single fault would cost every
            remaining answer of the run. Recycling the owned server cannot change an answer
            that already succeeded; it only gives the next request a healthy process.
            """
            try:
                return pipeline.decide(scene, text, instruction_id)
            except ModelCallError as exc:
                print(f"  {instruction_id}: {type(exc).__name__} on the pinned server; "
                      f"recycling it and retrying once", flush=True)
                if runtime is not None:
                    runtime.deactivate()
                return pipeline.decide(scene, text, instruction_id)

        start = time.perf_counter()
        for name in names:
            try:
                graph = scene_with_recycle(name)
            except Exception as exc:  # noqa: BLE001 - failed scenes remain unanswered
                scene_failures[name] = {
                    "status": "failed", "error": str(exc), "error_type": type(exc).__name__,
                }
                print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            else:
                graphs[name] = graph.model_dump(mode="json")
                print(f"{name}: {len(graph.objects)} objects, {len(graph.relations)} relations", flush=True)
            save_json(args.audit, audit_report)
            publish()
        for name in names:
            if args.command in {"perceive", "inspect"}:
                continue
            if args.instruction:
                jobs = [(name + "_interactive", args.instruction)]
            else:
                try:
                    jobs = instructions(pipeline.folder(name))
                except (ValueError, OSError) as exc:
                    # A missing/empty instructions file must not discard answers
                    # already computed for other scenes.
                    failure = {
                        "status": "failed", "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                    scene_failures.setdefault(name, failure)
                    audit.append({"scene": name, "instruction_id": f"{name}_instructions",
                                  "instruction": "", "stage": "instructions", **failure})
                    print(f"  {name}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    save_json(args.audit, audit_report)
                    publish()
                    continue
            for instruction_id, text in jobs:
                if instruction_id in answered:
                    print(f"  {instruction_id}: already answered (resume)", flush=True)
                    continue
                if name in scene_failures:
                    failure = scene_failures[name]
                    audit.append({"scene": name, "instruction_id": instruction_id,
                                  "instruction": text, "stage": "scene", **failure})
                    print(f"  {instruction_id}: FAILED {failure['error_type']}: "
                          f"{failure['error']}", flush=True)
                    save_json(args.audit, audit_report)
                    continue
                try:
                    decision = pipeline.decide(name, text, instruction_id)
                except Exception as exc:  # noqa: BLE001 - log-and-skip; an unanswered
                    # instruction scores 0, but one failure must not abort the run.
                    print(f"  {instruction_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    audit.append({"scene": name, "instruction_id": instruction_id,
                                  "instruction": text, "status": "failed", "stage": "decision",
                                  "error": str(exc), "error_type": type(exc).__name__})
                    save_json(args.audit, audit_report)
                    continue
                results.append(decision.official())
                audit.append({"scene": name, "instruction": text, **decision.model_dump(mode="json")})
                print(f"  {instruction_id}: {decision.kind} {decision.target_pixel} "
                      f"{decision.reasoning_seconds:.2f}s", flush=True)
                # Keep diagnostics separate from the pure official answer payload.
                save_json(args.audit, audit_report)
                publish()
                if args.limit > 0 and len(results) >= args.limit:
                    break
            if args.limit > 0 and len(results) >= args.limit:
                break
            # One image per scene, and the pinned llama-server enters its grammar fault as
            # a *state* once enough constrained requests accumulate (llama.cpp #30419), so
            # the owned runtime is recycled between scenes instead of being left to reach
            # it: restart the model after each image. A development run against the remote
            # endpoint owns no server and is unaffected.
            if runtime is not None:
                runtime.deactivate()
        save_json(args.audit, audit_report)
        publish()
        print(json.dumps({"output": str(args.out), "answers": len(results),
                          "evaluation_mode": evaluation_mode,
                          "elapsed_seconds": round(time.perf_counter() - start, 2)}))
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        if runtime is not None:
            runtime.deactivate()
        print(f"Grounded Intent: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
