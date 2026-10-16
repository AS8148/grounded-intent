# Report and slides

The submission deliverables are `report/technical_report.pdf` (two A4 pages) and `report/slides.pdf` (five 16:9 slides). Their readable Markdown counterparts are generated from the same prose, not maintained as a separate draft.

## Rebuild the deliverables

With the project's `report` extra installed, run from this repository:

```sh
python scripts/report.py --summary report/benchmark_summary.json --outdir report
```

This uses the committed summary and performs no inference, download, scoring or Docker build. The renderer checks page counts, measures wrapping and rejects overflowing panels instead of clipping them. Both PDFs are rendered before either is written. The source prose is in `docs/report-prose.json`; metric placeholders resolve from the summary's measured values.

To rebuild the summary from the saved cold local run, when those ignored working artifacts are available:

```sh
python scripts/summarize.py \
  --report artifacts/report_fresh_scratch.json \
  --audit artifacts/audit_fresh_scratch.json \
  --prose docs/report-prose.json \
  --tests-passed 635 --model-tests-passed 0 \
  --out report/benchmark_summary.json
python scripts/report.py --summary report/benchmark_summary.json --outdir report
```

The test arguments are a recorded suite result, not a live-model benchmark. `summarize.py` derives score components from the scorer and audit.

## What the five slides cover

1. **Team and approach.** The first-pick score can be correct while the requested placement is missing. State the hybrid approach and emphasize that the complete local pipeline ran on one RTX 4070 Ti with 12 GB VRAM, using sequential model loading.
2. **Pipeline diagram.** Image-only Qwen/YOLOE/SAM2, instruction-only Apertus semantics, Apertus grounding, downstream corrections and symbolic verification, then JSON. Arrows show the separate image and instruction paths.
3. **Development results.** The local 65/65 snapshot, grounding/safety breakdown, held-out evidence limits and an example of an incomplete plan that still scores. No held-out accuracy is invented.
4. **Apertus difficulties.** Failed rich schemas and narrower relation calls; the Apertus image-observer branch (37.625/65 versus the saved Qwen 65/65); a prompt fix that did not transfer; the 70B substitution. Remote diagnostics are labelled separately from local submission measurements.
5. **Lessons and next experiments.** Keep Qwen on vision in this tested pipeline, preserve the original task independently, test semantic transfer, regenerate results from the exact shipped configuration, and avoid treating symbolic checks as robot safety certification.

The report expands the same argument. It does not claim that all model-first work failed: later controller and abstention-contract variants made measurable progress without passing their acceptance gate.

Keep the report to two pages and the deck to five slides. The language experiments
and the gap between first-pick score and task completion are the main story. The
vision substitution gets a short paragraph and one slide panel; its full loss
ledger belongs in the evidence files, not in extra report pages. The single-GPU
claim covers the local submission pipeline, not the remote CSCS diagnostics.

## Evidence and boundaries

`report/findings.md` maps claims E1–E10 to notebook entries, source code and saved experiments. `report/evidence.json` indexes the preserved records, raw responses and artifact hashes. `report/vision_comparison.json` retains both scorer reports, the complete treatment audit/results and input-image integrity checks for E10. These are documentation data, never runtime inputs.

The report deliberately distinguishes:
- public development score from hidden held-out accuracy;
- local measured runs from remote CSCS diagnostics;
- a decoded response from a valid semantic contract;
- component gains from accepted end-to-end behaviour;
- the saved measured snapshot from later code edits;
- symbolic action consistency from successful physical execution.

The scorer returns `robot_check="ok"` without a 3D check on RGB-only captures. The Dockerfile also differs from the measured host configuration in MTP drafting. Neither detail supports an output-equivalence or physical-safety claim.

The PDF footer links the two cited research papers and the pinned Apertus, Qwen and SAM2 model cards. The papers provide context; their results are not this project's scores. Code is Apache-2.0, benchmark data CC BY 4.0, and model weights retain their own terms.
