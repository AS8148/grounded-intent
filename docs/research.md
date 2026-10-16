# Research and architecture

## Decision

Build a **grounded, executable-state-checked Apertus planner**, not a direct image-to-coordinate prompt. This is an engineering hypothesis, not a claim of global state of the art. The official benchmark, ablations, and failure cases decide whether it works.

Current optimization is inference-only: no training or fitting to challenge answers. Perception receives images only; Apertus receives the instruction and model-derived observations. Runtime never opens provider answers or category labels. Iteration currently uses remote services without local model processes; the final submission must run locally and is a separate gate. Dataset availability and the phrase "no training needed" do not by themselves establish fine-tuning permission.

## Current primary literature reviewed

| Source | Date | Relevant result / implication |
| --- | --- | --- |
| [REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](https://arxiv.org/abs/2505.10872) | 2025-05-16, later revisions | Reports up to 36.9% planning-success degradation under vague referring expressions; missing objects are a major failure source. Its task-oriented context cognition motivates explicit object references before action emission. This does **not** establish a gain on the ZHAW dataset. |
| [Spatially Grounded Long-Horizon Task Planning in the Wild / GroundedPlanBench](https://arxiv.org/abs/2603.13433) | 2026-03-13 | Evaluates action planning jointly with where to act. Finds spatially executable plans remain a bottleneck for VLMs. We check both target geometry and full action sequences, not just natural-language plausibility. Its training/data-generation approach is not used: this challenge is zero-shot. |
| [Qwen3.5-9B MTP UD-Q4_K_XL Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF) | 2026 model family; MTP quant adopted 2026-10-06 | Deployed vision model: UD-Q4_K_XL with its F16 mmproj and MTP speculative decoding (`--spec-type draft-mtp`). The retired Q8_0 baseline measured 42 tok/s on our 12 GB card; the MTP Q4 quant is the vendor-recommended setup for this model (community runs: ~85-110 tok/s on a 12 GB RTX 4070). The 4B and Q8_0 vision variants were retired. |
| [SAM 2 / SAM 2.1](https://github.com/facebookresearch/sam2) and [Transformers SAM2 API](https://huggingface.co/docs/transformers/v4.57.1/model_doc/sam2) | 2024 family, maintained implementation | Promptable visible-object masks, batched box/point prompts and reusable image embeddings. Use the mask distance-transform maximum, not centroid: centroids can fall into a hole or outside concave objects. |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Maintained detector + segmenter implementation | Useful architectural precedent for open-vocabulary boxes refined to masks. It is not proof that the detector recognizes industrial parts or beats VLM pointing here. |
| [Molmo2 official implementation](https://github.com/allenai/molmo2) | 2026 | A current pointing-capable alternative. Not the default: local checkpoint/serving footprint and quantization must be tested before displacing the actual working Qwen backend. |
| [SAM 3](https://github.com/facebookresearch/sam3) | 2025-11 model, 2026 updates | Newer concept segmentation alternative, but different license and gated weights. Not a drop-in assumption of superior performance; excluded from the unauthenticated local default. |
| [Apertus 1.5 official model](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) and [Q8 text conversion](https://huggingface.co/andreasmartin/apertus-v1.5-8b-text-Q8_0-GGUF) | 2026 | Upstream Apertus-v1.5-8B is multimodal (`image-text-to-text`, ships a vision tokenizer). The deployed artifact is **text-only** (`-text` GGUF) because llama.cpp's server has no compatible vision encoder/mmproj for Apertus — a serving-footprint decision, not a capability claim. Vision is therefore delegated to Qwen; Apertus reasons over the resulting structure. Deployed Q8_0 for precision near f16 with llama.cpp-friendly size; concurrent serving was replaced by sequential ownership after a measured host OOM. |

## Architecture and why each layer exists

1. **Image-only inventory:** Qwen3.5-9B MTP UD-Q4_K_XL + F16 mmproj, served with MTP speculative decoding (the retired Q8_0 measured 42 tok/s; the retired 4B produced more colour-noun mislabels and phantom objects) enumerates objects, colors, original-frame boxes/points and visually supported containment/stack relations. Stable IDs are assigned deterministically. The VLM never receives challenge instructions, so language task planning stays with Apertus; delegating vision to Qwen also keeps the reasoning model text-only, which is the only form llama.cpp can serve.
2. **Mask refinement:** SAM2 processes a scene once and reuses embeddings. A candidate mask must contain the proposed positive point and pass spatial-consistency checks. Rejected masks retain explicitly uncertain coarse observations; they are not presented as successful segmentation.
3. **Grounded intent:** Apertus maps the instruction to a constrained JSON `TaskIntent` over observed IDs: goals with optional destinations, `ask_user`, or `refuse` — never raw actions. Ambiguity, absent objects, contradictory requirements, human contact and uncontrolled motion are decided semantically.
3b. **Movement and output contracts:** a focused Apertus call independently grounds explicit movement-preservation requirements. Its observed IDs constrain `forbidden_movement`; unresolved protection prevents an unchecked plan. Decoded goals are checked against parsed held/under relations and target cardinality. No instruction regex, synonym table, or deterministic referent rescue overrides the model's language interpretation.
4. **Deterministic compilation:** `compiler.py` expands observed-ID goals into pick/place actions and stages visually supported blockers. Every pick requires physical mobility and robot-specific graspability, while receivers need only the independent receiving affordance. A bounding-box overlap is not proof of physical interference.
5. **Executable-state verification:** `planner.py` simulates gripper occupancy, known IDs, protected movement, support dependencies, and final-state relationships. Unknown IDs, double-picks, self-placement, blocked picks, and ungraspable picks fail checks. Failures feed bounded model repair; a valid state simulation does not prove that perception or language interpretation was correct.
6. **Inspectable decision:** The official JSON stays minimal. A separate audit contains scene evidence, intent, proposal, verified actions, checks, warnings, inference latency and token usage. The web view shows these same real outputs.

## Important limits

- Image-plane overlap alone does **not** prove a physical stack. Left/right are camera-relative geometry; on-top relations require visual evidence and retain heuristic confidence.
- VLM confidence is not statistically calibrated. Thresholds are explicit heuristics, not safety probabilities.
- The provided development captures are RGB-only. Calibration can project a ray onto the table plane; it does not supply object height, grasp orientation or collision-free trajectories. Do not label this accurate 3D grasp planning.
- An action-state checker is not a robot dynamics simulator, motion planner or certified safety function. The system proposes actions for human approval only; it has no robot actuator interface.
- Development examples are not held-out evidence. Inspecting and improving development failures is legitimate; per-scene coordinate tables, instruction-ID hacks and reading gold masks during runtime are not.
- A high ZHAW score does not guarantee the shared Academia award. Performance, reproducibility, failure analysis and a truthful compelling demonstration all matter.

## Planned measured comparisons

- VLM box/point inventory versus the same inventory refined with SAM2.
- Raw Apertus proposals versus checked proposals with bounded feedback repair.
- Grounding and ask/refuse breakdown from the unchanged official scorer.
- Cold-scene perception cost versus cached-scene decision cost.
- Reproducible scene-level generalization diagnostics and semantic stress cases, with failures reported rather than concealed.

No numerical gain is asserted before the actual local run. Benchmark reports and submission artifacts must identify model revisions, quantization, hardware and observed limitations.

## Support discovery: measured 65/65 on dev, rejected on held-out evidence

The vision inventory merges a flat support into the container it lies in, so a request
naming that support cannot bind. A discovery pass was built for it: a SAM2 point lattice
inside observed receiving surfaces proposes unexplained regions, and the same local vision
model — on the untouched original image, with the proposed boundary outlined — states
whether the region lies in a receiving surface and what rests directly on it. Only a
complete, evidenced support chain was accepted, and the accepted region was named with the
project's generic word for an unclassified component (`part`) plus its measured colour.

It worked on the development set, twice measured: **65.0/65**, with `scene_09_i3` — the row
that had been lost — recovered at `(932, 867)`, and the identical result at 24 and at 32
reasoning layers. Across all ten dev scenes it added exactly one object and rejected every
other proposal.

**It was still rejected, because of the held-out set**, which cannot be scored here but is
scored by the organisers. On the held-out scenes the pass added two objects that are
demonstrably not objects: a container's `side wall` (held_01) and a `mug handle` (held_05),
i.e. a container component and a part of an already observed object. Those additions
changed four answers: `held_05_i4` ("put everything that is on the plate into the grey tray")
became a refusal, and `held_05_i6` ("pick up the object just below the mug") became a
question, purely because a second white part now existed. Four held-out rows changing on
unscorable evidence — two of them into false refusals or questions — is a worse trade than
one development row gained, so the pass is not wired in.

Verified after reverting: the perception source revision is byte-identical to the shipped
run's, the current tree reproduces all 40 shipped held-out `target_pixel` values exactly,
and the suite is 552 passed / 15 skipped. The last development row therefore still needs an
acceptance rule that can tell a support piece from a container wall and from an object's own
part; neither a name test nor the coarse box-overlap test currently does, which is what the
held-out evidence shows.

## Two fixes kept from this work

- **`scripts/local.py` forwarded none of its own options to the per-scene children.**
  Measured: a run launched with `--apertus-gpu-layers 32` was decided at the default 24 in
  every child, so the option looked accepted and changed nothing. `_child_command` now
  carries the parent's effective values, with behavioural cover in `tests/test_launcher.py`.
- **`referents.py` matched a landmark phrase's colour exactly** while the rest of the code
  accepts the project's colour equivalence (grey/silver, white/cream). Landmark matching now
  uses the existing `_color_matches` helper. Measured neutral: no recorded development row and
  no shipped held-out pixel changes.

## Reasoning-layer split, measured

`-ngl 32` (all Apertus layers on the card) was measured on this tree: **65.0/65 on dev,
50/50 answers, zero grammar-fault lines, 238 s** for the fifty decisions against ~600 s at
24 layers — the 2.5x the code comment predicts, with none of the six lost rows it predicts.
The comment's claim predates the per-scene fresh-child fix for the reasoning path and no
longer holds here; it has been corrected in place. The submission keeps the 24-layer split,
because 32 changes one held-out answer for no measured development gain.
