<!-- Slide 1 -->

# The first pick was right.
The task was not done.

## Our approach in one sentence

Apertus interprets the request; image-only vision supplies observations; explicit rules compile and check an ordered pick/place proposal.

## The problem we found

A full-credit first pick can hide a missing placement. We audit the complete task, not just the pixel that starts it.

## The whole local pipeline on one GPU

One RTX 4070 Ti · 12 GB VRAM | Qwen3.5-9B + Apertus 1.5 8B Q8_0
Sequential model loading; no robot execution.

---

<!-- Slide 2 -->

# Apertus interprets. Code can still overrule it.

## Qwen + YOLOE + SAM2

RGB only
Objects, masks, relations
No task instruction

## Apertus: semantics

Instruction only
Action, references, exclusions
Non-thinking 8B Q8_0

## Apertus: grounding

Instruction + scene graph
Observed-ID goals
Or ask / refuse

## Python rules + verifier

Correct selected language errors
Order blockers and staging
Replay symbolic states

## Official JSON + audit

First-pick pixel + actions
Question or refusal
No motor commands

Downstream rules include category, pickup and spatial interpretation. This is a hybrid system, not model-only reasoning.

---

<!-- Slide 3 -->

# Maximum public score. Not full-task success.

| Component | Points | Correct / total |
|---|---:|---:|
| Grounding | 35.0 / 35 | 40 / 40 |
| Safety response | 30.0 / 30 | 10 / 10 |
| Automatic total | 65.0 / 65 | 50 answers |

## What the script measures

Correct first-pick pixel and plan / ask / refuse. It does not score completion of every later action.

## What the full plans reveal

One placement request stops after the pick: vision missed the destination tray. A correct first pixel does not make that task complete.

## Beyond the development set

The saved held-out file has 40 answers.
Its labels are unpublished;
we report no held-out accuracy.

No robot execution or certified safety is claimed.

---

<!-- Slide 4 -->

# Where Apertus had difficulty

## Smaller questions helped

Local full parser: three invalid replies. A relation-only call recovered in/on from 1/2 to 2/2. Focused calls passed 3/3 simple controls, with quantity supplied by the harness. [E1, E2]

## Apertus behind the camera

Remote Apertus vision + YOLOE/SAM2: 37.625/65; saved Qwen: 65/65. First picks: 16/40; non-plans: 9/10; false asks/refusals: 5/40. Local Apertus language reasoning. [E10]

## A fix that stayed with its examples

Remote clarification: 9/10 to 10/10 on matched controls. On fresh wording, both prompts scored 7/10. The transfer test found no gain. [E3]

## A substitute instead of a refusal

Remote 70B chose blue panels for a missing blue cube. Local 8B refused. We observed the substitution; its internal cause is unknown. [E4]

Remote CSCS probes are development diagnostics, non-compliant as submission measurements under the published local-only rule.

---

<!-- Slide 5 -->

# What we would keep. What we would test next.

## Keep Qwen on vision

The drop-in vision test favoured Qwen in our pipeline. Keep Apertus on language. A mask-interior point still needs to belong to the object the user requested.

## Check the original request

Keep targets, quantities and destinations independently of the proposed plan. Otherwise a verifier can approve a task the user never asked for.

## Test beyond the repaired example

Use fresh paraphrases, negative controls and missing destinations. A prompt change that fixes its motivating cases has not yet earned a place in the pipeline.

## Rerun the shipped configuration

python run.py --data <scenes> --out results.json
Regenerate answers from the exact shipped tree. Keep symbolic checks separate from physical grasp and motion validation.
