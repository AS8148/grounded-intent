# The first pick is not the whole task

## A perfect score, an unfinished task

We got a perfect development score with an unfinished task. Asked to put a yellow gear in a black tray, the pipeline picked up the gear and stopped. Vision had left the tray out of the scene graph. The automatic scorer checked the first pick and awarded full credit. That result gave us a better question to ask of every plan: after choosing the right object, does it actually finish the job? Our system separates image observation, language interpretation and symbolic execution so we can inspect where the request gets lost. [E5, E6]

## Who sees, who interprets, who checks

Qwen3.5-9B inventories the RGB photo without seeing the instruction. YOLOE-26s re-grounds observations and SAM2 refines masks; the pick pixel comes from a mask interior. Close-ups supply structure when a label such as 'part' is insufficient. Apertus 1.5 8B Q8_0 first extracts instruction semantics, then binds them to observed IDs or returns a question or refusal. Schemas prevent invented IDs, not mistaken selections. We use non-thinking decoding at temperature 0. [E6]

We also tried Apertus behind the camera. Remote Apertus 1.5 8B observations, with YOLOE/SAM2 and local Apertus reasoning, scored 37.625/65 against the saved Qwen 65/65: 16/40 first picks, 9/10 non-plans correct and 5/40 false asks/refusals. This favoured Qwen in our pipeline. The API diagnostic is non-compliant as a local submission measurement. [E10]

## Deterministic action compilation and state verification

Apertus proposes goals such as Goal(target='red_gear', destination='grey_tray'). The Python compiler orders blocker removal, stages obstructions on available trays and sequences picks and places for one gripper. A state verifier replays the actions: the gripper must be empty before a pick, holding before a place, and the support graph must stay acyclic. Plates and saucers cannot be picked; trays are destinations. Code also corrects selected language errors, including invented pickup destinations and extra objects in category requests. That makes this a hybrid system. The rules do some interpretation too, and can get it wrong. The output is a proposal for human review, not motor commands. [E5, E6]

## Measured local development result

On the ten public scenes, the official scorer reports grounding 35.0/35 (40/40 first-pick pixels) and safety-response scoring 30.0/30 (10/10 non-plan decisions, 0 false asks/refusals on plan rows). All 50 instructions have answers; the automatic total is 65.0/65. All six categories receive full credit. The missing placement above still passes: this score measures first-pick grounding and response kind, not completion of every later action. The held-out file contains 40 answers from an earlier saved run; without its hidden labels, we report no accuracy. [E5, E7]

## One 12 GB GPU, end to end

We ran the complete local pipeline on one RTX 4070 Ti with 12 GB VRAM. Perception and language models take turns on the GPU, using pinned local weights and llama.cpp. Per-scene child processes isolate observed server grammar faults. Run python run.py --data <scenes> --out results.json. Code is Apache-2.0, data CC BY 4.0. Model revisions and pinned weights are in src/grounded_intent/assets.py. [E5, E8]

---

# What we learned by asking Apertus

## The schema was precise. The answer was wrong.

We tried to make interpretation explicit enough to check: TaskProgram asked Apertus for operations, references, quantities, attributes and source spans. Even 'Pick up the green cone.' exposed trouble. Three local attempts returned complete, strict-schema JSON, but the quantity fields contradicted one another and absent values appeared as literal 'null' strings. A separate quantity-only probe granted arbitrary-choice permission on all five controls, including requests that granted none. The model had filled the form. It had not respected the distinctions the form was designed to express. Retrying the full parser repeated the invalid representation. [E1]

## Smaller questions and the limits of decomposition

In a local matched pair, changing only 'in' to 'on' produced on for both: 1/2 correct. Self-review left the error unchanged; asking for the relation alone recovered 2/2. Focused calls composed valid programs on three simple instructions, with quantity supplied by the harness. Later remote variants gave the final state sole ownership of the action and corrected abstention contracts. Interpretation passes rose from 1 to 3 to 6 to 9 of 11, but only 6/11 final decision profiles passed. We made progress without clearing the acceptance gate. Source quotes proved where words came from, not that Apertus understood them. [E1, E2, E9]

## The prompt repair that did not travel

With the rest held fixed, 'Choose any three compatible clips' granted choice, while 'Select any three compatible clips' did not. We clarified the difference between permission and enumeration. In remote CSCS diagnostics, the matched set improved from 9/10 to 10/10. On ten newly worded cases, both prompts scored 7/10. The fix stayed with its examples. These were component diagnostics, non-compliant as local submission measurements under the published local-only rule. [E3]

## 70B found a substitute for an object that was absent

In a remote CSCS smoke, Apertus 70B answered 'pick up the blue cube' with a plan to pick two blue panels. Its explanation called them 'rectangular panels' and the 'closest semantic match'. It described the mismatch and selected them anyway. Local 8B refused. This was not a controlled model-size comparison: serving location and model artifact differed. Earlier matched permission controls favoured 70B, 6/6 versus 5/6; both fell to 5/6 on fresh cases. We observed a substitution, not its internal cause or a general safety disadvantage of larger models. [E4]

## The next check belongs at the task boundary

A verifier can approve a consistent sequence for an incomplete interpretation. Keep the requested targets, quantities and destinations independently, then check the plan against them. Fresh paraphrases and missing-receiver cases belong in that check; regenerate answers from the exact shipped configuration. Physical deployment is separate. These captures are RGB-only: robot_check='ok' means there was no depth data to check. Neither a mask-interior pixel nor a symbolic plan establishes a successful grasp or collision-free motion. No robot execution or certified safety is claimed. [E5-E8]
