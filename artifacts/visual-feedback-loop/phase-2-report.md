# Phase 2 — Direct HTTP visual loop

**BLOCKED — one real-model attempt, zero completed image-guided corrections.** The evaluation driver is implemented, but the attempt stopped before the first assistant/tool response. This is not a model-quality failure or visual-loop qualification. No generation was retried, and no history was discarded.

## Separate outcomes

| Criterion | Outcome | Evidence |
| --- | --- | --- |
| A. Screenshot transport and interpretation | **Not established / BLOCKED** | No screenshot reached inference; HTTP 200 was followed by incomplete SSE, without `message_stop`. |
| B. Closed-loop execution | **Not established / BLOCKED; 0 iterations** | No completed assistant response, tool call, model artifact revision, or correction/render pair. |
| C. Artifact quality | **Not evaluated** | No model-created artifact exists. The retained renderer-control screenshot is not model output. |
| Real-model paired image continuation | **Not evaluated** | The only request was text-only initialization. Historical image reuse and encode counts cannot be inferred from this failure. |

Durable, unmodified attempt receipt: [run/receipt.json](phase-2-evidence/run/receipt.json). Supporting trace: [run/failure-traces.json](phase-2-evidence/run/failure-traces.json); SSE capture: [run/sse-00.json](phase-2-evidence/run/sse-00.json); server/client error: [run.log](phase-2-evidence/run.log).

## Starting state and identities

- Branch `Visual-Feedback-Loop`, clean initial HEAD `2fdea4fe7367e540300d7c81d4fed402d716a274`. No concurrent Python/model-serving process was observed initially. LM Studio's existing listener on 1234 and other operator services were left alone. Port **18079** was free and bound only on loopback.
- Apple M3 Ultra, 549,755,813,888 bytes RAM; macOS 26.5.2 arm64. Initial memory inspection showed roughly 404 GB free plus reclaimable file-backed memory; no competing model process was observed. Later checkpoint hashing increased inactive file-backed pages, not a second model allocation.
- Reused only the **disposable** `/tmp/ds41f-phase1-1017-tests` environment provisioned by the documented Phase 1 workflow. Rebuilt its Runtime wheel using `tools/build_execution_package.py --checkpoint-variant flash-1017 --glm-profile native`, then replaced only that disposable installation's Runtime package. Neither persistent operator installation was rebuilt, reinstalled, stopped, or modified.
- Runtime **0.23.0**, package provenance **2fdea4f**, numerical selection `flash-1017-pre-woa-fp8-v1`, GLM `native`. The three installed request-preparation/backend source files were byte-compared against this checkout before loading. The receipt retains their hashes, including the Phase 1 corrected preparation and enqueue implementations.
- Checkpoint: `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, original Flash-1017, source revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, 48 official shards. Full-byte checkpoint admission succeeded in isolated Python before the attempt. No original-Flash substitution or checkpoint edits.
- Python **3.13.15**, Recipe **0.2.0**, oMLX **0.7.0**, MLX **0.32.2**, MLX-LM **0.31.4.dev132+g94cdcae13**, httpx **0.28.1**, uvicorn **0.54.0**. Exact locked dependencies, native origins, checkpoint inventory and compatibility are retained once in [run/identity.json](phase-2-evidence/run/identity.json). Recipe uses the retained repaired wheel/source/patch authority, not an upstream substitution.
- Renderer: installed Chrome **155.0.8059.85**, Playwright **1.64.1**, separate renderer Python 3.13.15 with Pillow **13.0.0**. [Renderer dependencies](phase-2-evidence/renderer-dependencies.txt), font SHA and browser version in [renderer-smoke.json](phase-2-evidence/renderer-smoke.json). Runtime's independently pinned Pillow remains 12.3.0.
- Wheel/source hashes: [source-identities.json](phase-2-evidence/source-identities.json). Admission summary: [initial-admission-summary.json](phase-2-evidence/initial-admission-summary.json). [Pre-run process/memory inspection](phase-2-evidence/pre-run.txt) also records test-owned renderer setup processes; these were subsequently cleaned up.

## Driver and fixed evaluation

`tools/accept_visual_feedback_loop.py` is an external evaluation client. It reuses the production backend/app, isolated admission convention, real socket, health settlement and serving traces from `accept_mtp_vision.py`; native/paired acceptance utilities and Recipe Messages tests were inspected. Preflight calls the **existing** `prepare_request('messages', ..., ordinary=True)` for complete-body Recipe conversion, image geometry/aggregate budgets and expanded-context capacity. No Runtime protocol converter, Vision executor or cache authority was added.

The two client tools are `write_html({html})`, replacing one fixed workspace file, and `render({})`, invoking the external one-shot renderer and returning its actual PNG inline inside the matching native `tool_result`. Tool inputs come only from completed model output. No generated shell commands or arbitrary paths are accepted. Static HTML validation, CSP, disabled page JavaScript, aborted page network requests, DNS denial, a fresh sandboxed browser context and subprocess deadlines constrain rendering. All model files would live in a disposable workspace; evidence revisions/diffs would survive workspace deletion. The final driver additionally explicitly relocates its Runtime KV root into that workspace; the attempted run's default KV directory did not exist, and no persistence tool was used.

The fixed task and tool schemas are retained in [run/task.json](phase-2-evidence/run/task.json): a Field Notes dashboard with navy header, pale slate background, three equal white horizontal cards, 48 px margins, 24 px gaps, visible footer and no overflow. A deliberately mismatched starter is supplied; the model is asked to save/render it, then choose coherent, screenshot-grounded corrections. No discrepancy description, corrective patch or independent score is included in tool feedback.

Before starting, independent checks were fixed for element presence, overflow, header geometry, horizontal equal-width cards, card gaps, card geometry, palette, viewport visibility and footer placement. Quality requires all nine checks and improvement over the first render; the checks are a bounded layout/palette rubric, not a comprehensive typography assessment. The model's declaration is not a score.

Fixed render settings: **1280×720, DPR 1**, local embedded Arial with retained font hash, sRGB, light scheme, en-US, UTC, reduced motion, no external assets. Renderer failures, blank images and invalid artifacts are distinct from successful image observations. The controlled starter render succeeded: [renderer-control.png](phase-2-evidence/renderer-control.png), [source](phase-2-evidence/renderer-control.html). PNG SHA-256 `fc6dd2f551cb196b84eb64d1bd0c91b01633ce881403d4bc4f0a1d602c1176d9`. **This image was never transmitted to the model.**

## Budgets and actual attempt

Budgets saved as RUNNING before admission/generation:

- Total task: **5400 s**; startup/admission bound: **1200 s**.
- Each generation: **600 s**, **2600 output tokens**, temperature 0, Thinking OFF.
- **16** generation requests, **18** client tool effects, maximum **five** correction iterations.
- Each renderer subprocess: **25 s**; canonical settlement: **60 s**; HTML source: **65,536 bytes**.
- Complete serialized JSON: **16 MiB**, with existing Runtime image/pixel/patch/context admission applied to every preflight.

Only **one POST** occurred. Native `/v1/messages`, model alias `deepseek-v4.1-flash`, 3050 serialized bytes, 976 expanded prompt tokens, zero images. The receipt records its complete-body hash. HTTP headers reported 200, but the stream yielded **no native events** and closed incompletely. In particular there was no `message_stop`, no completed argument object, and **no tool effect was authorized**. Total attempt elapsed **115.63 s** including setup and cleanup.

The retained trace records `generated=0`, `cached_tokens=0`, `remaining_suffix_tokens=976`, `cache_published=false`, `cancelled=false`. Canonical frontier agreement, replay/repack counters and image-encoding metrics are absent at this early failure; **their absence is not evidence of correctness or zero replay**.

## Diagnosis and smallest local correction

The server raised **`numerical execution requires installed MTP admission`**. The new driver initially copied the older helper's subprocess-only admission pattern. `execution_identity.inspect_payload()` binds an immutable `_admitted_variant` **in the process calling it**; the isolated child could admit the checkpoint/dependencies, but could not bind the parent serving process. Setting `backend.dependency_identity` alone does not initialize that binding. The numerical fence correctly refused execution. This is an **evaluation-driver bootstrap defect**, not a reproduced Phase 1 image classification defect or a reason to weaken Runtime admission.

The committed driver now calls the existing `inspect_payload()` in its execution process, compares its variant/source to the isolated admission, and verifies `numerical_profile()` **before constructing the backend**. No private binding override, Runtime patch, cache workaround or protocol change was made. [Controlled diagnosis](phase-2-evidence/bootstrap-diagnosis.json) reproduces the rejection in fresh isolated Python, then demonstrates the existing admission call binding Flash-1017 successfully, without loading weights.

**No second real-model attempt was made.** The correction is controlled-test verified only. [driver-followup.diff](phase-2-evidence/driver-followup.diff) and hashes distinguish the executed driver from the committed follow-up, which also strengthens deadlines, model-error classification, renderer cleanup and explicit disposable KV configuration. The retained failed receipt was not rewritten to imply this follow-up ran.

Renderer setup initially hung on style injection with page JavaScript disabled. Moving fixed font CSS into the initial HTML resolved that controlled smoke test. Early forced control timeouts left test-owned detached Chrome processes; those exact processes were terminated, and timeout cleanup was strengthened to handle descendants plus graceful renderer termination. These were renderer-only setup probes, not additional model tasks.

## Focused verification and paired-cache limits

Final controlled verification:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/ds41f-phase1-1017-tests/bin/python -m pytest \
  tests/test_visual_feedback_loop.py \
  tests/test_recipe02_messages_serving.py::test_converted_image_frontier_at_real_paired_admission -q
```

**25 passed**: 17 client/bootstrap safety checks and eight existing focused Recipe/paired-admission cases. [Log](phase-2-evidence/final-tests.log). Checks cover missing/duplicate/mismatched opaque IDs, malformed arguments, forbidden active artifact output, oversized body/image pixels, incomplete/cancelled SSE (including tool blocks without `message_stop`), renderer failure/timeout, blank screenshot and process-local admission. The eight paired cases exercise fresh versus historical images with retained, missing, cleared/evicted and wire-mutated historical state; unavailable/incompatible historical pairs reject before encoding. These controlled checks are **not real-model paired-cache acceptance**, and the broad Phase 0/1 suites were not repeated.

The driver retains original native history and IDs, preflights all occurrences, records raw/transmitted PNG hashes and original-PNG variant, and checks existing trace encode counts against uncached spans, historical span coverage, canonical Target/DSpark offsets, cache publication, replay and repack. **None of those multi-image execution assertions was reached in this attempt.** Phase 1's first real-model continuation opportunity therefore remains open.

## Cleanup and stopping point

The owned server thread stopped, backend closed, driver PID 31172 exited and its disposable artifact workspace was removed. The previously default KV directory was absent. No listener remains on 18079 and no test-owned browser/model process remains; [final process check](phase-2-evidence/final-processes.txt). No credentials or base64 request bodies are retained. No changes to persistent operator installations, checkpoint files, Recipe, Runtime cache authority or model execution architecture.

Remaining gate: separately review the corrected driver bootstrap and authorize a new bounded real-model evaluation. Transport, two meaningful screenshot-guided corrections, independent final quality and real-model historical paired-state behavior are all still unqualified. No official Harness integration, Files API, Windows work, Phase 3, push or additional model attempt was started.
