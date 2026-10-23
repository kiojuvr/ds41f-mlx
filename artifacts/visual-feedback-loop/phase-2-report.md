# Phase 2 — Direct HTTP visual loop

**PASS, bounded native-Messages static Web loop.** Continuing from `f0f29be87b189c477d3777041cdcdf8b66e16446`, the real original Flash-1017 model completed **five image-guided edit/render iterations** in one conversation. No task goal, rubric, threshold or generation budget was changed. Earlier failures remain retained, not replaced by favorable results.

## Separate outcomes

| Criterion | Outcome | Independent evidence |
| --- | --- | --- |
| A. Actual screenshot transport and interpretation | **PASS** | Six rendered PNGs were inline native tool results and encoded once each. Canonical responses identify visible colors, stacking, clipping and card spacing. Every request body and native assistant response reconstructs exactly. |
| B. Closed-loop execution | **PASS — five completed iterations** | Six model-created artifact revisions, six screenshots, all 13 native `message_stop` boundaries, 12 original tool IDs and paired canonical settlements retained. |
| C. Artifact quality | **PASS under the unchanged bounded rubric: 1/9 → 9/9** | Independent layout/palette/visibility checks and final screenshot review. **Not literal full-goal compliance:** subtitle remains light slate rather than white; later polish is not monotonically better. |
| Real-model paired continuation | **PASS for this observed conversation** | Six fresh spans encoded; 30 historical image occurrences reused without encoding. Exact prefix restoration, Target/DSpark agreement and cache publication on every request; replay/repack zero. |

Authoritative reviewed summary: [receipt](phase-2-evidence/receipt.json). Original automatic run receipt is [model-run/receipt.json](phase-2-evidence/continuation-1/model-run/receipt.json); its transport field intentionally remains `REVIEW_OBSERVATIONS`. Independent external review is [review.json](phase-2-evidence/continuation-1/review.json), not the evaluated model's completion declaration.

## Starting state and execution identities

- Branch `Visual-Feedback-Loop`; clean initial HEAD **f0f29be**. No competing Python/model/browser process observed; loopback **18079** free. Apple M3 Ultra, 549,755,813,888 bytes RAM, macOS 26.5.2 arm64; about 324 GB free plus reclaimable memory. Existing operator listeners were not modified. [Preflight](phase-2-evidence/continuation-1/preflight.txt).
- Reused only disposable `/tmp/ds41f-phase1-1017-tests`, provisioned by `mtp_setup` per `docs/verification.md`. **No rebuild/reinstall was needed**: installed Runtime **0.23.0**, provenance **`2fdea4fe7367e540300d7c81d4fed402d716a274`**, still exactly matches the checkout's corrected request preparation/backend files. f0f29be changes the external driver/evidence, not those Runtime files. [Source/package check](phase-2-evidence/continuation-1/package-check.json).
- Original **DeepSeek-V4.1-Flash-1017**, checkpoint `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, 48 official shards; numerical profile **`flash-1017-pre-woa-fp8-v1`**, GLM `native`, MTP/DSpark ON. Isolated checkpoint admission used the existing full-byte SHA authority with its unchanged-file identity cache, not an original-Flash substitution or a new checkpoint enrollment.
- Python **3.13.15**, Recipe **0.2.0**, repaired Recipe wheel SHA `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`, oMLX **0.7.0**, MLX **0.32.2**, MLX-LM **0.31.4.dev132+g94cdcae13**. Full native origins, dependency/checkpoint identities and admission: [identity.json](phase-2-evidence/continuation-1/model-run/identity.json).
- Execution-process admission is recorded **before backend construction**. Runtime source hashes include the Phase 1 wire-origin correction. External client hashes actually executed: [executed-client-hashes.json](phase-2-evidence/continuation-1/executed-client-hashes.json).
- Separate disposable renderer environment: Chrome **155.0.8059.85**, Playwright **1.64.1**, Python **3.13.15**, Pillow **13.0.0** (Runtime retains its pinned Pillow 12.3.0). [Renderer dependencies, executable/framework/font hashes](phase-2-evidence/continuation-1/renderer-identity.json). All six renders record the same browser version and Arial hash.

Neither persistent operator Runtime installation was rebuilt, reinstalled, stopped or modified. No checkpoint files were edited.

## Bootstrap correction and retained failures

| Stage | Actual outcome | Retained evidence |
| --- | --- | --- |
| Original attempt, before f0f29be | One native POST, zero generated tokens, no settled assistant/tool effect. Child-only admission did not bind the serving process. | [Original immutable receipt](phase-2-evidence/run/receipt.json), [error](phase-2-evidence/run.log); original report remains in Git at f0f29be. |
| Continuation startup using unchanged f0f29be | **Zero HTTP generation requests and no model loading.** In-process admission rejected `transformers` after early backend imports changed import-spec search entries from strings to `Path` objects. | [Startup receipt](phase-2-evidence/continuation-1/run/receipt.json), [error](phase-2-evidence/continuation-1/run.log), [controlled import-order diagnosis](phase-2-evidence/continuation-1/import-order-diagnosis.json). |
| Corrected continuation | One actual conversation, **13 successful generation requests**, no retry, five corrections, clean completion. | [Run receipt](phase-2-evidence/continuation-1/model-run/receipt.json). |

The smallest correction is **driver-only**: move the backend/server imports after actual in-process admission, matching the supported order in `ds41f_mlx/serve.py`. No Runtime origin check was weakened, private numerical binding assigned, protocol converted, or cache authority changed. The regression now checks both actual admission followed by backend import and the real `run()` import ordering. The previous helper-only check missed that ordering issue.

This startup correction happened before any generation request. After it, only one actual model conversation was started; no uncertain generation or tool effect was resubmitted. The user explicitly authorized this continuation from f0f29be.

## Fixed task, tools, rendering and budgets

The [exact task/tool schemas](phase-2-evidence/continuation-1/model-run/task.json) equal the earlier task byte-for-byte as structured data; the offline audit asserts this. Goal: a Field Notes dashboard with a pale slate background, navy header, three equal white cards in one horizontal row, 48 px margins, 24 px gaps, readable content/footer and no overflow. The model saves the supplied mismatched starter, renders it, then independently chooses changes from screenshots. There are no injected discrepancy descriptions, corrective patches, hidden scores, or extra follow-up prompts.

Only `write_html({html})` and `render({})` execute. Actual complete model arguments create the HTML; every retained revision equals its canonical tool argument. The renderer receives one fixed workspace file, not a shell command. It uses static HTML/CSS validation, CSP, disabled page JavaScript, aborted page network requests, DNS denial, no external assets, sandboxed headless Chromium, embedded local Arial, **1280×720/DPR 1**, fixed sRGB/en-US/UTC/light/reduced-motion settings and a subprocess deadline. Failed, invalid or blank renders cannot become successful image observations. Artifact/KV paths are disposable; model revisions/diffs and screenshots survive in evidence.

Budgets saved before execution, unchanged from f0f29be:

- **5400 s** total; **1200 s** admission/startup bound; **600 s / 2600 output tokens** per generation; temperature 0, Thinking OFF.
- Maximum **16 requests**, **18 client effects**, **five correction iterations**, **65,536 source bytes**.
- **25 s** per render subprocess, **60 s** canonical settlement, **16 MiB** entire JSON body.
- Existing Recipe/Vision source, pixel, patch, expanded-context and output-reservation admission before every POST; no second converter or Vision implementation.

Actual run: **324.77 s**, **13 requests**, **12 effects**, **4642 output tokens total**. Maximum request body **278,814 bytes** and expanded prompt **9848 tokens**. At six images: 5,529,600 source pixels and 28,704 grid patches; each image grid `[52,92]`, span length **578** (below 1014). All fresh image-last prompts retain **six terminal text tokens**. This qualifies only the observed prompt sizes, **not** the Runtime's entire implemented 1,048,576-token multimodal envelope.

## Complete revision/render trajectory

All source files, diffs, screenshots, complete hashes and original IDs are retained under [model-run/](phase-2-evidence/continuation-1/model-run/). Hashes below are prefixes; full SHA-256 values are in its receipt. Raw screenshot bytes equal transmitted PNG bytes and the Runtime-admitted source digest; variant is original PNG with **no client normalization**.

| Iteration | Previously observed | Model revision / SHA prefix | Subsequent screenshot / SHA prefix | Independent review |
| --- | --- | --- | --- | --- |
| Initial creation | None | `artifact-01.html` / `eb299db3f019` | `render-00.png` / `fc6dd2f551cb` | Actual model tool saves starter; orange header, stacked yellow cards and clipped footer. Score 1/9. |
| 1 | render-00 | `artifact-02.html` / `3431705607ca` | `render-01.png` / `0c03c9f7a75a` | Corrects visible layout, palette, margins and overflow. Score 9/9. |
| 2 | render-01 | `artifact-03.html` / `6e7218b9891e` | `render-02.png` / `407264f3f63e` | Changes heading/value typography, padding and footer spacing. Claimed caption-gap correction remains incomplete (`auto` margin retained). |
| 3 | render-02 | `artifact-04.html` / `06466d3d6f0b` | `render-03.png` / `a694c0cd7b20` | Actually groups caption closer to value; substantial visible card-composition change. |
| 4 | render-03 | `artifact-05.html` / `4e84eb80bdbd` | `render-04.png` / `142fda327438` | Reverses centered content to top alignment with bottom divider: preference oscillation, not demonstrated quality improvement. |
| 5 | render-04 | `artifact-06.html` / `5794b3bfe9b7` | `render-05.png` / `12efe39fe582` | Small divider/padding refinement; model then ends naturally. Final score 9/9. |

[Assistant 02](phase-2-evidence/continuation-1/model-run/assistant-02.json) independently identifies the orange header, yellow vertical cards, viewport overflow and off-screen footer. [Assistant 04](phase-2-evidence/continuation-1/model-run/assistant-04.json) identifies disconnected card captions and tight footer spacing; [assistant 06](phase-2-evidence/continuation-1/model-run/assistant-06.json) follows the still-visible caption gap with the actual composition correction. Iterations 1 and 3 are particularly clear substantive examples, independently exceeding the minimum two; the weaker/oscillatory cycles were not deleted or presented as monotonic improvements. All five edit/render cycles use prior delivered screenshots, not driver-generated patches or forced no-op changes.

## Independent quality and interpretation

The **unchanged nine-check rubric** covers element presence, overflow, header geometry, equal horizontal cards, card gaps, card geometry, palette, viewport visibility and footer placement. Scores across **every** screenshot: **`[1, 9, 9, 9, 9, 9]`**. Final artifact, not a selected intermediate, passes all nine and improves over baseline. No model declaration is used in scoring.

Final independent metrics: header `(48,48,1184,120)`; cards start at y=200, height 220, widths about 378.67, gaps 24; footer box y=456..472; no document overflow. Screenshot pixel samples confirm slate `[241,245,249]`, navy `[15,23,42]` and white `[255,255,255]`. Adjacent screenshots differ in 747,783 / 13,177 / 12,735 / 14,076 / 3021 pixels. These pixel facts confirm actual changed renders, **not** an aesthetic quality metric.

Limitations are material: subtitle is **`#cbd5e1`, not literal white**; the fixed rubric does not inspect that foreground. Most objective gains occur in iteration 1. The caption correction takes two attempts, and later layout preferences oscillate. The model overstates complete specification compliance. This is bounded quality improvement, not perfect goal satisfaction or general design quality. No withheld-image/text-only ablation establishes exclusive causal dependence on Vision; acceptance rests on actual image encoding, accurate visual observations and their completed edits/renders.

## Image history, paired cache and settlement evidence

[Offline audit](phase-2-evidence/continuation-1/audit.py) performs **no generation, model loading, HTTP or client-tool effects**. It rebuilds complete request bodies in memory from original goal, canonical assistant blocks, exact tool results/IDs and raw PNGs, without storing base64 bodies. All 13 reconstructed hashes/sizes match. Every reconstructed client assistant equals the Runtime's complete canonical native response, and the actual SSE events end in `message_stop`. Results: [audit-results.json](phase-2-evidence/continuation-1/audit-results.json), [audit log](phase-2-evidence/continuation-1/audit.log).

| Request | Images (historical + fresh) | Restored cached frontier | Final canonical frontier | Newly encoded spans |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 1383 | 0 |
| 1 | 0 | 1383 | 1456 | 0 |
| 2 | 0 + 1 | 1456 | 2872 | 1 |
| 3 | 1 + 0 | 2872 | 2952 | 0 |
| 4 | 1 + 1 | 2952 | 4419 | 1 |
| 5 | 2 + 0 | 4419 | 4499 | 0 |
| 6 | 2 + 1 | 4499 | 5974 | 1 |
| 7 | 3 + 0 | 5974 | 6051 | 0 |
| 8 | 3 + 1 | 6051 | 7524 | 1 |
| 9 | 4 + 0 | 7524 | 7599 | 0 |
| 10 | 4 + 1 | 7599 | 9076 | 1 |
| 11 | 5 + 0 | 9076 | 9158 | 0 |
| 12 | 5 + 1 | 9158 | 10093 | 1 |

For **all 13 requests**: cached frontier equals the preceding complete canonical frontier; all 40 Target offsets and three DSpark offsets agree with the new canonical frontier. Authority disposition is `settled`, consumed/emitted/quiescence frontiers agree, pending predictions/queues retire, cache publishes and cancellation is false. Existing `prompt_replay`, `full_cache_repack` and quiescence history-replay/repack counters are **zero**. Historical image spans stay covered by restored paired state and are never re-encoded. New screenshot spans alone are encoded. No speculative historical reconstruction or hidden full-prompt replay was used.

Wire source occurrence ordering is invariant: `messages.4/8/12/16/20/24.content.0.tool_result`; exact image digests/grids/spans persist across every continuation. Original tool IDs match model calls, native call certificates, executed effects and reconstructed results. There is no Chat Completions projection, image deletion/replacement or normalization drift.

For compact retention, complete raw traces/SSE are **losslessly gzip-compressed** with original byte hashes in [retention-manifest.json](phase-2-evidence/continuation-1/retention-manifest.json). Readable `trace-*.json` files omit duplicated `protocol_events`, `response`, `canonical_generated` and `quiescence.recovery_suffix_tokens`; those remain intact in `trace-*.raw.json.gz`. The audit verifies the archive hashes and reads the originals. This is external evidence projection, not Runtime instrumentation or execution/cache architecture change.

## Focused verification, cleanup and limits

**26 passed** in the final focused run: 18 client/bootstrap/error-boundary checks plus eight existing Recipe/paired-admission cases. [Log](phase-2-evidence/continuation-1/final-tests.log). Cases cover IDs, malformed arguments, active/invalid artifacts, oversized body/image pixels, incomplete SSE, renderer failure/timeout, blank screenshots, process-local admission and import ordering. Missing, evicted and incompatible historical pairs reject in the controlled eight-case lane; **no destructive real-model eviction test** was added. Phase 0/1 broad suites were not repeated.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/ds41f-phase1-1017-tests/bin/python -m pytest \
  tests/test_visual_feedback_loop.py \
  tests/test_recipe02_messages_serving.py::test_converted_image_frontier_at_real_paired_admission -q
```

The owned server/backend shut down gracefully, model-run PID **97870** exited, both startup/model workspaces were removed, no test browser remains and **18079 is free**. [Final process/state record](phase-2-evidence/continuation-1/final-state.json). No credentials or full base64 request bodies retained; only the dedicated disposable workspace accepted model writes.

Remaining limitations: one static task, one viewport, Thinking OFF, isolated single conversation, no competing-cache pressure; multiple-result freshness remains controlled-test rather than real-model coverage. No broad context, endurance, 3D, official Harness, Files API, Windows or Phase 3 qualification is implied. **No further model attempt, operator installation change, or push. Stop at this Phase 2 report.**
