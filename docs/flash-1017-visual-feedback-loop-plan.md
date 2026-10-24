# Flash-1017 Visual Feedback Loop — Investigation and Implementation Plan

Status: **Phase 0 bounded PASS; complete visual feedback loop not yet qualified.** This documentation update changes no implementation or installed environment. The original source review remains historical; subsequent transport acceptance is scoped below.

## 1. Decision

Evaluate Vision as an agent capability: **create → render → observe → compare with the goal → edit → render again**, not merely image description.

The current priority is to establish the complete loop using the already demonstrated **inline transport**, in this order: **Inline Vision correctness → Direct HTTP visual loop → Official Harness inline loop → Windows normal-operation compatibility → Quality and endurance.** Phases 1–3 do not depend on implementing Files API. Existing Files-to-inline fallback is acceptable for the initial official Harness MVP within request-size, image-history and cache constraints.

**Observed acceptance:** commit `ca2d4ed` retains the [Phase 0 report](../artifacts/visual-feedback-loop/phase-0-report.md) and [acceptance receipt](../artifacts/visual-feedback-loop/phase-0-evidence/receipt.json). Official macOS headless Harness **0.2.1-rc.2** used its existing `read_image` attachment pipeline: `/v1/files` returned **HTTP 404**, existing inline fallback succeeded, native Messages received the base64 image, and Flash-1017 correctly interpreted the screenshot. The exact Tool Call ID and completed Harness turn were preserved. Completion is supported by Harness events; a direct wire `message_stop` was not retained. This does **not** qualify Windows GUI, ordinary image attachments/UI uploads, image history, Thinking ON or the complete visual feedback loop.

Existing Runtime Vision preparation, Recipe conversion, native Messages semantics and paired MTP remain execution authorities. Do not add a second image-processing engine, Messages converter, cache implementation or rendering service inside Runtime. Files compatibility is a later evidence-driven milestone, not an inline-MVP prerequisite.

## 2. Historical source review and evidence scope

| Repository | Reviewed revision |
| --- | --- |
| ds41f-mlx | `be44f393f6c1df7c2b93307ab587384dc172af0f` |
| official DeepInfer | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| official deepseek-recipe | `935817c66f5d6e64838f419692545f06787e485f` |
| official deepseek-harness | `687f282e8625c6b8d11ff58211a90524be2e4317` |
| packaged Recipe 0.2 upstream base | `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`, recorded in `third_party/mtp/normal-local.json` |

At the original source review, official checkouts were inspected under `/tmp/ds41f-visual-review`. These revisions describe that historical review, not the later Phase 0 installation; the report records its installed identities. Upstream HEAD is not the packaged Recipe/native payload or necessarily the user's installed Harness. Reproduce conversion behavior against the actual packaged Recipe 0.2 implementation before implementing a fix.

Primary official references, pinned to the reviewed revisions:

- [Recipe Messages conversion](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs): `image_block`, `MessagesToolResultContent::into_content`, `transform_messages`.
- [Recipe V4 encoding](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/v4/mod.rs): ordered image-source collection, including Tool messages.
- [DeepInfer multimodal processor](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/ve/multimodal_processor.py): placeholder expansion, image spans, and skipping fully cached images.
- [DeepInfer image processor](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/ve/processor/image_processor.py): image geometry and token/patch reference.
- [Harness read_image](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/fs/tool-fs/src/read-image.ts): route capability gate and image attachment output.
- [Harness serializer](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/serialize.ts): image blocks inside native `tool_result`, exact tool IDs and adjacent-role merging.
- [Harness adapter](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/adapter.ts), [image preparation](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/images.ts), and [Files resolution](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/request-files.ts): Files-first requests and bounded inline fallback.
- [Harness image-offload integration test](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/apps/cli/tests/profiles/acp/tests/image-offload.expected.e2e.ts): expected image results and inline fallback; this is upstream test source, not a ds41f acceptance receipt.

Runtime contracts and limits remain scoped by [capabilities](capabilities.md), [Vision integration](mtp-vision-integration.md) and the [development record](deepinfer-development.md). Their historical Harness image-acceptance statements precede the Phase 0 report above; that report is the authority for the later bounded transport PASS, not broader capability expansion. Neither earlier receipts nor Phase 0 establish the proposed creation loop.

The preliminary draft recorded failed local probes: missing `ConversionOptions.with_model_versions_by_name`, and 111 setup errors in four targeted suites because isolated installed-payload checks could not import `ds41f_mlx`. These are environment failures, not passed regressions or reproduced image defects. They were not rerun in this source-only review. Prepare a disposable, admitted installation following [verification](verification.md); do not bypass admission or modify the everyday installation to obtain a green result.

## 3. Official implementation versus current Runtime

| Boundary | Official behavior | Current ds41f behavior / consequence |
| --- | --- | --- |
| Image-return tool | Harness `read_image` reads through its filesystem service, commits an attachment and returns text plus an image block; the active model route must advertise image input | Existing Harness tool is the preferred MVP path. A screenshot pathname in ordinary tool text alone is not image feedback |
| Native wire encoding | Harness serializes tool images as `user.content[].tool_result.content[]`; image source is base64 or file ID | `serving/messages_admission.py:blocks` recursively permits base64 images and rejects Files/URL sources |
| Transport selection | Adapter tries Files resolution first; `FileResolutionFailure` switches to inline and rechecks image budgets | Phase 0 observed HTTP 404 followed by successful inline Messages delivery; the exception classification is source-correlated, not directly instrumented. No provider fix or explicit inline-only switch was required for that path |
| Protocol ownership | Recipe converts nested image/text results to Tool messages, with image sources retained | Keep native Messages conversion, thinking/signatures, exact IDs and JSON/SSE. No Messages-to-Chat translation or second parser |
| Vision execution | DeepInfer expands image placeholders from processor configuration and skips only completely cached spans | `runtime/multimodal.py`, existing MTP adapter and paired Target/DSpark path are the implementation owners. Verify geometry and source order rather than porting GPU workers |
| Historical image reuse | DeepInfer computes cache coverage using complete image spans | ds41f additionally requires a retained compatible pair for historical images; eviction/edit rejects rather than silently re-encoding history |
| History management | Harness can normalize request images and offload older occurrences, replacing images with text; serializer merges adjacent user/tool turns | Both transformations affect exact prompt/image identity. They must be tested against paired continuation, not assumed compatible |
| Render/iterate | Harness owns tools and workflow; protocol/engine code alone does not define a production task | Use external deterministic rendering and the existing image-return tool. Runtime remains an inference service |

### Source path; Phase 0 qualifies only read_image-to-observation

```text
Harness creates/edits workspace files
  → rendering tool writes screenshot.png
  → existing read_image(file_path) returns text + attachment
  → prepareImages reads deterministic request-image bytes
  → Files resolution, then bounded inline fallback if resolution fails
  → serialize: user.tool_result.content[text, image(base64)]
  → /v1/messages → Recipe → multimodal preparation
  → paired lookup → encode uncached new image → existing MTP
  → completed assistant response → next edit/render/read_image
```

The create/edit and repeated render/observe stages above remain proposed; Phase 0 used one externally rendered screenshot. A combined render-and-return-image tool can follow later if needed. Reuse the official attachment service rather than inventing a path-to-image interpretation inside Runtime. Ordinary UI uploads remain unverified; Windows uploads and tool images require separate probes in Phase 3B.

## 4. Priority gaps

### P0 — New-image classification after conversion

`ds41f_mlx/serving/production_mtp.py:_enqueue` counts image sources in `conversation.messages[:-1]` as historical, then derives `required_image_frontier` from those image identities.

Recipe can expand one wire user message containing multiple tool results into several Tool messages. Harness merges adjacent tool/user turns and puts results before other user blocks. Consequently, a **new screenshot in an earlier result followed by a text result or user text** can be outside the final converted message and be required to exist in an earlier pair. This is a concrete source-level risk, not yet an execution-reproduced defect.

Implementation approach:

1. Reproduce with the actual packaged Recipe 0.2 implementation, covering multiple tool results, source ordering, trailing text, historical image reuse and paired-cache constraints.
2. Only if a real defect is reproduced, implement the smallest correction so freshness follows the latest wire input turn rather than merely the last normalized message.
3. Investigate Recipe-provided origin/source metadata first. If needed, carry minimal non-executable origin metadata through request preparation, with explicit source-count/order assertions. Do not reimplement Recipe normalization or infer freshness from image bytes alone.
4. Retain complete-span/digest checks and all historical-image identity/compatibility fences. Add only directly affected Chat/system/branch regressions; this is not a general Vision refactor or full regression campaign.

Likely owners: `serving/messages_admission.py`, `serving/server.py` preparation, `serving/production_mtp.py`, and the prepared-request representation selected after the reproduction.

### P0 — Files-first behavior and real wire compatibility

Phase 0 qualifies one installed macOS headless tool-image path, including Files 404 and inline fallback; it does not qualify all attachment/history behavior. Reuse its recorded Harness/provider/plugin identities and wire facts. In subsequent loop receipts, retain:

- Files requests against the local endpoint, failure classification, latency, and the subsequent Messages request.
- Inline base64 inside the actual tool result, not a local path, `file_id`, URL or stringified JSON.
- MIME after normalization, decoded dimensions, serialized body size and normalized-image digest.
- Stable historical image bytes/variant across requests, preserved thinking/signatures and exact tool IDs.

Use the demonstrated fallback first; no source fix is indicated for the Phase 0 path. Only if further evidence shows unreliability or costly repeated Files probes, separately propose a small explicit **inline-only route policy in the Harness provider**, with provider tests and no Runtime Files emulation. No such setting is established. Distinguish pre-generation Files fallback from generation POST retry: the latter must not hide an unknown execution outcome. Windows normal-operation contracts are a separate Phase 3B gap.

### P1 — Offload, compaction and availability

Harness's default inline image budget is 20 MiB, whereas Runtime bounds the **entire JSON request to 16 MiB**. An image-only budget cannot guarantee acceptance. Runtime also counts cached images against aggregate preparation budgets.

Harness image offload or compaction can mutate the old prefix and remove image occurrences. Remaining historical images may then lack compatible paired state. Do not treat this as transparent continuation. For the MVP, keep runs below every offload/compaction threshold and detect any projection change. For longer tasks, use an explicit new-conversation checkpoint containing project files/revision, goal, current screenshot and outstanding issues. Record lost conversational context; this is new inference, not restored KV.

Ordinary Messages has no logical-session/pinning/persistence contract. The shared cache holds 18 complete pairs and can be affected by other conversations. Eviction, restart and historical image changes need explicit failure/restart handling, not blind resubmission or silent re-encoding.

## 5. Minimal wire and rendering contract

Append this block only after a matching completed assistant `tool_use`:

```json
{
  "role": "user",
  "content": [{
    "type": "tool_result",
    "tool_use_id": "<original-id>",
    "content": [
      {"type": "text", "text": "render=r2; revision=...; viewport=1280x720; sha256=..."},
      {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "<base64>"
      }}
    ]
  }]
}
```

Use a fixed 1280×720 viewport, DPR=1, deterministic fonts/assets and headless Chromium/Playwright for Web. Start 3D with Three.js in the same browser, fixed camera, lighting, seed and simulation time. Separate render failures/blank frames from successful observations. Blender and other renderers are later extensions.

Retain render ID, code revision/hash, raw screenshot hash, normalized request-image hash/variant, dimensions and camera parameters. Normalization may change pixels; the model's actual input must be traceable, not just the original screenshot.

The existing P5 handoff requires two terminal text tokens after all image spans. Validate the actual encoded prompt for image-only and image-last results; do not assume the sample wire block alone proves this invariant.

## 6. Implementation phases and exit gates

### Phase 0 — Reproducible environment and transport capture — bounded PASS

The [report](../artifacts/visual-feedback-loop/phase-0-report.md) and [receipt](../artifacts/visual-feedback-loop/phase-0-evidence/receipt.json), retained by `ca2d4ed`, record the admitted installation, dependency identities, isolated official macOS headless Harness 0.2.1-rc.2 profile, actual Files 404 → inline delivery, accessible external screenshot and correct observation. No provider source fix was indicated for this tested path.

**Exit met, narrowly:** one `read_image` result reached nested native Messages as base64 with exact tool ID and a completed Harness turn. The original UI-image probe ambition was not completed; ordinary uploads and Windows remain separate scope. Historical evidence and its SSE-tail limitation remain unchanged. This is transport/image-informed inference acceptance, not loop or artifact-quality acceptance.

### Phase 1 — Inline image correctness and minimal fixes

Investigate new versus historical images after conversion using the actual packaged Recipe 0.2 implementation. The suspected new-image misclassification is **not yet a reproduced defect**. Add focused cases to `tests/test_recipe02_messages_serving.py`, `tests/test_multimodal_contract.py`, `tests/test_metadata_first_vision.py`, `tests/test_expanded_vision_capacity.py` and paired-cache tests only where relevant:

- Multiple results in one wire turn; new image first/middle/last; source ordering, trailing text and tool-call versus returned-result ordering.
- Historical image plus fresh screenshot; cached old spans plus uncached new spans; identical bytes in distinct occurrences.
- Historical identity/normalization changes and unavailable/evicted pairs must retain existing rejection fences; validate image-last terminal-text constraints.

Implement the smallest correction only if a real defect is reproduced. Preserve Recipe semantics, exact tool IDs, historical digest/grid/span identity, compatibility fences and existing cache/Vision/MTP ownership. Use focused affected regressions, not a general Vision refactor or full regression campaign. Thinking ON and broader negative/compatibility coverage are not implied by this gate.

**Exit:** retained reproduction/test evidence establishes consistent ordered sources and required frontier; newly supplied images do not incorrectly require an old pair, and incompatible historical images still reject. Any correction passes directly affected regressions in an admitted environment; no speculative fix is required if the risk is not reproduced.

### Phase 2 — Direct-HTTP real-model loop

Proposed new evaluation driver: `tools/accept_visual_feedback_loop.py`, following the real-socket, trace and durable-receipt pattern in `tools/accept_mtp_vision.py`. This isolates Runtime correctness; it is not a substitute Harness product.

- One bounded conversation, deterministic rendering environment, bounded tool set, maximum five correction iterations and fixed time/token budgets. Use inline images; Files API is not a dependency.
- Model creates or edits an artifact; an external renderer produces the actual screenshot; the driver delivers it as image content through native Messages. The model independently observes the result and chooses subsequent changes. The driver must not inject the correct patch or describe the visual discrepancy to it.
- Include a controlled initial mismatch and retain at least two actual model-directed edit/render correction iterations, each guided by delivered image content.
- Preflight the entire JSON body (16 MiB), aggregate image/context budgets and paired-cache availability, including history. Preserve historical identities; stop or explicitly restart rather than silently replaying after eviction/projection changes. Execute tools only after settled native completion as required in §8.
- Check old-image encode count is zero on reuse, new-image encoding corresponds to uncached spans, Target/DSpark frontiers agree, and forbidden historical replay/full repack does not occur. Add narrow trace fields if existing telemetry cannot prove these facts.
- Separate negative cases: SSE disconnect/cancel, tool timeout/error, blank screenshot, oversized body and pair eviction. Fence unknown tool outcomes; do not rerun effects automatically.

**Minimum inline MVP exit:** image identities (raw and transmitted), exact tool IDs, source revisions/diffs, rendered artifacts and Runtime traces prove at least two model-directed correction iterations in one bounded conversation. Record transport success, loop execution success and independently assessed artifact-quality success separately; a completed loop does not itself prove quality. No official Harness acceptance is claimed yet.

### Phase 3 — Official Harness inline MVP

- Run the same bounded Web loop through the official Harness, initially in the qualified macOS environment, using its existing tools and `read_image`/attachment architecture with external rendering.
- Reuse Phase 0's demonstrated Files-to-inline fallback, not new Files infrastructure. Observe whole-request, image-history and cache constraints; Phase 3 has no Files API implementation dependency.
- Verify actual wire traffic, not only UI previews or attachment metadata. Require at least two image-guided edit/render correction iterations without human/driver descriptions of discrepancies or supplied patches.
- Retain the same auditable images, exact tool IDs, source revisions, rendered artifacts and Runtime traces as Phase 2. Record transport, loop and artifact-quality outcomes separately. Run a fixed-camera 3D task next as additional scoped evidence.

**Exit:** a durable receipt proves the bounded inline loop and explicitly identifies the tested OS, Harness/provider versions, image transport and Runtime installation. Only then narrowly update README/[capabilities](capabilities.md) image-workflow qualification language. macOS headless acceptance does not establish Windows normal-operation support.

### Phase 3B — Windows normal-operation compatibility

**Known failure (user observation):** ordinary image handling in Windows DeepSeek Harness currently fails. The precise failing protocol boundary has not yet been established by a retained Windows wire trace. A successful macOS inline fallback does not remove this compatibility requirement.

**Gate W1 — static audit: PARTIAL overall, contracts mapped.** The
[rc.2 source audit](flash-1017-harness-files-compatibility.md) establishes the
Files upload/reference contract and conditional lifecycle calls, common user/tool
image preparation, and the independent default-on `dsh_session_log` admission gap.
Windows-specific source is available, but the failing Windows installation's
identity/configuration and wire boundary remain unknown. No Windows or macOS
execution was performed, and Phase 0–3 acceptance evidence is unchanged.

**Gate W2 — bounded model-free implementation work completed:** image Files
upload/delete and native Messages file-reference resolution are implemented and
model-free verified; see [W2 scope](harness-image-files.md). This in-memory work
is a bounded prototype, not official Harness Files/Windows qualification or a
finalized session-lifecycle design. Existing support and qualification exclusions
remain unchanged.

**Separate future Files campaign:** defer Files redesign/full implementation,
Windows Files qualification and `dsh_session_log` compatibility decisions. The
known Windows failure remains unresolved. Prefer **session-scoped, volatile file
ownership**, not general durable persistence; this is a future direction, not an
implemented guarantee. Process-local storage is not session-owned storage.
Completion/cancellation, active request snapshots, forks, expiry and cache
eviction require explicit ownership rules. File-byte lifetime must remain
independent of paired KV validity. Do not redesign W2 in the endurance campaign.

After the inline MVP and static audit:

1. Reproduce and characterize actual Windows ordinary image-upload and tool-image behavior through its normal interaction flow; retain sanitized wire evidence and installation identities.
2. Identify the API/transport contracts actually required by those workflows rather than inferring them from macOS headless behavior.
3. Evaluate Files API upload, file-ID references, Messages file-reference resolution, required lifecycle operations and any other observed missing boundaries together. Do not assume `POST /v1/files` alone is sufficient.
4. Define a separate evidence-driven design and implementation task, then implement only the subset necessary for supported Windows workflows under that task's authorization. This plan does not pre-authorize a full Files API, storage architecture, persistent KV or unrelated protocol extensions.
5. Resolve any supported file-backed images into existing Vision preparation and paired MTP, preserving Recipe/native Messages ownership and image/cache fences; do not create another Vision executor.
6. Re-run the same bounded visual feedback task through Windows Harness's normal interaction flow, including at least two image-guided correction iterations and auditable evidence.

**Exit:** required Windows contracts and the supported workflow scope are explicit; a version/transport/Runtime-scoped receipt demonstrates normal-operation compatibility and the same loop. Unimplemented or untested boundaries remain excluded, not implied by upload success.

### Phase 4 — Quality and endurance

**Current gate Q1 — capacity investigation:** quantify the qualified Phase 3
inline path from retained evidence and current admission/history source, without
another model conversation or production changes. The [Q1 envelope](flash-1017-visual-endurance-envelope.md)
finds six images observed, seven admissible at the measured geometry, and eight
rejected by aggregate patches. Its bounded Gate Q2 proposal is not authorization
to run it; 10–20 full-size uninterrupted observations are not currently supported.
Files redesign/full implementation and Windows qualification remain separate.

The planned integrated progression follows Phase 3B for Windows workflow claims. Independent quality/endurance experiments may proceed on the qualified macOS inline path without waiting for Windows compatibility; scope each result to its actual environment and transport.

- Measure 10–20 iterations, competing conversations/cache pressure, history/body growth, normalization/preparation/Vision time, prefill/TTFT, total task time and failure rate.
- Preflight whole-body bytes, image count/bytes, pixels, patches and expanded context. Current limits include aspect 1:2–2:1, 4,194,304 source pixels and 1014 expanded positions per image; aggregate preparation limits remain authoritative in [capabilities](capabilities.md).
- Start with one PNG/JPEG per iteration, targeting at most 512 KiB when readable; this is a proposed operating target, not a new Runtime limit. Use viewport captures or explicit crops rather than unsupported tall full-page screenshots. Record crop coordinates and scale.
- Compare overview-plus-crop for small UI text within existing budgets. Qualify any resolution/preprocessing/limit changes separately.
- Exercise explicit fresh-conversation checkpoints. Phase 3B owns evidence-driven Windows Files-contract evaluation; broader Files/URL support, Messages session semantics or persistence still require separate designs justified by measurements.

## 7. Evaluate agency, not just transport

Record **transport success**, **closed-loop execution success** and **artifact-quality success** separately. HTTP 200, an accurate image description or the model declaring completion is insufficient.

Use a fixed task suite with predetermined goals, budgets and independent success criteria:

- Web: layout, spacing, colors, typography, overflow and responsive views.
- 3D: object presence/position/color, composition, occlusion and lighting.
- Compare text-only, first-image-only and per-iteration image feedback under the same task/budget. Include render-time discrepancies not directly revealed by source/DOM, and a withheld-image control. Do not put the answer in tool logs.
- Score initial-to-final improvement, success rate, corrections, regressions, stagnation, false completion and latency across multiple trials.
- Combine visual rubric review with DOM/object checks and auxiliary pixel differences; fonts/GPU differences make raw pixel equality insufficient.

Persist RUNNING/FAIL/PASS receipts with dependency identities, task/goal hashes, code diffs, raw and transmitted image hashes, transcript location, tool outcomes, traces and independent evaluation. Specify retention/privacy rules for workspace files and screenshots.

## 8. Safety and non-goals

- Execute tools only after a fully settled assistant response; argument deltas or `tool_use` stop reason alone do not authorize effects. Native Messages completion uses `message_stop`.
- Sandbox generated code and renderer processes; constrain workspace paths/network access, time, process lifetime and execution count. No unrestricted host execution.
- Treat page text, screenshots and console logs as untrusted observations, not instructions or proof of success.
- No silent image deletion/replacement, automatic historical re-encoding, speculative effect execution or generation retries to mask uncertain outcomes.
- Do not revive retired Browser/server-side Web Tool applications or embed a renderer in Runtime.

**Current development gate:** Phase 4 / Gate Q1, the model-free endurance-envelope
investigation linked above, using the retained qualified Phase 3 official Harness
inline loop. W2 records model-free implementation only; Files redesign/full
implementation, Windows qualification and session-log decisions are deferred to
the separate Files campaign. Gate Q2 requires subsequent authorization; this
update starts no live experiment or implementation.
