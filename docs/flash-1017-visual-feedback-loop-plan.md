# Flash-1017 Visual Feedback Loop — Investigation and Implementation Plan

Status: **Proposal; not end-to-end qualification.** No Runtime implementation or installed environment was changed by this review. This document replaces the preliminary investigation with an English plan including the official Harness source review.

## 1. Decision

Evaluate Vision as an agent capability: **create → render → observe → compare with the goal → edit → render again**, not merely image description.

The necessary building blocks already exist. The first milestone should connect the official Harness's existing `read_image`/attachment pipeline to native Messages, bounded Vision and paired MTP, then retain reproducible evidence of at least two model-directed edit/render iterations. Do not start by adding another Vision executor, a browser inside Runtime, Files API, URL fetching or persistent KV.

Source inspection establishes an image-return path, **not that the installed Harness and Runtime have successfully exercised it**. The highest-priority integration issues are Files-first transport, image-history mutation/offload, and the Runtime's distinction between new and historical images after Recipe conversion.

## 2. Reviewed revisions and evidence scope

| Repository | Reviewed revision |
| --- | --- |
| ds41f-mlx | `be44f393f6c1df7c2b93307ab587384dc172af0f` |
| official DeepInfer | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| official deepseek-recipe | `935817c66f5d6e64838f419692545f06787e485f` |
| official deepseek-harness | `687f282e8625c6b8d11ff58211a90524be2e4317` |
| packaged Recipe 0.2 upstream base | `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`, recorded in `third_party/mtp/normal-local.json` |

Official checkouts were inspected under `/tmp/ds41f-visual-review`. Upstream HEAD is not the packaged Recipe/native payload or necessarily the user's installed Harness. Run conversion regressions against the pinned installed payload before implementing a fix.

Primary official references, pinned to the reviewed revisions:

- [Recipe Messages conversion](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs): `image_block`, `MessagesToolResultContent::into_content`, `transform_messages`.
- [Recipe V4 encoding](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/v4/mod.rs): ordered image-source collection, including Tool messages.
- [DeepInfer multimodal processor](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/ve/multimodal_processor.py): placeholder expansion, image spans, and skipping fully cached images.
- [DeepInfer image processor](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/ve/processor/image_processor.py): image geometry and token/patch reference.
- [Harness read_image](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/fs/tool-fs/src/read-image.ts): route capability gate and image attachment output.
- [Harness serializer](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/serialize.ts): image blocks inside native `tool_result`, exact tool IDs and adjacent-role merging.
- [Harness adapter](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/adapter.ts), [image preparation](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/images.ts), and [Files resolution](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-deepseek/src/request-files.ts): Files-first requests and bounded inline fallback.
- [Harness image-offload integration test](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/apps/cli/tests/profiles/acp/tests/image-offload.expected.e2e.ts): expected image results and inline fallback; this is upstream test source, not a ds41f acceptance receipt.

Existing local evidence is scoped by [capabilities](capabilities.md), [Vision integration](mtp-vision-integration.md) and the [development record](deepinfer-development.md). Small Chat-image/cached-follow-up and text/read-only Harness tool receipts do not establish the proposed creation loop.

The preliminary draft recorded failed local probes: missing `ConversionOptions.with_model_versions_by_name`, and 111 setup errors in four targeted suites because isolated installed-payload checks could not import `ds41f_mlx`. These are environment failures, not passed regressions or reproduced image defects. They were not rerun in this source-only review. Prepare a disposable, admitted installation following [verification](verification.md); do not bypass admission or modify the everyday installation to obtain a green result.

## 3. Official implementation versus current Runtime

| Boundary | Official behavior | Current ds41f behavior / consequence |
| --- | --- | --- |
| Image-return tool | Harness `read_image` reads through its filesystem service, commits an attachment and returns text plus an image block; the active model route must advertise image input | Existing Harness tool is the preferred MVP path. A screenshot pathname in ordinary tool text alone is not image feedback |
| Native wire encoding | Harness serializes tool images as `user.content[].tool_result.content[]`; image source is base64 or file ID | `serving/messages_admission.py:blocks` recursively permits base64 images and rejects Files/URL sources |
| Transport selection | Adapter tries Files resolution first; `FileResolutionFailure` switches to inline and rechecks image budgets | Local Files failure may already activate compatible fallback. Capture actual requests/statuses before proposing an adapter change; no explicit inline-only switch was established by this review |
| Protocol ownership | Recipe converts nested image/text results to Tool messages, with image sources retained | Keep native Messages conversion, thinking/signatures, exact IDs and JSON/SSE. No Messages-to-Chat translation or second parser |
| Vision execution | DeepInfer expands image placeholders from processor configuration and skips only completely cached spans | `runtime/multimodal.py`, existing MTP adapter and paired Target/DSpark path are the implementation owners. Verify geometry and source order rather than porting GPU workers |
| Historical image reuse | DeepInfer computes cache coverage using complete image spans | ds41f additionally requires a retained compatible pair for historical images; eviction/edit rejects rather than silently re-encoding history |
| History management | Harness can normalize request images and offload older occurrences, replacing images with text; serializer merges adjacent user/tool turns | Both transformations affect exact prompt/image identity. They must be tested against paired continuation, not assumed compatible |
| Render/iterate | Harness owns tools and workflow; protocol/engine code alone does not define a production task | Use external deterministic rendering and the existing image-return tool. Runtime remains an inference service |

### Confirmed source path

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

A combined render-and-return-image tool can follow later if needed. Reuse the official attachment service rather than inventing a path-to-image interpretation inside Runtime. UI uploads and tool-returned images require separate transport probes.

## 4. Priority gaps

### P0 — New-image classification after conversion

`ds41f_mlx/serving/production_mtp.py:_enqueue` counts image sources in `conversation.messages[:-1]` as historical, then derives `required_image_frontier` from those image identities.

Recipe can expand one wire user message containing multiple tool results into several Tool messages. Harness merges adjacent tool/user turns and puts results before other user blocks. Consequently, a **new screenshot in an earlier result followed by a text result or user text** can be outside the final converted message and be required to exist in an earlier pair. This is a concrete source-level risk, not yet an execution-reproduced defect.

Implementation approach:

1. Reproduce conversion with the packaged Recipe, including tool-result ordering and image-source ordering.
2. Define freshness by the latest wire input turn, not the last normalized message.
3. Investigate Recipe-provided origin/source metadata first. If needed, carry minimal non-executable origin metadata through request preparation, with explicit source-count/order assertions. Do not reimplement Recipe normalization or infer freshness from image bytes alone.
4. Retain complete-span/digest checks for every earlier image; new-image admission must not weaken historical-image fences.
5. Cover Chat, system updates, branches and mixed result ordering as regressions.

Likely owners: `serving/messages_admission.py`, `serving/server.py` preparation, `serving/production_mtp.py`, and the prepared-request representation selected after the reproduction.

### P0 — Files-first behavior and real wire compatibility

The source path is no longer unknown, but its installed behavior is unqualified. Record the exact Harness/provider/plugin revisions and observe:

- Files requests against the local endpoint, failure classification, latency, and the subsequent Messages request.
- Inline base64 inside the actual tool result, not a local path, `file_id`, URL or stringified JSON.
- MIME after normalization, decoded dimensions, serialized body size and normalized-image digest.
- Stable historical image bytes/variant across requests, preserved thinking/signatures and exact tool IDs.

First test the existing fallback. If it is unreliable or repeats costly Files probes, propose a small explicit **inline-only route policy in the Harness provider**, with provider tests and no Runtime Files emulation. Do not claim such a setting already exists. Distinguish pre-generation Files fallback from generation POST retry: the latter must not be used to hide an unknown execution outcome.

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

### Phase 0 — Reproducible environment and transport capture

- Prepare an isolated admitted Flash-1017 installation; record checkpoint, wheel, Recipe/native identities and Harness/provider/plugin inventory.
- Reuse the documented `deepseek-official` native Messages route, image capability advertisement, session-log restrictions and supported tool additions.
- Capture one UI image and one `read_image` tool result, including Files-to-inline behavior and normalization metadata. Redact credentials/base64 bodies from ordinary logs.
- Verify that render tools can write a screenshot and `read_image` can read it in the same execution filesystem.

**Exit:** actual installed image transport is known, image bytes reach nested Messages content, and the need (or lack of need) for a provider patch is established. This is not model-quality acceptance.

### Phase 1 — Contract regressions and minimal correctness fixes

Extend `tests/test_recipe02_messages_serving.py`, `tests/test_multimodal_contract.py`, `tests/test_metadata_first_vision.py` and `tests/test_expanded_vision_capacity.py` plus paired-cache regressions as appropriate.

Required cases:

- Text+image, image-only, multiple images, text before/after images.
- Multiple results in one wire turn; new image first/middle/last; trailing user text; tool-call versus returned-result ordering.
- Historical target image plus fresh screenshot; cached old spans plus uncached new spans.
- Thinking on/off, signatures, exact IDs, JSON and SSE completion.
- Invalid base64/MIME, unknown/duplicate IDs, unsupported URL/file sources, resource excess and special-token-like tool text.
- Same image bytes in different occurrences; changed pixels with identical placeholder tokens; changed historical image; cancellation and pair eviction.
- Harness normalization stability, Files fallback and offload/compaction boundary behavior.

Fix origin classification only after reproducing the problematic conversion. Keep the existing cache owner, Vision executor and native semantic owner.

**Exit:** ordered sources, digest/grid/span and required frontier are consistent; new images do not require an old pair; incompatible old images still reject. Existing Chat/text/paired regressions pass in the admitted environment.

### Phase 2 — Direct-HTTP real-model loop

Proposed new evaluation driver: `tools/accept_visual_feedback_loop.py`, following the real-socket, trace and durable-receipt pattern in `tools/accept_mtp_vision.py`. This isolates Runtime correctness; it is not a substitute Harness product.

- One conversation, bounded tool set, maximum five correction iterations and fixed time/token budgets.
- Model creates or edits code, renders, receives the image, judges discrepancies and chooses the next edit. The driver must not inject the correct patch or describe the visual defect for it.
- Include a controlled initial mismatch and retain at least two actual model-directed correction/render iterations.
- Check old-image encode count is zero on reuse, new-image encoding corresponds to uncached spans, Target/DSpark frontiers agree, and forbidden historical replay/full repack does not occur. Add narrow trace fields if existing telemetry cannot prove these facts.
- Separate negative cases: SSE disconnect/cancel, tool timeout/error, blank screenshot, oversized body and pair eviction. Fence unknown tool outcomes; do not rerun effects automatically.

**Exit:** images, tool IDs, code diffs, renders and Runtime traces form one auditable loop. No claim of official Harness acceptance yet.

### Phase 3 — Official Harness end-to-end acceptance

- Run the same Web task through the actual official Harness using its existing render/write tools and `read_image`; add only the smallest provider change justified by Phase 0.
- Run a fixed-camera 3D task next.
- Verify actual wire traffic, not only Harness UI previews or attachment metadata.
- Require multiple image-guided edits/renders without human image descriptions or patches. Record provider version and OS scope.

**Exit:** a durable receipt proves the complete official-Harness loop. Only then narrowly update README/[capabilities](capabilities.md) image-workflow qualification language.

### Phase 4 — Quality and endurance

- Measure 10–20 iterations, competing conversations/cache pressure, history/body growth, normalization/preparation/Vision time, prefill/TTFT, total task time and failure rate.
- Preflight whole-body bytes, image count/bytes, pixels, patches and expanded context. Current limits include aspect 1:2–2:1, 4,194,304 source pixels and 1014 expanded positions per image; aggregate preparation limits remain authoritative in [capabilities](capabilities.md).
- Start with one PNG/JPEG per iteration, targeting at most 512 KiB when readable; this is a proposed operating target, not a new Runtime limit. Use viewport captures or explicit crops rather than unsupported tall full-page screenshots. Record crop coordinates and scale.
- Compare overview-plus-crop for small UI text within existing budgets. Qualify any resolution/preprocessing/limit changes separately.
- Exercise explicit fresh-conversation checkpoints. Evaluate Files/URL support, Messages session semantics or persistence only in separate designs justified by measurements.

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

**First deliverable:** reproducible evidence that Flash-1017, running through the official Harness, observes its own rendered output and performs at least two subsequent code-edit/render iterations. Capability claims remain bounded by that receipt, not by source inspection alone.
