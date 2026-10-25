# Phase 1 — Multimodal Runtime source gap audit

**Disposition: source audit COMPLETE; execution qualification NOT YET VERIFIED.**
No C2–C9 completion, changed execution contract, implementation, model run or
release approval is implied. This report answers Phase 1 of the
[CURRENT plan](../../docs/multimodal-runtime-plan.md), not Phase 2.

## 1. Identity, method and authority

Preflight: branch `Multimodal-Runtime`, HEAD
`f72c0645594d5db675b57ca118a4df8e57e0f4d8`, clean working tree, one existing
worktree. Process inspection found one `pi` process (PID 2432), no other listed
coding agent or Runtime process. This is a process-list observation, not a global
lock against concurrent work. HEAD did not require an intervening-change review.
No existing worktree, installation, wheel, service or model asset was changed.
Only documentation was edited. Temporary upstream source exports/bare clones
were used for inspection; no dependencies were installed or imported for inference.

| Authority / reference | Exact inspected identity |
| --- | --- |
| Local code, all `L` references below | `f72c0645594d5db675b57ca118a4df8e57e0f4d8` |
| `H`: [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317) | `687f282e8625c6b8d11ff58211a90524be2e4317`, existing clean source tree |
| Material newer Harness comparison | Official HEAD observed by `git ls-remote`: `22ddee3792402c3e70ec5f86a63c3e3e895a2f60`; bounded diff of offload, basic compaction, DeepSeek adapter/config, content projection and read-image paths |
| `D`: [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, clean source tree; also official HEAD at remote check, so no newer fix to inspect |
| `R`: [deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f) | `935817c66f5d6e64838f419692545f06787e485f`; also official HEAD at remote check |
| Delivered Recipe, **not R as a whole** | 0.2.0; `third_party/mtp/normal-local.json:7885–7889`: source `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` + `docs/recipe-02-native-semantic-boundary.patch` |
| `O`: pinned oMLX substrate | [jundot/omlx](https://github.com/jundot/omlx), 0.7.0 base `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, exported candidate `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`; `third_party/mtp/sources.json:4–9` |

Inspected repository-owned source archives, not assumed equality with an installed
package. SHA256: oMLX archive
`26cc224a5fa77d8576589764a56ba8053ac31fe9f4904d8ba60305ed841f33e6`;
Recipe 0.2 archive
`f9e36b29efc91ace7c06378765947b456d99ed55a87d8dbcbcad01dcb88c2cc6`;
Recipe native patch
`9d0b08e9b88de0bf41fb237db04f45205addbd7fb412d81eacc6d05955a383f2`.
`ds41f_mlx/mtp_setup.py:24–31,52–67` applies the ordered compressed-growth,
bit-exact-kernels, template-reuse and context-copy overlays, then the Recipe
native patch. Those repository-owned overlays and the package-fixed numerical
profile remain part of any future executable identity. No installed-payload
qualification was performed here.

Read the nine requested CURRENT contracts. Read the decision/identity/resource
boundaries of [cached admission](../cached-image-admission/report.md), the terminal
refusal and cleanup sections of [repaired loop](../repaired-visual-loop/report.md),
and the relevant caller/accounting analysis in the
[earlier upstream audit](../visual-feedback-loop/upstream-cached-image-accounting-audit.md).
These are revision-scoped EVIDENCE, not fresh HEAD certification. In particular,
request 28's missing rejection values cannot identify which byte guard failed.

Ownership stays upstream-first: Harness owns the event log, projection and tools;
Recipe owns canonical protocol conversion/encoding/IDs; DeepInfer supplies
reference preparation/recomputation architecture; ds41f owns MLX admission,
complete paired integrity and settlement. Harness session history is **not** a
Runtime Chat logical session or executable KV lease.

## 2. Source map and verified caller paths

Paths below are relative to the named exact repository/export. Local links resolve
to the checkout; line ranges refer to the audited HEAD, before documentation edits.

### Harness lifecycle

| ID | Source owner, function and actual caller |
| --- | --- |
| H1 | `packages/fs/tool-fs/src/read-image.ts:193–351`, `imageReadContent`, `applyReadImageTool`: registered tool checks routed image capability, reads via mounted filesystem, saves through attachments, returns text + attachment image reference. `packages/attachment/attachment-local/src/index.ts:203–280`, `saveImage`, `readImage`, `imageHostPath`, `readImageRequest`; `store.ts:186–226,424–460`, publication and `readImageFile`, supply content-addressed normalized bytes with read verification. |
| H2 | `packages/llm/llm-deepseek/src/adapter.ts:90–170`, `request` → `images.ts:38–61` `prepareImages` → `projectOffloadedImages` → `attachments.readImageRequest` → `assertImagesFit`; then `prepareFileIds` / `RequestFiles.resolve` → `serialize.ts:58–155` `serialize` → HTTP Messages. Serializer emits image handle text + image, nests tool images inside `tool_result` with exact `tool_use_id`, merges adjacent roles and checks completed call/result pairing. Files failure selects bounded inline fallback, which runs its own budget check. |
| H3 | `packages/llm/llm-deepseek/src/config.ts:33–44,99–104,240–273`, schema and `resolveAdapterOptions`: positive safe-integer byte/count thresholds and quanta; quantum must not exceed its corresponding bound. `images.ts:13–20,78–90` passes these into `requiredImageOffload`; `packages/llm/llm/src/content.ts:276–349` computes oldest-occurrence removal from represented count/bytes and quanta. |
| H4 | `packages/compaction/compaction-image-offload/src/index.ts:25–49`, `apply`: actual `agent/request-error` and `compaction/summary-error` listeners handle only `IMAGE_OFFLOAD_REQUIRED` with `offloadImages`; `image-offload.ts:16–49` `offloadOldestImages` selects user/tool surface occurrences, skips already offloaded images and appends `image/offload`. `projection.ts:43–86` `imageOffloadProjection.project` validates current node/index targets and calls `project-message.ts` to mark occurrences. |
| H5 | `packages/llm/llm/src/content.ts:109–118,230–273`, `offloadedImageText`, `replaceOffloadedImages`, `projectOffloadedImages`: request projection changes selected image blocks to deterministic text, not deletion of whole tool messages. `packages/core/agent-loop/src/agent.ts:500–526,631–650`: request-error retry loops through fresh request-header preparation; this is not tool-effect replay. |
| H6 | `packages/compaction/command-compact/src/index.ts:58–107`, registered `/compact` → `ctx.compaction.compactNow`; `compaction-basic/src/index.ts:383–445` → idle `agent.runMaintenance` → `selectCompactableRange` → `compactSurfaceRegion`. `region.ts:117–156` preserves system head and balanced tool boundaries; `386–426` summarizes with offload-error recovery and requires smaller framed summary; `471–509` appends summary record and text user checkpoint with `surfaceOp: replace`. |

**Observed projection semantics.** Offload is occurrence-specific: original
message nodes/seqs, non-image text, tool-call IDs/results and the durable source
attachment reference remain; projection marks the selected occurrence, then the
adapter substitutes text naming the attachment and, where available, its read-only
normalized path. Other occurrences of the same attachment are not implicitly
removed. Old expanded image positions become text positions; every later span may
move. Increasing a budget does not unmark an occurrence. Offload does not delete
attachment bytes. A later real `read_image` can create a **new occurrence** from
that normalized path; absent mapping/bytes the placeholder asks for reattachment.
This is availability conditional on the attachment/filesystem provider, not a
promise of permanent retention or Runtime Files persistence.

Compaction shadows a balanced selected surface range with a framed **text-only
user summary checkpoint**; the event log still contains source events,
`compaction/start`, `summary`, `end` and source relationships. The system head and
unselected tail remain projected. Tool identities in an unselected tail remain;
shadowed tool messages are no longer model inputs. A prose summary is not a
replacement executable tool ledger. No historical tool is rerun by prefill.

Neither lifecycle sends provider KV handles, invalidation messages or reconstruction
commands. The provider sees a newly serialized canonical Messages request (plus
Harness attribution/compaction headers), not a leased pair. Exact identity/prefix
comparison must detect a hit or miss. The offload README's future provider-report
of uncacheable images is an intention, **not wired recovery**. Implemented recovery
is adapter-local budget error → durable selection → new projection retry, Files
refresh/fallback, and supported compaction; arbitrary Runtime resource errors do
not acquire that meaning.

**Newer material check:** at Harness `22ddee3…`, offload implementation, image
preparation, read_image and content projection in the bounded comparison are
unchanged. Basic compaction adds proposed-model-switch pressure handling and
refactors pressure calculation (`compaction-basic/src/index.ts`,
`_registerAutomaticCompaction`, `compactPressure`); this does not supply a provider
KV-rebuild protocol or change the inspected manual replacement transaction.
DeepSeek config additions concern model discovery, not these thresholds. This is
not an audit of every newer caller or installed Harness version.

### Recipe canonical semantics

`R:deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs`,
`MessagesRequest::convert` → `transform_messages` (`426–729`) → `image_block`
(`266–290`) and `MessagesToolResultContent::into_content` (`327–359`), preserves
text/image block order, records source paths, creates one canonical placeholder
per image occurrence, expands tool results into canonical Tool messages, and
checks pending/exact/duplicate/foreign tool-use IDs. `R:deepseek-recipe-encoding/src/v4/mod.rs:109–135`,
`collect_image_sources` → `render`, collects System/User/**Tool** sources in
message order; the official encoding owns conversation delimiters and continuation.

The two files above in the delivered Recipe source archive are byte-identical to
R (direct source comparison). The native patch does not replace these request
conversion/encoding files; it supplies native semantic sessions, response/tool-ID
and streaming settlement changes. Thus local Recipe 0.2 already provides the
required canonical ordering, source origins, tool-result images and fresh encoding
of a changed/shortened **valid** Messages history. It has no retained KV-history
obligation and needs no second parser. Arbitrary malformed tool continuation
remains invalid, even after history shortening.

Important separation: `image_block` rejects direct File-ID conversion. Local
[`image_files.py:206–275`](../../ds41f_mlx/serving/image_files.py),
`resolve_messages`, first validates the original Recipe request structure and
resolves only supported user/nested-tool-result file image sources to request-local
base64 snapshots before the existing conversion. This is bounded transport glue,
not an upstream Recipe File-ID resolver. Missing/expired local IDs reject even if
an executable pair might exist; current admission still requires all projected
image containers. Attachment ID, provider File-ID, encoded-byte digest and image
**occurrence** (position/grid/length) are different identities.

### DeepInfer execution, including the handoff limitation

| ID | Exact D source / caller / conclusion |
| --- | --- |
| D1 | `deepinfer/engine/llm_engine.py:543–625`, request sequence construction → `MultimodalProcessor.overwrite_prompt` → `Sequence` → `_enqueue` → `scheduler.add_sequence` → **`multi_modal_processor.add(request_id, multi_modal_data)`**, no frontier argument. |
| D2 | `deepinfer/ve/multimodal_processor.py:105–169,219–263,287–315`, `overwrite_prompt`, `_compute_and_strip_cached_images`, `add`, `fetch`: expand absolute image spans; complete span means `start+count <= num_cached_tokens`; incomplete image retains its whole binary. `add` defaults to zero. Worker mapper result is installed before `set_schedulable`. |
| D3 | `deepinfer/model_executor/models/deepseek_vl.py:41–203`, `multi_modal_mapper` → `_split_cached_prefix` → processor conversion → `_load_image_raw_bytes`: if coverage is supplied, replace only fully covered segment prefix with equal-length dummy text, zero cached crop/size/patch offsets, retain full absolute spans. Partial text segment stays; partial image is fully prepared, not fractionally discounted. |
| D4 | `deepinfer/worker/model_runner.py:131–146` → `ModelMetadata.prepare_inputmetadata_dict` (`model_executor/model_metadata_utils.py:146–188`) → `ve/prepare_tensor_input_vl.py:31–172`, `prepare_input_metadata_vl`: slice by actual query/key chunk bounds, include whole overlapping images, discard zero crops, sum active patches. `deepseek_vl.py:210–243`, `forward` → `get_input_embeddings` → Vision tuner only when raw image data exists, then language forward/KV. |
| D5 | `csrc/engine/scheduler.hpp:350–465,477–534,691–705`, `_schedule_prefill` / sequence metadata: execute `get_len()-cache.num_tokens`, allocate on cold entry, respect schedulable/block/state/chunk constraints; `1059–1118` `fork_seq` clones/pins valid block/state and caps reused frontier at `min(parentC, N-1)`. `750–790` `_preempt` can reset and queue recomputation; `sequence.hpp:300–312` `reset_for_recompute` resets cache/schedule/MTP rolling state, not tools. |

**Not an end-to-end cache-aware preparation claim:** D1 still supplies zero to
D2; scheduler `add_sequence` (`scheduler.hpp:135–142`) enqueues, it does not return
a cache frontier to Python. Consequently normal initial requests prepare all
images at this reference. D2/D3 prove conditional complete-image skipping, not its
activation on that caller. Official remote HEAD remains D: there is no newer
revision at the observation to repair this finding.

For a whole logical-prefix hit, the inspected fork path still retains an
uncomputed final row (`N-1`), not a zero-work whole-prompt bypass. For a partial
text hit, scheduler query length is the remaining suffix and D3 preserves absolute
metadata when a frontier is supplied. The base request constructor does **not**
establish arbitrary cross-request content-addressed prefix lookup; its cold cache
and the explicit retained/fork path must not be conflated. In particular the
scheduler variable `prefix_caching_hit` is also set on an **empty block table**
(`scheduler.hpp:374–375`); that name is not evidence of a real complete-prefix hit. Broader external cache
service hit discovery is **NOT YET VERIFIED**, not required as a new local owner.

On a complete initial miss D1–D4 provide the current prompt, all image inputs,
fresh Vision and language/KV production; no distinction based on old wire-message
age forbids that recomputation. Recompute after preemption resets execution
rolling state and follows scheduling again, but universal multimodal preemption
recovery is **NOT YET VERIFIED**: processed metadata/pool lifetimes
(`ve/schema.py:53–77`) and any stripped payload must be revalidated. Do not infer
fresh bytes can be reconstructed from token IDs or a stripped cached-image marker.
Neither this CUDA/state mechanism nor its frontier values qualify local DSpark.

### Local physical flow (ordinary native Messages)

| ID | Local source and actual path |
| --- | --- |
| L1 | [`server.py:326–359,619–623`](../../ds41f_mlx/serving/server.py), registered `/v1/messages` → `api_handler` → `prepare_request` (`45–156`) → `backend.ordinary_response`; [`mtp_public.py:185–240`](../../ds41f_mlx/serving/mtp_public.py) bounds ingress/preparation. Preparation does bounded Messages/Files resolution → `messages_admission.py:39–86` `convert_messages` → Recipe native conversion/render/encode → `prepare_multimodal`. Source wire paths determine `image_is_historical`, not canonical Tool-message role guesses. Capacity uses expanded tokens. |
| L2 | [`runtime/multimodal.py:31–119,198–268`](../../ds41f_mlx/runtime/multimodal.py), `prepare_multimodal`: validate every container/byte/pixel/identity; plan expanded span/grid/kinds; retain encoded bytes only. `validate_prefix`, `validate_workload`, `encode_new`: require whole-span frontier, skip covered spans, recheck digest, prepare/encode all spans at or after C using current request bytes. Encoder does not inspect wire age. |
| L3 | [`production_mtp.py:373–411`](../../ds41f_mlx/serving/production_mtp.py), `_enqueue`: image identities + required historical maximum end → same Scheduler request. [`paired_checkpoint.py:156–186,236–289`](../../ds41f_mlx/serving/paired_checkpoint.py), capture/restore/acquire: image-aware upstream hashes, complete tokens and execution identity, validate 40 Target/3 rings, independent handles; invalid payload seeks shorter pair or misses. Lookup through `prompt[:-2]`. |
| L4 | `production_mtp.py:52–117`, `_schedule_waiting`: optional **pre-clone** conditional byte admission, ordinary or exact retained acquisition, whole-span/required-frontier/fresh-work checks, then `_start`. Historical coverage rejection is at **80 and 106**, with message `historical image pair unavailable; image replay/re-encode forbidden`. |
| L5 | `production_mtp.py:338–365`, `_encode_new_images` → L2 with pinned Model adapter → `O:omlx/patches/deepseek_v41/model.py:26–69`, `get_input_embeddings`: ViT/aligner + delimiters, no language forward. `_prefill_tap_start` selects bounded DSpark tail. |
| L6 | [`internal_mtp.py:196–337`](../../ds41f_mlx/serving/internal_mtp.py), `_start`: make fresh Target/rings on miss; consume full prepared IDs and sparse embeddings → `DeferredPrefillAppend.create/continue_from_commit` → taps → `dspark_append_context` → N-2 pending pair capture → N-1 sealed commit → P5 `handoff_to_generation` with held-out terminal once. [`prefill_fp8_mlx/p6_append.py:220–251,315–338,633–658`](../../ds41f_mlx/prefill_fp8_mlx/p6_append.py) accepts complete history/committed frontier, sparse absolute embedding merge and image-aware Engram hash/masks. |
| L7 | `internal_mtp.py:391–458`, `_settle` → existing canonical quiescence/guard finish; `production_mtp.py:133–246`, `_retire`: only settled healthy state captured/published; failures after native start burn and clear cache/fail engine. Same worker/BatchGenerator/native UID, no alternate executor. `production_mtp.py:290–299,431–450,450–477` holds Messages tool suffix until settlement, shields cancellation and keeps logical sessions Chat-only. |

`O:omlx/patches/deepseek_v41/language.py:886–1055`, `_forward`, supplies an
independent ordinary layer-loop path: image mask/Engram handling and post-Engram,
pre-layer mean DSpark taps. `mtp.py:151–195`, `DSparkMixin.__call__`, routes
ordinary prefill capture but permits verification only on text rows.
`omlx/patches/mlx_lm_mtp/deepseek_v4_dspark.py:151–219`, `capture_prompt`,
`install_committed_context`, validates native common frontier; rings are bounded
and absolute-modulo, not rewindable transcript storage. Local L6 reuses these
physical append/install mechanisms, not the generic Scheduler `vlm_mtp` route.

## 3. Central finding and protected invariants

The minimum obstruction is **historical-origin coverage policy at L3/L4**, not
missing pixels, a missing Messages grammar, or absence of a cold prefill engine.
At any miss/shorter hit `C < required_image_frontier`, `_schedule_waiting` rejects
before `_start` even if every current projected image byte is available. There is
a second rejection in the >36,864 all-history conditional-admission callback.
Removing only one check would not create a safe reconstruction path.

Source establishes that C=0 L2/L6 can receive the **complete new canonical** token
sequence and encode all its supplied images. At C>0 it can append from an earlier
valid **whole-image-safe text boundary** and skip images fully covered by that
pair. P6 inputs are absolute spans and current encoded bytes, not saved Vision
embeddings; removed occurrences need no obsolete bytes. Source plausibility does
not establish changed-history numerical correctness, particularly across dense
bulk → bounded DSpark tail → N-2/N-1 → P5. That is the necessary qualification gap.

The guard protects against treating prior-message images as executable from bytes
alone without qualified reconstruction, using a mismatched/target-only pair,
undercharging fresh work with historical patch deferral, and accidentally applying
ordinary miss recovery to a retained logical-session failure. Preserve exact
retained Chat lease failures, post-mutation burn and protocol/effect fences.

Partial image frontiers are separately forbidden by L2 `validate_prefix` /
`validate_workload` and paired capture at `paired_checkpoint.py:167–168`.
A valid local capture never bisects its own image. Under a changed valid request,
image hashes/identity comparison must reject a candidate spanning a changed image.
Thus an ordinary lookup should normally select an earlier whole-span boundary or
miss, **not** return a usable partial-image checkpoint. Do not trim/round live rings
or reuse half an image. The smallest safe fallback is an earlier valid pair (or
C=0) and whole-image encoding. DeepInfer's whole-image processing plus chunked
embedding slice is an alternative semantic reference, not permission to graft
fractional state into a local pair. Whether a real admitted path can expose a
partial frontier remains a focused verification question, not an asserted defect.

## 4. Case matrix

`SUPPORTED` below means existing bounded execution evidence plus matching source,
**not renewed qualification of this HEAD/package**. `GAP` identifies a concrete
required transition blocked in source; the proposed replacement remains unverified.
`BLOCKED` means unavailable prerequisites or deliberate physical refusal, not a
request to enlarge limits. Mixed rows separate positive and negative cases.
Evidence abbreviations: **N** = numerical reference checks in §6; **S** = exact
wire/Recipe/span/tool-ID/settlement joins; **Rsrc** = resource/cleanup checks in §5.
All positive reconstruction rows require N + S + Rsrc, not just successful HTTP.

| Case | Official expected behavior / references | Existing local behavior and status | Minimum extension point / evidence / ownership |
| --- | --- | --- | --- |
| A — exact complete paired hit | D2/D3 skip fully covered images if frontier supplied; D5 retains final work. Recipe canonical identities unchanged. | **SUPPORTED**, bounded compatible reuse: L3 restores complete **pair**, never exact N-token zero-work bypass; eligible C≤N-2. Scoped Vision/MTP repeat evidence. | No new mechanism. L3/L7; S must prove actual saved IDs/images and 40/3 state, zero covered-image encodes; N remains needed for broader C1 numerical scope. |
| B — partial text prefix + suffix | D3/D5 retain absolute coordinates, prefill missing suffix. | **SUPPORTED** for text and wholly covered historical images/new last-message images, in bounded evidence. **GAP** if suffix contains uncovered historical images (L4). | Keep L3 earlier-pair selection; qualify L2/L6 suffix reconstruction, N/S/Rsrc. Project integration gap only for uncovered images. |
| C — cold miss + fresh images | D1–D4 prepare all images and execute prompt. | **SUPPORTED** in bounded seed/new-image evidence, subject to fresh guards and two terminal text tokens. Not maximum capacity qualification. | Reuse L2/L5/L6; N/S/Rsrc for expanded reconstruction scope, no new cold engine. |
| D — historical spans wholly covered | D2/D3 conditional complete-image skip; D1 wiring caveat. | **SUPPORTED**, L3 saved identity + L4 coverage → L2 skips; scoped cached-admission/loop evidence. | Preserve pre-clone byte checks for discount; S verifies zero redundant encodes, Rsrc includes retained/capture coexistence. |
| E — historical span partially covered | D2/D3 treat whole image as uncached; D4 slices active chunk. | **NOT YET VERIFIED** as a reachable normal local pair outcome: capture forbids intersection, keys/identity exclude changed spans, and L3 already retries shorter after invalid restore. If that fallback leaves an uncovered historical image, the concrete **GAP is F**, not evidence that a fractional-image executor is missing. | Reuse existing earlier safe-pair/miss selection; whole image re-encode via F's L2/L6 integration, no ring trim. N must exercise partial-text vs image boundaries and verify fallback. No demonstrated upstream blocker. |
| F — historical images, no compatible pair | D1–D4 cold current-projection computation with available inputs; not agent replay. | **GAP**, exact L4 guard, including admission callback; not equivalent to lost retained Chat lease. | L3/L4 ordinary independent-request admission + existing L2/L6. Charge all uncovered spans fresh, require N/S/Rsrc before guard replacement. |
| G — changed identity/order | R converts actual order; helpers D2/D3 alone do not prove image-content cache identity. | **SUPPORTED** safe rejection of stale pair; **GAP** for positive historical reconstruction. L3 keys bind digest/start/length/grid, restore compares identities; earlier text hit/miss currently fails L4. | Keep identity invalidation; reconstruct only actual new projection. N/S include same token IDs/geometry but different bytes, reorder and moved occurrence; detect false hit. |
| H — official offload changes prefix | H2–H5 substitute only selected occurrences with text, retry current projection, retain attachments. | **GAP** when surviving historical images fall beyond new compatible C. Text-only remainder can use existing ordinary miss; that lifecycle execution is **NOT YET VERIFIED**. | No Runtime offload. L4 ordinary reconstruction + N/S/Rsrc; prove changed tokens and unavoidable uncovered spans, not simply old complete hit. |
| I — official compaction replaces history | H6 emits summary user checkpoint, shadows range and preserves balanced tail. Summarizer itself is an ordinary image-bearing request. | **GAP** where summary request or retained image tail needs historical recomputation. All-text post-compaction path exists but lifecycle qualification is **NOT YET VERIFIED**. | Same L2/L4/L6 integration, not a compactor. S includes real summary request, replacement events and exact projected tail/IDs; N/Rsrc. |
| J — offloaded image reacquired | H1/H5 read normalized path as new occurrence, do not unmark old occurrence. | Bounded `read_image`/Files transport is evidenced; full offload → reacquire → inference/continuation **NOT YET VERIFIED**; may encounter F for other surviving images. | Existing Files/Recipe/Vision path, plus F if needed. S binds attachment/request-byte digest, new position and original completed IDs; independent visual fact + N/Rsrc. |
| K — required image bytes absent | H1/H2 attachment read error or placeholder asks for reattachment; no recovery from IDs/KV alone. | **BLOCKED** positive reconstruction. Current local Files/source/container/digest checks reject; even hits require projected source validation. | Deliberately unsupported missing prerequisite, not a new image manager. S proves no native start/publication; operator/client reattachment where possible, never fabricate embeddings. |
| L — otherwise valid input exceeds physical admission | Harness route budget offload only for its own classified error; no generic provider memory guarantee. | **BLOCKED** physical execution outside admitted envelope; bounded refusal supported, general automatic recovery absent. | Keep all limits. Rsrc + S prove pre-start refusal/no invalid pair and retained healthy state, exact guard/values retained. Runtime-specific auto-recovery is absent upstream, not a license for custom HTTP/offload protocol. |

Bounded support evidence: [Vision/MTP delta](../../docs/mtp-vision-integration.md#delta-evidence)
(seed, repeat, append, branch, cancellation; 40/3 frontier checks),
[cached admission](../cached-image-admission/report.md) (12 occurrences and narrow
resource envelope), and [repaired loop](../repaired-visual-loop/report.md)
(27 generations, seven encodes, 85 historical reuses). None compared reconstructed
historical state against an independent canonical-prefill reference. Safe rejection
in E–I does not satisfy positive C2–C5/C8.

## 5. Resource, ownership and recovery boundary

Observed current admission has **two notions of historical**. L1 wire age permits
early patch deferral, but only actual complete paired coverage authorizes the
physical discount at L4. `validate_workload(C)` charges **every** uncovered image,
regardless of its message age. It already has the correct fresh-work sum once a
safe reconstruction frontier is selected. This existing part must be reused.

A changed prefix cannot inherit the old coverage discount. If all required work
fits 36,864 fresh patches, L2 can enforce it at C=0/earlier pair; if it does not,
refusal is legitimate. However, today's >36,864 all-history path requires an
existing checkpoint with full historical coverage before byte projection. It
cannot simply be bypassed for reconstruction, especially mixed covered-history +
uncovered-history requests. `projected_pair_bytes` (`paired_checkpoint.py:36–74`)
is based on an existing known layout/frontier ≥128; it is **not a cold-start
memory estimator**. A reconstruction path needing conditional historical discount
must either prove its existing safe candidate's layout/projection and actual
fresh sum, or conservatively refuse outside the qualified route until a bounded
cold/layout projection is justified. This is a narrow admission seam, not a new
allocator or a proven need for new limits.

Unchanged independent guards: 600 occurrences, 64 MiB encoded occurrence bytes,
16,777,216 source pixels (cached included), per-image validation/bytes/pixels,
16 MiB entire body, context/output, 128 stored Files/64 MiB storage, and queue
serialization. No partial patch discount. **36,864 fresh patches, 12 MiB projected
complete pair and 192 MiB retained logical roots remain unchanged in meaning.**
The latter byte ceilings currently qualify the conditional discount; they are
not universal text/cold admission limits. All-current-image transport costs remain
even on a cache hit. No deleted/offloaded occurrence requires old bytes.

`pair_bytes` counts shared backing repeatedly. Projection includes requested
output plus five speculative positions. Retained roots + active/restored handles
+ **two** pending captures coexist (historical qualification reserves
192 + 3×12 MiB logical arrays). Cold prefill arenas, Vision patches/activations,
all fresh embedding rows, transport/Recipe trees and model weights are additional;
logical byte accounting is not physical footprint. The existing source does not
provide a comprehensive cold multimodal peak estimator. This is an evidence gap,
not demonstrated OOM or a reason to raise ceilings.

Ownership reuse: L2 validates transient decoded containers on preparation, retains
encoded immutable bytes in queued requests, prepares normalized patches only on
the worker and evaluates embeddings before dropping inputs. All fresh embedding
rows coexist until append completes. L6 restores layer wrappers and releases
embedding references in `finally`; P6 seal/failure releases its own references.
L4/L7 release sources on scheduling failure, retirement and queued cancellation.
`_cancel` waits under shielding for worker mutation and settlement. Native-start
failure burns/clears/fails the engine; pre-native missing coverage rejects without
burning a healthy engine. Clearing `_sources` alone does not prove Recipe/body/
Files references retired. Reconstruction cancellation must verify every owner,
not count Python dictionaries or RSS alone.

Future Rsrc evidence must separately measure MLX active/cache/peaks, host reserve/
pressure, swap and lifetime physical high-water through Vision+cold bulk+tail+
capture+restore+retained roots; distinguish weights/allocator retention from leaks.
Include queued/running cancellation, admission boundary refusal and cleanup;
retain the **rejected** request's frontier, fresh/covered patches, projected pair,
root bytes and error before archival assertions. The request-28 trace retention
failure is a demonstrated observability gap in prior evaluation, not a new cache
subsystem requirement.

Recovery wiring: `H:llm-deepseek/src/transport.ts:22–40` classifies provider errors;
`request-files.ts` retries recognized missing/expired Files and quota/transport
cases, not arbitrary paired-byte/fresh-patch rejection. H4 requires its exact
budget error/count. Context-overflow compaction has a separate recognized policy.
Runtime-specific resource refusal has **no observed automatic official offload
recovery**. An operator can use supported `/compact`, supported route budgets and
reattachment, but success is not guaranteed: the summary request itself may need
historical-image reconstruction or exceed resources. Do not retry uncertain
output/tool effects or translate arbitrary 400 into `IMAGE_OFFLOAD_REQUIRED`.

## 6. Independent numerical reference strategy (not executed)

A practical reference is the **pinned ordinary layer-loop prefill** (`O:Model.get_input_embeddings`
→ `LanguageModel._forward` with full per-layer computation, no CED/deferred decoder
cone) on **fresh caches**, under the same admitted Flash-1017 activation/profile
patches, plus native `dspark_append_context`/committed rings. This bypasses paired
lookup and the candidate's DwarfStar/P6 scheduling. It is an existing independent
execution path, not a second production selector or a universal official oracle.
The official checkpoint/DeepInfer precision semantics remain correctness authority.
First establish this reference's agreement with the accepted bounded cold baseline
before using it to approve changed-history recovery; no complete 40/3 multimodal
numerical comparison is claimed to exist already.

Reuse `tools/run_prefill_fp8_mlx_p6_matrix.py:221–249` matched-control comparisons,
`tools/qualify_prefill_fp8_mlx.py` ordinary-Attention source/index control,
`tools/qualify_vision_reference.py` official Torch-vs-MLX Vision comparisons,
`tools/accept_mtp_vision.py` ordinary HTTP joins, and `tests/test_production_mtp.py`
immutable snapshot/wrapped-ring checks. Their current checks alone are insufficient:
P6 matrix source/index comparisons cover selected slots, Vision reference checks
tower/aligner, and HTTP acceptance checks frontiers/output, not every paired value.
Extend these finite existing collectors only after separate approval; no new test
framework is justified.

Proposed comparison contract to freeze **before acceptance**, not inferred PASS:

1. Save exact wire Messages and Recipe conversion/rendered source order, canonical
   IDs, expanded `(start,length,digest,grid,kinds)` and numerical/profile identities.
   Both candidate and reference consume that **same current projection**, not the
   old pre-offload history or a reconstructed guessed transcript. Reference has
   no candidate cache lookup. Compare prepared image patches/layout exactly;
   compare independently prepared embeddings per occurrence, including delimiters,
   aligned by absolute expanded positions. Then a shared-embedding diagnostic can
   isolate language-state disagreement, but cannot replace independent preparation.
2. At candidate reused C, N-2 capture, N-1/P6 seal, P5 post-terminal N, and a bounded
   generation continuation, compare all **40 Target layers**: seven slots,
   offsets/layout/padding/lengths, window physical position, compressed KV/index,
   compressor tails and Engram history. Compare all **3 DSpark rings**' keys,
   capacity, offsets and absolute-modulo physical order. Normalize only explicitly
   unused padding, never meaningful rows or ring rotation to hide a mismatch.
3. Compare logits at matched canonical frontiers and teacher-forced short token
   continuation, then bounded greedy continuation/semantic completion. Avoid RNG
   equality as oracle. A margin/tie-sensitive token divergence requires diagnosis,
   not automatic tolerance enlargement. Record proposal/rollback/commit and
   P5 terminal consumed once; settled frontier agreement alone is insufficient.
4. Exact equality is required for IDs, digests, source origins/spans, masks/kinds,
   integer histories/offsets, shape/dtype/layout and immutable snapshot stability.
   For same-operation snapshot restore, exact value equality is justified.
   Different FP8/BF16 operation/chunk schedules need **predeclared numerical
   tolerances**, not assumed bit equality. Existing Vision reference bounds
   (`qualify_vision_reference.py:108–116`: fp32 RMS-relative <0.0002/cosine >0.99999;
   bf16 <0.1/>0.995) are scoped cross-backend Vision checks, **not** blanket
   tolerances for Target KV, rings or logits. Establish per-state max-absolute,
   RMS-relative and cosine envelopes using unchanged accepted cold/append controls
   against this reference first; reject unexplained error. No defensible universal
   Target/ring tolerance value can be selected from source alone.
5. Detect wrong hits/history substitution by same placeholder IDs + different
   image bytes, order changes, changed text before images, shortened projection,
   and old-cache roots still resident. Log selected saved prefix/image metadata,
   actual restored C and fresh encode count; compare candidate state against the
   fresh current-projection reference. A forged hit with plausible text fails.

The smallest unresolved numerical gate is independent baseline establishment and
per-slot tolerance calibration using the existing controls, **before** changed
path acceptance. This is explicit future verification, not an identified kernel
defect or authorization to execute it now.

## 7. Feasibility of the official three-round workload

**Source-feasible with supported configuration; not executed or installed-version
qualified.** Default count 600, raw Files 128 MiB and base64 fallback 20 MiB do not
make nine images automatically offload. Supported DeepSeek provider configuration
can set `maxImagesPerRequest: 3` and **`imageOffloadCountQuantum: 1`**. Both accept
positive integers including one; the default quantum 20 must not be left above
a lowered count bound. Keep byte thresholds at supported values or, if deliberately
lowered, also keep each byte quantum ≤ its bound. Count-only reduction is simpler
and independent of compression variance. It uses the **same** H2–H5 selection,
durable event and projection semantics, not fake events or Runtime overrides.

With three distinct acquisitions per round and cached continuations, the fourth
retained occurrence triggers official offload of the oldest, leaving three. During
round two this can leave surviving **historical** occurrences after a rewritten
earlier prefix: a concrete reconstruction case, not just an all-text miss.
Offload happens when a request exceeds the supported threshold, **not an arbitrary
operator offload command between rounds**. Smallest clarification to the acceptance
sequence: allow the next round's first acquisition to cause offload, then observe
continuation; do not promise a manual offload operation. Cache continuation before
the threshold crossing demonstrates reuse; reacquisition in round three is a new
occurrence, possibly causing another real offload.

Official mounted `command-compact` exposes `/compact` → H6 while agent idle and
without waking queued work. `retainTokens=0` still keeps a tail node and walks back
to balanced tool boundaries; one trivial exchange may have no useful range or a
summary not smaller than source. The planned multi-tool/image rounds should offer
a range, but this is not guaranteed by the word “round.” Freeze a finite set of
completed exchanges with enough ordinary text/tool/image content for a smaller
summary; if selection returns null or the real summary is too large, report that
transition unexercised. The smallest adjustment is an explicit bounded additional
completed exchange/text budget, not fabricated summary/history. The summary's
own request must be included in generation/resource budgets and may need F first.

Proof reconstruction was necessary: retain event/wire snapshots before/after
`image/offload` and `surfaceOp: replace`, Recipe token/span digests, selected pair
C and image metadata. For each required lifecycle transition show at least one
current image beyond compatible C (or C=0) actually encoded and complete new state
compared to §6. Old roots may remain resident: exact token/image mismatch must
exclude them; a useful earlier text hit is allowed but must not mask uncovered
image computation. No cache clearing, Runtime admission override, artificial
Messages rewrite or replayed tool effects is needed to create that distinction.
If all surviving images are covered or removed, that transition does not qualify
the required multimodal reconstruction case even though the request succeeds.

Freeze actual Harness/plugin versions/configuration, attachment normalization
and request-byte identities, Recipe/native/oMLX/Runtime/profile identities and
all Phase 0 stop/budget rules before future execution. Retain current supported
Harness exclusions (`session-log-deepseek.enabled=false`, no unsupported tool
removal) and a declared image-capable route; these are not lifecycle replacements. Source feasibility is not
proof that the installed CLI has these exact plugins enabled or that model-directed
reacquisition/summarization will complete within the finite budget.

## 8. Ranked minimum gaps and Phase 2 recommendation

1. **Necessary: qualify ordinary independent-request reconstruction, then replace
   only the uncovered-historical coverage prohibition with a proved execution
   decision.** Reuse L2/L6 from a compatible earlier whole-span pair or fresh C=0,
   consuming only the current projection. Preserve exact token/image/profile
   checks and actual restored-count reporting. Never extend this to lost retained
   Chat leases, uncertain native mutation or tool-effect repair.
2. **Necessary: admission integration for that selected frontier.** Reuse
   `validate_workload(C)`; account uncovered old-message images as fresh. Preserve
   all limits, pre-clone conditional-byte checks and root/capture coexistence.
   For no-pair/cold or mixed historical reconstruction, qualify the reachable
   resource route explicitly; do not pretend `projected_pair_bytes` works without
   a qualifying layout or that historical wire age buys a discount. A bounded
   refusal remains valid outside the admitted envelope.
3. **Necessary qualification, not a new engine: establish §6 independent baseline
   and numerical comparisons**, including partial-image fallback to an earlier
   pair/miss. Existing source is sufficient to identify the seam, not to approve
   guard deletion. Reuse existing finite verification infrastructure.
4. **Necessary evidence integration: retain reconstruction and pre-start rejection
   diagnostics** (selected frontier, identity reason, work sum and both byte
   values) through existing trace/acceptance owners. This closes the demonstrated
   archival ambiguity and makes false reuse/resource failure distinguishable.
5. **Later lifecycle qualification, not Runtime history implementation:** official
   configured count offload, actual `/compact`, real `read_image` reacquisition,
   incomplete protocol/cancellation and effect fences. No source-grounded need
   for a new Scheduler, cache subsystem, image manager or session framework.

Smallest justified Phase 2 scope is items 1–3 with the minimal trace support in 4,
limited to ordinary native Messages on the declared numerical profile. Establish
reference first; qualify cold historical input, earlier-text-boundary append,
changed bytes/order, shortened prefix and whole-image fallback separately. Keep
unsupported/missing inputs and physical refusal explicit. Do not start Phase 3's
full Harness campaign or Phase 4's resource acceptance implicitly.

### Blockers and remaining uncertainties

- **Project gap:** positive F/G/H/I continuity is guarded, despite reusable inputs
  and execution components. No demonstrated upstream architectural blocker to
  current-projection recomputation was found.
- **Upstream limitation:** D1 does not wire cache frontier into conditional cached
  image preparation; newer official DeepInfer HEAD does not fix it. This limits
  reference claims, not local ability to use its own paired frontier.
- **Missing prerequisite:** unavailable current image bytes block reconstruction;
  retained attachment bytes and Runtime paired state have independent lifetimes.
- **Upstream recovery absence:** arbitrary Runtime-specific admission refusal has
  no automatic offload protocol. C9 must disclose operator/client intervention;
  it cannot promise automatic recovery or turn physical refusal into positive C8.
- **NOT YET VERIFIED:** full independent 40/3 numerical baseline/tolerances,
  changed-projection physical peaks/cleanup, installed package/plugin equivalence,
  actual partial-image candidate reachability, and finite summarizer/reacquisition
  success. These are bounded future checks, not asserted implementation defects.
- DeepInfer generic service prefix-hit discovery and universal multimodal
  preemption reconstruction were not established by the bounded caller inspection.
  Neither is required to replace local paired ownership with another framework.

Phase 1 exit is satisfied by identified guards, reusable caller-wired execution,
explicit cases, independent-reference strategy and supported lifecycle controls.
C1's broader proof and C2–C9 remain unqualified. No follow-on implementation is
authorized by this report.

## 9. Documentation validation

Validation recorded after the documentation edits. The shell initially had no
`python` command; the requested Python checks were then run successfully with
`PATH="$PWD/.venv/bin:$PATH"` (also passed with system `python3`). No environment
contents were changed.

- `git diff --check`: PASS.
- `python tools/check_repository_self_containment.py`: PASS.
- `python tools/check_authority_labels.py docs/multimodal-runtime-plan.md artifacts/multimodal-runtime/phase1-gap-audit.md`: PASS (wording only).

No real-model tests, builds, numerical experiments, Runtime startup or acceptance
workload were run. The focused documentation commit is reported by the task's
final response; it cannot contain its own Git commit SHA.
