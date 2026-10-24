# Phase 4 / Gate Q1 — visual session capacity and endurance envelope

**PASS — capacity investigation, not new inference qualification.** Inspected at
`2865c5d230b7017e6dad811d64a5f3110307fb55` on `Visual-Feedback-Loop`.
No conversation, rendering, Harness launch, dependency installation or numerical
inference was performed. Original Phase 3 evidence is unchanged.

**Decision:** the measured 1280×720 image geometry permits **seven complete image
occurrences per request**, conditional on all other admission and paired-state
requirements. **Six are observed with real inference; seven is model-free
admission only. Eight fail the aggregate ViT patch budget.** Uninterrupted
10–20-screenshot sessions at this geometry are not supportable today. This is
not an image-count limit independent of resolution, nor a memory guarantee.

## 1. Authoritative baseline and scope

Use the successful external-renderer section of the [Phase 3 report](../artifacts/visual-feedback-loop/phase-3-report.md),
not its earlier blocked attempts or Gate P's incomplete trace set. Evidence root:
[`phase-3-evidence/visual-loop/`](../artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/)
(abbreviated **E** below).

- Official Harness **0.2.1-rc.2**, Node v24.5.0, 17 matching package identities.
- Runtime **0.23.0**, qualified provenance
  **`2fdea4fe7367e540300d7c81d4fed402d716a274`**, Recipe **0.2.0**.
  Execution identity `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`;
  all 113 installed Runtime Python sources matched the qualified checkout.
- Original Flash-1017 revision `1bae0e1088993b9b82de08f59de5be5a34e42170`,
  `flash-1017-pre-woa-fp8-v1`, Thinking OFF, native GLM and Target/DSpark paired MTP,
  ordinary `mtp-serving-v1`. The admission identity fixture's singleton label is
  not the actual serving profile or its limits; see E/runtime.py and report.
- macOS/M3 Ultra 512 GB, static Web, deterministic 1280×720/DPR1 PNGs;
  Chromium outside Seatbelt, official tool-image/native Messages inline fallback.
- **21 generations, six screenshots, six fresh encodes, 53 historical reuses,
  zero forbidden replay/repack, complete 21/21 traces**. Four explicitly grounded
  correction cycles; independent rubric 1/9 → 9/9, with literal-goal shortfall
  and regressions disclosed in the original assessment.
- Conversation 193.14 s, 20 tools, 5,804 protocol output tokens. E/identities.json,
  runtime-admission.json, source-package-match.json, trace-retention.json,
  audit-results.json and independent-review.json remain the receipts.

W2 is later **model-free implementation evidence**, not part of this live inline
qualification. It is a bounded in-memory prototype, not finalized session-owned
Files. No Files support is removed or newly qualified here.

## 2. All 21 requests: measured trajectory

Sources: E/wire.jsonl, prepared-images.jsonl, cache-summary.json and lossless
`trace-00.raw.json.gz` through `trace-20.raw.json.gz`. Joined by generation order
and exact body SHA-256. Wire IDs include intervening Files attempts (18 HTTP 404s).
The existing audit's identity/prefix/settlement checks are reused as the review
criteria; its `main()` was **not rerun**, since it writes into original evidence.
Read-only extraction checked all 21 body joins, decompressed trace SHA-256s,
paired-frontier continuity, settlement/publication, zero replay/repack and geometry;
aggregate results were six fresh encodes, 53 reuses and final frontier 15,230.

H/F = historical/fresh occurrences, also measured reuse/encode counts.
Image B = sum of source container bytes in that request (calculated from measured
wire lengths). Prompt is the **expanded** prepared prompt. Cached/suffix are
measured token spans, not bytes. Settled is the common frontier of all **40 Target
and three DSpark offsets**. Every row has settled authority, retired prediction,
empty queues, native `message_stop`, cache publication and zero prompt/quiescence
history replay/full repack. No row is a partial trace.

Times are seconds: prep = `request_prepare_s`; prefill = `prefill_handoff_s`
(includes suffix append/P5, not an isolated kernel benchmark); TTFT =
`total_ttft_s` (first canonical token relative to Runtime t0, not HTTP headers);
decode = accumulated `next_token` calls, including the first call; total =
Runtime `elapsed_s`. These timings overlap and must not be added together.
Request normalization/Files/render latency is not separately attributed by these
Runtime fields. Timestamp subtraction from fetch to native stop gives 2.843–27.941 s
per request (first 19.095, last 9.391), not full task latency.

|G / wire|H+F|Body B|Image B|Prompt|Cached / suffix|Settled|Prep|Prefill|TTFT|Decode|Total|
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
|1 / 1|0+0|14,326|0|2,633|0 / 2633|3099|0.0223|7.985|8.203|10.789|19.075|
|2 / 2|0+0|16,069|0|3,177|3099 / 78|3270|0.0131|0.514|0.598|2.324|2.932|
|3 / 3|0+0|16,813|0|3,434|3270 / 164|3521|0.0144|0.732|0.809|2.280|3.101|
|4 / 5|0+1|65,126|35,329|4,397|3521 / 876|5268|0.0342|2.339|2.807|21.839|24.768|
|5 / 7|1+0|67,890|35,329|5,346|5268 / 78|5440|0.0155|0.499|0.579|2.253|2.841|
|6 / 9|1+0|68,637|35,329|5,603|5440 / 163|5690|0.0160|0.746|0.827|2.012|2.847|
|7 / 11|1+1|119,082|72,257|6,566|5690 / 876|7564|0.0186|2.241|2.690|24.926|27.785|
|8 / 13|2+0|122,515|72,257|7,642|7564 / 78|7738|0.0195|0.501|0.583|2.317|2.914|
|9 / 15|2+0|123,270|72,257|7,896|7738 / 158|7983|0.0243|0.727|0.815|2.050|2.879|
|10 / 17|2+1|173,687|109,166|8,855|7983 / 872|9828|0.0286|2.323|2.786|24.973|27.930|
|11 / 19|3+0|177,038|109,166|9,906|9828 / 78|10000|0.0232|0.507|0.593|2.330|2.939|
|12 / 21|3+0|177,788|109,166|10,161|10000 / 161|10248|0.0264|0.734|0.827|2.067|2.907|
|13 / 23|3+1|228,769|146,498|11,126|10248 / 878|11462|0.0304|2.269|2.730|10.436|13.235|
|14 / 25|4+0|230,286|146,498|11,526|11462 / 64|11705|0.0257|0.458|0.549|4.824|5.400|
|15 / 27|4+0|231,042|146,498|11,769|11705 / 64|11963|0.0261|0.452|0.543|4.823|5.394|
|16 / 29|4+0|231,807|146,498|12,027|11963 / 64|12120|0.0308|0.453|0.550|2.352|2.916|
|17 / 31|4+0|232,556|146,498|12,280|12120 / 160|12367|0.0251|0.767|0.856|2.074|2.944|
|18 / 33|4+1|283,393|183,722|13,246|12367 / 879|13792|0.0349|2.324|2.794|14.685|17.566|
|19 / 35|5+0|285,325|183,722|13,856|13792 / 64|13950|0.0326|0.460|0.558|2.348|2.921|
|20 / 37|5+0|286,078|183,722|14,110|13950 / 160|14197|0.0351|0.740|0.840|2.083|2.937|
|21 / 39|5+1|337,023|221,025|15,074|14197 / 877|15230|0.0367|2.381|2.855|6.481|9.380|

### Image geometry and incremental work

The six raw PNG lengths are **35,329; 36,928; 36,909; 37,332; 37,224;
37,303 bytes**. Source = wire = Recipe digest for every occurrence. Every image
contains **921,600 source pixels**, has a resolved **52×92 patch grid**,
**4,784 planned ViT patches**, and **578 expanded image positions**. The latter
are a 18×31 aligned feature grid plus row delimiters and two boundary positions:
`18*(31+1)+2=578`. Patch padding is 728×1288 at patch size 14, not extra source
pixels. Distinguish these three counts.

|Images present|Source pixels (calculated)|Planned patches (calculated)|Expanded image positions|
|---:|---:|---:|---:|
|0|0|0|0|
|1|921,600|4,784|578|
|2|1,843,200|9,568|1,156|
|3|2,764,800|14,352|1,734|
|4|3,686,400|19,136|2,312|
|5|4,608,000|23,920|2,890|
|6|5,529,600|28,704|3,468|

This table plus H+F gives geometry for every request. Span starts stay exactly
**3813, 5982, 8271, 10542, 12662, 14490**, each length 578. All historical spans
are wholly covered by the restored pair; only the fresh span encodes. Fresh
Vision times at G4/7/10/13/18/21 are **0.368/0.366/0.368/0.366/0.370/0.372 s**
(rounded), 2.210 s total. Historical-only requests report zero encodes; small
`vision_s` bookkeeping values are not historical Vision execution. Preparation
still validates/decodes supplied containers, including historical images.

Preparation totals 0.533 s across the 21 requests (range 0.0131–0.0367 s);
prefill/handoff totals 30.151 s; decode calls total 150.266 s. The largest
latencies accompany longer model responses, not a demonstrated six-image
capacity slowdown. No extrapolated tokens/s, memory growth rate or endurance
failure probability follows from this short, variable-output run.

## 3. Independent limits and the first binding resource

Source owners: `runtime/multimodal.py:prepare_multimodal`,
`model_execution/processing.py:plan_image_grid/image_token_kinds`,
`serving/capacity.py`, `serving/production_mtp.py:_schedule_waiting/_enqueue`,
`serving/paired_checkpoint.py`, and [capabilities](capabilities.md).
Read-only inspection of the baseline checkpoint's `config.json` confirms
`text_config.max_position_embeddings=1048576`, patch size 14, downsample ratio 3,
minimum pixels 295936 and processor maximum image tokens 1024 (Runtime separately
enforces the stricter 1014-position bound). No checkpoint weights were loaded.
Harness references below are installed official rc.2 `@deepseek-ai/*/lib/index.js`
under `~/.local/share/deepseek-harness/node_modules/`, correlated with retained
identities and E/composed-config.yml; no installed files were edited.

|Independent boundary|Current rule and measured/prospective consequence|
|---|---|
|Raw Messages JSON|16,777,216 bytes for **the whole body**, including schemas, text, JSON and base64. Last body 337,023 (2.01%); remaining 16,440,193 bytes. No universal image-count conversion because text can grow independently.|
|Harness inline payload|`dsh-llm-deepseek` default 20,971,520 **base64 payload** bytes, not raw PNG bytes; `dsh-llm:requiredImageOffload` counts occurrences using `4*ceil(bytes/3)`. Six images total 294,708 base64 bytes. Runtime whole-body ingress would bind earlier for sufficiently large payloads.|
|Per-image encoded/pixels|Runtime ≤16,777,216 encoded bytes, ≤4,194,304 source pixels, aspect 0.5–2, supported single-frame PNG/JPEG/WebP, high/auto, ≤1014 expanded positions. All six satisfy these; largest PNG 37,332 B. Harness request-image normalization defaults to 2 MiB/4096-side bounds; actual original variants survived unchanged.|
|Aggregate encoded bytes|67,108,864 bytes, **including historical occurrences**. Six use 221,025 B; seven at measured maximum size use ≤261,324 B. Not binding.|
|Aggregate source pixels|16,777,216. `floor(limit/921600)=18`; 18 use 16,588,800, 19 use 17,510,400 and fail. This is **not** the admissible image count.|
|Aggregate ViT patches|36,864. `floor(36864/4784)=7`. Six use 28,704; seven 33,488; **eight 38,272 → reject**. First deterministic resource boundary for growing same-geometry history with bounded text/bytes and available pair.|
|Expanded image positions|578 each, seven 4046 (net expansion over seven placeholders 4039). No separate ordinary 8192-position/four-image cap. Per-image 1014 is not exceeded. Text/tool positions still count.|
|Context + output|Actual expanded prompt + reserved output must fit checkpoint context; evaluated route 1,048,576, output reservation 2600. G21 uses 15,074+2600=17,674, leaving 1,030,902 positions. Final settled frontier 15,230 is a different quantity. Paired capture must also remain extendable (`C+3≤1,048,576`). This investigation does not qualify maximum-context execution.|
|Historical paired state|A compatible retained pair must cover the last historical image end; bytes alone are insufficient. Current production cache has 18 complete immutable pair slots plus null block, shared with logical-session pins; active/restored handles are additional. It is **not 18 images**, nor an 18-request lifetime. Sequential publication can evict older roots while preserving a usable latest root, but competing conversations/restart can remove it. No Messages session pin/lifetime guarantee.|
|Harness attachment/history|Default maxImagesPerRequest 600, raw Files-path budget 128 MiB; inline fallback rechecks 20 MiB. No override lowering these is present in Phase 3 composition. Offload counts occurrences, not unique attachments. Compaction/projection can bind semantically before resource exhaustion; see §4.|

For seven images at the measured maximum PNG size, base64 is ≤348,432 bytes.
Holding G21's non-base64 body remainder (42,315 B) fixed gives **390,747 B**;
this is an illustrative estimate, not a serialized seven-image continuation
(the added tool/result/text JSON must also be counted). A 577-position net
expansion for one additional image is exact; future surrounding conversation
length and decode latency are unknown. No fixed maximum number of text-only
requests or minutes can be inferred from image count.

### Model-free boundary check performed for Q1

Used the existing disposable Python, checkout `prepare_multimodal` and
`ModelConfig(vision_n_layers=1, max_seq_len=1048576)`; no model instantiated.
Sources were the six retained PNGs followed by copies of render-00 and render-01.
`image_patch_array` was replaced in-process with an assertion failure so no
normalized patches could be materialized. Input tokens were
`[7] + [image_token_id]*n + [8,9]`, bytes sources wrapped with `kind='bytes'`.
This is a geometry/admission check, **not** native Messages history or inference.
An initial invocation supplied raw bytes rather than source objects and failed
with AttributeError before admission; corrected to the existing source contract.
No production code/test infrastructure or evidence files were modified.

```text
6 ADMIT 3471 prompt positions, 28704 planned patches
7 ADMIT 4049 prompt positions, 33488 planned patches
8 REJECT invalid image: aggregate image patch budget exceeded
```

The prompt counts here include three text tokens, not Phase 3's conversation.
Repeated bytes still consume occurrence budgets. No backend, encoder, HTTP or
cache lookup executed. **Paired-state outcome: NOT TESTED**; these inputs have no
required retained pair and cannot be called successful resumed inference.
No broader test harness/addition was needed to establish the source boundary.

## 4. History and paired-cache lifecycle

- `dsh-tool-fs`/attachment service retain attachment references in durable tool
  results. `dsh-llm-deepseek:prepareImages` projects offloaded occurrences,
  reads each attachment's request variant, then checks occurrence budgets.
  Normalization is request-time and model-target-dependent. Reopening a path
  or retaining source bytes does not prove identical request bytes, span start,
  grid, digest, canonical tokens or executable pair.
- `dsh-compaction-image-offload` logs `image/offload` for oldest occurrences on
  `IMAGE_OFFLOAD_REQUIRED`; the next request uses placeholder text. It preserves
  attachment ownership/history records, **not the canonical numerical prompt**.
  Runtime's patch-budget rejection is not that Harness error and does not itself
  provide offload recovery. Inline/count offload thresholds are far above seven.
- `dsh-compaction-basic` summarizes an old region and retains a recent tail,
  balancing tool pairs. Default threshold ratio .8, headroom 65,536; for the
  1,048,576 route and 2600 output reserve, pressure threshold is
  `floor(min(.8*1048576,1048576-2600-65536))=838860` estimated tokens.
  This is Harness pressure accounting, not Recipe token admission. It changes
  the prefix and may itself call the model; it was not exercised here.
  Manual `/compact` is not transparent KV-preserving continuation either.
- E/composed-config.yml also enables tool-result pruning (8192-char threshold,
  4096 head/1024 tail) and spilling (12500 inline tokens). These are further
  possible history changes, not solutions to aggregate image preparation.
  Phase 3's retained canonical-prefix checks establish no incompatible change
  for its actual requests, not universal projection compatibility.
- Runtime image cache keys include `(start,length,digest,grid)` in addition to
  canonical tokens and execution identity. `validate_prefix` rejects intersected
  spans/changed committed identity. `PairedCheckpoint.restore` checks exact
  prefix and images; Scheduler requires restored frontier ≥ historical frontier
  or explicitly rejects **“historical image pair unavailable; image
  replay/re-encode forbidden”** before native start.

**No existing safe general mechanism was found to shrink this live image history
while guaranteeing uninterrupted paired continuation.** If projection removes an
early image but leaves later historical images, a shorter compatible prefix will
ordinarily not cover them. If all old images are removed, a fresh miss might be
admissible, but that does not authorize replay/repacking of the old conversation
or qualify compaction. Do not silently delete images, downgrade them to text,
re-encode history or blindly retry on pair loss. A new conversation/checkpoint is
explicitly new inference with lost conversational context, not continuation.
Existing handling supports the demonstrated bounded sessions without changes;
it does not support arbitrary extended screenshot accumulation.

## 5. Headroom and practical operational limits

After Phase 3's last request: one same-geometry occurrence fits before the next
image fails; **8160 patches remain**, seven would leave only **3376**. Pixel
headroom is 11,247,616, but patches make it unusable for additional full-size
screenshots beyond seven. Seven images are 6,451,200 source pixels. Admission
capacity is conditional on text/body limits, stable normalization/history and
pair availability; only six-image execution is qualified.

The run ended naturally, not at capacity, using 21/24 request slots and all six
render slots of its application budget. Those are experiment limits, not Runtime
limits. No image/cache memory resident/peak bytes, allocator/wired high-water,
thermal behavior, long idle recovery, cancellation, eviction pressure or hours-
long latency trajectory was retained. The 512 GB machine does not make these
limits optional or establish safe memory consumption. Historical reuse saves
Vision execution, **not** JSON transport, base64 decoding, container validation
or aggregate admission charges.

## 6. One recommended Gate Q2 experiment (not authorized/executed here)

**One bounded macOS official-Harness inline conversation**, same admitted
Runtime/Recipe/Flash-1017/native paired MTP/Thinking OFF identities and supported
option exclusions. Retain the existing official tools and external-renderer
handoff, sandbox/network restrictions and full trace-retention procedure; no
Files API, Windows, persistent KV, new agent framework or compaction dependency.
Recheck installed identities before startup; do not treat this checkout's W2 as
renewing the qualified payload.

Concrete task: repair a static single-page conference operations dashboard in
`index.html`: overview cards, a dense schedule table and a registration summary,
under a shared header with a visible status footer. Supply fixed realistic
content, including long labels, and visible goals (readable text, aligned columns,
no clipping/overlap, consistent spacing and footer visibility at 1280×720).
Seed genuine layout defects, not instructions to make repeated edits. Table
wrapping, card heights and footer placement interact: correcting one section
can regress another, giving real reasons for repeated whole-page verification.
Do not supply diagnosis/patches/rubric scores through renderer logs. Reuse the
existing single-artifact renderer and tools, no scripts or network; freeze
independent per-section acceptance rubrics before starting.

- **Maximum seven total screenshot observations**; no eighth capture/read.
  One initial view and up to six useful edit/verification observations, selected
  by the model only to address actual remaining problems or regressions. Natural
  early completion is allowed; no minimum meaningless edits or forced seven-image win.
  Seven is below patch/pixel/body/count/context limits, but is the admission
  boundary count, not yet live qualified. A shorter run cannot establish seven-
  image endurance and must be reported as such.
- 1280×720/DPR1, lossless single-frame PNG, same deterministic fonts/renderer.
  Require readable text and no blank frame. Target ≤512 KiB **per image**; stop
  rather than silently recompress/resize history if this fails. Seven at this
  bound use ≤4,893,364 base64 bytes, well below 20 MiB. Pre-fetch cap the **whole
  serialized body at 8 MiB**, so text/JSON cannot consume hidden ingress headroom.
- One conversation, ≤32 generation requests, ≤45 admitted tool effects, ≤2600
  output tokens/request, ≤12,000 aggregate protocol output tokens, ≤65,536 actual
  expanded prompt positions/request. Prompt+reservation ≤68,136. Bound workspace
  source to 65,536 bytes; keep tool-result logs below pruning/spill limits.
  Use existing observer/budget facilities; verify exact prompt from preparation.
  Do not summarize/prune to meet a cap.
- ≤5400 s total including ≤1200 s startup; ≤600 s generation/stream idle;
  same 25 s request/22 s renderer deadlines. No automatic generation/effect
  retry. Stop on time/token/request/tool/body limit, render uncertainty, changed
  historical identity, offload/compaction/projection drift, unavailable pair,
  nonzero replay/repack, frontier mismatch or incomplete native settlement.
- For every request retain whole-body size/hash, normalized source bytes/hash/
  dimensions, source revision, occurrence location/order, expanded span/grid and
  historical flags, prompt/output reservation, cached/suffix spans, exact restored
  and all settled Target/DSpark offsets, fresh encodes/reuses, Vision/preparation/
  prefill/TTFT/decode/total timing, canonical stop, publication/queue retirement.
  Require complete trace count including last response, flush before backend
  destruction. Run alone (no deliberate competing cache-pressure campaign).
- Observe host process RSS/available Runtime memory telemetry at request
  boundaries if already available; record absent peak/wired attribution honestly.
  Do not add intrusive cache instrumentation merely to claim memory qualification.
- Independent assessor reviews actual transmitted screenshots, source revisions
  and diffs after completion: score each section, identify observation → edit →
  verification chains, substantive corrections and regressions, whole-page
  consistency and false completion. Tool success/model self-report is not quality
  acceptance. No score or corrective feedback reaches the model mid-run.
- Separate outcomes: natural native `end_turn`/Harness completion before a cap
  (possibly incorrect quality); externally budget-stopped; resource-admission
  rejected; paired-cache failure; renderer/tool failure; incomplete evidence.
  A `max_tokens` stop or natural response explaining a capacity blocker is not
  natural successful task completion. Preserve the exact stop trigger and last
  complete screenshot/request; do not relabel a seven-shot budget stop as an
  observed eight-image Runtime failure.

Q2 can test one extra image and longer bounded multi-section work, not 10–20
uninterrupted observations. A later different-resolution campaign would need its
own geometry/readability analysis and acceptance; this task does not propose a
Vision limit change.

## 7. Explicitly unresolved

1. **Seven-image resumed execution:** offline admission cannot prove physical
   execution/memory/latency or correct historical reuse. Smallest next live check
   is Q2's seventh useful observation with all six earlier spans covered and all
   seven-image traces retained. If the task ends earlier, leave this unqualified.
2. Long-idle, cancellation, branch/fork, competing conversations and eviction
   behavior under visual workloads remain unqualified; source says reject on
   missing pair, not that it will remain available. These need separate bounded
   experiments, not a wider Q2.
3. Maximum context and physical memory envelope cannot be inferred from this
   15k-token frontier or synthetic metadata preparation. No performance claims
   for 1M-context multimodal execution are made.
4. Request-time Harness normalization/attachment preparation and Files-fallback
   time lack separate measurements; smallest remedy, if latency attribution
   becomes material in Q2, is timestamps around existing preparation/fallback,
   not a new processing engine. Phase 3's raw variants were stable.
5. Safe offload/compaction continuation is not established. Byte retention alone
   cannot resolve the canonical-prefix/paired-state ownership problem.
6. Files full implementation/redesign, session-scoped volatile ownership,
   completion/cancellation cleanup, active request snapshots, forks, expiry/cache
   eviction rules, Windows qualification and `dsh_session_log` decisions belong
   to a **separate campaign**. Process-local W2 storage is not session ownership;
   file-byte lifetime must remain independent of paired KV validity. The known
   Windows failure remains unresolved.

No capability promotion, production modification, Gate Q2 execution or automatic
follow-on campaign is implied by Q1 PASS.
