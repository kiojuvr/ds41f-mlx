# Bounded ordinary Vision → native MTP seam

Status: **CLOSED** — bounded ordinary Vision integration into the existing paired
MTP production path, with real-image delta acceptance. The separate promotion
and delivery gates listed below are not part of this closure.

DeepInfer follow-up: request preparation is now metadata-first. Full encoded-image
validation remains, but normalized patches are created on the worker only after
paired lookup, for uncached spans. See [development notes](deepinfer-development.md#metadata-first-cache-aware-vision-preparation)
for the focused verification; the historical model-acceptance receipt below was
not regenerated.

Scope: checkout production implementation and affected-boundary acceptance, not
new release admission, Full R1, packaging, a new default, or removal of OFF.
At the original Vision delta the installed package remained unchanged; subsequent
local delivery is recorded in [runtime closure](runtime-completeness.md).
Application image-acquisition UI and the
isolated PDF-observation policy were not changed by this API delta. Their later
capability-driven promotion is recorded in historical application Vision integration (`c1d1f716:docs/web-mtp-vision-integration.md` in Git).
That application is now retired; direct bounded Runtime Vision is unchanged.

## Code audit: pinned oMLX 0.7.0 versus ordinary ds41f

The admitted MTP environment's `omlx/patches/deepseek_v41/` owns these paths:

* `model.py:Model.get_input_embeddings` embeds IDs, runs checkpoint ViT/aligner,
  adds delimiters and merges image spans. `Model.__call__` passes those embeddings
  to the existing language model only during image prefill.
* `language.py:LanguageModel._forward` derives image masks for hashing/Engram/MoE,
  and captures mean hyperconnection streams **after Engram, before each selected
  target layer**. Its returned DSpark hidden concatenates the configured taps.
* `mtp.py:DSparkMixin.__call__` primes DSpark from those same target hidden rows.
  Verification expressly requires a text decode row with no `inputs_embeds`.
* `mlx_lm_mtp/deepseek_v4_dspark.py:capture_prompt` appends native bounded rings
  at the target's absolute offset. `install_committed_context` validates a common
  target/ring frontier; the native BatchGenerator consumes that context.
  Scheduler's separate generic `vlm_mtp` drafter route is **not** DeepSeek's
  qualified DSpark topology and is not introduced here.

Ordinary ds41f already replaces generic Scheduler prefill with DwarfStar/P6:
selected layer wrappers tap the same post-Engram/pre-layer mean streams, native
`dspark_append_context` reaches the target frontier, N-2 captures a pending pair,
N-1 seals a genuine append, and P5 transfers the held-out terminal to the sole
Scheduler batch. This does not need upstream full-prompt priming or VLM decode.

Actual missing connections were:

1. Ordinary admission/response rejected all recipe image sources despite existing
   inline preparation and expanded-capacity resolution.
2. `_start` did not pass existing sparse image embeddings into P6. P6's same-live
   `continue_from_commit` also needed an optional embedding argument between its
   existing segments/capture boundaries.
3. `MultimodalInput.encode_new`'s default OFF path requires an OFF resource lease
   and first-party `encode_image_span`; pinned MTP Model has neither. MTP now
   supplies a checked pinned-Model span adapter to **its existing**
   `get_input_embeddings` API. This runs Vision for new spans, not language
   forward, and adds no Vision math or alternate weight owner. The default OFF
   resource-admission path is unchanged. The pinned and OFF `vision.py` are
   identical; CPU patch preparation differs only in deferring MLX conversion to
   the worker (plus a legacy diagnostic prompt import).
4. Token IDs alone cannot distinguish different image bytes. Existing paired
   payload metadata now binds digest/grid/span, and upstream PagedCache's existing
   `extra_keys` / `extra_key_ranges` discriminate those same identities. No new
   index, eviction policy, cache layout, packing or persistence is added.

An initial real-image attempt exposed missing OFF-resource admission on the
pinned MTP model, before Vision/target mutation. The span adapter closes this
interface mismatch rather than pretending an MTP model owns an OFF lease.

## Connected execution and ownership

```text
ordinary /v1/chat/completions
 → recipe conversion/render/encode
 → inline CPU validation + geometry/identity metadata + bounded expanded capacity
 → ProductionScheduler paired lookup (IDs + image discriminator)
 → patch preparation + encode_new via pinned Model API, only spans beyond restored C
 → existing P6 absolute embedding merge + image masks
 → existing target-layer taps → native committed DSpark rings at the same C
 → N-2 paired capture → N-1 sealed append → existing P5 terminal transfer
 → unchanged native DSpark/MTP BatchGenerator + recipe guard
 → existing canonical quiescence → immutable paired publication / retirement
```

Image embeddings/patches are request inputs, never a retained execution authority.
After retirement only target/DSpark pairs and non-executable image identity
metadata remain reusable. Capture clones metadata and arrays; restore validates
both. Text-only cache hashes and numerical decode are unchanged.

Historical (prior-message) images ordinarily require an eligible paired prefix
covering those complete spans. The **deployed independent Messages path** also
admits independent Messages on an actual C=0 miss with complete validated source
bytes, charging every image fresh through the same cold MTP path. It introduces
no alternative numerical schedule or forward algorithm. Whole-image-safe reconstruction
also admits independent Messages at a genuinely restored C>0 whole-image-safe
pair: covered occurrences skip encoding; uncovered historical occurrences encode
in full. The [finite integrated E2E](../artifacts/multimodal-runtime/phase4-e2e-native.md)
exercises official offload, automatic compaction and reacquisition through paired
settlement/publication. The [promotion record](../artifacts/multimodal-runtime/production-completion.md)
binds installed acceptance; broader C2 qualification remains PARTIAL.
The aggregate patch-discount callback still requires complete historical coverage and supported paired-array layouts. The independent
[paired-byte bugfix](../artifacts/paired-byte-qualification/report.md) removes only the former fixed logical-byte ceilings. A selected pair is
never discarded to force C=0. Fresh last-message images
retain their safe-miss path. Branches never rewind live rings. See the
[ordinary admission contract](mtp-production-serving.md#paired-capture-and-restore)
for scope and model-free versus prior native evidence. The normal N-2/P5 seam requires two terminal text tokens.
Cancellation still shields the same worker mutation and then settles or burns
that Scheduler request. Logical sessions use their existing exact leased pair,
with image identity validation and the same recipe reconstruction certificate
(preparation only, no historical Vision execution).

The original delta below used the OFF four-image / 8192-position bound.
Ordinary MTP now admits actual expanded prompt + output reservation up to checkpoint
context, without OFF's four-image cap (ordinary occurrence ceiling: 600). Per-image limits remain 16 MiB, 4,194,304 pixels,
1014 expanded tokens, inline PNG/JPEG/WebP, aspect 1:2..2:1 and high/auto detail.
Aggregate encoded residency is bounded by 64 MiB, including cached inputs.
Per-image validation/decoding remains sequential and fully checked. Fresh work
is bounded by 16,777,216 source pixels and 36,864 planned ViT patches at the
validated paired frontier; historical wire age alone never grants this discount.
The [historical-image aggregate correction](../artifacts/historical-image-aggregate-admission/report.md)
has isolated native evidence and subsequent
[b2f7654 installed Production acceptance](../artifacts/production-multimodal-b2f7654/report.md). Flash-1017 historical patches may be deferred only after complete
compatible paired coverage and supported-layout validation before restoration.
See [current capability bounds](capabilities.md#bounded-inline-images) and the
[narrow physical qualification](../artifacts/cached-image-admission/report.md).
The retained original real-model delta below does **not** qualify expanded capacity. Legacy OFF remains four images / 8192 positions. No URL/video,
batch Vision, Responses multimodal or MTP persistence is enabled. Existing ordinary JSON body admission remains 16 MiB
in total, independently of per-image resource ceilings.

## Delta evidence

`tools/accept_mtp_vision.py` runs one model over real loopback TCP HTTP using the
ordinary `create_app`/ProductionScheduler path, not injected state or a prototype
executor. It first admits unchanged installed dependencies/assets in isolated
Python, then records checkout production-source hashes separately. The receipt
must not be interpreted as installed-payload admission of the changed checkout.

**PASS:** [the retained receipt](mtp-vision-acceptance.json) records these finite
real-image results. Considered counts sum native `depth_drafted`; accepted counts
are native `accepts`, not a profile flag.

| Ordinary case | New Vision calls | Paired cached C | Settled C | Draft considered / accepted |
| --- | ---: | ---: | ---: | ---: |
| corn image seed | 1 | 0 | 243 | 25 / 17 |
| same image repeat | 0 | 205 | 243 | 25 / 17 |
| text after image | 0 | 243 | 314 | 43 / 20 |
| text branch after image | 0 | 243 | 293 | 29 / 20 |
| TCP SSE disconnect | 0 | 293 | 309 | cancelled during initialization |
| same prompt after cancellation | 0 | 304 | 402 | 77 / 42 |
| append new carrots image | 1 | 402 | 895 | 22 / 14 |

All cases have all 40 target offsets and all 3 DSpark offsets at their reported C,
zero prompt replay and zero full-cache repack. Corn was correctly identified;
repeat output was identical. Historical corn→carrots replacement with prior
assistant history returned 503 explicitly forbidding replay; the engine remained
healthy. Plain text seed/hit/final outputs and draft statistics were identical
(44 considered / 29 accepted); decode times were 2.208 / 2.178 / 2.184 seconds.
This is a finite regression, not a new throughput benchmark or performance claim
for every multimodal/max-context workload. Production changes are seven files,
about **98 added / 42 removed lines**; the retained finite driver is 143 lines.

Existing qualified numerical, DwarfStar/P5, paired-state and MTP lifecycle evidence
remain authority; this acceptance checks their changed image boundary only.

Affected source regressions: **144 passed, 12 subtests passed** across production
MTP/sessions, ordinary recipe parts, multimodal contract/restore ownership and P6. Three unaffected OFF cancellation controls also pass.
An additional OFF cancellation fixture check has a pre-existing mock missing
`conversation` at `generation_policy.reasoning_policy`; its untouched fixture
and execution code were not changed to broaden this task.

## Promotion boundary

The original Vision delta did not change the default or installed package.
The subsequent [runtime completeness/local promotion review](runtime-completeness.md)
now approves ordinary MTP as the local default. At that decision, Browser image
acquisition and isolated PDF Vision still retained historical OFF-only routing.
The later historical application integration (`c1d1f716:docs/web-mtp-vision-integration.md` in Git) historically closed that gap;
Browser Chat/PDF workflows are now retired, without changing direct Runtime Vision.
Installed delivery retains
its separate immutable-payload boundary, not a source override.
Historical-image cache expiry remains an availability limitation outside the
narrow independent Messages C=0 / whole-image-safe C>0 source rules. It is not permission to replay tools,
recover a lost exact logical lease or retry uncertain state. Broader contexts/protocols/persistence and universal visual quality
remain unqualified. OFF is retained as an explicit fallback/profile.
