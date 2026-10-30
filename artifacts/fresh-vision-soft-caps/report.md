# Fresh-Vision soft caps — PARTIAL

Scope: ordinary Flash-1017 MTP only. **REMOVE aggregate source-pixel rejection;
DEFER fresh-patch removal**, retaining its guard and conditional paired admission.
No new resource manager, execution path or Production change.

## Verified revisions / isolation

- Clean `master` at `4429f7d83acf4d378e682bc6431bfa03d805db65`; remote master/HEAD
  matched before edits. No branches/history changed.
- Production `/Volumes/SDXC-512/ds41f-1017`: ds41f-mlx 0.23.0, Recipe 0.2.0,
  MLX 0.32.2, python-multipart 0.0.32. Read-only manifest/provenance and actual
  relevant installed source/native-owner hashes are in [identities](identities.json).
  Manifest provenance is `69689d514fc5acb2cbe229f8c863da1e0a855e4a` plus subsequent
  promotions; it is not a complete current source identity. Production had no
  running ds41f server in the initial process listing. Nothing there was installed,
  imported for execution, restarted or edited; checkpoints are read-only assets.
  Final read-only recheck confirms all recorded Production hashes are unchanged.
- Official DeepInfer reviewed at the CURRENT plan's pinned
  **`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**, fetched directly from official GitHub.
  Exact reviewed file hashes are retained. Current advertised remote HEAD and clean
  local reference checkout: `553741557a57df44a46fb3ab72b6aafeaf945337`; not silently
  substituted for the pinned architecture. Harness remote HEAD
  `ce7073d51254c26cd63b22a60835b8f948deb447`, installed-workflow release tag rc.2
  `687f282e8625c6b8d11ff58211a90524be2e4317`; no Harness changes.
- Source tests use admitted disposable `/private/tmp/ds41f-step4ar1-tests`.
  Native candidate is independent `/private/tmp/ds41f-fresh-vision-candidate`:
  fresh venv, copied disposable dependency closure (not symlinked Production),
  freshly built Flash-1017/native wheel installed without dependency changes.
  [Build log](build.log). Native runner uses `-I`, genuine installed/checkpoint
  admission, and verifies all installed Runtime Python bytes against this dirty
  source snapshot. No test-only admission threshold overrides.

## Complete rejection-path inventory

| Site | Baseline behavior / decision |
| --- | --- |
| `prepare_multimodal` | Sums nonhistorical source pixels; supplies `MAX_REQUEST_PIXELS - fresh_pixels` to image validation. **REMOVE only for admitted ordinary Flash-1017**. Still checks every container/decode and per-image pixel limit sequentially. Other profiles retain the policy. |
| `validate_encoded_image(... remaining_pixels=...)` | Rejects `aggregate decoded pixel budget exceeded` before decode if dimensions exceed remaining allowance. **RETAIN optional legacy/direct-call boundary**, but Flash-1017 preparation supplies `None`. Per-image bytes, MIME, frames, full verification/decode, decompression warnings and 4,194,304 pixels remain unchanged. |
| `prepare_multimodal` patch sum | Rejects `aggregate image patch budget exceeded` when nonhistorical planned grids exceed 36,864. Historical origin defers this check only; it does not prove cache coverage. **RETAIN pending resource replacement**. |
| `MultimodalInput.validate_workload(C)` | Checks whole-image-safe frontier, sums genuinely uncovered spans. **REMOVE pixel sum for admitted ordinary Flash-1017**; retain 36,864 actual fresh-patch rejection. A miss C=0 charges every image regardless of wire age. |
| `encode_new` | Calls `validate_workload(C)` again before patch allocation, then consumes only uncovered immutable sources. Retain frontier, source digest, geometry and patch checks. Its `finally` releases sources and temporary row-list ownership on success/error/cancellation. |
| `ProductionScheduler._schedule_waiting` | After actual pair lookup/restore, checks historical reconstruction conditions, whole-image frontier, actual fresh workload and source digests before `_start`. Unchanged. Cold/partial-cache Messages use existing source reconstruction rules; exact logical Chat leases remain exact. |
| Conditional historical-patch callback | When all-history patches exceed 36,864, callback runs **before clone/restore**: admitted numerical profile, exact prefix/image identity, complete historical coverage, fresh workload, supported target/ring/compressor layout via `projected_pair_bytes`. Retained unchanged; byte projection is not a physical allocation reservation. |
| `vision_capacity` | Source-only metadata now reports `max_request_pixels: null`, `source_pixel_validation: sequential_per_image`, scope `fresh_patches_with_complete_pair` for Flash-1017; other profiles retain aggregate pixels. Patch metadata unchanged. |

`MAX_REQUEST_PIXELS` and `MAX_REQUEST_PATCHES` remain defined for unaffected
profiles / retained patch policy. Neither number is a Processor-intrinsic limit.
No Files, HTTP, per-image pixel/encoded-byte, context/output, logical-session,
Harness, model math, Scheduler or paired-cache behavior was changed.

## Resource units, actual lifetime and upstream comparison

- **Source pixels:** CPU decoder size, not ViT work or KV. Full image validation
  opens/verifies and reopens/loads one image at a time; only immutable compressed
  bytes and metadata survive. Worker `_encode_span` likewise owns one decoded
  image, RGB resize/normalization arrays and CPU patch array at a time. Cumulative
  original pixels do not correspond to concurrently live decode buffers.
- **Resized Vision pixels / patches:** Processor resize/grid derives these
  independently of source pixels. Normalized float32 CPU patches, MLX conversion
  and bfloat16 inputs plus ViT QKV/attention/MLP/aligner temporaries can coexist
  *within one image*. `mx.eval(value)` completes each image before the next and
  `finally` drops inputs. Existing per-image bounds and Processor limits remain.
- **Expanded LLM image rows:** unlike input patches, output embeddings from **all
  fresh images** accumulate in `ImageEmbeddings.rows`. P6 consumes/merges them
  across suffix segments; references are dropped on seal/failure and by `_start`
  after its segment loop. Sequential Vision is not zero cumulative output residency.
- **Target/DSpark:** selected immutable root, restored arrays, fresh embedding rows,
  P6 state and pending prompt/final captures can overlap. Paired root counts and
  token limits are not available-byte reservations; logical `nbytes` cannot
  assume zero-copy snapshots. Restored covered images never encode again.
- **Existing resource controls:** ordinary MTP executes one Scheduler request, but
  its overridden `_schedule_waiting`/`step` do not use generic oMLX preflight or its
  ProcessMemoryEnforcer. Generic `_prefill_memory_guard` defaults false; this
  backend does not configure it. `MTPWiredLimitLease` sets/restores the recommended
  wired limit and is acquired **after Vision** in `_start`, not before it. It is
  not a memory quota/reservation. MLX `set_memory_limit` is documented as a
  guideline, default 1.5x recommended working set, not a safe host-reserve gate.
  OFF resource admission is a separate model-lifetime owner, not an MTP replacement.

Official DeepInfer at the reviewed revision:
`multimodal_processor._compute_and_strip_cached_images` counts only completely
cached spans and strips their payloads. `deepseek_vl._split_cached_prefix` preserves
exact cached segment lengths; its mapper skips cached images, computes int64 raw
RGB byte offsets and patch prefix sums from actual fresh inputs. Lazy CPU image
loaders can run in parallel, then packed resized RGB persists in the existing
shared pool. `prepare_tensor_input_vl._chunk_mm_metadata` slices inputs by actual
prefill token/image ranges; batch preparation filters zero cached crops and
computes actual chunk patch counts, not a fixed total-history threshold.
`vl_aligner.forward` normalizes/pads raw RGB into the current chunk's patches,
runs batched variable-length Vision, downsamples/aligns and merges into that
chunk's token embeddings. Chunk raw inputs, patches, features and embeddings can
coexist; it is **not** universally one-image-at-a-time execution. Pool-handle
release is deferred until asynchronous H2D consumption is safe (`ve/schema.py`).
Worker peak-memory profiling and KV/state block managers own GPU resource capacity.
CUDA pools, asynchronous copies and profiling are not an MLX memory admission proof.

**Concrete patch-removal blocker:** local Vision is already per-image, so 36,864
cumulative patches are not simultaneous ViT residency. But its all-fresh output
rows remain live across full-request P6 preparation, unlike upstream's chunk-local
fusion, and no configured local admission owner accounts for their overlap with
restored/prefill/captured paired state. Context/600 occurrences make allocations
finite (up to ~5.86 GiB of image rows at 600x1024x5120x2 alone), not physically
reserved. Merely removing the patch guard or copying a CUDA budget would not meet
this task's resource-safety requirement. Wiring a generic estimator without
qualifying this custom P6/pair lifetime would be a new unverified admission path.
**DEFER** that replacement; retain the historical guard as a temporary conservative
fallback, not a claim that exactly 36,864 patches is a measured memory cliff.

## Exact changes and focused acceptance

Runtime changes are limited to `runtime/multimodal.py` (internal admitted-profile
selection for sequential pixel validation, skips only aggregate pixel charging)
and `serving/capacity.py` (truthful capability metadata). Existing direct/legacy
pixel checks, patch checks, numerical operations and paired callback are intact.

[Model-free focused checks](model-free-tests.log): **170 passed**, 9 existing
deprecation warnings, no skips/xfails, 40.61 seconds. They cover real Pillow/MLX geometry,
source identity, live input/output weakrefs, genuine misses, all-image fresh
charging, complete reuse, incompatible/partial coverage, legacy profile admission,
pre-native rejection ordering, cancellation/failure and cleanup. The new lifetime
test explicitly proves earlier CPU patches disappear while all output rows coexist.

**PASS — finite isolated native pixel-boundary acceptance.**
[Result](native/result.json), [request receipts](native/requests.json),
[resource summary](native-summary.json), [raw MLX/process samples](native/samples.jsonl),
[raw host samples](native/host.jsonl), [admitted identity](native/identity.json),
[all Runtime source hashes](native/sources.json). Runner
`tools/qualify_fresh_vision_pixels.py` observes actual execution and injects one
failure; it never changes admission thresholds or numerical operations.

| Case | Observed result |
| --- | --- |
| Below both former limits | One 512x512 image: cold encode 1, aligned Target/DSpark C=195, canonical settlement and pair publication. |
| Above former pixel threshold | Four 2048x2048 + one 512x512 image: **17,039,360 pixels, 36,117 patches**, genuine C=0, all **5** encoded, aligned C=4175, settled and published. This exact request formerly failed on aggregate pixels. |
| Complete compatible reuse | Same five images plus actual response/text continuation: restored C=4175, **0** re-encoded, aligned final C=4185, settled and published. |
| Failure during Vision | Changed first image forces C=0; injected exception after second real image computation. Sources released, no waiting/running requests, no pair publication, all retained roots cleared, existing engine-fatal retirement preserved. Process then shut down/exited; it is not claimed recoverable for another request. |
| Above retained patch threshold | Five 2048x2048 images reject with `aggregate image patch budget exceeded` **before model load** ([receipt](native/patch-rejection.json)); no above-patch-threshold native execution or qualification is claimed. |

All successful requests have zero prompt replay/full-cache repack, empty prediction
queue, prediction retirement, and all forty Target offsets equal all three DSpark
offsets. Model-free tests additionally cover partial and incompatible paired
coverage and cancellation after four accumulated image outputs.

Resource observations on M3 Ultra 512 GiB (not a maximum-capacity search):

- Above-pixel case MLX peak **293.182 GiB**; maximum sampled process physical
  footprint **300.274 GiB**; lifetime footprint high-water **300.734 GiB**.
  CPU-normalized image buffers are proven sequential by weakrefs; unified process
  footprint is not a separately attributed CPU heap measurement.
- Individual large output rows are 10,178,560 bytes; the smaller output is
  1,884,160 bytes. All five embeddings coexist (~40.625 MiB), not five resident
  ViT input/activation sets. Image-step samples expose bounded active-memory
  changes but do not replace allocator-event tracing.
- Host reserve proxy minimum **139.434 GiB**, pressure free percentage minimum
  **40%**, swap **0.00 MiB** throughout observed native phases. Reserve proxy is
  free+inactive+speculative, not guaranteed allocatable RAM.
- After injected failure/shutdown, request/pair ownership is empty. MLX model
  weights remain active until process exit; their residency is not a cleanup leak.
- Observer safety stops (360 GiB process/MLX active+cache, 100 GiB reserve proxy,
  25% pressure free, zero swap) were not triggered. These are finite qualification
  stop conditions, **not** newly installed Runtime resource admission.

Reproduce the focused model-free checks and finite native acceptance (native
output directory must be absent; candidate must be independent and matching):

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_fresh_vision_pixels.py tests/test_expanded_vision_capacity.py \
  tests/test_historical_image_aggregate.py tests/test_cached_image_admission.py \
  tests/test_metadata_first_vision.py tests/test_multimodal_cancellation.py \
  tests/test_multimodal_restore_ownership.py tests/test_harness_image_capacity.py \
  tests/test_multimodal_contract.py tests/test_production_context.py \
  tests/test_documentation_authority.py -q --disable-warnings
PYTHONDONTWRITEBYTECODE=1 DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  /private/tmp/ds41f-fresh-vision-candidate/bin/python -I tools/qualify_fresh_vision_pixels.py \
  --source /Volumes/SDXC-512/ds41f-mlx --output /private/tmp/new-absent-pixel-result
```

No general numerical equivalence, arbitrary host pressure, maximum-context,
600-image execution, Windows/Harness, or Production promotion claim is made.

**PASS for the scoped pixel correction, PARTIAL overall:** aggregate pixel blocker removed narrowly; full arbitrary
fresh-Vision cap removal is not established. Stop at the identified patch
resource-ownership gap. No automatic next-cap investigation or Production promotion.
