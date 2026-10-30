# Per-image source-pixel cap — RETAIN / BLOCKED

## Decision and stop condition

**RETAIN `MAX_PIXELS = 2048 * 2048`.** It is a local qualification policy, not a
Processor limit or a demonstrated physical memory cliff. Removing it is not a
safe small correction with the existing allocation/admission contract. No Runtime,
capability metadata, CURRENT documentation, numerical operations, Scheduler,
paired ownership, Production or Harness changes were made. The only source-tree
addition is `tests/test_per_image_source_pixels.py` (16 bounded model-free cases).
Stop at the precise CPU decode protection gap below; do not investigate another cap.

**Verdict: BLOCKED for higher-resolution admission; PASS for the focused
model-free witnesses.** A 2600×2000 image fits the unchanged Processor geometry
and patch constraints, but remains rejected without overrides. No unrestricted
resolution, general memory safety or new real-model qualification is claimed.

## Verified identities and isolation

Before additions, checkout was clean `master`, HEAD
`aefbb95d59a2d47739b4f01032843bfd816d446a`, origin
`https://github.com/kiojuvr/ds41f-mlx.git`. [Before](identities-before.json) and
[after](identities-after.json) capture source and actual installed identities using
the prior investigation's read-only identity script. No branch/history operations
or installs occurred. Remote observations are in [upstream.log](upstream.log).

Production `/Volumes/SDXC-512/ds41f-1017` remains Runtime 0.23.0, Recipe 0.2.0,
MLX 0.32.2, mlx-lm 0.31.4.dev132+g94cdcae13, oMLX 0.7.0, MTP
`flash-1017-pre-woa-fp8-v1`. Manifest provenance plus promotions is not treated as
this checkout's identity. All recorded Production package/native hashes match
before/after and the previous investigation. No server appeared in the initial
process listing; Production was only read, not imported for execution or restarted.
[Supplementary hashes](supplementary-identities.json) identify processing,
validation, Files, capacity, Pillow Python and native decoder sources. Source
and installed snapshots are distinguished, not assumed equivalent.

Tests used existing disposable `/private/tmp/ds41f-step4ar1-tests`, with source
imports asserted and independent installed payload/numerical selection authenticated
by `tests/conftest.py`. Its older installed Runtime is not the checkout under test.
No checkpoint or model weights loaded. No test changes admission limits, except
tiny scoped Pillow thresholds to exercise its existing warning/error branches.

Official DeepInfer reference remains the plan's pinned
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`. The retained eleven reviewed source
hashes were rechecked. Relevant `ve/processor/image_processor.py` was additionally
fetched directly from official GitHub at that revision: SHA-256
`8b8a0a021795b6648ac372ccc7e6ca935ffb71d5d92313c9b6fcd4833761ba21`, identical to
the inspected retained file. Advertised upstream HEAD and clean local reference
checkout are `553741557a57df44a46fb3ab72b6aafeaf945337`; not substituted for the pin.
Harness advertised HEAD is `ce7073d51254c26cd63b22a60835b8f948deb447`; wire/reference
rc.2 revision remains `687f282e8625c6b8d11ff58211a90524be2e4317`. Actual installed
Harness reports **0.2.1-rc.2**; manifests do not prove an exact build Git SHA.
Its executable bundle hashes, not an assumed checkout identity, accompany the
[installed executable evidence](installed-harness.json). Node v24.5.0. Nothing
was written to official package/store/config locations.

## Exact restriction and transport paths

- `runtime/multimodal.py:18`: defines 4,194,304 source pixels, independent of
  encoded size, resized geometry, normalized patches or expanded positions.
- `validate_encoded_image`, lines 187–220: byte type/1..16 MiB; PNG/JPEG/WebP
  magic and optional MIME; Pillow warning-to-error scope; open/format;
  single-frame policy; positive dimensions/product <= `MAX_PIXELS` at 208–209;
  optional remaining legacy aggregate pixels; encoded SHA-256; complete
  `verify()`; reopen and `load()` (needed for actual JPEG/WebP decoding).
  Source-pixel failure is `ImageConstraintError('image pixel limit ... exceeded')`.
- `prepare_multimodal:263`: invokes this same validator for every inline/resolved
  occurrence, including historical images; wraps failures as `invalid image`.
  Ordinary Flash-1017 removes only aggregate source-pixel charging, not this check.
- `serving/image_files.py:125`: Files upload calls the same validator with MIME
  before publication. Files acquisition preserves immutable bytes; preparation
  validates them again. Tested above-limit upload and inline Messages both return
  **400**, no file publication/no prepared request. Neither path can bypass the cap.
- `serving/capacity.py:50`: advertises `max_image_pixels=MAX_PIXELS`. This remains
  truthful and unchanged. OFF/other profiles and optional aggregate pixel checks
  remain unchanged.

`image_bytes` separately bounds decoded base64 bytes, URL syntax/length and detail;
request encoded-byte, HTTP body, occurrence, storage, context/output limits are
untouched. Processor expansion and the **36,864 fresh-patch guard** remain exactly
as before. Source SHA-256/span/grid identity, frontier validation, whole-image-safe
paired reuse and Target/DSpark semantics are unchanged.

## Resource boundary and actual lifetimes

Apple CPU/MLX share physical memory; these categories describe owners, not separate
RAM pools. Pillow here is 12.3.0 (exact installed sources in supplementary evidence).

| Category / phase | Size dependence, overlap and end of lifetime |
| --- | --- |
| Encoded bytes | <=16 MiB each and <=64 MiB/request, queued immutably. Not a decoded-memory bound: a transparent 5000×4000 PNG is <100 KB. Base64/transport copies are separate. |
| Open/header/verify | PNG/JPEG normally expose size before full raster load; warning/error checks run during `Image.open`, before the local product check. PNG verification traverses chunk checksums; JPEG/base verify alone is not decoding. WebP `_open` reads the encoded stream and constructs native `WebPAnimDecoder` before Pillow's generic bomb check/local check. Do **not** claim all plugin/native allocation is after header admission. Multi-frame WebP can have decoder setup before rejection. |
| Validation decoded source | Reopen/`load()` fully materializes original dimensions before Processor planning. PNG/JPEG cores depend on source dimensions/mode, not future resized grid. WebP also obtains raw frame bytes from its native decoder and creates Pillow core storage. Decoder scratch/metadata/encoded copies can overlap. Sequential validation avoids cumulative normal-success raster residency; no decoded image is queued. |
| Encoding original + RGB | `_encode_span` reopens exact immutable bytes. `image_patch_array` calls `convert('RGB')` **before** planning/resize. `convert` forces `load`; even an already RGB source is copied by Pillow. Caller `_encode_span` still owns the original image. Thus source core + full source-size RGB core coexist, with decoder scratch. In the inspected Pillow, RGB/RGBA core storage is nominally 4 bytes/pixel, not compressed size nor merely three packed bytes. Other modes/plugin buffers differ. |
| Resize temporaries | RGB source, original core and resize result/intermediate can coexist until assignment returns. `ImageOps.pad` may also allocate contain and padding images. Stretch/degenerate branches differ. Temporary resampling buffers can depend on source axis sizes as well as destination. Moving planning earlier alone does not avoid source decode/copy. |
| Resized image | Dimensions depend on Processor `plan_image_grid` and aspect handling: p=14, downsample=3, min pixels and 1024-token envelope here. Full-size RGB temporary is gone before normalization in the observed ordinary pad path, but caller's original source core remains. |
| CPU normalization/packing | Float32 H×W×3, arithmetic temporaries and packed reshape/transposition storage depend on resized dimensions. At least normalization and packing can overlap; not safe to count just final `nbytes`. Original decoded core remains live throughout this function. |
| MLX input/ViT | After patch function returns, CPU patches survive while MLX creates/casts input to bfloat16 and runs encoder. Graph/activation size depends on resized patch grid, architecture/dtype, not original pixels. Original source Python object can still own its decoded core (see below). `mx.eval(value)` precedes input cleanup/next image. No numerical path was changed or newly qualified. |
| Output embeddings | Depend on expanded rows and model width/dtype, not original source pixels; all fresh rows accumulate. Previous patch-cap report already establishes their P6/pair overlap. It is not re-investigated or relabeled as a source decode budget here. |

**Important context-manager finding:** installed Pillow's `__exit__` closes image
file pointers, not `Image.close()`'s ImagingCore destruction. Validation normally
releases its last decoded local on return. In `_encode_span`, `image` is a local
past the `with` block, so the original decoded core can overlap MLX work until
function return. Its `finally` clears `data`, `patches`, `pixels`, `value`, but not
`image`. On conversion/decode failure a retained traceback can retain the image
core after streams close. The test holds such an exception and demonstrates this;
once traceback and observer handles are dropped, weakref dies after GC. Context
exit is not proof of immediate CPU buffer reclamation. This strengthens the
reason not to remove protection; no cleanup redesign is made in this investigation.
Validation's verify/reopen also need not prove zero native decoder overlap during
replacement of the previous local, particularly for WebP.

`encode_new` clears all queued source handles/temporary row-list ownership in
`finally`, including BaseException; scheduler retirement drains work and releases
request/native ownership as previously tested. Synchronous decode/convert/native
startup is not an interruptible cancellation rescue point. Python decref, GC and
MLX cache clearing do not guarantee immediate physical page return.

## Bounded observations (not admission or model qualification)

[CPU observer](observe-cpu.py) calls the existing Processor directly, once per
fresh process, with 512×512 and 2600×2000 transparent RGBA fixtures. It does not
modify `MAX_PIXELS`, call an encoder or claim the larger image passed Runtime.
The larger diagnostic is a controlled 5.2MP allocation, not a bomb/capacity search.
RSS is `ps` resident bytes, not macOS physical footprint; `ru_maxrss` is bytes on
Darwin. `tracemalloc` sees Python/NumPy allocations but not Pillow native cores.
Fixture creation affects baseline/allocator reuse; these are not exact per-buffer
physical attribution or memory reservation measurements.

- [2600×2000](cpu-2600x2000.json): encoded 20,262 bytes; nominal original and RGB
  cores **20,800,000 bytes each**, concurrently live. Planned grid 80×105 patches,
  resized 1120×1470, **974 expanded rows**, packed float32 **19,756,800 bytes**.
  RSS baseline 72,073,216; both source/RGB 93,028,352; source/RGB/resize result
  111,296,512; patches returned with source still alive 155,451,392 bytes.
  Traced peak 39,543,625 bytes, dropping to 17,731 after explicit diagnostic owner
  cleanup. RSS did not immediately drop. Weakrefs establish RGB object release,
  not page release. Source load reused fixture allocator storage: zero apparent
  RSS growth is not zero decode allocation.
- [512×512](cpu-512x512.json) provides the independent below-limit comparison.
- 5000×4000 RGBA test (~80 MB nominal core) is rejected at the local dimension
  check before full PNG load, verified with a load-forbidden witness. It is not
  decoded in the test. Pillow warning/error tests use tiny 40×40 files with scoped
  lowered thresholds; no bomb-size image is decoded and safeguards stay enabled.

## Official behavior and compatibility

Pinned DeepInfer `image_processor.py:21` explicitly sets
`Image.MAX_IMAGE_PIXELS = None`. `decode_image` opens, fully loads and converts to
RGB before `_prepare_image` contains/resizes (or stretches). No comparable 4MP
source guard exists in that implementation. Lazy `ImageTensorLoader` delays work;
uint8 resized tensors are packed and normalized/padded on GPU. That reduces later
float CPU residency relative to local NumPy normalization, **not** original decode
or full-size RGB conversion requirements. Worker/CUDA resource profiling is not a
Pillow host-decode reservation. Its disabled Pillow safeguard must **not** be copied.
Absence of an upstream check does not prove arbitrary decoding safe.

Installed official Harness was executed read-only by
[check-installed-harness.mjs](check-installed-harness.mjs), using its exported
`prepareImageFile` and `resolveRequestImageTarget`, not a replacement algorithm:

- Source admission defaults **64,000,000 pixels**, **8192 long edge**, **20 MiB**;
  `detectImage` checks dimensions before `sharp.raw().toBuffer()`. It uses libvips/
  sharp (`limitInputPixels:false`) with its own explicit admission. Runtime cannot
  assume equivalent Pillow buffer behavior or trust an unverified client.
- Default provider-independent normalization: **4,194,304 pixels**, 8192 long edge,
  4 MiB target. 2600×2000 becomes **2335×1796**, 25,017 bytes. Default DeepSeek route
  further projects Processor geometry; target for a 2600×2000 reference is
  **1470×1131**, 2 MiB encoding target. Normally it resizes before sending.
- A supported explicit normalization budget 6,000,000 plus explicit model
  `imagePixelBudget:6000000` yields a byte-identical valid **2600×2000** normalized
  image and request target, below all source limits. Therefore valid **configured**
  official Harness output can exceed Runtime's 4MP restriction. This is not default
  behavior, nor a claim that the user's current live model configuration does so.
  The check passed explicit in-memory options; it changed no installed configuration.

Pillow full-source CPU decode/copy occurs before CUDA/MLX differences matter.
Local complete verification, byte/MIME checks, single-frame policy, bomb warnings
and immutable identity are necessary safety/integrity choices; matching upstream's
unchecked decode is not a necessary MLX compatibility change. No Harness
transformation is modified or proposed as a substitute for server safety.

## Smallest missing resource protection

Existing Processor/token/fresh-patch limits bound **post-resize** work, not these
pre-resize buffers. Encoded limits cannot bound expansion. Default Pillow warning
threshold is **89,478,485** pixels (warnings are errors locally), permitting on the
order of **683 MiB for just two four-byte source/RGB cores** below the warning,
plus decoder/resize temporaries, normalized arrays and native work. This is finite
but not an established host-capacity guarantee, especially with weights/restored
state and concurrent ingress validation. The current 4MP policy is also not a
universal safety guarantee under arbitrary host pressure.

The smallest missing contract is **pre-decode admission for source-sized CPU peak
bytes**, including mode/decoder setup, original+RGB+resize overlap and concurrent
Files/inline validation, carried until those buffers actually retire on success,
error or cancellation. It must use an existing execution/ingress resource owner
with qualified conservative allocation bounds or allocator-backed protection;
no such configured shared authority exists here. A free-memory snapshot, encoded
byte multiplier, larger pixel constant, moving `plan_image_grid`, or MLX wired
limit cannot supply that contract. Scheduler's wired lease starts after Vision
and does not reserve CPU allocations in any event. See the prior resource-admission
follow-up for existing controls; no broad resource redesign is repeated here.

A local explicit-close cleanup could shorten core lifetime, but would still leave
unaccounted full source/RGB allocation **before resize** and unleased concurrent
validation; it is not sufficient grounds for cap removal. Decoder-side shrinking
is not a common PNG/JPEG/WebP drop-in equivalent preserving complete decode and
exact Processor semantics. No new decoder pipeline, allocator, cache or resource
manager is introduced. Retain the shared cap rather than expose different unsafe
transport envelopes or implement a substantial mechanism under this task.

## Focused tests and native scope

[Focused run](focused-tests.log): **165 passed**, 361 warnings, 42.17 seconds:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_per_image_source_pixels.py tests/test_metadata_first_vision.py \
  tests/test_multimodal_contract.py tests/test_multimodal_cancellation.py \
  tests/test_cached_image_admission.py tests/test_image_files.py -q --disable-warnings
```

Coverage: below/above 4MP with valid Processor geometry, portrait/landscape,
compressed high-expansion rejection, malformed/truncated JPEG/MIME/animated PNG,
all three supported formats (existing suite), Pillow warning and error protections,
Files/inline boundary consistency, source digest/identity and partial-frontier
rejection, exact paired reuse without patches/encoding, decoder stream closure,
conversion traceback core lifetime and source cleanup, sequential input release,
MLX eval failure and cancellation draining. Existing tiny random ViT test is
model-free adapter coverage, not checkpoint acceptance.

Initial new-test run had three incorrect expectations (Files returned 400, not
413; Pillow context exit does not destroy cores). Tests were corrected to assert
observed ownership/HTTP contracts rather than mask the lifetime finding. Final
run above has no failures/skips. Runtime behavior was never patched.

**Real-model trials NOT RUN**, as required when a safe correction cannot be
established. No above-cap admission override or native capacity search. Hence no
new encode-count/canonical-settlement/Target–DSpark alignment/replay-repack result,
MLX memory/physical-footprint/host-pressure/swap result or native safety-stop trial
is claimed. Prior evidence is preserved, not credited as per-image-cap removal.
Only bounded model-free CPU allocation observations and existing cleanup tests
are new execution evidence. All remaining guards and Production remain unchanged.

Artifacts are in the repository's normally ignored `artifacts/` tree and the new
test is untracked, left available for review. No commit, promotion or next-cap work.
