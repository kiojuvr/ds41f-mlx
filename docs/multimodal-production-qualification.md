# First-party standard-OFF multimodal production boundary

**Qualified, bounded:** `QUALIFIED_BOUNDED_FIRST_PARTY_OFF_MULTIMODAL_4_IMAGES_8192`.
Initial execution, fresh-process restore, official reference and affected regression
closeout passed. This is development/core production capability, not full R1,
release, packaging or `ds41f-runtime` promotion.

## Envelope and API

The existing standard-OFF Chat Completions endpoints carry text/image content:
`POST /v1/chat/completions` and `POST /v1/sessions/{id}/chat/completions`. No Vision
execution/debug endpoint, new generation engine, cache representation, scheduler
or model selector is introduced. Text-only Responses/Messages remain text-only;
multimodal requests on those protocols reject. MTP/DSpark/speculation are unchanged.

The supported bounded image envelope is:

- 1–4 images **in the full conversation**, not four additional images each turn;
- inline base64 PNG/JPEG/WebP (or recipe's internal bytes source), one frame,
  high/default/auto detail; no external URL/file fetch and no low-detail resizing;
- at most 16 MiB encoded bytes per image, 4,194,304 decoded pixels, aspect ratio
  1:2 through 2:1, and 1,014 expanded image positions per image;
- at most 8,192 total multimodal consumed positions. Admission reserves
  prompt + requested output (128 by default), including expanded image spans.
  Future turns must stay within the same bound. This does **not** reduce the
  independently qualified bounded 1M text-only core.

Protocol image placeholders are interleaved with text in official recipe order.
Clients continue with the full ordinary conversation and the same historical
inline image bytes. Dropping/replacing/reordering a historical image rejects,
including when its dimensions and expanded token IDs would remain identical.
Text-only **turns inside an image conversation** retain those earlier image parts
in the full conversation request; they do not execute the vision tower again.
Additional images append normally, subject to the total four-image/context limits.

The limits are a finite representative envelope, not an all-image model-quality
claim. Safety limits alone are not evidence: the maximum matrix must actually
execute four photographs transformed to 2048², 1024×2048, 2048×1024 and 128²,
including PNG/WebP, a fresh 8,160-position prompt and consumed frontier 8,192.
The 2048² square produces 994 positions; the portrait produces 1,014. Neither
checkpoint `max_image_tokens=1024` nor `max_seq_len=1048576` grants Vision support.
The 16-MiB encoded-byte cap is a preflight safety budget, not a measured worst-case
container-decode latency claim; geometry/count/context are the measured limits.

## Actual execution and ownership

```text
canonical recipe conversion/render/encode (placeholder + ordered image sources)
 -> bounded inline decode / CPU patch array + expanded IDs / span identities
 -> existing single-flight inference worker / admitted first-party Model
 -> Model.encode_image_span: ViT -> 3×3 aligner -> learned start/newline/end rows
 -> sparse absolute-position embedding injection into existing P6 setup
 -> DENSE_P0_P7 on the same packed 40-layer list / segment-local image mask
 -> P5 text terminal exactly once -> M44 generation / M45 transaction / M46 state
 -> same-list idle -> new suffix only (including only newly introduced image rows)
```

`model_execution/processing.py`, `vision.py`, `model.py`, `language.py` and
`engram.py` own checkpoint-specific geometry, BF16 normalization/patch packing,
2-D split-half RoPE, full per-image attention, RMSNorm/SwiGLU, channel-first
3×3 zero-padded unfold order, GELU aligner, learned image delimiters, VL MoE bias
and image exclusion from n-grams/Engram contributions. The full image span uses
one ID, 129264; IDs alone are not its numerical representation. Image masks cover
**delimiters too**, not just feature positions. They are sliced for each segment,
encoder command, Engram micro-tile and decoder suffix query. Decode after the
text terminal uses the existing text target-forward authority.

Pillow supplies generic decode/resampling; NumPy CPU transport and MLX/Metal/nn
primitives supply subordinate arithmetic. Recipe owns protocol conversion,
placeholder/source ordering and response parsing, not this checkpoint's Vision
executor. Its optional OpenCV/WebP resolver was reviewed but not selected: that
transport preprocessing/recompression is not a replacement for the checkpoint's
`inference/image_processor.py` semantics. Low detail, network fetching, animated
images, EXIF auto-transpose and optional resolver behavior are not implicitly
inherited. The original official PIL physical-pixel recipe is the fidelity oracle.
PyTorch is qualification-only; no production PyTorch/donor model runtime is used.
The existing mlx-vlm result container remains an ordinary adapter dependency.

M47 pins/binds the first-party model and preprocessing implementations, all
checkpoint payloads, tokenizer/native profiles and concrete SSD handles. Vision
weights were already part of the M48 inventory/residency; they are now executed.
The wired/32-GiB free-allocator lifetime established at 200K/1M remains unchanged.
No admission bypass or production oracle/fallback selector is added.

## State, persistence and zero hidden replay

After commit, the only executable image-derived session representation is the
same physical KV/compressor/index/candidate/normalized-history state, plus the
ordinary committed token history. No pixels, patches, image embeddings, hidden
prompt or second cache are retained as a session reconstruction authority.
Request-local sparse image rows disappear with the append/request. The existing
P5 certificate backlink retirement is preserved.

Small ordered image identities `{start,length,sha256,grid}` are semantic session
metadata: byte SHA256 refers to the original encoded image. The M11 idle manifest
stores them in `diagnostics.image_identities` (reserved against caller diagnostic
override). Restore validates count, order/non-overlap, hex digest, grid/layout
length and exact coverage of image positions in committed history. A legacy
text artifact needs no image metadata; an image history without corresponding
metadata fails closed. The M9 tensor schema and all 280 physical slots are unchanged.
This is same-backend idle persistence, not a tamper-proof signed artifact format.

Restore materializes and synchronizes **all exact loaded slot objects on its
owner worker before publishing idle state**. This removes deferred file/CPU-stream
I/O dependencies, without changing dtype, representation or tensors. Raw-byte
qualification reads are also scheduled on that worker: creating a new numerical
view of CPU-loaded arrays on another thread is not the serving architecture.
Restore does **not** need/re-run images or prompt. Subsequent canonical full
conversation input supplies historical bytes for identity/preprocessing validation,
not for hidden encoding/replay. CPU revalidation is allowed; committed images
never enter `encode_image_span` again. Newly introduced images are encoded once
before their suffix append. Artifacts are dormant storage, not live authorities.
Active generation persistence, cross-runtime portability, automatic completion of
protocol-unrepresentable partial assistant turns and transparent prompt rebuild
remain unsupported, as in the established text architecture.

## Failure and cleanup

Malformed base64/container, missing/count-mismatched placeholders, animation,
unsupported protocol/detail/URL/file, resolution/aspect/token/context excess and
changed image identity reject before cache mutation. HTTP preparation is CPU-only;
no request-thread MLX graph is transferred into the inference worker. Stateless
stream cancellation closes the existing target session and releases its count/lock.
Cancellation inside a running worker operation waits for that same operation to
finish before releasing ownership. Canceled encoding drops only temporary rows;
a canceled completed prefill explicitly discards/burns its ready one-shot lease,
including the passive certificate backlink. A canceled terminal bootstrap closes
the newly created target lease. Queued cancellation clears only the waiting
record's busy flag, never another request's lock.
Core cancellation settles between complete target transactions and returns the
same list. Stateful execution publishes its completed ordinary turn and protocol
record **inside the worker**, before canceled transport can lose the result; GET
can recover the committed `last_turn`. It retains the existing commit-before-
protocol-event policy, not an immediate GPU interrupt. No arbitrary protected-
phase rollback or partial-protocol automatic completion is claimed.

Vision encoding happens before append mutation. A pending P6 execution or P5
bootstrap failure burns/closes the continuation rather than advertising idle
state. Initial assistant/parse failure closes a newly created M8 lease even when
it could not yet be published into the backend's session record. No failure
retries via donor forward, fresh prefill, replay or repack. Tests cover these burns
and CPU ownership as well as invalid inputs and command-local mask slicing.

## Fidelity and qualification evidence

Evidence is under `artifacts/vision/`; collectors write RUNNING/FAIL durably and
closeout refuses partial receipts. `tools/qualify_multimodal.py` exercises actual
backend loading, canonical image + text generation, text continuation, additional
image continuation, historical-image rejection without changed slots, core/async
stream cancellation, persistence, fresh-process restore, affected text serving,
maximum matrix and policy retirement. Displaced donor numerical imports/calls are
forbidden; profile counts identify the first-party Vision executor and detect
re-encoding of old images. Every save/restore/probe comparison includes history
and shape/dtype/raw-byte SHA256 of **40 × 7** slots, not just generated text.

`tools/qualify_vision_reference.py` independently loads the actual checkpoint's
Vision/aligner tensors into the official PyTorch CPU implementation. On the
checkpoint's real corn and carrot photographs:

- normalized BF16 patches, patch grids and start/row-newline/end types match exactly;
- FP32 tower relative RMS error is 1.47e-5 / 2.76e-5; aligner 8.48e-6 / 1.56e-5;
- BF16 CPU-versus-MLX tower relative RMS differs by 7.54% / 5.73%, aligner
  4.65% / 3.22%, with high cosine similarity. Layerwise receipts expose drift.

The initially assumed 4%/0.999 BF16 gate **failed**. Its gate revision is recorded
explicitly; the original raw failure collector output was overwritten during
requalification, so it is not presented as a preserved receipt. Final receipts
retain the same BF16 metrics and the FP32 diagnostic proof.
It is not claimed as an implementation fix or exact official BF16/CUDA parity.
FP32 isolates the mathematical/layout implementation from accumulated backend-local
BF16 rounding; the final oracle gates FP32 at 2e-4/0.99999 and separately bounds
observed BF16 drift at 10%/0.995. Production remains BF16, not a precision-switch
workaround. This is explicit local numerical fidelity evidence, **not** official
CUDA full-LM logit/token parity or universal visual reasoning quality. Real
conversation responses correctly identify corn/yellow-green, then carrots/orange
and contrast the second image with the first.

## Defects/qualification corrections

1. Real initial Vision execution exposed an inference-thread stream error from
   lazy MLX patches constructed on the input thread. CPU request patches and
   worker-owned BF16 conversion fixed this; no new stream/cache authority was added.
2. Production block mask forwarding had not been sliced by command/query range.
   Integration now supplies command-local masks, including decoder suffixes; tests
   cross the 8192 encoder-tile boundary and the final one-row suffix.
3. Pending append/bootstrap errors and unpublished initial assistant failure had
   lease/publication cleanup gaps. They now close/burn rather than retain an idle
   or unrecorded active authority.
4. The initial cancellation collector used an unframed suffix that naturally
   emitted EOS before three tokens. Its FAIL is retained; a canonical new user turn
   now cancels while active. This was a fixture defect, not model recovery.
5. A square-only 994-position candidate bound rejected the portrait matrix's
   official 1,014-position layout. The bound is subject to real matrix requalification,
   not raised to the checkpoint's configured 1,024 maximum by assumption.
6. Fresh backend restore exposed an uncaught `Stream(cpu, 1)` abort in the
   cross-thread byte observer (exit 134, RUNNING is not PASS). Restore had exported
   deferred CPU loads, and the observer created new numerical views off-owner.
   All restored slots now materialize/synchronize before publication; observers
   stay on the same worker. Tests delete the artifact after restore and verify
   the exact loaded objects, without replay/repack or a new cache format.
7. Cancellation could release a busy/lock fence while executor work still ran,
   lose an unpublished initial/complete turn, or strand a queued record as busy.
   Protected-operation drain, ready-prefill discard and worker-side protocol
   publication fix these lifecycle defects under the existing single worker and
   generation authority. Real encoding/prefill/stateful/stream cancellation is
   requalified, with GC-disabled burn/backlink and repeated-cancellation tests.

## Canonical evidence and practical operation

`artifacts/vision/summary.json` is the fail-closed closeout decision; it hashes the
canonical `initial.json`, `restored.json`, `official-reference.json` and
`affected-tests.txt` receipts. Both production processes verify current source
hashes and the admitted resource-set identity. Historical failed/intermediate
receipts are not substitutes for these final passes.

- Real corn → text color continuation → added carrots: correct food/color answers,
  same cache list, image encode counts **1 / 0 / 1**, zero replay/repack, coherent
  40-layer frontiers. End-to-end worker turn times **2.05 / 0.52 / 2.86 s**.
- Warm single-image streaming, including CPU request preparation, first token
  **0.953 s**. `first_token_latency_s` in the core trace is only the first decode
  report after bootstrap, **not** image request TTFT.
- Maximum matrix: real photos transformed to 2048² PNG, 2048×1024 JPEG,
  1024×2048 PNG and 128² WebP; **8160 prompt → 8192 consumed positions**.
  CPU preparation **0.131 s**, four image encodes **2.550 s**, complete worker
  turn **20.487 s**, 32-token decode **17.78 tok/s**. First-use shape/bootstrap
  overhead remains; no universal latency guarantee or micro-optimization claim.
- Peak MLX allocation **310.989 GB**, active **301.122 GB**, free cache
  **32.354 GB**; sampled system available **105.739 GB**, corrected swap **0**.
  RSS is not GPU residency. Loaded/closed active allocation is about **301.104 GB**:
  passive model references remain, so retirement is not claimed as physical
  deallocation. Admission retires, generation count is zero, FDs drop **13 → 9**,
  and caller wired/allocator policy restores exactly in both processes.
- Cold admitted model loads **310.5 / 314.2 s**. Daily use assumes a resident
  long-lived model; cold readiness and warm image TTFT are distinct.
- Initial P7 overlap records **26 logical-match Engram consumes**, no activated
  prefetch foreground fallback. Fresh restored M8 uses its existing direct
  SSD/P6 path (no overlap events); exact canonical bytes/tokens and resource
  binding pass, but restored P7 overlap is **not** claimed.
- Save/fresh restore: exact saved history/identities and all **280 physical slots**;
  same canonical text probe yields identical tokens and all 280 slots, zero old
  image encoder calls. A new post-restore image is encoded exactly once and
  answered “Yes” to whether it is the first food.
- Stream, core, in-encoding, in-prefill and in-stateful-worker cancellation pass.
  Discarded ready prefill burns all 40 aliases and detaches its certificate;
  stateful canceled transport retains an idle, completed `last_turn`.
- **216 tests + 34 subtests pass**, including affected M44–M48/P5–P7, resource
  policy, input/identity/mask, M8/M11/termination and cancellation/restore contracts.
  Additional historical evidence checks yield 17 passes and one pre-existing
  M35 source-pin mismatch (`internal_mtp.py`, unchanged from master); static
  self-containment stops on a pre-existing `M2` token in `session-state.md`.
  These are recorded, not fixed by rewriting old evidence or counted as a fresh
  R1 pass. Full R1 is deliberately not rerun. The prior 1M text qualification is retained,
  not relabeled as a new full-context multimodal or HTTP proof.

Reproduce with the documented admitted Python environment:

```bash
python tools/qualify_vision_reference.py --output artifacts/vision/official-reference.json
(python tools/qualify_multimodal.py --output artifacts/vision/initial.json &&
 python tools/qualify_multimodal.py --restore artifacts/vision/initial.json --output artifacts/vision/restored.json)
echo $? > artifacts/vision/sequence.exit
python tools/close_multimodal_qualification.py
```

Only the independent Torch CPU oracle needs the qualification-only Torch package;
production does not import it. Collector CLI checkpoint/input paths and actual
source/dependency identities are recorded in the receipts. The idle tensor
artifact is generated locally; large weights, raw images and safetensors are not
bundled into the repository or promoted into another runtime.

## Deliberate exclusions

Remote/local file loading, video/audio, animation, low-detail policy, extreme
aspect ratios, >4 images, >4-Mpixel sources, >1,014 image spans, >8,192 multimodal
history, multimodal Responses/Messages, batching/concurrency, active-state or
cross-backend restore, automatic partial-protocol recovery, all-format metadata
edge cases, unbounded soak, full visual benchmark/CUDA parity, MTP/DSpark promotion,
full R1, release/packaging/clean-room and runtime promotion remain unsupported or
unqualified. The measured bounded 1M text core and its ownership/resource contracts
are not redefined by this finite image boundary.
