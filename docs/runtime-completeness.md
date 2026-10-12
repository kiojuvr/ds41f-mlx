# DeepSeek-V4.1-Flash runtime closure and local default decision

## Decision

**COMPLETE** for the official checkpoint's model/runtime core in the current
single-operator Apple Silicon lane. **No current default-MTP promotion blocker
was found.** Ordinary `mtp-serving-v1` is now the default for `ds41f start`,
`inspect`, `accept` and the direct launcher. OFF remains an explicitly selected
fallback/profile; singleton and historical OFF `quick/full` qualification are
unchanged. This is an operational decision, not a claim of general Internet,
multi-user, all-client or maximum-workload production-release completion.

This review composes retained qualification with current owners/call sites. It
does not rerun or relabel immutable model/state/real-image evidence. Installed
packages remain immutable: a checkout default does not change an older installed
wheel. Use operations' existing one-way publication route, never a source override.

## Completeness against official architecture and reference runtimes

Official checkpoint `inference/config.json` and `inference/model.py` are the
semantic comparison basis, not ds4/oMLX feature lists. The released geometry is
40 target layers, four HC streams, FP8 projections/FP4 experts, 384 routed experts
(top six), shared expert, SWA-128 plus compressed/shared KV and index/candidates,
Engram at layers 1/14, ViT/aligner/delimiters, and three DSpark layers reading
post-Engram/pre-layer means at 37/38/39. CUDA/distributed reference mechanisms
are implementations of those semantics, not required local execution modes.

| Classification | Current boundary / disposition |
| --- | --- |
| COMPLETE | Official weights/config/tokenizer and precision/packing; target projections, HC/Sinkhorn/collapse, MoE routing/shared experts, SWA/compressed KV/index/candidate reuse and RoPE/YaRN. OFF `model_execution` and admitted pinned MTP model execute these, not reduced/requantized substitute checkpoints. |
| COMPLETE | SSD Engram/hash/history and image exclusion; dense DwarfStar/P6 prefill/continuation, exact consumed frontier and same-live P5 transfer. Long text core and ordinary MTP serving evidence have distinct retained scopes. |
| COMPLETE | ViT/aligner/new-span embedding merge, image masks/VL routing, bounded ordinary Vision→DSpark/native MTP and image-discriminated paired reuse. Real-image ordinary closure is separate from the original OFF receipt. |
| COMPLETE | Native DSpark proposal/verification/rollback/commit, request-owned sampling, recipe tools/reasoning/JSON/SSE, paired snapshots, edits/branches/cancellation/retirement and optional logical sessions. No second executable owner, replay repair or backend-finish publication. |
| DEFERRED CAPABILITY | Multi-request native batching/fairness, larger multi-user admission, multiple logical sessions and restart-crossing MTP persistence. Not needed by one active native generation with ordinary queued requests. Persistence would need complete target+DSpark and application-outcome ownership, not target-only SSD reuse. |
| DEFERRED CAPABILITY | Broader multimodal context/input/API, ordinary MTP Responses/Messages, attachments/product UI and separate-machine LAN acceptance. Current clients use ordinary Chat Completions; OFF retains its separate protocols/persistence and isolated PDF Vision. No automatic cross-profile routing. |
| ALTERNATE MODE / OUT OF SCOPE | Multi-Mac/TP/RDMA, CUDA/DGX Spark, GGUF/affine/oQ quantized checkpoints, MoE expert offload and other hosts/models. SSD Engram is implemented; expert offload is a different residency mode, unnecessary for the admitted full-resident backbone on M3 Ultra/512 GB. |
| PERFORMANCE BACKLOG | Kernel compilation amortization, exact fused kernels, copied proposals and compressed/index growth discussed below; no missing operator or execution capability. |
| ACTUAL BLOCKER | None remaining for current primary local operation. OFF-only protocol/persistence users must continue selecting OFF explicitly. |

Comparison inspected ds4 at `0aaea5a` (`ds4.c`, model/server documentation) and
oMLX main at `3f99250d`, including changes after `v0.7.0`. ds4's batched slots,
GGUF/quants, TP, disk-prefix replay and additional API projections are not model
requirements. Its own V4.1 session-batched serving disables DSpark; importing that
mode is not necessary to complete ordinary ds41f MTP. ds41f deliberately keeps
recipe authority and forbids historical-image/session replay on unavailable state.

## oMLX 0.7.0+ delta disposition

No upstream dependency/numerical payload is changed by this promotion.

| Change | Classification and ds41f relevance |
| --- | --- |
| `63917f5f` / #4354: fewer Metal recompilations | Semantics-preserving PERFORMANCE BACKLOG (category 3). Runtime shapes replace changing token/row counts in template keys for FP8 activation, HC, routing and top-k. Applicable ideas to pinned decode and first-party modules, but DwarfStar prefill dispatch is separate; not a missing capability or correctness repair. |
| `f0d4cb53` / #4360: bit-exact decode/long-prefill kernels | PERFORMANCE BACKLOG (3). Packed FP8 activation, MXFP8 GEMV, lane Sinkhorn and attention/index dispatch preserve arithmetic/storage boundaries and retain fallbacks. Decode ideas are relevant; generic long-prefill kernels do not automatically replace ds41f's dense/suffix topology. Upstream bit-exact tests are evidence for upstream inputs, not permission to silently replace admitted kernels here. |
| `a2ee4a2c` / #4363: DSpark context-copy drafts | Semantics-preserving speculative PERFORMANCE BACKLOG (3): full copied windows still target-verify and use sparse proposal distributions; no new model operator. No universal RNG/token bit-identity claim. Adoption needs current recipe horizon/quiescence/sampling compatibility, not just a cherry-pick. |
| Same change: per-request priming | Correctness/isolation fix for upstream interleaved request/UID priming (1), topology-inapplicable in the current ds41f lane (4), not missing capability (2). `ProductionScheduler(max_num_seqs=1)` starts only when no request is running; qualified P6 taps populate that request's rings, P5 installs complete committed context into the sole batch, and retirement drops context before another request starts. Waiting requests do not prime shared host slots. Multi-active generic upstream prefill is not selected. |
| `09be4f42` / #4365: compressed KV/index in-place growth | Semantics-preserving PERFORMANCE BACKLOG (3), particularly long-context decode. Alternating buffers avoid full-context concatenation on short appends; values and fallback copies remain unchanged. Adoption must reconcile single-row extract registry sharing, rollback/truncate and immutable paired capture/acquisition; no mutable alias or unbudgeted double-buffer authority may leak into snapshots. Current concatenation is correct, though slower. |
| `0eb8df1f` / #4311: MLX 0.32.3 compatibility | Conditional correctness/ABI compatibility (1) for an MLX upgrade, currently inapplicable (4): ds41f admits its exact locked MLX/native payload. Upgrading upstream is not promotion work. |
| `3f99250d`: replay `ced_tail` in oQ sensitivity | Correctness fix (1) in alternate quantization/calibration tooling, inapplicable (4) to official-weight serving. ds41f does not run oQ sensitivity replay in production. |

Older upstream index-cache estimation, prefill memory-peak, echoed reasoning,
batched-cache unpadding, expert-offload and alternate-quant fixes were also
checked for ownership applicability. ds41f uses actual recipe capacity, four
complete paired payloads, existing resource leases and DwarfStar/P6 prefill—not
generic oMLX pool estimation/prefill/offload or batched padded caches. Current
recipe reasoning conversion/projection remains authority. No unclosed failure
mechanism in the supported lane follows from those upstream changes.

## Default promotion review

- **Implementation:** selects the existing ordinary backend, not singleton or a
  new model path. TUI already launches this profile. Generic `create_app` keeps
  its OFF default for OFF/reference callers; production launcher passes the
  selected profile explicitly. `ops` also propagates explicit OFF to the child,
  so the new launcher default cannot accidentally override fallback selection.
- **Ordinary API/clients:** health/models, ordinary Chat JSON/SSE, generic tools,
  generation-policy-v1, Browser/TUI logical continuity and accepted OpenCode
  configuration are connected. Supported client/version evidence remains scoped;
  no claim for Rust's OFF seam, Responses/Messages or arbitrary clients is added.
- **Resources:** installed exact execution/semantic assets and host ABI/capacity
  admission remain fail closed. Prompt+reserved output, bounded inline images,
  body limits, four-pair retention, one active native batch and wired lease remain
  unchanged. Large output is a termination budget, not advance KV allocation.
- **Failures:** cancellation shields mutation and settles the same Scheduler ID;
  uncertain retirement burns reuse and fails the engine. Historical-image expiry
  rejects before native start without replay; logical-session loss fails rather
  than reconstructing history. No silent retry, automatic OFF fallback or restart
  persistence is promised. Operator selects OFF intentionally for its contracts.
- **Validation boundary:** affected CLI dispatch/lifespan tests cover omitted
  profile, explicit MTP/OFF/singleton, ordinary acceptance selection and unchanged
  historical qualification defaults. Existing real-model ordinary and Vision
  acceptance remains the execution evidence; no math/state change warrants
  reopening it just to change a selector.

Affected dispatch/lifespan and ordinary API/state/identity regressions pass.
The broader selected test run also exposed four stale fixtures in
`test_production_session_owner.py` / `test_runtime_capacity.py` (missing prepared
image/reasoning fields and the obsolete blanket ordinary-image rejection).
All four reproduce unchanged at parent `fb6f1cd`; they are not production failures
or introduced by the selector change. They were not used to reopen qualified core
or weaken current image/reasoning admission.

Progress is the completed default decision and retained correct fallback, not a
new probe/receipt/test count. Future performance work can be selected independently:
prioritize long-context compressed/index growth and top-k pipeline amortization,
then exact decode kernels and copied proposals only with affected ownership,
numerical and workload validation. Promotion does not wait for that campaign.
