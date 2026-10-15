# Stable runtime implementation-ownership feasibility

**Decision: bounded source transfer is feasible; physical MTP transfer is not a
low-risk first step.** This is EVIDENCE / feasibility, not implementation
permission, a new release, or a replacement architecture.

## Authority and method

Started clean at `a18b46c4514b33361908c6acb9a60cf027eb96ed`.
`DS41F-STABLE-e8c719c` is the behavioral/architectural authority: peeled commit
`68efc635754199cf1252b46ae0ec84fba6477598`, qualified implementation
`e8c719c8a91bd40d92257608c303100e90c7854a`. Its
[baseline](../release/stable-e8c719c/baseline.json), four frozen wheels and embedded
manifests prevail over historical version labels or `runtime_release.json`.
The only starting-tree delta from the tag was upstream-reconciliation evidence.

Direct upstream tracking is closed at oMLX
`f82821e2ba750cd7cad14e5152d6b4559960457d` and DwarfStar
`fc80bd695da76ee14bcb86820bf900a5cdfda806`. Neither is a transfer donor merely
because it is newer. No network/upstream-tip investigation was performed here.

Method: trace admitted callers and branch conditions, then inspect their existing
implementations. The oMLX export was extracted into temporary scratch and its
four **existing** overlays applied in order there, not to production. Twenty-two
selected reference identities match the frozen MTP-native wheel manifest;
[identity evidence](../artifacts/ownership-feasibility/evidence.json) records them.
MLX-LM `generate.py` and shared sampling were also checked against frozen OFF
identities. This is static reachability/implementation review, not a new exhaustive
dynamic trace, native-symbol coverage campaign or performance qualification.

## Ownership terms (not interchangeable)

- **Source ownership:** maintaining the delivered source and its changes. The
  repository already delivers attributed archives, patches and binary artifacts;
  that alone does not make their engine ds41f-implemented.
- **Implementation ownership:** responsibility for the executing algorithm/state
  implementation, including debugging and faithful maintenance.
- **Architecture authority:** who determines scheduling, topology, ownership
  transitions and permitted recovery. The stable contract and ds41f admission
  govern this; externally implemented physical mechanisms remain binding.
- **Semantic authority:** DeepSeek recipe determines conversion, prompt grammar,
  parsing, thinking/tool termination and response projection; official assets
  determine model meaning. A local adapter must not become a second grammar.
- **External library dependency:** MLX, MLX-LM, recipe and input/transport libraries
  can remain dependencies without owning ds41f application architecture.
- **Reusable primitive dependency:** e.g. tensor operators, cache base classes,
  route sort and stateless kernels. These are not a second scheduler.
- **Packaging/build dependency:** compilers, CMake, Cargo, SDKs, locks and wheel
  assembly. Build provenance is not runtime admission or semantic authority.

External license/attribution survives a source transfer. MLX dependence is
compatible with ds41f architecture ownership. Removing repository names is not
an independence criterion.

## Admitted graph and minimum execution surface

### Normal MTP (`mtp-serving-v1`)

`serve.main` selects `ProductionMTPBackend`; inherited `internal_mtp.load`
imports `omlx.scheduler` (patch installation), applies MTP/rollback patches,
uses `omlx.patches.deepseek_v41.loading.load(preserve_mtp=True)` and configures
DSpark depth 5. The loaded target, blocks, cache, SSD Engram and DSpark stages
are donor implementations, **not** OFF's `model_execution` objects.

`ordinary_response -> _enqueue -> Scheduler.add_request -> waiting`;
`ProductionScheduler.step -> _process_pending_aborts -> _schedule_waiting`
refuses a start while `running` is nonempty. Paired lookup/restore feeds
`internal_mtp._start -> DeferredPrefillAppend` (ds41f P0–P7/P6), target taps and
`dspark_append_context`; P5 inserts only the held-out terminal into the sole
Scheduler-owned MLX-LM `BatchGenerator`. `OMLXMTPGenerationSession.start` installs
committed rings and calls `insert(caches=..., all_tokens=...)`, then `next` for
bootstrap. Subsequent `next_token -> BatchGenerator.next -> GenerationBatch`
uses the frozen oMLX MTP patches.

Reached physical core: `mlx_lm_mtp.batch_generator.apply`, `_MtpState`,
`_run_verify_cycle_chain`, `_chain_next_drafts`, `_context_copy_drafts`,
`_chain_rollback`, emission queue and semantic horizon; V4.1 `mtp.DSparkMixin`,
`dspark.proposal_forward` and shared `deepseek_v4_dspark.DSparkContextCache` /
committed-context hooks. The chain verifies `[next_main, drafts...]` with
`n_confirmed=1`, chooses the accepted prefix, queues original source-row
logprobs, rolls back the target and schedules successor proposals in the existing
order. AcceptanceDepthController follows acceptance counts, **not elapsed time**.
Consumed-source context-copy is an existing reachable proposal source, not a
new algorithm to design.

`_advance_application -> _settle -> owner.quiesce` uses ds41f canonical connection
and bounded quiescence around those native owners. `extract_cache`, remove/filter
and `close` still execute MLX-LM and patched lifetime mechanics. Recipe consumes
canonical events; ds41f validates aligned target/rings, closes the executable
owner, captures immutable pairs, publishes through PagedCache, and retires.
Cancellation enters that same settlement lane; socket progress is not canonical
acknowledgment or application completion.

### Explicit standard-OFF

`DeepSeekRecipeRuntimeBackend.load -> OmlxRuntime.load_model` with
`preserve_mtp=False`, SSD Engram and no expert-offload override selects admitted
`ds41f_mlx.model_execution.loading`, with no donor-model fallback.
`DwarfStarMLXPrefillSession -> DenseP0P7PrefillSession -> DeferredPrefillAppend`
(or tool-boundary suffix append) -> P5 -> `TargetGenerationSession ->
TargetForwardTransaction -> DecodeStateProducer` owns same-list, all-40-layer
state production. OFF continuation/persistence reconstructs ds41f's seven-slot
cache. Neither ordinary `ProductionScheduler` nor patched `BatchGenerator`
is its production generation driver. Historical `OMLXGenerationSession` and
failed first-party MTP code are not selected just because they exist.

External OFF execution is narrower: MLX; MLX-LM `ArraysCache`, `SwitchLinear`,
SwiGLU and logits processors; oMLX `utils.sampling`, GLM `fast` / native bundle,
NAX detection, `m5_gather_qmm` (including its conditional NAX helper),
`moe_routes.sort_routes`; tokenizer/input libraries and recipe/native semantics.
Donor `encoding`/`tool_parser` metadata imports do not own ordinary recipe
encoding/parsing. MLX-LM generation stream support may be imported; this does
not mean its generation driver runs OFF.

### Actual minimum, versus installed closure

The **minimum current external implementation-ownership surface** is the MTP
physical target/DSpark engine + patched MLX-LM driver + PagedCache metadata and
lease machinery, shared sampling and acceleration primitives, MLX numerical
execution, and recipe/input/transport primitives. OFF has no external model
execution-policy owner; MTP does. There is **no DwarfStar executable dependency**.

The semantic/image/support and import/link closure is larger than the mechanisms
above. Scheduler initialization imports generic cache/VLM/model-patch helpers,
copies the tokenizer, probes model properties, initializes statistics, and installs
patches. Shutdown calls priming/row-registry cleanup. These are real source/load
obligations, not generic serving authority. `paged_ssd_cache_dir` is unset;
BlockAwarePrefixCache/SSD-tier async stores are not instantiated. Upstream
`preflight_or_raise` is called by admission but returns with its memory guard
disabled; ds41f capacity/resource admission remains authoritative.

Do **not** prune packages from this map: frozen manifests bind complete installed
payloads. Recipe image preparation can reach OpenCV core/imgproc/imgcodecs and
codec/compression dependencies; other loaded OpenCV modules are ABI/loadability,
not a second model engine. Transformers/tokenizers, NumPy, Pillow, the mlx-vlm
embedding-result container, FastAPI/Starlette/AnyIO and HTTP/SSE helpers execute
input or delivery functions. See [admission](runtime-admission.md) for the exact
semantic-native versus host-link distinction. Build tools and unused optional
package contents do not become runtime authority by lockfile membership.

## Component disposition

Paths below refer to the **frozen exported/overlaid** implementation, not upstream
HEAD. Classification concerns a future move, not permission to rewrite.

| Component / actual boundary | Classification | Reason and practical value |
| --- | --- | --- |
| ds41f admission, executor serialization, generation-policy normalization, resource/wired leases, canonical connection/quiescence, recipe guard, application certificates, logical sessions and delivery | **ALREADY_DS41F_OWNED** | Independently decides admission, lifecycle, failure containment and outcomes. `internal_mtp`'s historical name does not negate production inheritance. Preserve these owners; no duplicate controller needed. |
| DENSE_P0_P7/P6/P5: `prefill_fp8_mlx` runner/sweep/carry, deferred decoder suffix, P7 Engram overlap, handoff certificate and revocation | **ALREADY_DS41F_OWNED** | DwarfStar topology already internalized as MLX implementation. `dwarfstar_prefill` compatibility name is not a ds4 call. Transfer of ds4 C/CUDA/TP code has no practical value here. MTP still supplies external block/math/storage objects underneath it. |
| OFF `model_execution`: loading/checkpoint, target math, cache specialization, Engram hash/store/prefetch, `TargetGenerationSession`, target/state production, OFF persistence | **ALREADY_DS41F_OWNED** | Attributed structure already transferred and qualified; external base classes/kernels remain intentional. Useful precedent, not evidence that MTP is transferred. |
| `Scheduler` request/config/output records and exceptions | **SAFE_SOURCE_TRANSFER** | Faithful attributed records/exceptions can move with minimal adaptation. Benefit only as support for an eventual substrate extraction; moving DTOs alone does not remove physical ownership and is not the first experiment. Preserve constructor/post-init/status contracts. |
| Inherited Scheduler init, `add_request`, `_resolve_generation_prompt_start`, abort set/drain, shutdown and associated helpers | **STRICT_CONFORMANCE_TRANSFER** | Small *authority* boundary, large import/initialization closure. FCFS queue cap is 32; duplicates rejected; abort set drains on worker into ds41f override. Extract existing structure only after tracing initializer/cleanup helpers and dormant fields. Value: eliminate temporary generic server substrate and clarify maintenance, not add a new scheduler. |
| `ProductionScheduler.step`, `_schedule_waiting`, `_do_abort_request`, `_retire`, paired/session publication | **ALREADY_DS41F_OWNED** | These override generic upstream physical admission, decode orchestration, response/parser and cache-finish policy. Active cancellation is ds41f settlement; external abort enqueue is only plumbing. |
| MLX-LM `BatchGenerator` / `GenerationBatch`, token buffers, matchers, cache merge/extract/filter/remove, patched constructors and next | **HIGH_RISK** | Constructor already forwards; pending token pipeline, row extraction, async evaluation and finish filtering affect physical state lifetime. Whole attributed source relocation might later be possible, but reducing it to a 'simple singleton loop' is redesign. Value: direct maintenance of critical lifetime code; not currently low-risk. |
| oMLX MTP proposal/verify/rollback/commit, semantic horizon, prompt-priming/committed-context hooks, context-copy and acceptance controller | **HIGH_RISK** | Coupled to queue-ahead state, original verification rows, processor snapshots/RNG, lazy patch installation and target/ring offsets. Preserve complete reached structure and overlay order; an independently coded MTP transaction is not transfer. Future benefit is removal of temporary physical substrate and Flash-family maintenance. |
| MTP `deepseek_v41` target loading/block/math/head/Engram/vision and `dspark` stages | **STRICT_CONFORMANCE_TRANSFER** | Existing implementation can plausibly be attributed/extracted, as OFF precedent demonstrates, but MTP mixins, hidden taps, preserve-MTP tensor inventory, row geometry and verification flags must remain exact. Reusing OFF classes without those contracts is not safe. Benefit: coherent Flash-family source maintenance; no arithmetic unification in this task. |
| MTP seven-slot `DeepseekV41Cache`, `growth` capacity registry and `DSparkContextCache` rings | **STRICT_CONFORMANCE_TRANSFER** | Container source move is plausible only with exact from_state/extract/advance/rollback/registry behavior. Capacity views and COW snapshots are performance/lifetime contracts, not interchangeable lists. Value: maintenance of paired critical state. Changing buffer strategy/physical ring layout is **HIGH_RISK**. |
| `cache.paged_cache`: CacheBlock, free queue, hash map, tail index, locking/refcount/LRU, callbacks; `interface`/`stats` helpers | **STRICT_CONFORMANCE_TRANSFER** | Host metadata, not tensor KV paging in this use. Whole-pair root tails use the existing manager. Allocation, atomic `(id,hash)` acquisition, refcount-zero eviction and callback cleanup must retain one authority. Useful later to remove generic package coupling; extracting a fresh 'tiny LRU' would lose structure and be **HIGH_RISK**. |
| `PairedCheckpointAuthority`, snapshots, validation, image identity, immutable capture/restore and logical-session pins | **ALREADY_DS41F_OWNED** | Payload/paired integrity is local; indexing and physical lease authority remain external PagedCache. No independent lookup/LRU/rewind is needed. |
| Shared `omlx.utils.sampling` | **SAFE_SOURCE_TRANSFER** | Self-contained Apache-2.0 module importing only MLX, math and typing. Verbatim transfer preserves filter composition and MTP sampler attributes/densities; no new sampler. Value: one directly maintainable shared request-sampling boundary for both profiles, without relocating the engine. Exact numerical/RNG comparison remains mandatory. |
| MLX-LM logits processors, ArraysCache/SwitchLinear/SwiGLU; MLX core/nn/utils, allocator, streams, random, compilation and Metal | **KEEP_EXTERNAL** | Appropriate reusable/numerical primitives with strong exact-byte qualification. Transferring them offers no useful runtime-policy independence. MLX is numerical authority, not merely host plumbing. |
| oMLX stateless `moe_routes`, gate/up activation, NAX probe and GLM `fast` Python adapters | **KEEP_EXTERNAL** | Existing stable primitives are already subordinate to local OFF dispatch / frozen MTP math. Source transfer is technically plausible, but currently little practical benefit. A packaging namespace move alone is not improved state ownership. Preserve selected native/portable behavior. |
| GLM extension/C++ custom ops, common headers, dylib/metallib and generated Metal kernels for packed attention, grouped/MXFP4 experts, DSpark/index projections | **KEEP_EXTERNAL** as primitives; modified implementation **HIGH_RISK** | Frozen attributed bundle and ds41f build tooling already provide distribution control. Native synchronization/ABI/layout/rounding are not LLM rewrite targets. If later source extraction is justified, transfer all required existing inputs/build structure; no replacement kernels or repacking. Not all 23 admitted symbols or compiled model kernels execute on a given request. |
| DeepSeek recipe Python/Rust native conversion/encoding/parser/projection, consuming EOF and preview extensions; tokenizer/assets and image semantics | **KEEP_EXTERNAL** | Desired semantic/protocol authority, already exactly delivered with separate MTP/OFF native identities. Local wheel custody does not justify duplicating grammar/template/parser. Future Flash support should use official semantic authority, not a parallel interpretation. |
| Input/transport libraries and OS SSD/Metal/framework services | **KEEP_EXTERNAL** | Stable functional dependencies; ds41f owns application policy around them. Input libraries can change model meaning, so pinning/qualification remains. No meaningful independence from vendoring their internals. |
| Archive/overlay/locks/native-bundle delivery and execution admission tooling | **ALREADY_DS41F_OWNED**; compilers/Cargo/CMake/SDK/wheel tooling **KEEP_EXTERNAL** | Source/build custody is already local. Reproducible distribution can improve separately without changing execution ownership. Unused transitive audio/evaluator/server/build packages are not candidates by name alone. |
| Replacement scheduler/cache/MTP algorithm, multi-active generation, transparent replay/resume or unified redesigned OFF/MTP engine | **ARCHITECTURAL_RESEARCH** | Requires changing accepted topology, not moving source. Outside this program; historical failed first-party MTP line is negative evidence and must not be revived. |

## Required preservation references

1. Frozen four profile wheels/manifests and stable evidence, **not a same-version
   install**. MTP recipe uses the consuming-EOF native candidate; OFF retains its
   separately qualified native wheel. Never substitute one for the other.
2. `third_party/mtp/omlx-source.tar.gz` plus ordered compressed-growth,
   bit-exact-kernels, template-reuse and context-copy overlays and their JSON
   before/after hashes. Base revision `4d4f5a...` alone is insufficient. Preserve
   per-file MIT/Apache notices, common MLX/MLX-Serve notices and build attribution.
3. Frozen MLX-LM `generate.py` (`4a3bf57f...`) and cache/sample utility structure,
   alongside oMLX patches; do not preserve one side of a monkey-patched interface
   while silently replacing the other. Shared sampling hash is
   `8cd19ae59dd5802b38ae1825098f0ecdfdb13c7ed821bfcab32a24864b6fc131`.
4. ds41f qualified `prefill_fp8_mlx`, OFF `model_execution` and lifecycle/state
   owners, with DwarfStar-derived architecture notes and historical numerical
   evidence as provenance. DwarfStar remains a topology/reference oracle, not a
   fresh executable transplant candidate.
5. R1 immutable seams and exact protocol/recovery records, plus stable
   `reference-reconciliation.json`: the original verifier's five interface
   mismatches remain recorded FAIL. Maintained equivalents are necessary, not
   permission to rewrite expected records or claim unmodified R1 PASS.

## What would prove an architecture-preserving transfer?

A passing text example is insufficient. Record baseline and candidate side by
side under the same profile, weights, tokenizer, MLX/native bytes, policy, seed,
prompt/cache frontiers and generated-token count. Preserve:

- **Structural trace:** one worker/active executable batch and UID; FCFS queued
  readiness, 32 waiting cap, duplicate-ID refusal; pending abort before admission;
  no generic SSD store/memory requeue/parser path introduced. Same P0–P7 segment
  modes/frontiers/cone rows, P7 tile/read/release order and P6 complete tail taps.
  P5 executes the held-out terminal exactly once, revokes producer capability,
  passes the existing cache/rings, with replay=0 and full-cache-repack=0.
- **MTP trace:** same proposal source/native-versus-copy counts, acceptance-driven
  depth and verify shapes, `n_confirmed`, target taps and DSpark feature positions;
  same processors and snapshots; original source-row logprobs, queue ordering,
  accepted-prefix rollback, successor draft submission/eval/sync order. No extra
  per-token forward, sampling draw, host readback, tensor materialization, copy,
  weight owner or synchronization barrier. Greedy makes no RNG draws; stochastic
  comparison must include filtered q/p densities and exact draw ordering/state,
  not merely matching output frequency. Timing must not select verification depth.
- **State invariants:** 40 target caches with seven slots (offset, packed window,
  compressed KV/index, incomplete compressor KV/gate, Engram history); correct
  ratio/layout/dtype/shape, logical versus allocated capacity, growth registry
  sharing/truncation and snapshot COW. Three DSpark physical rings of capacity
  128 with keys length `min(C,128)` and absolute offsets C; drafts never enter
  committed context. At settlement all target sizes/ring offsets equal canonical
  consumed C; queue empty, pending prediction retired, owner closed exactly once.
  OFF retains same cache objects/list and publishes history only after its full
  all-layer commit barrier; unconsumed lookahead never becomes idle history.
- **Pair/lease invariants:** complete immutable target+ring payloads, independent
  mutable handles on restore, same image/prefix identity; null block reservation,
  hash/tail selection and bounded hints, atomic expected-hash acquisition,
  reference changes and LRU victim order. Pins survive hint loss; no pinned
  eviction, double free-link, stale-hash acquisition, target-only hit or rewind.
  Release-to-evict is not free-list enqueue. Callback eviction/clear removes
  payloads and accounts tokens once. Same extendability/holdout envelope rule.
- **Semantics/failure:** exact recipe JSON/SSE/tool/thinking/EOF and reconstruction
  records; canonical versus consumed/emitted/delivered frontiers distinct.
  Cancel queued, before lazy activation, during verification, after native
  terminal and before transport completion; branch/append/eviction/pin churn,
  resource exhaustion and injected forward/retirement failures. Failure burns
  aliases and fences outcomes, never recovers by replay or publishes partial
  tools/pairs. Restore wired/allocator leases and drop priming/row registrations.
- **Performance/resource comparison:** matched cold, shape-transition and warm
  prefill; TTFT split into queue/lookup/restore/suffix/P5/first native decode;
  per-cycle/overall decode, draft considered/accepted statistics, settlement,
  paired capture/restore, SSD Engram submissions and fallback count. Track peak
  active/cached/wired memory and residency, allocation/copy/readback/compile
  counts and growth-buffer retention; compare explicit native and portable lanes
  separately. Repeated paired-continuation and bounded long-context fixtures
  must retain the qualified behavior, not trade memory for throughput silently.

Use stable `evidence/performance-comparison.json`, MTP/OFF acceptance and retained
long-context/native/template/context-copy evidence at their recorded scope.
The frozen comparison was a practical sanity check, not a statistical SLA.
Future work must predeclare matched fixtures/repetitions and a material-regression
budget before measurement; there is no defensible invented universal percentage
in this feasibility study. Exact structural/state discrepancies fail even if
average throughput improves. A noisy timing result is inconclusive, not PASS.

## At most one first experiment (recommended, not performed)

**Attributed verbatim transfer of the complete shared `omlx.utils.sampling`
module into a ds41f-owned module, with only the shared backend sampler import
redirected.** Keep MLX primitives and the physical MTP implementation external.
Do not simultaneously move Scheduler records, change filters/processors,
compile the functions, rename sampler attributes or prune apparently unused
helpers. Retain license and donor SHA; any compatibility label changes must not
alter the generation-policy API or create a second effective sampler factory.

Why this one: 257 lines, no queues/cache/lifetime ownership, one common caller
(`DeepSeekRecipeRuntimeBackend.make_sampler`, inherited by MTP), both profiles,
and a real future sampling-maintenance/Flash-family boundary. It exercises
attributed transfer, admitted import routing and conformance without recreating
an engine. It **does not remove the oMLX package** or claim MTP independence;
its value is direct source maintenance at a shared qualified decision boundary.

Before redirecting, compare the frozen module with the transferred source;
allow only attribution/import-location adaptation. Compare all admitted filter
combinations, tie/cutoff/zero/large-k cases and invalid inputs, shapes/dtypes,
`_mtp_sampling_logits`, `sample_with_logprobs(rowwise=...)`,
`_mtp_batch_sampling_key` and exposed parameter attributes. Preserve float32
mass/density calculations, chunked top-k, filter order, absence of `mx.compile`,
seed/thread/stream RNG behavior and draw counts. No 'equivalent' new sampler.
Then run maintained generation-policy and MTP lifecycle/semantic tests plus
matched installed MTP/OFF greedy/stochastic/tools/cancellation fixtures; compare
proposal acceptance, outcomes, latency and memory against the frozen baseline.
Version/hash/import admission must be deliberately qualified in a separate
candidate, never bypassed or automatically resealed. Stop on any discrepancy.

## Task validation

Selected frozen-reference hash checks: 22/22 match. Focused development tests
(generation policy, production MTP/session owner, P5, OFF target/forward):
**79 passed, 14 subtests passed, 4 failed**. All four failures are HTTP-route
parametrizations stopped by `mtp_identity.config`: development `.venv` resolves
oMLX outside its selected environment, correctly refusing installed-profile
admission. No bypass/environment repair was attempted. This is not an installed
qualification or full test PASS; no new model/performance claim is made.
Production/model/native/dependency/tool/test/release diff against the stable tag
is empty. Command and limitation are recorded in the identity evidence.

This recommendation authorizes no implementation in this task. Larger physical
transfers remain bounded feasibility candidates, not a selected backlog.
Sequence remains feasibility → bounded preserving transfers → independent
qualification baseline → later extraction into `dsflash-mlx` → sole production
source authority there. No runtime implementation, stable artifact, replacement
kernel/scheduler/cache/speculation machinery or new repository was created here.
