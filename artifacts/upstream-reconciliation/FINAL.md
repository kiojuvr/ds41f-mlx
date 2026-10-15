# Final bounded upstream reconciliation

## Boundary and decision

**CLOSED — no new adoption justified.** Starting clean HEAD was
`68efc635754199cf1252b46ae0ec84fba6477598`. At investigation start
(`2026-10-15T01:52:16Z`), upstream default-branch HEADs were resolved once:

| Repository | Fixed cutoff |
| --- | --- |
| `https://github.com/jundot/omlx.git` | `f82821e2ba750cd7cad14e5152d6b4559960457d` |
| `https://github.com/antirez/ds4.git` (DwarfStar) | `fc80bd695da76ee14bcb86820bf900a5cdfda806` |

Only these exact objects were subsequently fetched into isolated temporary
repositories. No later upstream tip was consulted. Existing donor checkouts,
vendored archives, locks, overlays, installed environments and release material
were not modified.

`DS41F-STABLE-e8c719c` remains unchanged: tag object
`b7151db160df3af8d4743f86e8dbc1f6add43eca`, peeled commit
`68efc635754199cf1252b46ae0ec84fba6477598` (the starting HEAD).
The tag name describes the qualified implementation; it does not peel directly
to implementation commit `e8c719c`.

## Investigation basis

- oMLX source base: `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, attributed
  candidate `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`, and the four selective
  growth/kernel/template/context-copy overlays in `third_party/mtp/`.
- Previously closed oMLX investigation through `3f99250d` was reconciled with
  `docs/runtime-completeness.md`, `docs/performance.md` and
  `docs/context-copy-closure.md`. Source diffs for V4.1/cache mechanisms were
  checked against the actual exported archive, not assumed present from the
  upstream base label. New commit/path inventory covers `3f99250d..f82821e2`.
- DwarfStar comparison is `0aaea5a238fb41a35106a551e73c8409dfb751ac..fc80bd6`:
  all nine intervening commits were screened; shared `ds4.c` prefill, target
  history, verifier frontier and response-boundary changes were inspected
  separately from CUDA/TP kernels and standalone-server policy.
- Admitted MTP graph: `ProductionMTPBackend` -> `ProductionScheduler.step` ->
  `_schedule_waiting` -> qualified DENSE_P0_P7/P6 -> P5 -> singleton
  `OMLXMTPGenerationSession` -> pinned native proposal/verify -> canonical
  quiescence -> immutable paired capture/publication -> retirement.
  `_schedule_waiting` refuses another start while `running` is nonempty.
- OFF retains first-party `model_execution`, the same qualified prefill seam,
  its own target/lifecycle authority, and admitted stateless donor primitives.
  Directory membership in upstream is not production reachability.

## Material dispositions

| Upstream change | Mechanism and disposition |
| --- | --- |
| oMLX `63917f5f`, `f0d4cb53`, `09be4f42`, `a2ee4a2c` (#4354/4360/4365/4363) | **Already selectively solved.** Retain qualified runtime-shape keys, adapted MXFP8 GEMV/lane Sinkhorn, compressed capacity growth and consumed-source context-copy. Rejected FP8 packing bytes, grouped/radix score materialization and unrelated sparse scheduling do not become a backlog. No replacement of the four overlays. |
| oMLX `0eb8df1f`, `3f99250d` | **Conditional/unreachable.** MLX 0.32.3 compatibility is not an upgrade requirement for locked 0.32.2; oQ `ced_tail` calibration replay does not run on official-weight serving. |
| oMLX `ce862486` (#4180), `92d5fcea` / `b01cc5f4` (#4163) | **Failure mechanism not reached in current paired topology.** These absent archive deltas guard double enqueue of an already free-linked victim and clear stale metadata on `free_block(s)`. Paired publication calls permanent eviction only when free count is zero: `get_evictable_blocks` then selects allocated, unlinked refcount-zero blocks. Invalid-pair eviction also selects an allocated published block. Paired release decrements references without enqueueing; clear resets the pool. Production does not invoke generic free-block/SSD-tier pressure loops. Do not copy these as presumed repairs to a demonstrated current failure. SSD staging/temp-file/vision-cache fixes belong to different owners. |
| oMLX `1a270b9d` (#4443) | **Already solved differently.** Background dedup used an unleased lookup then incremented a possibly recycled block. `PairedCheckpointAuthority.acquire`, `acquire_retained` and `retain` use `acquire_cached_block(id, hash)`; paired publication/retention is serialized, with immutable target+DSpark payloads. Generic asynchronous BlockAwarePrefixCache store is not used. |
| oMLX `a789c89a`, `c756df2a` | **Already solved differently.** Generic freshness/store gates lost sub-block tails at parser/text stops. ds41f publishes whole paired root tails after canonical settlement; no full-block rounding or generic scheduler-finish store gate. Existing short-prompt and eviction tests exercise that authority. |
| oMLX `6d175229`, `a987407f` (#4509) | **Relevant, differently owned / reference-only.** Hard-watermark requeue, progress timeout reset and cancelled-prefix persistence can matter even with one active request; they are NOT dismissed as batching-only. However their resume contract is generic KV/SSD plus upstream parser/thinking state. ds41f owns admitted resources, cancellation, complete paired state and fail-closed outcomes; no matching generic memory-admission wait/requeue loop runs in its overridden step. Adopting target-only resume or partially prefetched cancellation snapshots would change accepted ownership/semantics. Settled cancellation already preserves eligible complete pairs; do not add transparent rebuild/replay. |
| oMLX `60fb1ea6`, `d34d17c2` (#4352), `32bfc494` | **Different semantic owner / unsupported extension.** Reasoning-aware stop automata and partial assistant continuation repair upstream parser/Processor paths. ds41f uses DeepSeek recipe conversion, parser, guard and projection, not these generic scheduler response hooks or the new Decisions API. Do not introduce a parallel grammar owner or claim these upstream fixes prove a recipe defect. |
| oMLX `ec13b9ec`, `0c6fd953` (#4510/4376) | **Unsupported policy extension / deferred batching benefit.** New no-repeat-ngram controls are outside generation-policy-v1; sampler interning primarily groups equal-parameter active rows. Keep request-owned qualified sampling/processors and the published policy surface; no new sampling semantics. |
| oMLX `125ca681` (#4366) | **Reachable optimization, reference-only.** Inverts a route permutation with `put_along_axis` instead of a second argsort above 4096 routes. DENSE prefill can reach this; singleton decode/verify cannot at current depth. Original upstream implementation and tests were checked before considering any local implementation. A bounded primitive probe found exact permutation equality and only ~0.10–0.15 ms savings per invocation, including host submission, not a real-model speedup. No material end-to-end benefit established relative to multi-second qualified prefill. Do not change the admitted graph/payload on this evidence or begin a general prefill campaign. This is an optimization intentionally declined, not an unreachable path or correctness issue. |
| oMLX `2c53654f` (#4475) | **Unreachable optimization.** Partial-row padding shares large affine 8-bit QuantizedLinear weight streams in `row_exact_qmv`. Official V4.1 dense projections use FP8/MXFP8 operators, not these affine 8-bit linears; routed experts use their separate MXFP4 gather mechanism. No transplant of a similarly named kernel. |
| oMLX `d128c118`, `7a4ea399`, `10192010`, `41166afa` | **Other models/modes.** Qwen command-buffer scoping, DFlash adaptive block drafting, GLM boundary alignment and padded ArraysCache metadata graphs do not repair current V4.1 singleton state. The rollback delta adds mlx-vlm rotating-cache classes for Gemma, not V4.1 DeepseekV41Cache. UID/batched priming and fairness remain deferred. |
| DwarfStar `77570b4`, `43a91aa`: seed/history/confidence | **Important cross-backend knowledge, already differently implemented.** Incoming seed at pos pairs with target feature pos-1; merge only current capture, maintain rolling history through skips/restores, and use pre-norm HC-head confidence input. ds41f uses qualified target taps and pinned oMLX DSpark context with explicit start/expected offsets, P6 priming and P5 complete-context transfer; it does not use ds4's target-position/capture bookkeeping. No algorithmic reconstruction of DSpark or wholesale DwarfStar runtime replacement. |
| DwarfStar `43a91aa`: indexer threshold | **Correctness issue checked, already absent.** ds4 removed its 1024-row decode sparse threshold because selecting more than model top-k changes attention. ds41f's loaded/first-party indexer uses configured `index_topk` and `min(index_topk, score_length)`, with qualified packed-index dispatch; no extra DwarfStar decode threshold exists. |
| DwarfStar `fc80bd6` | **Already solved differently.** Standalone server could end inside an accepted block, invalidate state and rebuild full context, or retain stale next logits on a sampling-mode change. New retained verifier frontier enables bounded restore/replay with provenance guards. ds41f already settles the canonical/consumed frontier from existing queue/rollback ownership, performs only bounded required suffix materialization, synchronizes, and captures both target and rings; stale/unavailable state burns rather than silently rebuilding. No import of a second snapshot owner, ds4 fallback rebuild or TP acknowledgment protocol. |
| DwarfStar `962ac19`: Engram conversion | **Different reached implementation.** CPU scalar FP8/scale conversion becomes a 128-KiB lookup preserving BF16 rounding. Production ds41f SSD Engram uses its qualified packed-row/MLX path, not `ds4_engram_read` scalar conversion. No table port or numerical-boundary change. |
| DwarfStar `962ac19`, `77570b4`, `43a91aa`, `63059b6`, `f0962e3`, `b14bf62`, `b3afc8e`, `d6b516a` | **Unsupported execution/reference.** CUDA/Spark projection, attention, scratch/copy/stream optimizations, native MXFP4 drafter formats, split heads/TP/RDMA/transport failure containment, ROCm fixtures and Score API are not the admitted Metal/MLX path. Shared C changes were screened rather than classified by commit title alone. No new DENSE sweep/carry/holdout mechanism was identified that warrants changing qualified P0–P7/P5. Fused HC rounding and standalone fallback policy are not accepted numerics. |

Unrelated upstream UI, authentication, evaluator, model-family, uploader and
embedding endpoint changes were screened out by path/scope. This is a bounded
mechanism review, not a claim that every upstream feature has been qualified.

## Validation and stable comparison

- Focused existing tests: **107 passed, 26 subtests passed**, 5.39 s; only existing
  Starlette/FastAPI deprecation warnings. Command and primitive observations are
  retained in `evidence.json`. Tests cover paired snapshots/leases/eviction,
  cancellation/settlement/burn, P5/P6/P7, request sampling and logical sessions.
- The route probe used MLX 0.32.2 on M3 Ultra, not model inference. It does not
  qualify a performance claim or justify adoption. No model benchmark, long-context,
  multi-active, R1, package/build or release campaign was launched.
- **Production code changed: none. Adoption reference: none newly imported.**
  The reviewed reference implementations are listed above. Independent rewrites
  of route inversion, cache pinning, speculative rewind or prefill machinery
  were considered unnecessary and avoided.
- Starting HEAD equals the peeled frozen stable baseline. Task changes are only
  this reconciliation evidence: model/runtime/native sources, source archives,
  overlays, locks and admission identities are byte-identical to that baseline.
  Consequently physical topology, MTP/OFF behavior, ownership and performance
  are unchanged by this task; tests are regression evidence, not fresh real-model
  qualification. No silent fallback or newly measured model speedup is claimed.

## Handoff disposition

**Ready as the starting point for architecture-preserving ownership transfer**
within the existing qualified scope; that phase has not begun. No new implementation
checkpoint or stable release is necessary because no implementation changed.
This report closes direct upstream tracking at the fixed cutoffs; oMLX and
DwarfStar become reference/research sources for the later transition, not ongoing
synchronization targets.

Logical concurrency with **serialized model inference** remains intentional:
concurrent agents/sessions/tools/application work, one active generation, queued
ready generations. Multi-active decode/batching/fairness, transparent MTP restart
persistence, alternate quantizations, expert offload, distributed/TP/RDMA/CUDA,
other models and broader standalone-server capabilities remain outside this lane.
DeepSeek recipe semantic authority and ds41f admission, paired integrity,
lifecycle, resources, containment and application outcomes are unchanged.
Neither ownership transfer nor creation of `dsflash-mlx` occurred.
