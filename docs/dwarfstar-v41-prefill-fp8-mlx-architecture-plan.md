# DwarfStar V4.1 prefill on official FP8/MLX — architecture plan

Status: **P0-P7 implemented/qualified within the new path; P8 optimization search is complete; Milestone 6 performance is qualified through 200K.**  This document started as an architecture plan; current status notes distinguish implemented package evidence from remaining serving/finalization work.

## Decision

Port the DwarfStar V4.1 prefill architecture onto the official DeepSeek-V4.1-Flash FP8 checkpoint by using the already-loaded oMLX/MLX model modules and cache formats as the mathematical/data substrate, while moving execution ownership, lifetime, chunking, publication, and handoff to a DwarfStar-style prefill package.

Do **not** port DwarfStar GGUF kernels or quantization formats directly.  DwarfStar is the prefill topology authority; official checkpoint semantics, oMLX FP8/MLX operators, and reviewed ds41f contracts are the model-math authority.

## Non-negotiable guardrail

Until the complete architecture package is connected end-to-end, performance observations are diagnostic only.  A 2-token or short smoke may be used to prove “not broken”, cache-frontier progression, or no-replay handoff, but speed deltas must not be used to adopt/reject pieces before the connected package includes:

1. DwarfStar sweep/chunk planner;
2. request-owned carry arena;
3. official FP8/MLX block execution seam;
4. publication/frontier ownership;
5. live `DeepseekV41Cache` handoff to `OMLXGenerationSession`;
6. no reference vertical-slice fallback;
7. the DwarfStar structural acceptance gate below is satisfied.

## Current-state analysis

### Present and reusable

- `ds41f_mlx/native_prefill.py` and `ds41f_mlx/native/ds41f_prefill_native.*` preserve DwarfStar sweep planning, chunk geometry, checkpoint-invalid transaction states, carry-buffer roles, encoder/deferred-decoder phases, and prefetch/read-ahead command vocabulary.
- `ds41f_mlx/runtime/dwarfstar_prefill.py` already executes the official oMLX `LanguageModel` layer loop with request-local `DeepseekV41Cache`, skips final prefix logits for serving, and hands cache directly to decode.
- oMLX DeepSeek V4.1 provides official-FP8-compatible MLX modules: `LanguageModel`, `Block`, `Attention`, `Compressor`, `Indexer`, `MoE`, `Engram`, `QuantizedProjection`, packed activation formats, and `DeepseekV41Cache` slots.
- Existing qualification artifacts cover primitive semantics, publication ownership, Engram, attention, MoE/HC, final norm/head, no-replay decode admission, and backend-local correctness policy.

### Not acceptable as production prefill

- `DwarfStarPrefillVerticalSliceExecutor` / `OfficialModelMath` / tools validators are correctness fixtures and must remain forbidden on the serving hot path.
- Current `DwarfStarMLXPrefillSession` is a practical oMLX one-chunk prefill substrate, not the final DwarfStar architecture: it lacks static carry arena, real chunked sweep, deferred decoder suffix, graph reuse, weight/expert scheduling, and strict materialization boundaries.
- Native C/Metal prefill currently owns buffers/plans and small primitives only; it does not execute full official model math.

## Target package layout

Add a production prefill package, separated from validation and historical prototypes:

```text
ds41f_mlx/prefill_fp8_mlx/
  __init__.py
  planner.py          # DwarfStar sweep/chunk plan adapter over native_prefill
  arena.py            # request-owned carry/state slots; no validation copies
  executor.py         # top-level prefill session and transaction lifecycle
  block_runner.py     # official FP8/MLX block execution seam
  publications.py     # shared kv/index/candidate/top-k ownership model
  handoff.py          # live DeepseekV41Cache result for OMLXGenerationSession
  guards.py           # no-reference-hot-path and materialization guards
  telemetry.py        # structural timings/counters, not success metrics
```

The existing `ds41f_mlx/runtime/dwarfstar_prefill.py` should become a compatibility facade that delegates to this package once the package is fully connected.  Until then it may remain the current substrate.

## Runtime architecture

### End-to-end pipeline

```text
recipe/deepseek token IDs
  ↓
DwarfStarFP8MLXPrefillSession.prefill(prefix)
  ↓
SweepPlanner: native DwarfStar V4.1 sweep plan
  ↓
RequestArena: token ids, hashes, HC carry ping-pong, pre-mix, shared publications
  ↓
OfficialFP8MLXBlockRunner: oMLX layer/Engram/block modules over official FP8 checkpoint
  ↓
PublicationManager: kv/index/candidate/top-k/source ownership commits
  ↓
P5 live-cache admission: validate committed cache; transfer the existing list once
  ↓
LivePrefillResult(live_cache, complete prefix token IDs, frontier)
  ↓
OMLXGenerationSession.from_prefilled_cache(...)
```

### State authority

`DeepseekV41Cache` is the only production decode handoff state.  The prefill package may maintain transient arena aliases during the sweep, and those aliases must not be forced into full cache materialization/repacking at intermediate sweep or chunk boundaries.  By final commit/handoff, every persistent item required for continuation must converge once into the live cache:

- slot 0: layer frontier/offset;
- slot 1: packed window KV;
- slot 2: packed compressed KV;
- slot 3: packed index K;
- slot 4/5: compressor pending KV/gates;
- slot 6: Engram hash history.

Publication records are metadata/provenance over those cache slots and shared dictionaries; they must not become a second persistent tensor authority.

### Block execution seam

Initial production block math uses oMLX modules directly, not NumPy validators:

- embedding: `LanguageModel.embed`;
- Engram hashes: `LanguageModel._hasher`;
- Engram layers: `layer.engram`;
- transformer block: `layer(h, pre, row_cache, shared, start, image_mask)`;
- packed KV/index/candidates: oMLX `Attention`, `Compressor`, `Indexer`, packed kernels;
- HC/MoE: oMLX `Block`, `MoE`, fused HC kernels where selected by MLX;
- decode handoff: existing `OMLXGenerationSession`.

The runner may introduce chunked row execution only where oMLX cache semantics can be preserved exactly: each chunk executes at absolute `start + offset`, consumes prior committed local/shared state, and publishes only according to the sweep transaction rules.

## DwarfStar mechanisms to port

### P0 — package skeleton and guards

- Introduce production package names and no-reference guard.
- Explicitly forbid imports/calls to `dwarfstar_prefill_slice`, `official_model_math`, and validation-tool block helpers.
- Add telemetry fields for architecture completion, but no success/failure performance gate.

### P1 — planner as execution authority

- Wrap `native_prefill.native_sweep_config_static/build_sweep_plan` into `SweepPlanner`.
- Normalize command kinds into Python enums: begin-invalid, prefetch-engram, SSD read-ahead, begin-layer, encode-rows, decoder-prepare-suffix, swap, publish, output/read, commit.
- Preserve DwarfStar chunk policy: 2K/4K/8K caps, encoder full-row phase, decoder suffix phase, deferred decoder candidate.

### P2 — request arena

- Own request-local tensors: tokens, input IDs, hashes/history, `h` HC tensor, `pre`, optional chunk views, and `shared` publication dictionary.
- Use ping-pong/alias semantics for `h` and `pre`; do not retain per-layer copies or digests in hot path.
- Define materialization boundaries: setup, command batch, publication, handoff.  No CPU conversion except explicit diagnostics.

### P3 — official FP8/MLX block runner

- Reuse the reviewed official oMLX/MLX mathematical operations and cache semantics from the current substrate.
- Replace the current substrate's execution structure with DwarfStar sweep/lifetime topology; do not preserve or wrap a whole-prefix `for layer in layers` loop as production prefill.
- Drive block/Engram/cache work from sweep commands and publication frontiers, passing correct absolute positions and cache/arena views for chunks.
- Keep final logits optional/off by default for serving prefill.

### P4 — publication manager

- Implement source/consumer ownership metadata matching the model topology:
  - source compressed KV generations at configured source layers;
  - index K generations and refreshes;
  - candidate source and consumer blocks;
  - top-k/index ownership advancement.
- Publication is transaction-local until commit; failed sweeps must not expose partially committed state.

### P5 — live-cache handoff

- P3/P4 maintain the request-local live `DeepseekV41Cache` directly; P5 validates and transfers that same cache list once to decode. Do not create a fresh cache, merge/repack, translate tensors, or reconstruct state from arena/publication records.
- Require complete request-owned `prefix_token_ids` explicitly; the last sweep arena owns only that sweep's tokens. Validate committed prefill/publication transactions, equal offsets matching complete prefix length, expected packed/pending slot geometry, and required Engram history without tensor-content inspection.
- Hand the same cache to `OMLXGenerationSession.from_prefilled_cache` without `PrefillContinuationState` export or adapter re-admission. Revoke prefill execution authority; after bootstrap the `BatchGenerator`/`GenerationBatch` scheduler is the sole active decode authority. Repeated handoff/start fails closed.
- Terminal-token holdout contract: `full_prompt = prefix_token_ids + [terminal_prompt_token]`. P3/P4 cache only the prefix and suppress final prefix logits. P5 calls `session.start(terminal_prompt_token)` exactly once: pre-start frontiers equal `len(prefix_token_ids)`, post-bootstrap frontiers equal `len(prefix_token_ids)+1`, and prefix replay is zero. Never prefill the terminal token and then pass it again to `start()`.

Implementation/real P5 qualification: [live-cache handoff closeout](p5-live-cache-handoff.md). Production selection remains a separate task; P7 is now qualified for the scoped base backend below.

### P6 — deferred decoder / long-context sweep

**Implemented and real-qualified in the new path.** The controlling package-level record is [P6 deferred-decoder authority and architecture](p6-deferred-decoder-architecture.md). The implementation is request-owned `DeferredPrefillAppend`: true encoder/source-only sweeps, final new-range decoder completion with a 2541-row input cone, explicit sealed-commit continuation, canonical public slot0, failure/rebuild behavior, and same-cache P5 handoff.

Real qualification environment: Python 3.13.15, MLX 0.32.2, NumPy 2.3.5, oMLX 0.7.0.dev2 at `b390b31e0c6831225fed0f24d278eb1db7fcb68b`, official DeepSeek-V4.1-Flash checkpoint, `preserve_mtp=False`, `engram_ssd_offload=True`.

Qualified cases: `complete-16384`, `pending-16384`, `tiny-16385`, `A-24577`, matched-geometry non-deferred control, `B-fresh-49155`, `B-continued C24578 -> T49155`, failure/rebuild, and P5 bootstrap/decode. Evidence records same live-cache handoff, `prompt_replay_count = 0`, `full_cache_repack_count = 0`, and no `PrefillContinuationState` export/repack path.

ArchitectureCompletion status local to the new path: `deferred_decoder_suffix_lifetime = qualified`; `dwarfstar_carry_lifetime = qualified`.

### P7 — overlap and scheduling

**Qualified for scoped base backend:** `P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM = QUALIFIED`; see [P7 overlap/scheduling architecture](p7-overlap-scheduling-architecture.md).  Qualified environment: Python 3.13.15, MLX 0.32.2, NumPy 2.3.5, oMLX 0.7.0.dev2 at `b390b31e0c6831225fed0f24d278eb1db7fcb68b`; loader config `preserve_mtp=False`, `engram_ssd_offload=True`, `moe_expert_offload_resident_fraction=None`.  The qualification scope is not generalized to `EXPERT_OFFLOAD`.

Qualified cases: `P7 complete-2048`, `P7 complete-8192`, `P7 complete-16384`, `P7 pending-16384`, `P7 A-24577`, `P7 scheduling-disabled control`, and `P7 failure/drain/reuse`.  Real Engram evidence: 2048 background reads = 2 / foreground fallback = 0; 8192 background = 8 / foreground = 0; 16384 background = 16 / foreground = 0; all qualifying P7-enabled paths have foreground fallback = 0.  `P7_ENGRAM_TILE = 2048` because pinned oMLX `EngramPrefetch` has a 16 MiB request limit: 8192-token full Engram requests are rejected, while 2048-token microrequests are donor-admissible.

ArchitectureCompletion status local to the new path: `deferred_decoder_suffix_lifetime = qualified`; `dwarfstar_carry_lifetime = qualified`; `p7_full_resident_backbone_ssd_engram = qualified`; `p0_p7_structural_gate = qualified`. This does not promote P8, production selector, performance gate/global runtime completion, or expert offload.

### P8 — graph/reuse optimization pass

**Complete.** P8 optimization search is closed with decision `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`.

Tile-native carry remains implemented as experimental code behind disabled switch `DS41F_P8_TILE_NATIVE_CARRY=1`; default is OFF and it is not production-selected.  The candidate eliminated dense source/encoder `_write_rows` and dense layer-transport slices for the 16384 A/B case, but warm median improvement was only ~0.092659 s (~0.57%), inside same-revision dense run-to-run noise, while the sampled final-boundary cache proxy was ~+1.5 GB.  Correctness/P5/P6/P7 evidence passed.

Do not pursue another P8 optimization for this milestone: no Attention/MoE/HC profiling, no `mx.compile`, no custom Metal, and no tile-native default enablement.  Milestone 6 qualified the existing dense P0-P7 path through 200K.

## DwarfStar structural acceptance gate

This gate prevents “P0-P7 connected” from being satisfied by wrapping the existing oMLX whole-prefix layer loop in DwarfStar-shaped abstractions.  Before any performance evaluation, all of the following must be true on the production prefill path:

- The sweep planner actually owns execution order: command iteration, chunk offsets, layer phases, publication points, deferred-decoder transitions, and commit/rollback are driven from the DwarfStar sweep plan, not from an independent model loop.
- The current whole-prefix `for layer in layers` execution structure is absent from production prefill.  A diagnostic or compatibility path may retain it, but it must not be the implementation selected by production serving or performance qualification.
- Request carry (`h`, `pre`, suffix rows, compressor pending state, and publication state) follows the DwarfStar lifetime/alias model.  Carry may be viewed/sliced for chunk execution, but it must not be reconstructed from scratch or copied into validation-style per-layer records between chunks.
- Deferred-decoder mode actually avoids retaining unnecessary full-prefix decoder intermediates.  When the sweep selects suffix/deferred execution, only required suffix rows and persistent continuation/publication state may survive.
- Source/consumer publication frontiers drive execution and visibility.  They must determine when compressed KV, index K, candidates, top-k refreshes, and consumers are produced/visible; they are not merely telemetry metadata.
- Engram prefetch, expert/weight residency, and read-ahead scheduling hooks perform their intended lifetime/scheduling role.  A no-op hook does not count as architecture-complete.
- `mx.eval`, `mx.async_eval`, synchronization, and materialization occur only at explicitly defined DwarfStar command/materialization boundaries: setup, command batch, publication, deferred boundary, final handoff, or explicit diagnostics outside the hot path.
- No forbidden CPU activation/cache fallback exists on the production hot path. Forbidden: h/pre activation CPU fallback; KV/index/cache CPU fallback; model-weight diagnostic conversion; hot-path tensor digest/list conversion. Allowed and qualified only as an explicit storage boundary for SSD Engram: Engram hash/index host representation, SSD/mmap selected-row I/O, and selected Engram rows returning to MLX. This exception is not a general CPU fallback.
- `DeepseekV41Cache` remains the final decode-handoff authority, but it must not force the in-flight DwarfStar arena into full cache materialization or repacking at intermediate sweep/chunk boundaries.
- Final handoff converges once into a live request-local `DeepseekV41Cache` and enters `OMLXGenerationSession.from_prefilled_cache(...)` without prompt replay.

Passing correctness smoke without this structural gate is not performance-evaluation readiness.

### P0-P7 structural-gate audit closure (real evidence)

| gate item | implementation authority | real qualification evidence | status | remaining caveat |
|---|---|---|---|---|
| DwarfStar planner owns execution order | `P6AppendPlanner`, `SweepPlan`/`SweepCommand`, `DeferredPrefillAppend.execute_segment`, `OfficialFP8MLXBlockRunner.execute_batch` | P6/P7 long cases execute segment command streams with progress from command kinds, offsets, phases, deferred transitions, commit/rollback. | PASS | Production selector unchanged. |
| no production whole-prefix independent layer loop | `OfficialFP8MLXBlockRunner.execute_command` only invokes one `ENCODE_ROWS` layer/chunk per sweep command; no runner-level `for layer in layers` hot loop. | P7 2K/8K/16K/A records contain command-indexed layer work; reference/vertical-slice paths are guarded out by `assert_no_reference_hot_path`. | PASS | Diagnostic tools may still contain loops outside production path. |
| request carry follows bounded DwarfStar lifetime | `RequestArena`, `_write_rows`, `detach_final_decoder_cone`, P6 owner token and sealed/revoked cache capability. | P6 complete/pending/A/failure and P7 complete/A/failure runs validate frontiers, source coverage, failure rejection, rebuild/reuse. | PASS | P8 may optimize writes but must preserve ownership. |
| deferred decoder actually skips unnecessary work | P6 source-only segments plus final decoder suffix commands and private/public frontier split. | `pending-16384`/`A-24577`: source-only segment has `D=0` and public frontiers at 0; final segment seals suffix and P5 handoff. | PASS | Suffix math remains oMLX-backed scoped implementation. |
| publication frontiers drive visibility | `PublicationManager.shared_for_span`, `producer_shared_for_span`, `publish_frontier`, capture of source/index/candidate keys. | P6/P7 evidence validates source coverage, layer20 slot2/slot3 exact against controls, and pending raw-generation rejection before commit. | PASS | Python publication metadata is not optimized yet. |
| Engram/residency/read-ahead policy active | `SchedulingCoordinator`, `ResidencyPolicy`, `ReadAheadPolicy`, `EngramPrefetchController`, `P7_ENGRAM_TILE=2048`. | P7 enabled cases show donor submissions/consumptions, resident read-ahead, drain/reuse; reads: 2K=2, 8K=8, 16K=16 background and 0 foreground fallback. | PASS_WITH_SCOPED_BACKEND | Scope is only `FULL_RESIDENT_BACKBONE_SSD_ENGRAM`; `EXPERT_OFFLOAD` is not qualified. |
| materialization only at declared boundaries | `MlxEvaluationPolicy`, P6 source detach `mx.eval`, P7 Engram `mx.async_eval`, loader `mx.eval`, P5 handoff. | P7 qualification records async Engram boundaries; P6 detach materializes persistent source state; no general per-layer `mx.eval` added. | PASS_WITH_SCOPED_BACKEND | oMLX/MLX internals may synchronize internally; P8 audits/grouping only, no semantic weakening. |
| no forbidden CPU activation/cache fallback | Runner does not convert h/pre, KV/index/cache, or weights to NumPy/list/digest on hot path; Engram storage boundary explicitly uses host IDs/SSD rows. | P7 foreground fallback count is 0 for enabled paths; qualification instrumentation only patches storage reads and metadata. | PASS_WITH_SCOPED_BACKEND | Allowed CPU use is scoped to SSD Engram storage boundary; not expert offload/general fallback. |
| DeepseekV41Cache remains final authority | Live cache created by oMLX `make_cache`, mutated in runner, validated by `validate_committed_cache`; no `PrefillContinuationState` export. | P5/P6/P7 artifacts: frontiers equal prefix length, `full_cache_repack_count=0`, `exported=false`. | PASS | Cache internals remain owned by pinned oMLX. |
| P5 same-cache handoff with zero prompt replay | `LivePrefillResult.from_committed`, `handoff_to_generation`, `OMLXGenerationSession.from_prefilled_cache`. | P5 checks in P7 2K/8K/16K/A: `prompt_replay_count=0`, same frontiers advance from prefix+terminal through decode. | PASS | Production selector not promoted. |

Gate result: **P0-P7 structural gate = QUALIFIED** for the new path and scoped P7 backend. No gate item failed. P8 completed without selecting an optimization; Milestone 6 measured and qualified the existing dense P0-P7 path without changing production selection.

## Correctness and qualification gates

### Allowed early smoke checks

- 2-token and small prompt execution to verify no exceptions;
- cache offsets equal prefix length;
- `prompt_replay_count == 0` after decode admission;
- forbidden-reference modules absent;
- Engram history slot filled when Engram is enabled;
- deterministic structure under repeated run.

These checks must not classify performance success/failure.

### Promotion gates after package connection

- DwarfStar structural acceptance gate passed.
- Backend-local determinism for fixed model/build/input.
- Structural parity with current live-cache prefill: cache slot shapes, offsets, compressor pending slots, Engram history, publication metadata.
- No reference vertical-slice calls.
- No prompt replay in `OMLXGenerationSession`.
- Bounded behavioral smoke under M4 backend-local policy; no cross-backend hidden/logit bit-identity requirement.
- Then, and only then, long-context performance measurement against oMLX-class baseline.

## Explicit exclusions

- No DwarfStar GGUF weight format adoption.
- No final-prefix-logits computation on the serving path unless explicitly requested for qualification.
- No API/protocol changes as part of this prefill port.
- No DSpark/MTP enablement during base prefill architecture connection.
- No performance-based adoption/rejection of partial pieces before P0-P7 are connected.

## Implementation start criterion

Implementation may begin when this architecture is accepted.  The first coding step should be P0/P1 package scaffolding and planner/guard integration, not kernel optimization or benchmarking.
