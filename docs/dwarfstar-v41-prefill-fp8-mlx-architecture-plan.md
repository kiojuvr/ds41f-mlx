# DwarfStar V4.1 prefill on official FP8/MLX — architecture plan

Status: **planned / architecture frozen for implementation start**.  This document is a design artifact only; it does not implement the runtime.

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

Implementation/real P5 qualification: [live-cache handoff closeout](p5-live-cache-handoff.md). Production selection, P6 and P7 remain separate tasks.

### P6 — deferred decoder / long-context sweep

- Enable encoder-only and resume/deferred-decoder phases from the native sweep plan.
- Retain only DwarfStar-required suffix rows for decoder layers rather than full prompt rows when the plan selects suffix mode.
- Preserve compressor pending state and publication frontiers across encoder/resume boundary.

### P7 — overlap and scheduling

- Add Engram prefetch/read-ahead command handling using oMLX Engram prefetch hooks.
- Add expert/weight residency scheduling around layer commands.
- A placeholder/no-op scheduling hook may exist during scaffolding, but it does not count toward architecture completion or performance-evaluation readiness.
- Group `mx.eval`/`mx.async_eval` according to command batches and explicit materialization boundaries rather than per validation layer.

### P8 — graph/reuse optimization pass

Only after P0-P7 are connected and the DwarfStar structural acceptance gate passes, evaluate whether to add MLX compile/custom-kernel graph reuse, command-buffer grouping, or native Metal kernels.  This phase may use performance to guide choices because the architecture package is then structurally connected.

## DwarfStar structural acceptance gate

This gate prevents “P0-P7 connected” from being satisfied by wrapping the existing oMLX whole-prefix layer loop in DwarfStar-shaped abstractions.  Before any performance evaluation, all of the following must be true on the production prefill path:

- The sweep planner actually owns execution order: command iteration, chunk offsets, layer phases, publication points, deferred-decoder transitions, and commit/rollback are driven from the DwarfStar sweep plan, not from an independent model loop.
- The current whole-prefix `for layer in layers` execution structure is absent from production prefill.  A diagnostic or compatibility path may retain it, but it must not be the implementation selected by production serving or performance qualification.
- Request carry (`h`, `pre`, suffix rows, compressor pending state, and publication state) follows the DwarfStar lifetime/alias model.  Carry may be viewed/sliced for chunk execution, but it must not be reconstructed from scratch or copied into validation-style per-layer records between chunks.
- Deferred-decoder mode actually avoids retaining unnecessary full-prefix decoder intermediates.  When the sweep selects suffix/deferred execution, only required suffix rows and persistent continuation/publication state may survive.
- Source/consumer publication frontiers drive execution and visibility.  They must determine when compressed KV, index K, candidates, top-k refreshes, and consumers are produced/visible; they are not merely telemetry metadata.
- Engram prefetch, expert/weight residency, and read-ahead scheduling hooks perform their intended lifetime/scheduling role.  A no-op hook does not count as architecture-complete.
- `mx.eval`, `mx.async_eval`, synchronization, and materialization occur only at explicitly defined DwarfStar command/materialization boundaries: setup, command batch, publication, deferred boundary, final handoff, or explicit diagnostics outside the hot path.
- No CPU round-trip exists on the production hot path.  Tensor-to-NumPy/list/digest conversion is allowed only for explicitly gated diagnostics, not for production execution or qualification timing.
- `DeepseekV41Cache` remains the final decode-handoff authority, but it must not force the in-flight DwarfStar arena into full cache materialization or repacking at intermediate sweep/chunk boundaries.
- Final handoff converges once into a live request-local `DeepseekV41Cache` and enters `OMLXGenerationSession.from_prefilled_cache(...)` without prompt replay.

Passing correctness smoke without this structural gate is not performance-evaluation readiness.

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
