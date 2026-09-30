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
6. no reference vertical-slice fallback.

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
CachePublisher: fill request-local DeepseekV41Cache slots
  ↓
PrefillResult(live_cache, token_ids, structural telemetry)
  ↓
OMLXGenerationSession.from_prefilled_cache(...)
```

### State authority

`DeepseekV41Cache` is the only production decode handoff state.  The prefill package may maintain transient arena aliases, but by commit time every persistent item required for continuation must be in the live cache:

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

- Lift the current loop from `DwarfStarMLXPrefillSession` into `OfficialFP8MLXBlockRunner`.
- Replace sequential “for every whole layer over whole prefix” control with sweep-command-driven calls.
- Preserve oMLX absolute-position semantics by passing correct `start` and cache slices for chunks.
- Keep final logits optional/off by default for serving prefill.

### P4 — publication manager

- Implement source/consumer ownership metadata matching the model topology:
  - source compressed KV generations at configured source layers;
  - index K generations and refreshes;
  - candidate source and consumer blocks;
  - top-k/index ownership advancement.
- Publication is transaction-local until commit; failed sweeps must not expose partially committed state.

### P5 — live-cache handoff

- Publish arena/cache rows into a fresh request-local `DeepseekV41Cache` list.
- Validate only structural invariants on hot path: equal offsets, expected slot geometry, no prompt replay, Engram history present when required.
- Hand to `OMLXGenerationSession.from_prefilled_cache` without `PrefillContinuationState` export/repack.

### P6 — deferred decoder / long-context sweep

- Enable encoder-only and resume/deferred-decoder phases from the native sweep plan.
- Retain only DwarfStar-required suffix rows for decoder layers rather than full prompt rows when the plan selects suffix mode.
- Preserve compressor pending state and publication frontiers across encoder/resume boundary.

### P7 — overlap and scheduling

- Add Engram prefetch/read-ahead command handling using oMLX Engram prefetch hooks.
- Add expert/weight residency scheduling hooks around layer commands.  These hooks may initially be no-ops but must be architecturally present before performance decisions.
- Group `mx.eval`/`mx.async_eval` according to command batches rather than per validation layer.

### P8 — graph/reuse optimization pass

Only after P0-P7 are connected, evaluate whether to add MLX compile/custom-kernel graph reuse, command-buffer grouping, or native Metal kernels.  This phase may use performance to guide choices because the architecture package is then connected.

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
