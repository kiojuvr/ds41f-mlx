# Milestone 2 prefill restoration status

Status: **INCOMPLETE — BLOCKED ON INTEGRATED PRODUCTION PREFILL EXECUTOR**.

This document records the result of attempting to proceed from Milestone 1 into Milestone 2. It is not a development diary and it does not change the runtime architecture decision. Milestone 2 remains the next implementation milestone.

## Governing constraint

Milestone 2 requires a real full-model prefill path for the official DeepSeek-V4.1-Flash checkpoint that executes with DwarfStar-derived `ds41_graph_prefill_sweep` execution/lifetime topology. A wrapper around the current native correctness/reference runtime is not sufficient if it preserves the native reference execution topology.

Correctness authority remains official checkpoint/data, reviewed official DeepSeek semantics, official-source-derived `ds41f` validators/contracts, and implementation-scoped regression evidence. DwarfStar and oMLX remain architecture/implementation sources, not semantic authorities.

## Implementation attempt result

The repository does not yet contain an integrated executor that can compose the already validated native primitives and reusable native/reference components behind the DwarfStar-derived sweep ABI without reverting execution control to the current native reference topology.

Two available paths were rechecked:

1. **DwarfStar-derived native prefill line**
   - Has the right planner/lifetime/submission frontier.
   - Has bounded official checkpoint primitives.
   - Does not yet execute full HC, attention, Engram, MoE, output head, publication, and transaction semantics as one official full-model prefill.

2. **Current `native/*` reference runtime**
   - Executes the official checkpoint and has correctness/reference value.
   - Contains sweep/deferred-decoder inspired functions and reusable model math.
   - Still owns execution through the native reference `TextBackboneReference` / `TextEncoderReference` / `TextDecoderReference` control flow, MLX graph lifetime, and 128-token internal chunking, not through the DwarfStar-derived native sweep allocation/command/submission seam.

Promoting path 2 as Milestone 2 would violate `docs/runtime-strategy.md` by reclassifying the correctness/reference runtime as production prefill architecture.

## Current DwarfStar-derived frontier still valid

Milestone 2 must continue from:

```text
ds41f_mlx/dwarfstar_v41_sweep.py
ds41f_mlx/native_prefill.py
ds41f_mlx/native/ds41f_prefill_native.{h,c}
artifacts/m2/dwarfstar-prefill/native-sweep-reconciliation.json
artifacts/m2/dwarfstar-prefill/native-metal-submission*.json
artifacts/m2/dwarfstar-prefill/native-official-embedding.json
artifacts/m2/dwarfstar-prefill/native-official-projection.json
```

Additional repository evidence now available from existing artifacts includes bounded official-reference-derived native validations for FP8 linear, RMSNorm, rotary, attention q-prelude, sparse attention, window KV prelude, compressed KV, candidate/indexer pieces, MoE pieces, HC pieces, and parallel head. These are important implementation ingredients, but they are still isolated or bounded validations. They do not by themselves constitute full-model DwarfStar-topology prefill.

## Exact missing production executor pieces

The blocker is the absence of an integrated production executor that, under the DwarfStar-derived sweep command/lifetime model, performs all of the following against official checkpoint data:

1. initialize full prompt embedding/carry rows in the native sweep arena;
2. execute HC pre/post for all layers in sweep order;
3. execute attention q/kv projections, RoPE, local window KV, compressed KV, index K, candidate/index selection, sparse attention, and output projection while updating/publishing persistent state;
4. execute Engram lookups/projection at the required layer/order with SSD-backed semantics and overlap opportunities preserved;
5. execute MoE gate/shared/routed expert math and selected expert scheduling without adopting incompatible DwarfStar checkpoint or quantization formats;
6. publish source-layer and reuse-layer state transitions at DwarfStar-style layer/frontier boundaries;
7. execute final collapse/norm/output head/read-logits in the DwarfStar final publication/commit phase;
8. preserve invalid partial sweep, encoder-only/deferred decoder, commit, reset, fork, and continuation semantics;
9. hand committed prefill state to a neutral production session/decode boundary without selecting the final Milestone 3 decode architecture;
10. pass implementation-scoped correctness gates for this new path.

## Why this is a genuine blocker

The missing work is not a small selector change, documentation update, or local optimization. It is the central Milestone 2 implementation: integrating complete official model computation behind the DwarfStar-derived sweep executor.

The repository has enough evidence to define the work precisely, but not enough implemented code to claim completion responsibly. The available alternatives are invalid shortcuts:

- calling current native `TextBackboneReference` prefill would execute the reference topology, not the DwarfStar-derived production topology;
- calling the oMLX layer-major adapter would reproduce diagnostic/prototype behavior, not the native DwarfStar sweep executor;
- adding more isolated primitives would not satisfy full-model prefill or state handoff;
- changing selectors would not create the missing execution/lifetime implementation.

## Required next implementation slice

Milestone 2 should remain open and proceed with a first integrated production-executor slice, not another audit. The next slice should connect one real layer span through the existing native sweep ABI while preserving DwarfStar lifetime and official semantics:

1. extend `ds41f_prefill_native` from command/buffer submission into typed sweep row/carry buffers for official model data;
2. bind official embedding/carry initialization to the sweep arena;
3. integrate the minimum complete layer-0 path behind `DS41F_SWEEP_ENCODE_ROWS` using existing validated primitive code where possible;
4. publish/commit state through the sweep plan rather than `TextBackboneReference` control flow;
5. gate the slice against official-source-derived fixtures and native/reference regression evidence;
6. only then widen to Engram/source/reuse groups and finally all 40 layers.

This is still Milestone 2 work. It should not be recast as Milestone 3 decode selection or current native optimization.

## Milestone 2 completion determination

Milestone 2 is **not complete**.

The milestone completion criteria are unmet:

- real full-model DwarfStar-topology prefill: **not implemented**;
- official checkpoint through the DwarfStar-derived full prefill path: **not implemented**;
- production state handoff from that path: **not implemented**;
- implementation-scoped qualification of that path: **not available**;
- bounded performance observation of the real production path: **not available**.

No runtime code was changed for this status update, because any small change that merely routes through the current native reference runtime would create the architecture drift the strategy was written to prevent.
