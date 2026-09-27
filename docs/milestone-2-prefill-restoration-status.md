# Milestone 2 prefill restoration status

Status: **INCOMPLETE — ENGRAM@1, ENGRAM@14, CANDIDATE SOURCE@20, AND FIRST INDEX-REFRESH@24 IMPLEMENTED**.

This document records the current Milestone 2 state. It does not change the runtime architecture decision. Milestone 2 remains open until full-model official-checkpoint prefill executes through the DwarfStar-derived topology.

## Governing constraint

Milestone 2 requires a real full-model prefill path for the official DeepSeek-V4.1-Flash checkpoint that executes with DwarfStar-derived `ds41_graph_prefill_sweep` execution/lifetime topology. A wrapper around the current native correctness/reference runtime is not sufficient if it preserves the native reference execution topology.

Correctness authority remains official checkpoint/data, reviewed official DeepSeek semantics, official-source-derived `ds41f` validators/contracts, and implementation-scoped regression evidence. DwarfStar and oMLX remain architecture/implementation sources, not semantic authorities.

## Implemented vertical slice

The repository now contains a vertically integrated DwarfStar-derived production-prefill executor slice through Engram@1, Engram@14, the first four full source/reuse dependency groups, and the first candidate-backed index-only refresh:

```text
ds41f_mlx/dwarfstar_prefill_slice.py
ds41f_mlx/official_model_math.py
tools/run_m2_dwarfstar_prefill_vertical_slice.py
artifacts/m2/dwarfstar-prefill/vertical-slice-layer0-27-candidate20-indexrefresh24.json
```

The slice demonstrates:

- native DwarfStar-derived sweep plan ownership;
- executor-owned typed token, embedding, HC carry, pre-mix, and shared publication storage;
- official checkpoint embedding gather through the native prefill library;
- complete real model execution for layers 0..27 using narrow production-facing official model-math and Engram seams, without calling `TextBackboneReference`, `TextEncoderReference`, or `TextDecoderReference`;
- real Engram@1 execution using regenerated token/ngram hashes, sparse SSD-backed Engram row reads, FP8 WKV projection, source-defined gate arithmetic, and residual update;
- layer-2 compressed/index/top-k source publication owned by the executor;
- layer-3..7 reuse/consumer attention reading the executor-owned layer-2 publication rather than recomputing producer state;
- layer-8 compressed/index/top-k source publication as a distinct generation;
- layer-9..13 reuse/consumer attention reading the executor-owned layer-8 publication and not the stale layer-2 generation;
- real Engram@14 execution using regenerated layer-14 hashes, sparse SSD-backed Engram row reads, FP8 WKV projection, source-defined gate arithmetic, residual update, and preserved pre-mix/shared publication state;
- layer-14 compressed/index/top-k source publication as a distinct generation after Engram@14 and Block14;
- layer-15..19 reuse/consumer attention reading the executor-owned layer-14 publication and not stale layer-2 or layer-8 generations;
- layer-20 full source/candidate-source publication producing compressed KV, index K, top-k indices, and the first real candidate state on the connected trajectory;
- layer-21..23 consumers reading generation @20 without recomputing source state or synthesizing candidates;
- layer-24 represented as an index-only/top-k refresh generation that consumes layer-20 compressed/index/candidate state, preserves those field owners, and refreshes only top-k ownership;
- layer-25..27 consumers reading mixed ownership: compressed KV/index/candidates from source @20 and top-k from index-refresh @24;
- candidate lifecycle explicitly scoped as not-applicable before candidate source layer 20;
- invalid-during-sweep and commit-after-publication transaction events;
- exact Engram-aware comparison against official-source-derived connected validation evidence for layers 0..27, including the newly added candidate@20→index-refresh@24→consumers25..27 reference fixture.

This is an implementation-scoped Engram-aware two-generation source/reuse slice, not full-model prefill.

## Remaining implementation gap

The repository does not yet contain a full integrated executor that composes all layers and final output behind the DwarfStar-derived sweep ABI without reverting execution control to the current native reference topology.

Two broader available paths remain classified as follows:

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

## Why full Milestone 2 remains incomplete

The remaining work is not a small selector change, documentation update, or local optimization. It is the rest of the central Milestone 2 implementation: widening the vertically integrated slice into complete official model computation behind the DwarfStar-derived sweep executor.

The available shortcuts remain invalid:

- calling current native `TextBackboneReference` prefill would execute the reference topology, not the DwarfStar-derived production topology;
- calling the oMLX layer-major adapter would reproduce diagnostic/prototype behavior, not the native DwarfStar sweep executor;
- adding more isolated primitives would not satisfy full-model prefill or state handoff;
- changing selectors would not create the missing execution/lifetime implementation.

## Required next implementation slice

Milestone 2 should remain open and proceed by widening the implemented vertical slice, not by starting another audit. The next slice should:

1. move more typed row/carry ownership into the native C sweep executor where needed;
2. replace Python-orchestrated official-source-derived helper calls with production-facing native seams stage by stage;
3. widen from layers 0..2 to the first source/reuse group, then Engram layer 1/14 handling, then decoder/source groups;
4. keep publication/commit state owned by the DwarfStar sweep executor rather than `TextBackboneReference` control flow;
5. gate each widening step against official-source-derived fixtures and connected validation evidence;
6. only then execute all 40 layers plus final collapse/norm/head/logits.

This is still Milestone 2 work. It should not be recast as Milestone 3 decode selection or current native optimization.

## Milestone 2 completion determination

Milestone 2 is **not complete**.

Current completion criteria:

- Engram-aware DwarfStar-topology source→publication→reuse executor slice: **implemented and validated for layers 0..27, including Engram@1, Engram@14, full source generations at 2, 8, 14, and 20, candidate publication at 20, and index-only top-k refresh at 24**;
- real full-model DwarfStar-topology prefill: **not implemented**;
- official checkpoint through the DwarfStar-derived full prefill path: **not implemented**;
- production state handoff from full prefill: **not implemented**;
- implementation-scoped qualification of full path: **not available**;
- bounded performance observation of the full production path: **not available**.

No runtime code was changed for this status update, because any small change that merely routes through the current native reference runtime would create the architecture drift the strategy was written to prevent.
