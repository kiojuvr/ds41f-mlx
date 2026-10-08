# Runtime strategy

This document defines the durable **standard-off** production strategy for the scoped text release.
The separately selected [bounded local MTP candidate](mtp-local-release-candidate.md)
is governed by [M41 evidence](milestone-41-local-mtp-release-candidate.md), not OFF
feature parity. Omitted profile/default production MTP remains OFF.

## Qualified release scope

```text
official DeepSeek-V4.1-Flash checkpoint
  -> DENSE_P0_P7 production prefill
  -> P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM
  -> P5 terminal holdout/handoff exactly once
  -> ds41f TargetGenerationSession single-stream decode
  -> MTP OFF / DSpark OFF / speculative decode OFF
  -> official deepseek-recipe local HTTP serving
```

Target hardware is the Mac Studio M3 Ultra 512 GB class system. Input scope is text. Serving is local single-flight.

M20 pins clean upstream oMLX `v0.7.0` at `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`; CED remains OFF/unused. M22 records this and the active package/dependency expectations in `release/ds41f-release.json`. The versioned release checkout is the qualification authority, dev2 is the unchanged rollback checkout, and `~/omlx` is reserved for a separately constructed identity-matching operational checkout. See [M20](m20-omlx-release-migration.md) and [M22](m22-release-packaging.md).

## Authority hierarchy

1. Official checkpoint/data.
2. Reviewed official DeepSeek semantics and `deepseek-recipe` protocol behavior.
3. ds41f backend-local fidelity contracts and validators.
4. Implementation-scoped regression evidence.
5. Optimized production implementation.

Backend-local determinism is required for fixed checkpoint/runtime/backend/build/config/input/session state. Cross-backend hidden-state, logit, or greedy-token identity is not a release requirement under official-compatible floating-point semantics.

## Production prefill/decode selection

The runtime-facing prefill facade selects the dense P0-P7 FP8/MLX path with P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM` and `P7_ENGRAM_TILE=2048`. P8 tile-native carry is retained as experimental/default OFF because it did not produce an end-to-end gain. The old one-chunk oMLX layer-loop substrate is diagnostic/legacy and is not selected by serving or release performance qualification.

P5 is the release handoff seam: serving holds out the terminal prompt token, prefill commits the prefix to one live `DeepseekV41Cache[40]`, and ds41f `TargetGenerationSession` consumes the terminal token exactly once. The engine owns target-call scheduling, normalized-logprob sampling, consumed-token history, EOS/length/cancel, failure invalidation, and return of the exact live list to idle continuation. Sampled lookahead is not committed until consumed. No scheduler row extraction, cache repack, prompt replay or hidden reconstruction occurs.

[M44](milestone-44-target-generation-ownership.md) transfers the generation lifecycle. [M45](milestone-45-target-forward-ownership.md) adds owned single-token target-forward orchestration and the all-40-layer mutation/commit transaction. No external LanguageModel forward or row extraction/merge executes on this production decode path. [M46](milestone-46-state-production-ownership.md) transfers the standard-off block/attention/compressor/index producer boundary into `DecodeStateProducer`: no external state-producing module call receives the live cache. ds41f owns slot writes, compressor tails, compressed/index/candidate reuse publications, admission-metadata advancement and structural completion. M48 subsequently owns checkpoint loading, projections/norms/MoE/HC/head/quantization math, packed storage and SSD Engram read mechanisms for standard-OFF; generic MLX/native primitives remain subordinate dependencies. [M47](milestone-47-execution-resource-admission.md) owns admission of that numerical/native implementation and the full checkpoint/tokenizer/SSD backing set. OFF startup verifies content and loaded native identities, then binds model/configuration and concrete SSD descriptors; OFF execution requires this live capability. Cold payload verification is explicit (~242 s for ~510 GB); no resource-file hashing occurs in the token loop. The OFF model/prefill math is first-party; bounded MTP and diagnostic donor execution/admission remain separate. The old `omlx_generation.py` engine remains only for R1 legacy fixtures/comparisons; bounded MTP is unchanged on its separate scheduler. Failed transactions invalidate every cache alias and cannot publish continuation; cancellation occurs between completed transactions.

## Canonical speculative block producer (bounded development qualification)

The development accepted-prefix producer may construct multiple causal inputs in
one owned 40-layer numerical region on the sole pending packed list. Window undo
records the chronological row lost at **each** input (including rows introduced
inside a block), compressor receipts retain each pre-pooling projection, and
history/taps/logits are indexed by consumed-input prefix. Receipts are bounded,
physically detached at block completion; all-layer settlement still prepares and
evaluates every slot before any alias becomes executable. Candidate scheduling,
verification-state objects and cache extraction/merge are not used.

[M56](milestone-56-canonical-accepted-prefix-block.md) qualifies independent
row-oracle state/tap/logit parity and selects this entry in M52. Short blocks use
existing batched GEMV primitives to preserve OFF reduction geometry. Attention
alone groups rows by canonical sparse-list width: block-end padding would move
valid keys across BF16 softmax tiles and is NOT equivalent to OFF. Embedding,
projections, HC, MoE and head remain layer-major block execution. Blocks are at
most eight inputs, journals at most 32; cancellation drains protected blocks
before prefix-zero settlement, and faults burn all aliases.

OFF retains its one-input contract, and the M54 semantic horizon remains the
admission authority. Matched performance is now faster than OFF in the bounded
M54 envelope. This permits a separate normal-local-profile approval evaluation,
not automatic profile/default/release promotion or broader capability admission.

## Model-lifetime allocator resource policy

The admitted standard-OFF lifetime bounds MLX's **free allocation cache** to at
most 32 GiB (and never increases a caller's smaller limit). This policy is acquired
before model allocation, covers dense prefill, decode and idle P6 continuation,
and restores the previous process setting on retirement/load failure. Overlapping
OFF allocator-policy owners reject rather than restore each other's global state.
This addresses measured free-buffer accumulation at 200K; it does not evict live
KV/weights, clear caches per token/turn, change precision, or create a second state
owner. Live working-set feasibility remains an evidence/admission concern, not a
promise of arbitrary context operation.

Very-long qualification also requires GPU residency across the **whole admitted
OFF lifetime**, not only active target decode. The default MLX wired limit is
zero; an idle/P6 prefill must not depend on a previously closed generation lease
to keep resident weights available. Admission owns the device-recommended wired
budget alongside the existing allocator policy, restores both settings at
retirement/load failure, and synchronizes before changing residency. This changes
resource lifetime only; no live state is evicted, replayed or repacked. Supported
frontiers still require actual lifecycle qualification.

Very-long turns also require deterministic retirement of the **passive** P6
certificate/setup reference cycle after P5 transfer (including failed bootstrap).
The setup-to-certificate backlink is needed for admission while ready, not after
the runner is revoked. Retiring that backlink does not alter the live packed list,
publication semantics, single-transfer/burn guards or explicit diagnostic aliases;
it prevents old source buffers from depending on Python cyclic GC for release.

## Session and persistence strategy

Stateful Chat Completions sessions preserve exact-prefix continuation, zero prompt replay, zero full-cache repack/reconstruction, and coherent all-layer cache frontiers at idle boundaries. Persistence is for idle same-backend artifacts only. Restore validates provenance, schema, shape, dtype, checkpoint identity, and frontier before re-entry, and fails closed on mismatch/corruption.

## Bounded multimodal strategy

[Multimodal production qualification](multimodal-production-qualification.md)
closes four inline images / 8,192 consumed positions on the same first-party
standard-OFF core; 1M text/resource semantics remain unchanged. CPU-only bounded
preprocessing produces expanded spans and byte identities. The owner worker
encodes only new images, injects sparse absolute embeddings and slices the VL /
Engram-exclusion mask per command. P5/P6 and first-party target generation remain
the only cache/execution authority; no image-specific generation stack exists.

Idle artifacts preserve image identities and all packed slots, never pixels or
encoder activations. Restore materializes exact loaded arrays on its worker
before publication. Cancellation drains protected work before releasing fences,
burns discarded ready leases, and publishes completed stateful protocol records
inside the worker. These are lifecycle repairs, not replay/repack or a new format.
Official preprocessing/FP32 fidelity and BF16-local drift limits are explicit.
This is bounded development/core support, not R1/release/runtime promotion.

## API strategy

HTTP protocol compatibility is delegated to official `deepseek-recipe`; ds41f implements only the runtime backend. Stateless text Chat Completions, Responses, and Messages are in scope. Stateful sessions are Chat Completions only, including client-side function-tool/result loops. The server does not execute tools.

M21's production Rust boundary is a Rust crate over this same loopback HTTP seam plus optional child-process startup/readiness/shutdown control. Rust clients do not receive executable cache handles, do not perform prompt replay, and do not duplicate DeepSeek protocol semantics; they submit JSON/SSE requests to the qualified server authority.

Arbitrary stateful request stop strings are rejected before mutation. The qualified stateful stop mechanism is token-level DeepSeek EOS plus length/cancel handling.

## Optional / experimental

- Diagnostic endpoints and telemetry.
- Native C++ model core as reference/qualification and reusable components.
- P8 tile-native carry and other optimization probes, default OFF.

## Unqualified / non-goals

Vision beyond the separate bounded core envelope, batching, MTP, DSpark, speculative decode, distributed serving, authentication, server-side tool execution, MCP/plugins, web search, shell tools, sessionized Responses/Messages, cross-runtime KV portability, new kernel optimization, and arbitrary stateful stop rollback are outside the release strategy.
