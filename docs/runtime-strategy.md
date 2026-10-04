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

[M44](milestone-44-target-generation-ownership.md) transfers the generation lifecycle. [M45](milestone-45-target-forward-ownership.md) adds owned single-token target-forward orchestration and the all-40-layer mutation/commit transaction. No external LanguageModel forward or row extraction/merge executes on this production decode path. [M46](milestone-46-state-production-ownership.md) transfers the standard-off block/attention/compressor/index producer boundary into `DecodeStateProducer`: no external state-producing module call receives the live cache. ds41f owns slot writes, compressor tails, compressed/index/candidate reuse publications, admission-metadata advancement and structural completion. Attributed loading, projections/norms/MoE/HC/head/quantization math, packed storage, kernels and SSD Engram read mechanisms remain subordinate dependencies. [M47](milestone-47-execution-resource-admission.md) owns admission of that numerical/native implementation and the full checkpoint/tokenizer/SSD backing set. OFF startup verifies content and loaded native identities, then binds model/configuration and concrete SSD descriptors; OFF execution requires this live capability. Cold payload verification is explicit (~242 s for ~510 GB); no resource-file hashing occurs in the token loop. The general batched/prefill/MTP LanguageModel path is not replaced, and bounded MTP admission remains separate. The old `omlx_generation.py` engine remains only for R1 legacy fixtures/comparisons; bounded MTP is unchanged on its separate scheduler. Failed transactions invalidate every cache alias and cannot publish continuation; cancellation occurs between completed transactions.

## Session and persistence strategy

Stateful Chat Completions sessions preserve exact-prefix continuation, zero prompt replay, zero full-cache repack/reconstruction, and coherent all-layer cache frontiers at idle boundaries. Persistence is for idle same-backend artifacts only. Restore validates provenance, schema, shape, dtype, checkpoint identity, and frontier before re-entry, and fails closed on mismatch/corruption.

## API strategy

HTTP protocol compatibility is delegated to official `deepseek-recipe`; ds41f implements only the runtime backend. Stateless text Chat Completions, Responses, and Messages are in scope. Stateful sessions are Chat Completions only, including client-side function-tool/result loops. The server does not execute tools.

M21's production Rust boundary is a Rust crate over this same loopback HTTP seam plus optional child-process startup/readiness/shutdown control. Rust clients do not receive executable cache handles, do not perform prompt replay, and do not duplicate DeepSeek protocol semantics; they submit JSON/SSE requests to the qualified server authority.

Arbitrary stateful request stop strings are rejected before mutation. The qualified stateful stop mechanism is token-level DeepSeek EOS plus length/cancel handling.

## Optional / experimental

- Diagnostic endpoints and telemetry.
- Native C++ model core as reference/qualification and reusable components.
- P8 tile-native carry and other optimization probes, default OFF.

## Unqualified / non-goals

Vision, batching, MTP, DSpark, speculative decode, distributed serving, authentication, server-side tool execution, MCP/plugins, web search, shell tools, sessionized Responses/Messages, cross-runtime KV portability, new kernel optimization, and arbitrary stateful stop rollback are outside the release strategy.
