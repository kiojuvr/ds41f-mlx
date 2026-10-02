# Runtime strategy

This document defines the durable runtime strategy for the scoped text release.

## Qualified release scope

```text
official DeepSeek-V4.1-Flash checkpoint
  -> DENSE_P0_P7 production prefill
  -> P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM
  -> P5 terminal holdout/handoff exactly once
  -> oMLX GenerationBatch decode
  -> MTP OFF / DSpark OFF / speculative decode OFF
  -> official deepseek-recipe local HTTP serving
```

Target hardware is the Mac Studio M3 Ultra 512 GB class system. Input scope is text. Serving is local single-flight.

M20 pins clean upstream oMLX `v0.7.0` at `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`; CED remains OFF/unused. The versioned release checkout is the qualification authority, dev2 is the unchanged rollback checkout, and `~/omlx` is reserved for a separately constructed identity-matching operational checkout. See [M20](m20-omlx-release-migration.md).

## Authority hierarchy

1. Official checkpoint/data.
2. Reviewed official DeepSeek semantics and `deepseek-recipe` protocol behavior.
3. ds41f backend-local fidelity contracts and validators.
4. Implementation-scoped regression evidence.
5. Optimized production implementation.

Backend-local determinism is required for fixed checkpoint/runtime/backend/build/config/input/session state. Cross-backend hidden-state, logit, or greedy-token identity is not a release requirement under official-compatible floating-point semantics.

## Production prefill/decode selection

The runtime-facing prefill facade selects the dense P0-P7 FP8/MLX path with P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM` and `P7_ENGRAM_TILE=2048`. P8 tile-native carry is retained as experimental/default OFF because it did not produce an end-to-end gain. The old one-chunk oMLX layer-loop substrate is diagnostic/legacy and is not selected by serving or release performance qualification.

P5 is the release handoff seam: serving holds out the terminal prompt token, prefill commits the prefix to one live `DeepseekV41Cache[40]`, and oMLX `GenerationBatch` receives the terminal token exactly once. After bootstrap, GenerationBatch-owned cache is the single executable authority.

## Session and persistence strategy

Stateful Chat Completions sessions preserve exact-prefix continuation, zero prompt replay, zero full-cache repack/reconstruction, and coherent all-layer cache frontiers at idle boundaries. Persistence is for idle same-backend artifacts only. Restore validates provenance, schema, shape, dtype, checkpoint identity, and frontier before re-entry, and fails closed on mismatch/corruption.

## API strategy

HTTP protocol compatibility is delegated to official `deepseek-recipe`; ds41f implements only the runtime backend. Stateless text Chat Completions, Responses, and Messages are in scope. Stateful sessions are Chat Completions only, including client-side function-tool/result loops. The server does not execute tools.

Arbitrary stateful request stop strings are rejected before mutation. The qualified stateful stop mechanism is token-level DeepSeek EOS plus length/cancel handling.

## Optional / experimental

- Diagnostic endpoints and telemetry.
- Native C++ model core as reference/qualification and reusable components.
- P8 tile-native carry and other optimization probes, default OFF.

## Unqualified / non-goals

Vision, batching, MTP, DSpark, speculative decode, distributed serving, authentication, server-side tool execution, MCP/plugins, web search, shell tools, sessionized Responses/Messages, cross-runtime KV portability, new kernel optimization, and arbitrary stateful stop rollback are outside the release strategy.
