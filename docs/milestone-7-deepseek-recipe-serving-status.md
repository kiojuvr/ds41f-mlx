# Milestone 7 DeepSeek recipe serving status

Status: **active qualification after selector promotion**.

Architecture decision:

```text
PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7
```

Scope:

- DwarfStar-derived command topology;
- P6 `DeferredPrefillAppend` for fresh prefixes;
- P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, `P7_ENGRAM_TILE=2048`;
- P8 tile-native carry OFF (`TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`);
- live `DeepseekV41Cache` authority;
- P5 zero-replay handoff by `handoff_to_generation()`;
- oMLX `OMLXGenerationSession` / `GenerationBatch` MTP-OFF.

Justification: P0-P7 structural gate qualified, P8 optimization search closed with no E2E gain, and M6 performance qualified through 200K (`M6_PERFORMANCE_QUALIFIED_200K`).

The runtime-facing `DwarfStarMLXPrefillSession` is now a compatibility facade over the production dense P0-P7 path. The former one-chunk oMLX layer-loop substrate is retained only as `LegacyOneChunkMLXPrefillSession` diagnostic evidence:

```text
ONE_CHUNK_SUBSTRATE = DIAGNOSTIC / LEGACY
```

The active serving path must never use the reference vertical slice or allow `DS41F_ALLOW_REFERENCE_VERTICAL_SLICE_SERVING=1` to change production selection.

Implemented serving flow:

```text
HTTP FastAPI endpoints
  -> official deepseek-recipe request conversion
  -> DeepseekV41Encoding.with_tokenizer(...).encode(conversation)
  -> prefix = tokens[:-1], terminal = tokens[-1]
  -> DwarfStarMLXPrefillSession facade
  -> DenseP0P7PrefillSession
  -> DeferredPrefillAppend.execute_all()
  -> LivePrefillResult.from_committed(...)
  -> handoff_to_generation(... terminal_prompt_token=terminal ...)
  -> OMLXGenerationSession / GenerationBatch
  -> InferenceChunk.token(token_id)
  -> deepseek-recipe StreamProcessor
```

Qualification gates now required before M7 can be marked qualified:

1. raw-token arbitrary-length dense P0-P7 matrix, including 29/30/31 ratio-2 regression closure;
2. representative P5 bootstrap checks with zero prompt replay/repack/export;
3. real loopback HTTP text serving for Chat Completions and Responses, non-stream and streaming;
4. cancellation/disconnect, invalid-request atomicity, single-flight behavior, model lifecycle, and 10 sequential requests;
5. `/health` and `/v1/models` behavior;
6. provenance and bounded trace retention recorded in `artifacts/m7/deepseek-recipe-serving/result.json`.

Current endpoint scope remains text-only, single-flight, MTP/DSpark OFF. `/v1/messages` qualification depends on the installed pinned `deepseek-recipe` feature surface.
