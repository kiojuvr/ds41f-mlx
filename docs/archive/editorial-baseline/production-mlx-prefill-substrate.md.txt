# Production MLX prefill substrate

This substrate is intentionally separate from the bounded reference vertical slice and the historical layer-major diagnostic prototype.

## Production API

`ds41f_mlx.runtime.dwarfstar_prefill.DwarfStarMLXPrefillSession`

```python
result = DwarfStarMLXPrefillSession(loaded_model).prefill(prefix_token_ids)
# result.live_cache is the request-local DeepseekV41Cache authority
```

The implementation uses the already-loaded oMLX/MLX `LanguageModel`, creates a fresh request-local `DeepseekV41Cache` via `language_model.make_cache()`, executes embedding/Engram/attention/HC/MoE layer operations, and returns the live cache without exporting through `PrefillContinuationState`.

## Handoff

`OMLXGenerationSession.from_prefilled_cache(...)` admits the live cache directly into `BatchGenerator` / `GenerationBatch`:

```text
live DeepseekV41Cache -> GenerationBatch
```

This is the same-backend production hot handoff. It avoids:

```text
DeepseekV41Cache -> NumPy PrefillContinuationState -> OMLXDecodeStateAdapter -> second DeepseekV41Cache
```

`PrefillContinuationState` remains the portable/qualification state contract for explicit export flows; production serving does not pay that cost.

## Guard policy

The serving guard against the reference vertical slice remains active. Production serving must not call:

- `DwarfStarPrefillVerticalSliceExecutor`
- `OfficialModelMath`
- `tools.run_m4_omlx_base_decode_qualification.build_prefill_state`
- official-source-derived validation helper `block()`

If that path is reached, classify as `PRODUCTION_PREFILL_REGRESSED_TO_REFERENCE_VERTICAL_SLICE`.

## Scope and provenance

Current substrate: **oMLX-operation based real MLX**, not native DwarfStar Metal. It establishes practical live-cache prefill and no-replay decode handoff. It does **not** yet restore DwarfStar performance mechanisms such as static Metal carry arena, command graph ownership, decoder suffix/deferred decoder, long-context chunk geometry, or expert/cache scheduling.
