# Milestone 7 DeepSeek recipe serving status

Status: **not qualified**.

Implemented integration seam:

```text
HTTP FastAPI endpoints
  -> official deepseek-recipe request conversion
  -> DeepseekV41Encoding.with_tokenizer(...).encode(conversation)
  -> split tokens[:-1] / tokens[-1]
  -> DwarfStar PrefillContinuationState builder
  -> OMLXGenerationSession
  -> InferenceChunk.token(token_id)
  -> deepseek-recipe StreamProcessor
```

Modules:

- `ds41f_mlx/serving/server.py`
- `ds41f_mlx/serving/deepseek_recipe_backend.py`
- `tools/run_ds41f_recipe_server.py`
- `tools/run_m7_deepseek_recipe_serving_smoke.py`

The server defaults to loopback (`127.0.0.1`), text-only, single-flight model execution, and MTP/DSpark OFF.  It exposes `/v1/chat/completions`, `/v1/responses`, and `/v1/messages` through official recipe conversion/response classes.  Image inputs are rejected before image fetching/preprocessing.

Qualification blocker: official recipe V4.1 text prompts encode to approximately 30+ tokens even for a minimal request.  The current available DwarfStar `PrefillContinuationState` builder used by this repository's qualification path is not yet a practical arbitrary-length production prefill path: a 30-token prefix did not complete within the bounded smoke window, and a 29-token prefix exposes the existing ratio-2 grouped-state reshape limitation.  The integration therefore intentionally does **not** fall back to oMLX prompt replay or native reference prefill.

Artifact: `artifacts/m7/deepseek-recipe-serving/result.json` records recipe revision/version, tokenizer source, prompt split accounting, parser-ownership proof using `InferenceChunk.token(...)` and `StreamProcessor`, implemented cancellation/streaming seams, and the final classification:

```text
DEEPSEEK_RECIPE_TEXT_SERVING_PATH_NOT_QUALIFIED_PREFILL_GENERALIZATION_BLOCKED
```

Next frontier: generalize/accelerate the real DwarfStar-derived production prefill path for arbitrary recipe-encoded prompt lengths while preserving the no-replay `PrefillContinuationState -> OMLXGenerationSession` contract.
