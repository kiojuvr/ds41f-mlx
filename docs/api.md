# API status

The eventual external serving layer should use official DeepSeek `deepseek-recipe` for protocol, prompt, response, tool-call, thinking, and streaming behavior. The project should not independently reinvent those conversions.

Target shape:

```text
HTTP transport
  ↓
deepseek-recipe
  ↓
ds41f backend interface
  ↓
production runtime
```

## Current endpoint scaffold

Existing scaffold code has used an OpenAI-compatible chat-completion surface:

### `GET /health`

Reports service health for the active backend.

### `GET /v1/models`

Returns the available model identifiers exposed by the server.

### `POST /v1/chat/completions`

Accepts chat-completion requests for the target model path. Supported request options depend on the currently wired backend.

## Backend/runtime ownership

Runtime architecture selection is governed by `docs/runtime-strategy.md` and `docs/implementation-plan.md`. Current M7 serving selection is `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`: official recipe encoding supplies complete prompt tokens, serving holds out `tokens[-1]`, prefill commits `tokens[:-1]` through `DeferredPrefillAppend`, and P5 `handoff_to_generation()` supplies the held-out terminal token exactly once to oMLX `GenerationBatch` MTP-OFF. Prompt replay, cache repack/export, MTP/DSpark, multimodal input, reference vertical-slice serving, and the legacy one-chunk prefill substrate are not production API paths.

A fully connected native HTTP serving path is not claimed until explicitly validated. Until then, API behavior should be described per backend used by a deployment.

## Error behavior

Invalid requests should fail atomically with respect to model/session state: no partial token commit, KV mutation, publication, or generation-state advance should survive a rejected request.

## Unsupported/unqualified options

Do not assume support for vision, speculative decode, long-context production serving, streaming parity, or PyTorch RNG parity unless a current qualification entry states it.

## Development guidance

Do not start API integration before runtime architecture restoration reaches the serving milestone. Future API work should bind through a narrow backend interface rather than creating another production session-state implementation.
