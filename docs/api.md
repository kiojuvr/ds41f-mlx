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

## Current endpoint surface

The server exposes both stateless compatibility endpoints and local sessionized endpoints over the same official DeepSeek recipe conversion/parsing layer:

### `GET /health`

Reports service health for the active backend.

### `GET /v1/models`

Returns the available model identifiers exposed by the server.

### `POST /v1/chat/completions`

Stateless Chat Completions request-local inference path qualified in M7.

### `POST /v1/responses`

Stateless Responses request-local inference path qualified in M7 for text smoke coverage.

### `POST /v1/messages`

Stateless pinned-recipe Messages text smoke path qualified in M7.

### `POST /v1/sessions`

Creates a local stateful session id for Chat Completions agent/tool loops.

### `GET /v1/sessions/{id}` / `DELETE /v1/sessions/{id}`

Inspect or close a local stateful session. Diagnostics are bounded metadata only and are not a second cache/token authority.

### `POST /v1/sessions/{id}/chat/completions`

Routes an official Chat Completions request through one `M11RecipeToolSession`, preserving M8/M9/M10/M11 cache/session invariants across tool-call and tool-result turns.

### `POST /v1/sessions/{id}/persist` and `POST /v1/sessions/restore`

Persist or restore an idle session through the existing M9 artifact model. Persisted artifacts are dormant storage, never a second live authority.

## Backend/runtime ownership

Runtime architecture selection is governed by `docs/runtime-strategy.md` and `docs/implementation-plan.md`. Current serving selection is `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`: official recipe encoding supplies complete prompt tokens, serving holds out `tokens[-1]`, prefill commits `tokens[:-1]` through `DeferredPrefillAppend`, and P5 supplies the held-out terminal token exactly once to oMLX `GenerationBatch` MTP-OFF.

M7 validates the stateless DeepSeek-recipe HTTP path for text-only single-flight serving. M8 validates repeated exact-prefix continuation. M9 validates same-backend idle KV persistence/restore. M10 validates repeated restored long sessions. M11 validates official-recipe Chat Completions function-tool boundaries. M12 exposes the M11 session object through local `/v1/sessions` HTTP endpoints, including streaming committed-boundary replay, overlap/unknown-session fail-closed behavior, and persistence/restore re-entry. M13 validates repeated Chat Completions function-tool agent loops through the same sessionized HTTP surface. M14 restores/qualifies DeepSeek V4.1 EOS token termination for the direct GenerationBatch path: EOS is retained in cache/all_tokens for exact-prefix continuity, hidden from protocol text, and reported as stop.

The HTTP/session layer does not own KV tensors, token history, prompt replay, or tool execution. A session id routes requests to exactly one live `M11RecipeToolSession` authority. Prompt replay, cache repack/export, MTP/DSpark, multimodal input, reference vertical-slice serving, and the legacy one-chunk prefill substrate are not production API paths.

## Error behavior

Invalid requests should fail atomically with respect to model/session state: no partial token commit, KV mutation, publication, or generation-state advance should survive a rejected request.

## Unsupported/unqualified options

Do not assume support for vision, speculative decode, batching, general tool execution, distributed sessions, authentication, sessionized Responses/Messages, or PyTorch RNG parity unless a current qualification entry states it. Stateless streaming text for Chat Completions and Responses is qualified in M7; stateful Chat Completions streaming in M12 replays official chunks only after a committed M11 boundary to avoid exposing tokens beyond the cache frontier.

## Development guidance

Future API work should keep binding through the narrow backend interface rather than creating another production session-state implementation.
