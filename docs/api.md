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

Runtime architecture selection is governed by `docs/runtime-strategy.md` and `docs/implementation-plan.md`. The current native implementation is the correctness/reference runtime and a source of reusable components; DwarfStar-derived prefill and the eventual selected decode architecture define the intended production path once qualified.

A fully connected native HTTP serving path is not claimed until explicitly validated. Until then, API behavior should be described per backend used by a deployment.

## Error behavior

Invalid requests should fail atomically with respect to model/session state: no partial token commit, KV mutation, publication, or generation-state advance should survive a rejected request.

## Unsupported/unqualified options

Do not assume support for vision, speculative decode, long-context production serving, streaming parity, or PyTorch RNG parity unless a current qualification entry states it.

## Development guidance

Do not start API integration before runtime architecture restoration reaches the serving milestone. Future API work should bind through a narrow backend interface rather than creating another production session-state implementation.
