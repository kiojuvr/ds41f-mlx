# Public API contract

This is the release-scope API contract for the qualified text runtime. It is intentionally narrower than all fields that the upstream `deepseek-recipe` request classes can parse.

## Qualified release scope

Transport is local HTTP. Protocol conversion, prompt rendering, response parsing, tool-call parsing, thinking fields, and stream formatting are delegated to official DeepSeek `deepseek-recipe`; `ds41f-mlx` supplies the backend runtime below that layer. Production paths and server settings are resolved through `ds41f_mlx.config.RuntimeConfig`; see `operations.md`.

Supported endpoints:

- `GET /health` — process/model health. Returns `alive` before model load, `ready` after the backend has a model, or `unavailable` with `fatal_error`.
- `GET /v1/models` — model id plus aliases. The release model id is `deepseek-v4.1-flash`; accepted aliases are `deepseek-v41-flash` and `deepseek-flash`.
- `POST /v1/chat/completions` — stateless text Chat Completions.
- `POST /v1/responses` — stateless text Responses smoke-qualified scope.
- `POST /v1/messages` — stateless text Messages scope pinned to the qualified recipe revision.
- `POST /v1/sessions` — create a local stateful Chat Completions session. Optional body field: `id`.
- `GET /v1/sessions/{id}` — bounded metadata for a live session; not a cache/token authority.
- `DELETE /v1/sessions/{id}` — close a live session and release its GenerationBatch/session ownership.
- `POST /v1/sessions/{id}/chat/completions` — continue one local stateful Chat Completions session.
- `POST /v1/sessions/{id}/persist` — persist an idle session artifact. Optional body field: `artifact_root`.
- `POST /v1/sessions/restore` — restore an idle same-backend artifact into a local session. Required body field: `artifact_path`; optional `id`.

## Runtime contract behind the API

Production serving uses `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`: official recipe tokens are rendered, `tokens[-1]` is held out, `tokens[:-1]` are committed through `DeferredPrefillAppend` with P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, and P5 hands the held-out terminal token exactly once to oMLX `GenerationBatch` with MTP, DSpark, and speculative decode OFF.

Stateful sessions route to exactly one live recipe tool session/GenerationBatch authority. The HTTP layer never owns KV tensors, all-token history, prompt replay, cache repack/export, or tool execution.

## Stateful behavior

- Scope: Chat Completions only.
- Continuation: exact-prefix continuation from the committed conversation/cache frontier.
- Prompt replay on stateful continuation: 0.
- Full-cache repack/reconstruction on stateful continuation: 0.
- Live authority: one executable cache authority per session.
- Idle boundaries: all 40 cache offsets must equal the committed frontier.
- Maximum live sessions: `DS41F_MAX_LIVE_SESSIONS`, default `4`.
- Single-flight: one backend request at a time; overlapping requests fail with conflict instead of creating a second active authority.
- Close: releases live session and GenerationBatch ownership.

## Persistence/restore

Persistence is qualified only for idle same-backend artifacts: manifest + safetensors + commit marker, with checkpoint/schema/shape/dtype/frontier/provenance validation. Restored artifacts are dormant storage until loaded; they are never a second live authority. Restore fails closed on mismatch/corruption and does not fall back to fresh prompt prefill.

Cross-runtime or cross-backend KV portability is outside the release claim.

## Function tools

Ordinary client-side Chat Completions function tools are supported. The model may emit tool calls; the client executes tools and sends tool results in the next session turn. Repeated tool/result loops are qualified for this scope. The server does not execute arbitrary tools, plugins, MCP, shell commands, or web search.

Invalid tool-result order, wrong ids, duplicate results, unknown sessions, and overlap conflicts fail before session mutation.

## Streaming

Stateless streaming follows official recipe chunk formatting for text Chat Completions/Responses in the qualified scope. Stateful Chat Completions streaming replays official chunks only after a committed session boundary, so protocol-visible chunks never outrun the cache frontier. Cancellation triggers cleanup and must not leave an active GenerationBatch owner.

## Termination

DeepSeek V4.1 EOS token semantics are qualified for the GenerationBatch path. EOS token id `1` is retained in cache/all-token history, hidden from protocol text by the recipe layer, and reported as finish reason `stop`. Length termination reports `length` according to the recipe response.

## Stateful `stop` policy

Arbitrary request stop strings are **not supported** on stateful session endpoints. A request body containing a non-empty/non-null `stop` field is rejected with `invalid_request_error` before recipe conversion can mutate session state. This avoids committing generated tokens/KV that the client protocol history would hide after detokenized stop-string truncation.

Stateless endpoints continue to use official recipe request behavior for fields the recipe supports. The qualified stateful termination mechanism is DeepSeek EOS token/length, not generic KV rollback.

## Error behavior

Rejected requests are atomic with respect to model/session state: no partial token commit, KV mutation, publication, or generation-state advance may survive. Unknown session returns not-found; active overlap/max-session conflicts return conflict; validation failures return invalid request errors.

## Optional / experimental

Diagnostic endpoints exist only when `DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS=1`. Their output is bounded operational metadata and is not a public cache API.

## Unqualified / unsupported

Vision, arbitrary batching, sessionized Responses/Messages, MTP, DSpark, speculative decode, distributed sessions, authentication, server-side tool execution, MCP/plugins, web search, shell tools, cross-runtime KV portability, and arbitrary stateful stop strings are outside the release contract.
