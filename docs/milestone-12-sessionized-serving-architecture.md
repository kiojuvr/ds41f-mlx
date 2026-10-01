# Milestone 12 sessionized HTTP / agent serving lifecycle

Status: **M12_SESSIONIZED_HTTP_QUALIFIED** for local Chat Completions function-tool sessions.

Primary evidence: `artifacts/m12/sessionized-http-qualification.json` (`schema: ds41f.m12.sessionized-http-qualification.v1`).

## Objective

Expose the already-qualified M11 session object through HTTP without creating another model-state authority:

```text
HTTP / agent client
  -> official deepseek-recipe request conversion and response/parsing
  -> session id routing
  -> one M11RecipeToolSession
  -> M8/M9/M10 cache lifecycle
```

The HTTP layer owns only bounded metadata: session id, timestamps, busy/closed flags, last diagnostics, and optional M9 artifact path. It never owns KV tensors, token history, or a prompt replay fallback.

## Public API

Stateless M7 endpoints remain unchanged:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`

M12 adds local stateful endpoints:

- `POST /v1/sessions` -> create an empty local session record; optional JSON `{"id": "..."}`.
- `GET /v1/sessions/{id}` -> bounded diagnostics and lifecycle metadata.
- `DELETE /v1/sessions/{id}` -> close/delete live cache authority.
- `POST /v1/sessions/{id}/chat/completions` -> route an official Chat Completions request to this session.
- `POST /v1/sessions/{id}/persist` -> persist the idle M11/M8 state through M9.
- `POST /v1/sessions/restore` with `{"artifact_path": "...", "id": "optional"}` -> create a session from a dormant M9 artifact.

Only Chat Completions is session-qualified for M12. Responses and Messages stay on the M7 stateless path until their continuation identity semantics are separately qualified.

## Lifecycle

First request to `/v1/sessions/{id}/chat/completions` creates the underlying `M11RecipeToolSession` through the normal DENSE_P0_P7 -> P5 -> GenerationBatch path. Later requests must be exact official-recipe token-prefix extensions of the committed `all_tokens`; otherwise M8 rejects before cache mutation.

At a model tool call, M11 uses the official parser-probe early boundary established in M11. The HTTP layer only receives the committed official response chunks/JSON. Hidden post-tool tokens are not streamed or stored.

Tool execution remains external. The caller uses official Chat Completions `tool_calls[].id/name/arguments`, executes a tool, then posts the next official Chat Completions request with the matching `role=tool` message to the same session id.

## Ownership and concurrency

The runtime remains globally single-flight. Additionally, each session has an explicit `busy` flag. A second overlapping request to the same session fails with HTTP 409 before mutating the session. Unknown or closed sessions fail with HTTP 404.

Live sessions are bounded by `DS41F_MAX_LIVE_SESSIONS` (default 4). Persistence uses the existing M9 artifact format; persisted files are dormant storage and never become a second live authority.

## Streaming

For stateful streaming requests, M12 buffers generation until the M11 committed boundary and then emits the official recipe-generated Chat Completions stream events as SSE followed by `[DONE]`. This deliberately prioritizes boundary correctness over lowest-latency token streaming. A client disconnect while replaying committed events cannot leave protocol-visible completion and cache frontier at different boundaries.

## Diagnostics

`/_ds41f/diagnostics` includes bounded `session_traces` and per-session diagnostics when enabled with `DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS=1`. These distinguish parser/tool completion, exact-prefix continuation, prompt replay, full-cache repack, cache-frontier mismatch, wrong session/unknown session, and overlap conflicts without becoming state authority.

## Qualification closeout

The M12 runner exercised real loopback HTTP with the official checkpoint and pinned recipe bindings:

- non-streaming session: create session -> tool-call request -> external deterministic stub -> same-session tool-result request -> final assistant response;
- streaming session: same tool-call request with `stream=true`; M12 replayed only committed official chunks and preserved the M11 early tool boundary;
- client disconnect after first committed stream event: session remained idle at a coherent committed frontier;
- overlap and unknown-session behavior: same-session overlapping request returned `409`, unknown session returned `404`;
- persistence/restore: first HTTP server committed tool-call boundary and persisted through M9, process was torn down, fresh HTTP server restored the M9 artifact, and the tool-result request continued through exact-prefix append.

Recorded invariant gates:

- tool-call frontier: `321`;
- persisted tensors: `280`;
- prompt replay: `0`;
- full-cache repack/reconstruction: `0`;
- all cache offsets equal frontier at idle boundaries;
- no stateful fresh-prefill fallback after tool result;
- exact-prefix continuation recorded in `session_traces`.

The post-tool assistant response still shows the same model/prompt tendency observed in M11: useful first answer followed by repetitive/no-progress text until length. M12 classifies this as model/protocol generation behavior with clean runtime lifecycle counters, not cache replay or reconstruction.
