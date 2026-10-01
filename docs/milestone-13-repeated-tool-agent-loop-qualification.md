# Milestone 13 repeated-tool agent-loop robustness

Status: **M13_REPEATED_TOOL_AGENT_LOOP_QUALIFIED** for local sessionized Chat Completions function-tool loops.

Primary evidence: `artifacts/m13/repeated-tool-agent-qualification.json` (`schema: ds41f.m13.repeated-tool-agent-qualification.v1`).

## Scope

M13 keeps the M12 HTTP/session architecture unchanged. Repeated tool use is qualified by repeatedly routing official Chat Completions requests through the same `M11RecipeToolSession`; no multi-tool runtime path, alternate prompt renderer, or HTTP-owned token/cache state is added.

Qualified lifecycle target:

```text
/v1/sessions/{id}/chat/completions
  -> model tool call A
  -> external deterministic result A
  -> same session request with result A and next user/tool instruction
  -> model tool call B
  -> external deterministic result B
  -> same session final assistant request
```

A persistence-interrupted variant persists after tool call B, tears down the HTTP process, restores through `/v1/sessions/restore`, then supplies result B and continues.

## Invariants at every boundary

- one executable cache authority: `M11RecipeToolSession` / M8 live cache or active GenerationBatch;
- official `deepseek-recipe` conversion/parsing/response semantics;
- exact-prefix continuation before mutation;
- prompt replay `0`;
- full-cache repack/reconstruction `0`;
- P5 terminal token consumed once;
- no stateful fresh-prefill fallback;
- bounded diagnostics only, never a second token-history authority.

## Post-tool behavior classification

M13 compares a stateful post-tool continuation with a stateless fresh production request over the same complete official Chat Completions body. This control uses the same M7 production path but intentionally fresh-prefills the whole prompt as a diagnostic oracle outside the stateful session. The comparison records finish reason, content prefix, token counts, and session replay/repack counters. It classifies the repeated/no-progress post-tool symptom as runtime/session defect only if stateful-only divergence correlates with cache/frontier/replay/repack evidence.

## Failure behavior

The runner validates official-conversion rejection for wrong/stale/duplicate tool results, retry after rejected continuation, unknown session, overlap conflict, and streaming disconnect around a committed tool boundary. Rejections must happen before committed session frontier changes.

## Qualification closeout

The real HTTP qualification executed two tool boundaries in one continuous session:

1. `lookup_weather({"city":"Paris"})` -> deterministic result `sunny 21C`;
2. `lookup_weather({"city":"Berlin"})` -> deterministic result `cloudy 17C`;
3. final assistant summary request in the same session.

Recorded live frontiers:

- after tool call A: `336`;
- after tool call B: `419`;
- after final assistant turn: `564`.

Persistence variant:

- persisted at committed tool-call B boundary/frontier `419`;
- artifact tensors: `280`;
- fresh HTTP server restored the artifact;
- tool result B continued with prompt replay `0`, full-cache repack `0`, all cache offsets equal frontier.

Failure/retry evidence:

- wrong `tool_call_id`: HTTP `400`, frontier unchanged;
- duplicate tool result: HTTP `400`, frontier unchanged;
- overlapping same-session request: HTTP `409` while the original completed;
- rejected requests did not mutate the committed frontier.

Post-tool behavior classification:

A stateless fresh production request over the same complete official recipe conversation produced the same useful answer prefix and the same length-finished continuation pattern, including decoded special-token-looking text such as `<｜end▁of▁sentence｜>`. Stateful and stateless prefixes matched for the first 120 characters, both finished by length, and stateful replay/repack/frontier counters remained clean. M13 therefore classifies the symptom as model/prompt/recipe/generation-stop behavior exposed by both production paths, not an M8/M11/M12 session-state defect. No evidence was found that a protocol parser completion event or cache boundary was being mismatched across the session layer.
