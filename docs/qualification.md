# Qualification status

This document records the current qualified state at HEAD. Historical milestone documents remain evidence, but this page is the canonical current-state summary.

M20's promoted decode dependency is clean upstream oMLX `v0.7.0` at
`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`. Fresh bounded lifecycle,
persistence/HTTP/tool/EOS, A/B, repeated-session and 200K endpoint evidence is
recorded in `artifacts/m20/promotion.json`. Unchanged source/prefill ladder
evidence is retained; old M18 whole-runtime inheritance is not valid for this
new dependency. See [M20](m20-omlx-release-migration.md) for exact scopes.

## Qualified release scope

| Area | Status | Scope |
| --- | --- | --- |
| Checkpoint provenance | QUALIFIED | Official DeepSeek-V4.1-Flash checkpoint identity and local checkpoint paths are recorded in artifacts and provenance docs. |
| Backend-local fidelity policy | QUALIFIED | Correctness is backend-local for a fixed checkpoint/runtime/backend/build/config/input/session state; cross-backend hidden/logit/token identity is not required under official-compatible floating-point semantics. |
| Production prefill | QUALIFIED | `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`, dense FP8/MLX path, P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, `P7_ENGRAM_TILE=2048`. |
| P5 handoff | QUALIFIED | Terminal prompt token held out once and handed to oMLX GenerationBatch; no prompt replay and no second cache authority. |
| Decode | QUALIFIED | oMLX `BatchGenerator`/`GenerationBatch`, MTP OFF, DSpark OFF, speculative decode OFF, practical single-stream decode above the release floor. |
| API serving | QUALIFIED | Local text-only single-flight HTTP: stateless Chat Completions, Responses, Messages; stateful Chat Completions sessions. |
| Long context | QUALIFIED THROUGH 200K | Production prefill/decode performance and memory class verified through 200K-token contexts in the dense P0-P7 path. |
| Long sessions | QUALIFIED FOR SINGLE-SESSION TEXT SCOPE | Repeated exact-prefix append/decode, cancellation recovery, bounded diagnostics, and stable memory/performance in the restored long-session evidence. |
| Persistence/restore | QUALIFIED FOR SAME-BACKEND IDLE ARTIFACTS | Idle `DeepseekV41Cache[40]` plus exact all-token history saves and restores with provenance/schema/shape/dtype/frontier validation and fail-closed corruption handling. |
| Tools/agent loops | QUALIFIED FOR CLIENT FUNCTION TOOLS | Chat Completions tool-call boundaries, tool results, repeated tool loops, invalid result rejection before mutation, persistence between tool steps. |
| Termination | QUALIFIED | DeepSeek V4.1 EOS token id `1` is retained in cache/history, hidden from protocol text, and maps to finish reason `stop`. |
| Stateful stop strings | RELEASE POLICY CLOSED | Non-empty/non-null request `stop` on stateful Chat Completions is rejected before mutation. |

## Optional / experimental

- Diagnostic endpoints when `DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS=1`.
- P8 tile-native carry code and probes; default OFF, not production-selected.
- Native C++ text runtime as reference/qualification implementation and source of reusable components.

## Unqualified

- Vision or image input.
- Arbitrary batching and distributed serving.
- Sessionized Responses or Messages.
- MTP, DSpark, or speculative decode.
- Server-side tool execution, MCP/plugins, web search, shell tools.
- Cross-runtime/cross-backend KV artifact portability.
- Arbitrary stateful request stop-string truncation/rollback.
- Authentication or multi-tenant service hardening.

## Non-goals

- Requantized or reduced checkpoint substitutes.
- Weakening production invariants to pass a benchmark.
- Hidden fresh-prefill fallback for stateful continuation or restore.
- Cross-backend bit identity as a universal correctness requirement.

## Historical / diagnostic evidence

Current dependency-migration authority: `artifacts/m20/promotion.json` and its
hashed gate artifacts. Historical artifacts retained or refreshed as classified
in M20 include:

- `artifacts/m6/performance-qualification/result.json`
- `artifacts/m7/deepseek-recipe-serving/result.json`
- `artifacts/m8/long-session-qualification.json`
- `artifacts/m9/kv-persistence-qualification.json`
- `artifacts/m10/restored-long-session-qualification.json`
- `artifacts/m11/tool-boundary-qualification.json`
- `artifacts/m12/sessionized-http-qualification.json`
- `artifacts/m13/repeated-tool-agent-qualification.json`
- `artifacts/m14/termination-qualification.json`

Those files preserve chronology and exact run details. New runtime paths or broadened API scope must be qualified separately rather than inferred from adjacent evidence.
