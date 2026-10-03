# Qualification status

This document records the current qualified state at HEAD. Historical milestone documents remain evidence, but this page is the canonical current-state summary.

M20's promoted decode dependency is clean upstream oMLX `v0.7.0` at
`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`. Fresh bounded lifecycle,
persistence/HTTP/tool/EOS, A/B, repeated-session and 200K endpoint evidence is
recorded in `artifacts/m20/promotion.json`. Unchanged source/prefill ladder
evidence is retained; old M18 whole-runtime inheritance is not valid for this
new dependency. See [M20](m20-omlx-release-migration.md) for exact scopes.

M25's [pinned upstream MTP decision](milestone-25-mtp-decision.md) is
**REJECT/DEFER**, not optional qualification. Actual accepted/rejected speculative
cycles and 200K diagnostics do not close idle extraction. Repeated MTP sessions,
protocol/tool gates and MTP restart/restore remain blocked; OFF evidence does not
transfer. M9 explicitly refuses preserved/active MTP models. New bounded OFF
regression/soak evidence is separate under `artifacts/m25/`.

## Guarded MTP qualification (internal, not release promotion)

[M33](milestone-33-protocol-qualification.md) closed the protocol gate.
[M34](milestone-34-operational-qualification.md) now qualifies **bounded guarded
singleton operation** on its exact isolated candidates plus a narrow native
cache-ownership transfer correction: 90-turn/three-session soak, 12-turn
cancellation/re-entry, 16 interruption/fault cases, and paired 4K/12K controls.
Observed replay/repack are zero; cache/frontier, semantic ownership and delivery
prefixes remain coherent. This supersedes the M25 blocker only for this guarded
internal path, not unmodified upstream or public serving.

[M35](milestone-35-http-sse-qualification.md) additionally qualifies **bounded
internal single-flight HTTP/SSE transport**: 18 real HTTP turns, ordinary and
semantic-terminal disconnect recovery, slow consumers, tool-result re-entry and
Rust iterator drop, with zero replay/repack. Explicit backend object injection
only; no normal public/release selector. Matched direct/HTTP model-phase rates
remain effectively equal.

[M36](milestone-36-client-recovery-qualification.md) is
**BLOCKED_CANONICAL_PROTOCOL_REPRESENTABILITY**: unfinished canonical tool DSML
cannot be assumed to be an ordinary reconstructable assistant result. Internal
settlement now poisons unfinished tool results; busy remains visible until cleanup
settles. A real partial-UTF-8 text recovery and finite restrictive socket workload
are observations, not full M36 qualification.
[M36R](milestone-36r-recovery-admission.md) qualifies a narrower **bounded certified
recovery** contract: official-recipe ordinary exact-prefix reconstruction plus
canonical tool completion, local sequence/body outcome fencing, and a living
single-client tool ledger. Negative certificates require DELETE/fresh session;
partial DSML is not repaired. The normal release API and Rust client are unchanged.
Client integration, release and longer-context gates remain separate.
Production/default MTP remains OFF; public MTP disabled; MTP persistence/restore,
token-exact immediate abort and shared/concurrent MTP remain unsupported.

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
| Long sessions | QUALIFIED FOR SINGLE-SESSION TEXT/AGENT SCOPE | Repeated exact-prefix append/decode, bounded diagnostics, persistence/restore, client interruption recovery, and stable bounded memory/performance in restored long-session and M24 operational-soak evidence. |
| Persistence/restore | QUALIFIED FOR SAME-BACKEND IDLE ARTIFACTS | Idle `DeepseekV41Cache[40]` plus exact all-token history saves and restores with provenance/schema/shape/dtype/frontier validation and fail-closed corruption handling. |
| Tools/agent loops | QUALIFIED FOR CLIENT FUNCTION TOOLS | Chat Completions tool-call boundaries, tool results, repeated tool loops, invalid result rejection before mutation, persistence between tool steps, and M24 client-side coding-tool soak. |
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
- Production/public MTP, DSpark, or speculative decode (bounded isolated guarded qualification is separate above).
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
- `artifacts/m24/operational-soak.json`

Those files preserve chronology and exact run details. New runtime paths or broadened API scope must be qualified separately rather than inferred from adjacent evidence.
