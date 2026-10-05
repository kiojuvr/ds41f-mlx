# Qualification status

This document records the current qualified state at HEAD. Historical milestone documents remain evidence, but this page is the canonical current-state summary.

Development qualification and runtime release promotion are separate. Ordinary
`ds41f-mlx` milestones run affected tests, applicable R1 conformance and necessary
real-model/lifecycle/performance/operational gates; multiple qualified milestones
may accumulate without promoting `ds41f-runtime`. Changed release-surface files
do not automatically require independent release environments, repeated
projection/determinism qualification or a new receipt. Only an explicit promotion
checkpoint applies M43's release rules. M44 remains PASS; its completed promotion
work is retained as evidence, not a mandatory future milestone sequence. See the
[development/promotion policy](reference-release-and-promotion-strategy.md).

## Current profile and reference authority

`standard-off` remains default qualified production. [M41](milestone-41-local-mtp-release-candidate.md)
separately qualifies explicit bounded `mtp-singleton-v1`; the internal historical
promotion statements below must not be read as reversing M41's scoped admission.
[M42 / Reference Release R1](milestone-42-reference-release.md) makes the established
semantics executable through [owned contracts and conformance](reference-release-r1.md).
It neither broadens either profile nor promotes a new numerical backend.

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

## Post-M48 first-party OFF long-session core

The [200K production qualification](standard-off-200k-production-qualification.md)
provides fresh first-party core evidence, not inheritance from M6/M20/M24: long
prefill followed by sustained exact-prefix append/decode, idle persistence and
fresh-process exact restore, cancellation/re-entry and resource behavior. It also
closes the measured free-allocation cache retention gap with a lifetime-scoped
32 GiB budget. The long turns exercise the actual core seams; this is not a new
200K HTTP/client tool-loop or unlimited-process-lifetime qualification. M48's
numerical ownership and M44–M47 state/failure boundaries remain intact. Full R1,
release/packaging and `ds41f-runtime` promotion were not performed.

## Very-long-context text core

[Fresh very-long qualification](very-long-context-production-qualification.md)
closes bounded 512K/768K operation and continuation to **1,048,576 consumed tokens**
with practical decode, SSD Engram, idle persistence, exact fresh-process restore,
cancellation/re-entry and zero replay/repack. A prefill/idle GPU residency-lifetime
defect exposed by 512K jetsam was repaired. P5 also retires a passive P6
certificate/setup cycle rather than retaining old publication buffers until GC.
All three frontiers were requalified with identical tokens and all checked slots.
The supported envelope accepts long batch ingestion latency; it is not physical
RAM exhaustion, arbitrary prompt-quality proof or long HTTP/client/tool recovery.
The configured 1M model range is supported here only because actual operation
reached it; no above-checkpoint extrapolation, new release or promotion is claimed.

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
[M37](milestone-37-local-client-integration.md) integrates that contract in a reusable
experimental local Python client above the existing browser HTTP boundary: frozen
requests, centralized certified reconciliation, bounded tool ownership and explicit
DELETE/fresh/expired results. The public browser loop and Rust raw API stay unchanged.
Current-source decision and exact scope are in its canonical artifact; release,
longer-context and release gates remain separate.
[M38](milestone-38-client-operational-soak.md) completes a 76-request living-client
soak and bounded fault/ledger matrix, but is **BLOCKED_OPERATIONAL_LIFETIME_RETENTION**:
server closed records/diagnostic payloads have no lifetime cap. Client ambiguity
now remains fenced across re-observation and lifecycle actions. This is not promotion.
[M39](milestone-39-lifetime-admission.md) resolves that retention blocker by narrowing
internal IDs to server-issued process-namespace serial lifetimes. No-wrap issuance
and live-only admission fence retired identities independently of 16 bounded summaries;
closed records/payloads are discarded. 20,000-lifetime structural stress and checkpoint
post-eviction integration pass, with explicit exhaustion denial rather than unlimited
issuance. This remains internal; the next task is dedicated release/admission/provenance
readiness evaluation, not further recovery micro-qualification.
Production/default MTP remains OFF; public MTP disabled; MTP persistence/restore,
token-exact immediate abort and shared/concurrent MTP remain unsupported.

## Qualified release scope

| Area | Status | Scope |
| --- | --- | --- |
| Checkpoint provenance | QUALIFIED | Official DeepSeek-V4.1-Flash checkpoint identity and local checkpoint paths are recorded in artifacts and provenance docs. |
| Backend-local fidelity policy | QUALIFIED | Correctness is backend-local for a fixed checkpoint/runtime/backend/build/config/input/session state; cross-backend hidden/logit/token identity is not required under official-compatible floating-point semantics. |
| Production prefill | QUALIFIED | `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`, dense FP8/MLX path, P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, `P7_ENGRAM_TILE=2048`. |
| P5 handoff | QUALIFIED | Terminal prompt token held out once and handed to ds41f TargetGenerationSession; no prompt replay and no second cache authority. |
| Decode | QUALIFIED | ds41f `TargetGenerationSession`, MTP OFF, DSpark OFF, speculative decode OFF; M44 records generation qualification; M45 records R1, owned-forward/all-layer lifecycle and matched pre-M45 real-model/cache/performance regression evidence. M46 adds owned block/packed state producers with R1 standard-off PASS, 112 affected tests (32 subtests), and matched M45 4K/32K token/cache identity and performance regression evidence; see `milestone-46-state-production-ownership.md`. M47 adds enforced execution-resource admission: 146 affected tests (32 subtests), all 24 R1 standard-off gates PASS, and admitted matched M46 4K/32K token/cache/performance regression evidence. See `milestone-47-execution-resource-admission.md`. M45–M47 qualification is development-only, not runtime promotion. |
| API serving | QUALIFIED | Local text-only single-flight HTTP: stateless Chat Completions, Responses, Messages; stateful Chat Completions sessions. |
| Long context | QUALIFIED DEVELOPMENT CORE THROUGH 1M | Fresh first-party 512K/768K/near-1M prefill, decode, continuation to 1,048,576 consumed tokens and exact fresh idle restore; bounded maintenance/document workload, not a new long-HTTP/client/release claim. |
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
- Default/implicit or generalized public MTP, DSpark, or speculative decode (explicit bounded M41 local profile is separately qualified).
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
