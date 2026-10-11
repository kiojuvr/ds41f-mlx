# Production MTP logical sessions

This page owns **`mtp-serving-v1` logical session identity, continuation, settlement
and observation**. Independent Chat Completions remains available without sessions.
This is not the OFF native-session or singleton fence contract. Limits/defaults
live in [capabilities](capabilities.md); [ordinary admission](ordinary-api-boundary.md)
owns recipe/capability validation. No persistence, image or release scope expansion
follows from logical sessions.

## Ownership model

```text
session count / application UUID / canonical-history fence
  -> ordinary recipe conversion and capacity admission
  -> same ordinary delivery/settlement broker
  -> ProductionScheduler and sole executable BatchGenerator
  -> exact paired lease acquisition -> P5 -> canonical decode/recipe settlement
  -> immutable pair publication + upstream pin -> native retirement
  -> atomic logical count/frontier/outcome -> HTTP/GET projection
```

`ProductionSession` retains identity, canonical ordinary history/token IDs, frozen
tools/reasoning, correlations and latest response/diagnostics. It has **no native
cache, DSpark rings, parser, physical queue, generation owner or model forward**.
Production dispatch explicitly uses `production_session_response` before inherited
qualification paths; legacy OFF/qualification response entry points reject.

Scheduler pins the latest extendable pair through the existing PagedCacheManager
refcount. The logical session holds no physical snapshot, separate cache/LRU or
second payload index. Ordinary work uses the same Scheduler/cache; pressure may
not evict the leased pair. A forgotten bounded root-tail hint is an optimization
miss, not loss of the known upstream hash/refcount lease. A longer ordinary hit
cannot replace the session's exact frontier.

Missing, corrupt or mismatched canonical pairs fail **before native start**.
They never authorize prefill of retained session history. Replay/repack remain
zero. Pin replacement follows successful paired publication; close releases its
lease, not unrelated ordinary reusable pairs. Native persistence/restart restore
is unsupported. Physical capture/restore definitions belong to
[session state](session-state.md) and [ordinary serving](mtp-production-serving.md).

## Continuation authority

The first request cannot reconstruct prior assistant/tool history. Subsequent
requests must extend the exact canonical ordinary message history **and** recipe
token prefix. Frozen tool declarations and reasoning mode/effort cannot change;
sampling is request-owned and may change on a new user workflow.

Certified completed call batches require this session's matching ordered tool
results. Foreign, duplicate or missing results cannot borrow another outcome.
Recipe remains the schema/content authority; these are retained-continuity rules,
not replacement generic grammar. First-turn ordinary prefix reuse is an
optimization, never session truth. Budget preview does not reserve or generate.

## API

| Endpoint | Contract |
| --- | --- |
| `POST /v1/sessions` | `{}` or optional `id`; 1–128 ASCII letters/digits/underscore/hyphen |
| `GET /v1/sessions/{id}` | Observe empty/busy/idle/unrecoverable state, count, active identity, latest canonical turn and bounded diagnostics; never execute |
| `POST /v1/sessions/{id}/chat/completions` | Text **streaming only**; mandatory single expected-count and canonical application-UUID headers |
| `POST /v1/sessions/{id}/budget` | Ordinary recipe tokenization, capacity and exact-prefix validation without native mutation; optional expected-count header checked |
| `POST /v1/sessions/{id}/cancel` | Body `{"request_id":"known-active-id"}`; stop that Scheduler ID and reconcile canonical retirement/publication |
| `DELETE /v1/sessions/{id}` | Empty body/application-json under production ingress policy; active work rejects, stop/reconcile first |

Generation headers are `X-DS41F-Expected-Request-Count` and
`X-DS41F-Application-Request-ID`; response supplies `X-DS41F-Request-ID`.
No singleton sequence header, event cursor or delivery ACK is required.
Nonstreaming legacy session execution, persist/restore and sessionized
Responses/Messages are explicitly unavailable. Production ingress does not
activate them merely because shared server routes exist.

Caller-chosen create IDs permit GET observation after lost acknowledgement.
Create is **not idempotent retry**; issued IDs are never reusable in the process,
even after close. One live logical session and finite process-lifetime/correlation
budgets are intentional resource policy; exact values live in capabilities.
Exhaustion fails closed without identity eviction/reuse. Process restart loses
in-memory history and identity authority; there is no database or replay recovery.
This is not another executable singleton: independent ordinary work still queues
through the same single-active Scheduler.

## Settlement and interruption

Before enqueue, reservation consumes the application UUID, including when its
acknowledgement is lost. An identical POST rejects; it does not resend execution
or reproject a retained response. Stale counts and reused correlations cannot
start native work. Observe through GET instead.

Scheduler settles the consuming recipe once, publishes coherent target/DSpark
state and retires executable aliases before logical publication. Request count,
frontier and canonical response advance **together**. Normal completion, explicit
cancel and disconnect use the same finalization callback, including committed
suffix not delivered on the socket. The broker releases the active fence only
after observing Scheduler completion. SSE yields and socket writes are not
canonical acknowledgement.

An unstarted cancellation leaves count/frontier/history unchanged but retains the
consumed correlation and explicit nonadmission error. Unfinished tool protocol or
nonrepresentable recipe history retains the actual diagnostic outcome and makes
the session unrecoverable, without effect/continuation authority. Native or pair
publication failure fails engine/session closed. Failed transport cleanup does
not prove nonadmission: a registered Scheduler ID stays fenced until retirement.
No outcome is automatically retried, discarded or replaced with guessed text.

`GET` exposes `last_turn.response_json`, native/application request IDs, cancelled
disposition, reconstruction certificate and `diagnostics.m8`: frontier, all target/
DSpark offsets, alignment and replay/repack counts. These are canonical bounded
observations, not exported executable state or a durable archive of every outcome.
Tools execute outside the runtime under application policy. Completed-call
certification is not real-world exactly-once execution; application reservation/
result observation governs effect uncertainty.

## Frontend boundary

`/health` advertises `capabilities.stateful_chat = production-sessions-v1`,
persistence support, live-session limit and current `generation-policy-v1` defaults.
Browser/TUI consume runtime capabilities rather than inheriting the original
milestone's greedy settings. Chat Auto output uses budget preview and admission
rechecks. Unsupported images/controls and thinking budgets reject rather than
silently rewriting policy.

A frontend workflow snapshots all eight sampling fields through tools, continuation
and cancellation/reconciliation. Sampling is not hidden inheritance from the
previous UI request; retained reasoning is a separate frozen protocol envelope.
See [frontend contract](generation-policy-frontends.md) and
[operator Chat](operator-chat.md).

## Scoped acceptance and earlier edition

[Production session acceptance](production-sessions-acceptance.json) records its
exact installed-payload scope. The
[original edition](archive/editorial-baseline/production-sessions.md.txt) preserves
canonical/cache-pressure/Chat Server turn counts, original regression results,
installation paths and failed root-tail-hint attempt. Those results are not fresh
HEAD approval or later generation-policy qualification. They do not broaden OFF,
long-context, image, TUI or release scope.

`tools/qualify_production_sessions.py` is the existing acceptance driver for an
already READY installed ordinary-MTP runtime, **not an instruction to run a new
campaign during compilation**. Focused tests cover identity/count rejection,
opaque pins and hint churn, no qualification dispatch, settlement-before-delivery,
missing-pair no-replay, fail-closed uncertainty and existing production boundaries.
Distinct native recipe environments must not bypass admission to make tests pass.
