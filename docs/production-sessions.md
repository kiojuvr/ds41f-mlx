# Production MTP logical sessions

Scope: `mtp-serving-v1` text Chat Server backend capability. Ordinary Chat
Completions remains available without a session. Singleton qualification,
standard-OFF, persistence/restore, images, long-context qualification and release
promotion are not expanded by this milestone. TUI presentation is a separate WIP.

## Ownership model

```
HTTP session count/application-UUID/history fence
  → ordinary authoritative recipe conversion/capacity admission
  → same ordinary broker (_deliveries / _settled)
  → ProductionScheduler queue and sole executable BatchGenerator
  → PairedCheckpointAuthority acquire → P5 → canonical decode/recipe settlement
  → immutable pair publication + upstream refcount pin → native retirement
  → atomic logical outcome/count/frontier publication → HTTP/GET projection
```

`ProductionSessions` / `ProductionSession` retain only logical identity, canonical
ordinary history, canonical token identity, frozen declarations/reasoning, request
correlations and the latest canonical response/diagnostics. They have **no native
cache, ring, parser, generation owner, physical queue or model forward**.

The existing ProductionScheduler continues to own physical request contexts,
canonical native settlement, and paired checkpoint publication. Its physical
hooks still reuse the existing qualified core, as ordinary serving already did;
there is no qualification response loop, singleton worker lease, or live native
session cache attached to a logical session. Production explicitly overrides the
qualification/standard-OFF/legacy response entry points to reject them. The
server's ordinary session branch dispatches explicitly to
`production_session_response` before any historical qualification detection.

The Scheduler pins the latest extendable canonical pair using **the existing
PagedCacheManager refcount**. The session stores no physical snapshot or separate
cache index. Capture/restore/publication/eviction remain PairedCheckpointAuthority
operations. Ordinary requests can continue to use the same cache and queue;
ordinary cache pressure cannot evict the pinned pair. Session continuation restores
its exact upstream hash/refcount lease, not an ordinary prefix hint. The upstream
bounded root-tail hint index can forget a live pinned block; that is an optimization
miss, not loss of the known leased physical pair. There is still one physical
payload/index/refcount authority, not a second session cache or LRU. A longer
matching ordinary prefix can never replace this session's frontier.
A missing/corrupt/mismatched canonical pair fails closed **before native start**;
it never becomes prefill of the historical session tokens. Native replay/repack
counters remain zero. Pin replacement occurs only after paired publication; close
releases this Scheduler lease, not unrelated ordinary reusable pairs.

A first session request cannot reconstruct previous assistant/tool history.
Subsequent requests must extend both the exact canonical ordinary history and
its exact recipe token prefix. Certified completed tool calls require matching,
ordered tool results; foreign/duplicate/missing results cannot borrow another
session's outcome. The recipe remains the message/content/schema authority.
Cached prefix reuse on a first turn is just an optimization, never session truth.

## API

* `POST /v1/sessions`: `{}` or `{"id":"client-chosen-id"}`. IDs are 1–128 ASCII
  letters/digits/underscore/hyphen. A known caller-chosen ID allows GET observation
  after lost create acknowledgement. IDs are never reusable within this process,
  even after close; create is not an idempotent retry.
* `GET /v1/sessions/{id}`: empty/busy/idle/unrecoverable, active request identity,
  request count, last canonical `response_json`, native/application request IDs,
  cancelled disposition, reconstruction certificate and `diagnostics.m8` frontier,
  target/DSpark offsets, alignment, replay/repack counters. GET never executes.
* `POST /v1/sessions/{id}/chat/completions`: text streaming only, with mandatory
  single `X-DS41F-Expected-Request-Count` and canonical-UUID
  `X-DS41F-Application-Request-ID`. The response supplies `X-DS41F-Request-ID`.
  No sequence header or delivery cursor/ACK is required. Nonstreaming legacy
  stateful execution is explicitly unavailable on this production session path.
* `POST /v1/sessions/{id}/budget`: real ordinary recipe tokenization and qualified
  context/output reservation, exact session prefix validation, no native mutation
  or generation. Optional expected-count header is checked. Chat Server preview
  and post-effect tool-result fitting use this endpoint; actual admission rechecks.
* `POST /v1/sessions/{id}/cancel`: `{"request_id":"known-active-id"}`. Aborts that
  Scheduler ID, waits for coherent native retirement and canonical publication;
  a late cancel of the last settled ID is observation, not another execution.
* `DELETE /v1/sessions/{id}`: empty body, application/json, rejects active work.
  Stop and reconcile before close. Retires logical history and the Scheduler pin.
* Persist/restore/sessionized Responses/Messages: explicitly unsupported, before
  I/O/mutation. They are not enabled by the production LocalBoundary allowlist.

Only **one live logical session** is supported. This is not another executable
singleton: the existing Scheduler remains single-active and queues ordinary
requests. Closed requested-ID tombstones are bounded to 4096 lifetimes/process;
correlation IDs to 65536 reserved requests/session. These are fail-closed resource
bounds with no eviction/reuse, not context or generation limits. Restart loses all
in-memory logical sessions; no database, native restoration or replay is provided.

## Settlement and interruption

Reservation occurs before native enqueue, consuming the application UUID even if
its acknowledgement is lost. An identical POST is rejected, not retransmitted or
reprojected. Stale counts and reused correlations cannot start native execution.

The native consuming worker settles the recipe exactly once, captures/publishes
both target and DSpark, and retires executable aliases before publishing a logical
snapshot. Request count, frontier and canonical response advance **together**.
SSE iterator yields and socket send progress are not canonical acknowledgement.
The same Scheduler finalization callback publishes normal, explicit-cancel and
disconnect outcomes, including any canonical suffix not delivered on the socket.
The broker releases the active fence only after observing Scheduler completion.

An unfinished tool protocol or nonrepresentable recipe prefix retains the actual
canonical diagnostic outcome but makes the session unrecoverable; it grants no
tool execution or continuation. Native/publication failure fails closed for the engine and
session. Failed transport cleanup is not proof of non-admission: an ID
still registered with Scheduler stays fenced until retirement is proven. An
unstarted cancellation leaves count/frontier/history unchanged and retains the
consumed correlation plus explicit nonadmission error. No outcome is automatically
retried, discarded or replaced with a guessed partial response.

Tools execute only in the existing application server/frontend policy. The runtime
certifies canonical completed calls; it does not execute effects. Existing Chat
Server reservation/result observation remains authoritative for effect uncertainty.

## Qualified controls and frontend boundary

`/health` advertises `capabilities.stateful_chat = production-sessions-v1`, one
live session, persist_restore=false and `chat_defaults` from current production
admission: **temperature 0, top-p 1, reasoning none, Auto output**. Current MTP's
actual qualified total envelope and 393216 output ceiling are reused unchanged.
Unsupported sampling/thinking/images are rejected, never silently rewritten.

Browser/TUI clients must consume these authoritative capabilities/defaults for a
new MTP conversation. Historical Browser UI defaults 1.0/.95 are not an admitted
MTP sampler. The existing Chat Server protocol passes supported requests without
protocol changes; frontend default handling belongs to the subsequent Chat/TUI
milestone, not a hidden backend normalization.

## Acceptance

Run `tools/qualify_production_sessions.py` with an already READY **installed**
mtp-serving-v1 runtime. It exercises real streams, canonical GET, exact multi-turn,
a generated generic multiply call and real application result, ordinary cache
pressure, disconnect/reconcile/continue, explicit cancel/continue, stale count and
application correlation rejection, unsupported persistence, and close/nonreuse.
Every settled session turn checks canonical frontier, target/DSpark alignment,
replay=0, repack=0 and continuation cached_tokens exactly equal to its previous
canonical frontier. Ordinary pressure settlements check the existing observability.
Final installed-wheel evidence is recorded in
[production-sessions-acceptance.json](production-sessions-acceptance.json).

The final backend-only wheel passed **8 canonical session turns**, **4 ordinary
cache-pressure turns**, and **4 turns through the existing Chat Server production
proxy**, including a real public `fetch_url` effect, GET-only effect-ledger
observation, model continuation and explicit cancel. Every settlement is aligned
with replay/repack zero. The wheel excludes TUI WIP; startup/drivers used the
non-editable installed package from `/tmp` with normal admission. **239 MTP
regressions pass**, plus the separate model-free budget HTTP regression below.

An earlier pressure attempt failed closed when the upstream bounded prefix-tail
hint forgot a still-pinned pair. That failed receipt/log is preserved locally;
it is not current acceptance. The exact retained upstream hash/refcount restore
fix is covered by the final model pressure pass and a hint-churn regression. No
replay fallback, enlarged upstream hint index, or second physical cache was added.

Focused regression tests cover the same protocol decisions, opaque pin eviction
behavior, forbidden qualification dispatch, worker-publication-before-delivery,
missing-pair no-replay, fail-closed uncertainty and existing production boundaries.
A standard-OFF-only budget HTTP test has a different native recipe pin in the MTP
provisioning environment and is excluded from this scoped MTP test gate; its
identity rejection is not bypassed or repaired. The same single budget regression
test also passes separately in the already provisioned standard-OFF test environment
without model execution. No OFF qualification is claimed.
