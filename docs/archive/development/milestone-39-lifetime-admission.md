# M39 — operational lifetime, identity and admission

## Design before implementation

Authority: clean M38 `998500b`. M33 horizon, M34 native cache transfer,
M35 response leases, M36 representability, M36R certificates/fences and
M37/M38 client authority and conservative ambiguity remain mandatory.

Selected internal contract: **server-issued process-namespace serial identities**.
Arbitrary client-selected IDs are no longer supported by the explicitly injected
internal MTP backend (the ordinary OFF backend/API is unchanged). An ID is
`mtp_<128-bit namespace hex>_<128-bit serial hex>`, canonical lowercase fixed width.
The namespace is generated once per backend lifetime. Serial issuance is contiguous,
starts at 1, increases only on successful create, and never wraps. At exhaustion
create fails closed. This is not an authentication capability or public security
scheme. The live dictionary is the sole execution authority, not ID syntax.

Within this namespace: the current live dictionary entry is live; any issued serial
absent from that dictionary is retired; serial zero or above the issuance high-water
mark is never valid. A different namespace is stale/foreign (the server cannot prove
whether it was ever issued elsewhere). Malformed IDs are never valid. GET/POST/DELETE
on nonlive IDs deterministically reject without mutation, whether recent diagnostics
exist or not. Create accepts no requested ID, including retired IDs. Thus delayed
(session, sequence) work cannot attach to a later lifetime. No timeout or diagnostic
eviction changes validity. Request sequence remains local to the unique lifetime.

Correctness state: one namespace, one fixed-width issuance high-water mark, at most
one live record and one active response lease. No per-retired-ID authority remains.
A finite machine cannot issue infinitely many unique identities: the precise claim
is history-independent storage for sequential bounded lifetimes **until explicit
128-bit exhaustion**, then permanent fail-closed admission, not counter wrap.
Request consumed sequence also has a fixed 64-bit ceiling; settled retry remains
observable at the ceiling but new work fails before reservation/mutation.

Recent diagnostics: 16 small retirement summaries (ID, closed state, request count,
frontier and final outcome classification), independent of authority. No closed
canonical IDs, certificate/witness, response, reconstruction body/tokenizer,
processor/guard/cache/rings/owner or last-turn trace survives in a session record.
Detailed server trace deques have fixed 32-slot budgets, independent of environment;
DELETE clears the retired lifetime's detailed traces/progress. No closed outcome
payload slots. One latest live fence/outcome, one latest reconstruction body, <=1 MiB
request bytes, <=8192 total token budget and <=768 responses; existing exact-prefix
and model limits remain. Diagnostics are intentionally narrower, not an outcome
archive. Input capacity checks precede lock acquisition, sequence consumption,
native/model/cache allocation. Singleton create denial precedes issuance.

Restart is outside qualification. A newly constructed backend gets a new namespace;
old namespace requests are rejected. Namespace randomness is not persistence or a
proof across crashes/restarts, and lifecycle transport ambiguity still stops the
client. No discovery or automatic replacement is introduced.

Qualification plan: actual cheap create/DELETE through the backend (synthetic native
retirement only where no model exists), >=10000 sequential lifetimes vs 16 diagnostic
slots; direct container/payload accounting and early/mid/late lifecycle/admission/
GET/stale lookup microbenchmarks; weak references proving discarded record payloads
are collectible; capacity denial snapshots; oldest stale session/request rejection
after eviction; real checkpoint fresh/retained/fresh integration; rerun affected
M33–M38 runtime, client, native and OFF gates. Decision pending evidence.

## Decision and exact contract

**QUALIFIED_FINITE_PROCESS_LIFETIME_ADMISSION**. Canonical machine authority:
`artifacts/m39/qualification.json`; structural stress: `lifetimes.json`; checkpoint
integration: `integration.json`; real lifecycle faults: `lifecycle-transport.json`.
Final commit resolves with `git log -1 --format=%H -- artifacts/m39/qualification.json`.
M38's closed-record retention blocker is resolved on the narrowed internal contract.
M38 evidence itself is unchanged; this does not retroactively relabel its result.

Within one qualified process with one injected backend instance, lifecycle authority
and diagnostics have explicit history-independent bounds. Retirement does not retain
a tombstone archive and cannot resurrect after diagnostic eviction. Sequential create
continues until the explicit `(2**128)-1` issuance budget; then it permanently rejects.
There is **no claim of mathematically infinite successful issuance on a finite machine**.
Any promise of inexhaustible distinct identities with permanent retirement and strict
finite bits is incompatible. The operational claim is finite storage independent of
history, with explicit no-wrap fail-closed exhaustion, not eviction or a hidden timeout.

### Identity proof and classifications

Inductively, successful create is the only issuer: it inserts exactly serial `issued+1`
and advances `issued`, without yielding. Capacity/requested-ID/exhaustion denial does
neither. DELETE removes the live entry, never decreases `issued`, never inserts a
replacement, and never changes the namespace. Consequently that serial cannot be
inserted by a later create. GET/POST/DELETE require live dictionary membership; merely
having a valid syntax or previously issued serial cannot execute. Diagnostics are not
consulted by any admission operation. Lookup parses a fixed 69-character identity and
uses a <=1-entry dictionary: neither lookup nor issuance scans historical sessions.

The most recently issued serial absent from the live map is the current retired
lifetime. Lower issued serials are older retired/stale lifetimes; the rejection API
intentionally reports both as `retired`, since both have the same permanent prohibition.
A foreign namespace is `stale_namespace` (possibly never issued elsewhere; that fact
is unknowable and immaterial). Zero/future serials and malformed IDs are `never_valid`.
Stale requests carrying any old `(session, sequence)` reject before protocol conversion
on the HTTP path, even when the recent summary is absent. No old sequence aliases a
new lifetime's sequence 1. The existing latest-slot fence and its digest, immutable
settled observation, expiration, mode-mixing prohibition and exact-prefix check are
unchanged. The final settled slot remains observable at the 64-bit ceiling; new work
cannot overflow the bounded counter.

These IDs are not authentication secrets. Process-local loopback internal qualification
is not public access-control qualification. Backend reconstruction/new process is a
new authority namespace and outside the proof. Random namespace generation is not
persistent/distributed collision-proof identity. No restart or backend-recreation
qualification is inferred.

### Explicit ownership and retention budgets

| Retained state | Bound / disposition |
| --- | --- |
| Live session/native cache authority | 1 record, 1 singleton response lease; failed retirement keeps the poisoned singleton and blocks fresh creation |
| Historical admission identity | Namespace + high-water serial, each <=128 bits; zero retired authority records |
| Request/lifetime metadata | <=64-bit consumed sequence and request count, one latest outcome/fence; request bytes <=1 MiB |
| Live model contents | Existing <=8192 prompt-plus-response token budget, <=768 responses; exact-prefix mandatory |
| Recent retired diagnostics | 16 summaries, <=199 JSON bytes each; <=3216 bytes serialized list, independent of authority |
| Closed outcomes / canonical IDs / witnesses | Zero retained payload slots after successful DELETE |
| Detailed traces | Each deque fixed at 32, not environment-expandable internally; cleared at DELETE, as are progress/last-trace references |

The summary's fixed ID, bounded count/frontier and fixed classification vocabulary
prove its byte bound; no truncation heuristic or arbitrary text enters it. Live trace
payload is bounded by admitted body/token/response contents and the pinned tokenizer/
protocol (not a claim of zero live diagnostics). At most 32 bounded turn snapshots plus
one current guard/body/outcome may exist. Static model weights are excluded. Transport
buffers/preparation are not a qualified unbounded ingress workload; flood resistance
and public admission remain release-readiness concerns.

Successful DELETE holds the singleton lease through shielded native retirement,
GenerationBatch/processor close, target/ring release, prompt-priming drop, stream sync
and allocator-cache clear. It then removes the record, clears canonical/guard/parser/
certificate/fence/reconstruction/last-turn references and detailed deques. Returned
summary is a copy, not execution authority. Repeated retired DELETE and GET return 404;
old POST returns 404; requested-ID create returns 400. None mutates issuance, a later
live session or diagnostic retention. Failure is not advertised as successful retirement.

`InternalLocalClient` requires no new identity authority or seen-ID set: existing
`create()` accepts the server result and starts sequence 1. Callers must stop supplying
names on this internal path. Its optional named-create transport remains useful to the
ordinary OFF backend, but the internal server rejects it; conservative client ambiguity
handling is unchanged. No automatic replacement or new lifecycle reconciliation API.

## Evidence and capacity qualification

Actual backend create/DELETE methods process **20,000 lifetimes**, **1250 times** the
16-summary budget. Native retirement alone is replaced with an assertion that no native
owner/cache exists. Each lifetime is populated with 8192 synthetic canonical IDs and
large fixed reconstruction/certificate/response payloads, reserves/settles one fence,
and is retired. Weak references prove the record becomes collectible after executor
result-delivery callbacks release on the next event-loop turn. Payload fields are already
cleared at successful DELETE. Every iteration checks capacity before issuance and
oldest-ID rejection; samples at 1/16/32/200/10,000/20,000 show:

- Live/native/retired authority records **0 after DELETE**, live peak **1**.
- Diagnostic records **1 → 16 → 16**, payload **180 → 2880 → 2880 bytes**.
- Namespace/counter storage **101 bytes** throughout this run, hard maximum **117**
  in this Python build at 128-bit exhaustion; namespace size 32 characters, no ID set.
- Live dictionary backing **184 bytes**, diagnostic deque backing **760 bytes** at
  sampled boundaries; closed outcomes and detailed trace slots **zero**.
- Process max RSS **46.55 → 46.86 MB**; synthetic MLX active/cache **0/0**. RSS is
  secondary evidence, not the invariant.

After eviction the oldest identity is still derivably retired; current sessions still
admit and serve. Zero/future/foreign/malformed and requested old identities also reject
non-mutatingly. Artificial boundary fixtures reach the final 128-bit issuance and 64-bit
request sequence, prove no wrap, and preserve settled observation at the ceiling. The
finite test is not the lifetime proof: contiguous monotonic issuance + no wrap + live-only
membership is the proof. Container budgets are independent of historical count.

Real checkpoint integration uses the actual reusable client and loopback uvicorn/h11:
**four model-backed fresh sessions, three retained continuations, seven requests / 25
returned responses**, plus **48 empty actual HTTP create/DELETE lifetimes**. This is a
small integration gate, intentionally not another recovery soak. Frozen request identity,
sequence advancement, repeated identical settled observations, expired sequence handling,
all 40 target/3 DSpark frontiers, DELETE payload release and fresh prefill are checked.
The first detailed identity is evicted, its old GET/DELETE/POST still return 404, and a
new model-backed session succeeds. Replay/repack counters remain **0/0**; fresh target
allocations exactly **4**, distinguished from retained continuation.

HTTP live-capacity (409), oversized body (400) and out-of-budget sequence (400) denial
leave the complete session snapshot, issuance, trace count and model/native counters
unchanged. Oversized body denial precedes official request conversion. Unit gates add
wrong body type, bool/oversized sequence, singleton/exhaustion denial, stale operations,
full-reference release and failed-retirement containment.

Actual DELETE/create socket-body loss after success headers leaves the client sticky
`stopped`. DELETE truly removed the record; unknown create truly issued a new identity
that the client cannot know. Further actions do not send or mutate. Only a labelled
external controller cleans the unknown created session. No guessed ID/discovery/retry.

Real post-DELETE MLX active bytes **309.155 / 309.160 / 309.160 / 309.172 GB**, cache
**zero** at every boundary. RSS **11.799 / 7.265 / 7.268 / 7.286 GB**, max RSS **20.009 GB**.
These model-dominated measurements corroborate native release; the structural CPU
retention plateau is the central result. Final diagnostics are **16 / 2849 bytes**,
closed payloads/native owners zero. Evidence copies live outside backend ownership.

## History-independent performance

200 observations per cohort: first 200, 10,001–10,200, final 200 lifetimes. Median µs:

| Operation | Early | Mid | Late |
| --- | ---: | ---: | ---: |
| Create | 1.125 | 1.083 | 1.083 |
| DELETE (includes worker dispatch/payload destruction) | 68.042 | 67.730 | 67.375 |
| Oldest stale-ID rejection | 2.584 | 2.583 | 2.583 |
| Current-session GET object observation | 2.833 | 2.834 | 2.833 |
| Settled fenced outcome observation | 3.458 | 3.333 | 3.334 |
| Live-capacity denial | 0.458 | 0.438 | 0.458 |

Raw p95/counts are in `lifetimes.json`. These are process-local microbenchmarks, not
HTTP latency SLAs or kernel optimization. GET returns the current bounded payload;
outcome observation uses a fixed synthetic response. No cost depends on a scan or
transfer of historical diagnostics. Different current-session contents can still
change observation/serialization cost, as M38 measured.

## Contracts, attempts and regressions

The incompatible promise was arbitrary client-selected IDs with permanent retirement
and indefinite successful arbitrary new admissions under finite memory. M39 narrows
only the internal ID contract, rather than forgetting tombstones. No upstream/native
math or semantic ownership defect was found. Diagnostic lifetime ownership was the
owning layer of M38's blocker. Fixed trace budgets and removal of all large closed
payloads are architectural policy, not a model-memory optimization.

Preserved excluded attempts: historical fixture/client labels initially rejected;
lease fixtures now resolve labels only inside a synthetic adapter while the actual
backend still issues IDs. A weakref assertion preceded executor callback release;
the next loop turn corrected its observation window. A ceiling test attempted a
forbidden no-header fenced retry; mode mixing correctly rejected it. An initial
checkpoint harness changed tools after a retained text-only prefix and correctly
failed exact-prefix admission; final integration keeps its envelope unchanged.
The passing preliminary integration is superseded by final source-hashed admission
integration. None is counted as successful qualification evidence.

Affected current-source gates pass: **7 M39 admission tests; 56 client/browser tests;
110 runtime/horizon/OFF tests + 32 subtests; 52 historical semantic/evidence tests +
137 subtests; five preserved release-OFF tests; seven Rust tests; 28 native official
recipe fixtures; all 16 checkpoint native interruption/fault cases; identity gate**.
Exact commands and logs are retained. Five historical source/hash assertions are
explicitly deselected, not rewritten. Current source hashes and fresh integration/
native gates provide M39 authority. The historical M35/M38 named-ID workload harnesses
are not silently run against an obsolete contract or used as new qualification;
M38's finite client qualification is already established and was not repeated as the
primary task. Dedicated M39 evidence-consistency tests validate current hashes separately.

## Unsupported scope and handoff

No public/default/release MTP, persistence/process restart/backend recreation,
distributed identity/recovery, crash-safe tool exactly-once, concurrent/shared MTP,
batching, token-exact immediate abort, 200K HTTP MTP, unbounded backpressure, portable
native wheels or universal partial DSML recovery is qualified. Production MTP remains
OFF; no selector, public API promotion, kernel change or replay/repack fallback.

**Next task: one dedicated release/admission/provenance readiness evaluation**, not
another recovery micro-milestone. It should decide whether an intentionally supported
public opt-in mode is justified: launcher/selector and supported envelope, ingress and
resource admission, local security/identity access policy, lifecycle/effect uncertainty
application workflow, provenance/installation/native packaging and remaining repository
wide audit housekeeping. Longer-context qualification is separate. M39 establishes the
internal lifetime policy only; it does not decide release readiness.
