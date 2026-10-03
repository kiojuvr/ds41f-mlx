# M38 — living internal client operational soak and fault matrix

Base: clean M37 `daf429e`. Decision: **BLOCKED_OPERATIONAL_LIFETIME_RETENTION**.
Canonical machine evidence: `artifacts/m38/qualification.json`; checkpoint workload:
`soak.json`; exact-boundary regressions: `client-tests.xml`, `ledger-exhaustion.json`;
real socket lifecycle faults: `lifecycle-transport.json`; retirement finding:
`retention-probe.json`. Final SHA resolves with
`git log -1 --format=%H -- artifacts/m38/qualification.json`.

The **bounded workload passes** after narrow client hardening. The stronger
structurally bounded client/server lifetime claim does not: successful DELETE
retires native authority but retains every closed server record and its diagnostic
payload without a lifetime cap. This is an explained retention-policy blocker,
not evidence of leaking target caches or incorrect recovery. No release promotion.
M33–M37 historical evidence is unchanged.

## Architecture and changes

The actual reusable `InternalLocalClient` drives loopback uvicorn/h11 through the
existing `RuntimeClient`; no qualification controller bypasses its transitions.
Server canonical/native state, official certificate, client application transcript,
and living effect ledger remain separate authorities. No alternate transcript,
parser, token history, session observation protocol or retry heuristic was added.
M33 horizon, M34 transfer, M35 lease/checkpoints, M36 representability blocker,
M36R certificates/fences and actual exact-prefix admission are unchanged.

Preserved regressions against the original M37 client expose three defects:

1. Re-observing a reserved failed effect changed `tool_ambiguous` into `tool_pending`.
   Reservation still prevented a duplicate callback, but direct retirement and
   fresh creation could bypass the unresolved effect and carry incomplete tool
   history. Reserved effects now keep their visible classification; execution,
   new work, tool-result submission, retirement and creation reject before mutation.
   No application-resolution API is invented. Application intervention is required;
   crash-safe or distributed effect resolution remains unsupported.
2. DELETE/create failures set only generic `stopped`. A later DELETE attempt could
   still reach the runtime, including after unknown creation. A sticky
   `lifecycle_uncertain` record now identifies the uncertain operation and any
   available requested/current ID. No further DELETE, create or new work is allowed.
   Reconciliation also rejects the stopped state. There is no automatic replacement
   and no client API that clears this uncertainty. External manual reconciliation
   is outside the qualified workflow; test-controller cleanup is labelled as such.
3. A non-dictionary user addition raised `AttributeError` rather than the explicit
   pre-mutation client rejection. Validation now rejects that shape consistently.

The pre-fix fault run has **10 failures / 8 passes**; the current client regressions
pass (**56 client/browser tests**). The server is not changed to manufacture a successful soak or to evict old
identities. Lifecycle failures are conservatively classified even when an HTTP
error might in principle prove rejection: no unsupported outcome inference.

## Coherent checkpoint workload

One living client, three sequential agent sessions, **76 requests / 14 certified
effects / 2,390 returned model responses**. Each phase has four planning/weather/
stored-result/note/interrupted-text/review rounds. Ordinary text budgets are 128,
forced weather budgets 96; temperature zero, unchanged schema and thinking envelope.
There are 12 non-streaming requests; the other 64 use the owned SSE abstraction.
All 14 effects use the deterministic application weather stub, not server tools.

The 25/26/25-request phases include 12 deliberately required calls and two valid
unsolicited calls on fresh reconstruction, each executed once and continued using
its stored result. The two first phases end with real argument-delta interruption:
negative certificate, no execution, omitted final native assistant, DELETE and
fresh creation preserving legitimate ordinary application messages. The third
retires after ordinary completion. Fresh sequence restarts at 1 and no native
cache/ring authority crosses retirement. The effect ledger stays in the same client
and keys include session ID, so sequence/call-index restarts cannot alias effects.

Transport cases recur: six content-event disconnects, three zero-body-event closes,
three iterator closes, three controlled completed-tool terminal delivery losses,
three slow consumers and two partial DSML cuts. Delay windows are harness controls,
not naturally sampled loss probabilities. The standalone `client-pressure.json`
adds real saturated h11 delivery through **this client**, using bounded synthetic
256-KiB SSE comments and an 8192-byte send buffer. No body is consumed during the
three-second pressure observation; dropping the only stream handle closes its real
socket, settles, replaces history from the certificate and admits ordinary retained
continuation. A real send blocks for **3.078 s**, canonical production plateaus at
two responses, and disconnect-to-certified settlement takes **2.111 s** (including
protected in-flight model work, not a pure cleanup timer). This is finite pressure
only, not unbounded backpressure qualification.

Every request records frozen bytes/hash, identity, display observations, certified
outcome, repaired transcript, native trace and retained-state measurements. Assertions
run after **every settlement and each retirement phase**, not just at the end:

- One current identity; submission does not advance sequence; positive settlement
  advances once; negative settlement does not authorize the next request.
- Two repeated settled observations per request are immutable, non-generating and
  transcript-neutral; tools have additional observations after execution.
- Display/ordinary non-certificate responses never mutate authoritative transcript.
  Positive outcomes replace with request messages plus exactly one certified
  assistant; negatives preserve request messages only.
- Expired first identities in all three phases return expired without generation;
  direct old POST rejects. Retired client/server identities and ID reuse reject.
- Native owners/caches/rings are absent after retirement. All applicable target
  (40) and DSpark (3) frontiers equal canonical H; queues/predictions are retired.
- All model replay/full repack instrumentation is zero; quiescence adds no proposal,
  verify, replay or repack. **Three fresh prefills and 73 retained suffix requests**
  are separate. Fresh prefill is not retained-session replay.

## Fault matrix and bounds

Synthetic exact-boundary faults supplement, not replace, the real checkpoint run.
JUnit retains individual cases and timings. It covers absent lookup/frozen retry,
active observation, an absent-GET/active-POST race, same identity observation,
body/sequence/next-sequence disagreement, displaced/expired identity, negative versus
internal poison, malformed/changed certificate/outcome/witness/tool permission,
ledger mismatch, malformed SSE, dropped owned handle and ordinary local errors.
Actual server fencing, protected native poison, response leases, socket pressure
and interruption gates are rerun from M33–M37; those older controller probes are
not relabelled as reusable-client evidence.

Both DELETE and create are tested before/after possible mutation synthetically.
The actual HTTP lifecycle test lets the server mutate and send success headers,
then injects an ASGI failure before the declared body: the client receives
`IncompleteRead`. DELETE really retired the ID; create really established the ID.
The client cannot know these controller observations and stays stopped. Repeated
retire/create actions do not send another operation. Main-soak header-send faults
add HTTP failure after actual mutation. No idempotent/session-discovery mechanism
was required to obtain the safest bounded behavior: stop rather than guess.

The capacity test drives the **actual default 128-entry client ledger** through
four synthetic certificate/runtime sessions, using the same call ID/index and
restarting sequences. All 128 results stay intact; call 129 exhausts capacity
before its callback, with no eviction or reuse. This is actual client ledger
behavior, not 128 checkpoint-sampled calls. The stopped state is conservative.
A 64-KiB result is accepted and reused; oversized/invalid results and a callback
that may have effected externally remain reserved, never retried or bypassed.
The request's 1-MiB pre-send bound and ledger full-call binding remain enforced.
Capacity was **not increased**. Long-lived applications must plan explicit bounded
client/effect ownership lifetimes; no indefinite-service ledger is promised.

## Resources, retention and performance

End-of-phase client payload estimates: **9,095 / 18,829 / 28,496 bytes**; measured
request/outcome/transcript/ledger peak **74,481 bytes**. This intended application
history growth is subject to the existing request/context limits, not an unbounded
transcript promise. Ledger entries **4 / 9 / 14**, serialized ledger payload
**1,772 / 3,985 / 6,198 bytes**. Timing deque reaches and stays at **128**; early
samples deliberately disappear. Payload estimates exclude Python allocator sizes
and the timing deque itself; current request/outcome sizes are recorded separately.
Evidence files retain the planned finite run outside the live client, not inside
runtime authority.

Server trace deque lengths **25 / 32 / 32**, configured limit 32. In contrast,
closed records retain **1 / 2 / 3** identities and **1,555 / 4,421 / 8,565** aggregate
canonical IDs. Serialized exposed session diagnostics grow **21,904 / 58,847 /
130,376 bytes**; these are lower-bound payload estimates, excluding retained
reconstruction bodies/native wrappers. Five closed records remain after lifecycle
fault cleanup. The synthetic actual-create probe reaches 256 closed records with
linear growth: `max_live_sessions` ignores closed records, and `sessions` has no
cap. `last_turn`, canonical IDs, certificates, reconstruction envelope and guard
references survive DELETE; deque eviction does not remove per-session retention.
Simply evicting identities would weaken permanent old-ID retirement. **This is the
open architectural blocker.** It cannot be hidden behind resident model memory or
called a universal leak-free result.

Post-retirement MLX active bytes **309.205 / 309.179 / 309.173 GB**, allocator cache
zero. Current process RSS **11.731 / 8.764 / 9.050 GB**, max RSS **20.006 GB**. Active
bytes are model-dominated and not interchangeable with RSS; neither shows monotonic
unexplained growth in this finite run. Diagnostic CPU retention is nevertheless
structurally monotonic and explained by ownership, not allocator noise.

Early/mid/late ordinary warm request wall medians **1.188 / 1.171 / 1.186 s**;
prefill/handoff medians **259 / 259 / 260 ms**; weighted decode **28.68 / 34.41 /
35.39 responses/s**. These cohorts exclude first requests and injected-delay/drop
cases. Acceptance/output variation is not interpreted as client acceleration or
degradation. Reconcile wall medians **2.93 / 4.29 / 5.89 ms** accompany growing
ordinary diagnostic response sizes; lookup currently transfers server diagnostics
before the client discards them. No model replay follows. Fresh prefill/handoff
**4.36 / 10.33 / 16.89 s** intentionally processes increasingly long application
history and shape work; initial model load is separate. Final retained timing
samples show effect/ledger work **133–143 µs**, repeated ledger lookups **7–11 µs**,
latest successful DELETE **18.7 ms**, and empty-session create **1.01 ms**. These
are local observations, not latency SLAs; bounded timing eviction prevents a claim
of complete early/mid/late lifecycle microbenchmarks.

## Gates, attempts and readiness gaps

Current-source M37 workflow, M35 HTTP (18 turns), M36 byte/tool/pressure probe,
M36R fenced HTTP (four sessions), 28 native recipe fixtures, 16 checkpoint native
interruptions/faults, identities, client/browser, **110 runtime/horizon/OFF tests +
32 subtests**, **49 historical semantic/evidence tests + 137 subtests**, **five
preserved release-OFF tests**, and **seven Rust tests** accompany the canonical
artifact. Historical HEAD hash
checks are explicitly deselected; current source hashes and fresh runs replace
those assertions, not historical artifacts. Exact commands/results are retained.
An initial script import error, fresh-tool harness expectation, interrupted/overlapping
in-development gate scheduling, and historical hash selection attempts are excluded
and preserved; none substitutes for the serialized final gate runs.

Release-readiness gap analysis: repeated certified client semantics are no longer
the main gap. Server lifetime admission/retirement budgets remain unspecified;
manual lifecycle/effect reconciliation has no released application workflow;
public selectors/admission and supported envelope policy remain unqualified;
portable installation/native provenance still needs an explicit release gate.
Longer-context HTTP qualification is separate. Persistence, distributed effects
and universal DSML recovery are not prerequisites silently brought into scope.

Recommend **operational lifetime/admission hardening before release-readiness**:
select and enforce a finite server lifetime/session budget, preserve permanent
retirement fencing without heuristic eviction, reduce unnecessary closed diagnostic
payloads, and qualify capacity denial before mutation. Then perform one dedicated
release/admission/provenance evaluation rather than more micro-milestones on already
exercised recovery semantics. M38 is BLOCKED until that ownership/retention policy
is established; successful finite workflows do not erase the finding.

Still unsupported: public/default/release MTP, persistence/process restart,
crash-safe tool exactly-once, distributed idempotency/recovery, concurrent/shared
MTP, batching, token-exact immediate abort, 200K HTTP MTP, unbounded backpressure,
portable native wheels and universal partial DSML recovery. Production MTP stays OFF.
