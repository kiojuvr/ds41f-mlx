# M54 continuation — semantic-authorized cycles and exact-byte reservation

Historical `fbca4bc` status. The [EOF-sensitive operational continuation](milestone-54-eof-sensitive-operational-continuation.md)
implements the next native primitive and supersedes this NOT PASS decision.

**NO / M54 NOT PASS (operational tool envelope).** Base: M51–M53 PASS,
M54 assessment `2b68a69`. The two previously identified primitives are now
implemented, exercised, and connected to the standard application boundary.
This is not another assessment stopping at those same blockers. The next
minimum primitive is **non-mutating, exact-provenance EOF-sensitive tool
completion preview matching the existing standard M11 decision**. Tools fail
closed before conversion/load/reservation/mutation until that primitive exists.

## One standard runtime and explicit strategy

```text
same standard FastAPI server / session reservation / application boundary
  -> StatefulStream (SSE) or protected JSON worker
  -> same LiveRecipeTurn / canonical StreamProcessor
  -> SemanticCycleAdapter + M8 whole-prefix accounting
  -> canonical TargetGenerationSession
  -> first-party DSpark proposal child/producer
  -> M52 verification/sampling/publication -> M51 settlement
```

OFF remains the constructor/operator default. The development selection is an
explicit backend API, not a per-request flag, environment fallback, promoted
profile or second server:

```python
backend = DeepSeekRecipeRuntimeBackend(
    runtime_config=config, execution_strategy='first-party-mtp-development')
app = create_app(backend=backend, runtime_config=config)  # same standard routes
```

`execution_strategy='off'` explicitly selects the control path. The strategy is
read-only for the backend lifetime. Both strategies share the standard response
reservation implementation. Loading still binds the exact OFF target, then
explicitly admits the first-party M47 proposal child before any prefill. No
native scheduler or candidate generation owner is imported by this selection.
Health/provenance expose the strategy; target `preserve_mtp` remains false.
Existing CLI profiles/defaults and `mtp-singleton-v1` are unchanged. The old
candidate/internal fence remains on its own historical records, not as another
ledger on a standard session. No runtime promotion or supported public MTP
profile is asserted by this partial integration.

## Semantic contract and publication order

`runtime/semantic_cycle.py` implements worker-confined authorization:

1. Require the same generation identity, coherent M8/generation history, no
   outstanding recipe terminal, and a non-mutating exact native preview.
2. Preview the pending canonical anchor before proposal arithmetic. A completing
   recipe token or backend control terminal is a **protected ordinary canonical
   step**, with no proposal or speculative verification. Terminal publication
   then retires the producer. This retains M52/OFF consume-and-sample semantics:
   an unconsumed sampled lookahead is not a committed response token.
3. Only a proven nonterminal anchor, full derived ring and sufficient remaining
   output budget permit DSpark proposals. Preview anchor + draft prefix again;
   submit only inputs **strictly before** a completing terminal and within the
   remaining generation budget. Discard unauthorized draft tails. Proposal width
   and semantic authorization horizon are not the same concept.
4. Bind the permission to generation/frontier, canonical emitted ordinal, pending
   anchor and processor snapshot. Cancellation before mutation consumes nothing.
   M52 handles cancellation during verification and deferred sampling cancellation
   with its existing semantics, without an OFF retry.
5. M52 verifies/samples, M51 settles, M52 publishes the sole generation truth.
   `M8.adopt_cycle_reports` checks contiguous complete reports, final-only
   terminal flags, all target offsets, exact history and no duplicate adoption.
   It adopts the **entire** consumed prefix, not a report queue.
6. Feed every report through the same canonical processor on the same worker,
   without an await/yield; compare semantic terminal identity/source provenance
   with authorization. Reports contain only consumed tokens, frontier ranges,
   canonical latency/terminal flags; accepted/rejected speculative tails and
   unconsumed correction/bonus do not become recipe history.
7. Only after complete coherent recipe advancement may response bytes be
   reserved for delivery. Finish/idle transfer precedes terminal publication.
   No proposal/verify during report delivery, completion, cancellation settlement,
   reconciliation, idle transfer or retirement. Receipt/ring advancement is
   derived-only and uses M53's same-forward committed taps.

JSON development execution uses the same LiveRecipeTurn progression as SSE,
not the old single-token M11 decode loop with a second MTP parser. OFF's retained
single-token JSON behavior is unchanged. Short continuation rings use explicit
protected canonical warm-up steps; they are not reconstructed from history and
never trigger an exception-based strategy switch.

Authorization failure retires the derived producer. Any failure after target
mutation in M8 adoption, recipe observation or receipt advancement burns the
generation/cache lease and closes re-entry. The cursor publishes no completed
turn, and the reservation is uncertain. A partially emitted SSE prefix is not
permission to regenerate or continue OFF. Failure cleanup releases a generation
even when M8 is already marked closed; it does not attempt burned idle extraction.

## Standard exact-byte reservation, retry and effects

`StatefulSessionRecord.response_reservation` is the sole bounded request/
serialized-response slot. It is used by **OFF and development first-party MTP**;
there is no MTP-specific response ledger. `X-DS41F-Request-Sequence` is optional
for unfenced OFF legacy clients, mandatory for the development strategy. Once a
session uses fencing it cannot return to unfenced mode.

- Identity is session + monotonic sequence + **exact request body bytes**, not
  semantic JSON equality. A whitespace-only body change under the same sequence
  rejects. The slot holds SHA256 certificates, original request identity and
  actual serialized JSON/SSE body bytes (not a regenerated response object).
- The standard route observes completed retry **before** recipe conversion,
  prefix admission, load or generation. Active/uncertain, expired/out-of-order
  and contradictory retries fail closed. A new sequence must pass exact-prefix
  admission; it is not a license to repeat the old prompt.
- JSON serialization/freezing runs on the protected worker after canonical turn
  publication, before a cancelled future can lose its result. A task cancelled
  after reservation can still observe/replay that exact completed outcome.
- SSE serializes each frame once into the reservation **before** iterator yield.
  On disconnect, the protected worker settles the current operation and cancels
  coherently. Undelivered pending/terminal frames and `[DONE]` are serialized into
  that same slot, then frozen. Retry returns the full frozen stream: already
  delivered bytes are its identical prefix, not a second generation or an ACK.
- A coherent but non-representable cancellation may freeze/replay its response
  while canonical reconstruction denies further model re-entry. This differs
  from uncertain target/recipe publication, which cannot complete or replay.
- Bounds: 1 MiB request, 4 MiB response, 8,192 serialized chunks; a ceiling burns
  the reservation and releases transport/worker ownership. Closing/deleting the
  session retires its reserved byte payloads. No persistence/restore contract.

Busy GET does not read worker-owned M8 diagnostics between adoption and recipe
feed. Reservation certificates snapshot state/chunks before hashing (which can
release the GIL), so a prefix hash cannot be paired with a later completed size.
These final observation guards have forced-interleaving regression coverage.
Direct development JSON/SSE backend entry also rejects an absent/uncertain
reservation before model load or mutation. Worker errors cannot freeze a
successful SSE EOF; retired producers cannot masquerade as short-ring warm-up.
Budget and reasoning gates reject unqualified modes before execution. These
last guards were finalized after the socket probes and have regression coverage,
not a new official-checkpoint socket trial.

The existing `StatefulStream.publish` reconstruction certificate and Web's
request-count/tool-effect ledger remain authoritative. UUID correlation is not
used as a retry fence. No tool execution or extra effect authorization occurs on
response replay. The injected effect-then-delivery-loss regression executes one
real registry counter effect, then retries/observes the same Web ledger outcome:
**duplicate effect 0**. It uses runtime/transport doubles, not MTP tool execution.

## The newly demonstrated next primitive: EOF-sensitive tool horizon

`tests/test_m54_tool_eof_horizon.py` uses actual installed recipe bindings and
actual tokenizer, not a proposed grammar. For a valid `lookup_weather(city=Paris)`
DSML stream of 38 token inputs:

- Standard M11's canonical recent-arguments + official EOF/Stop probe first
  recognizes completed tool identity at **ordinal 31**.
- Native `preview_tokens` at that ordinal returns **no terminal**, and canonical
  `semantic_terminal` is also absent. Native preview recognizes
  `DSML_TOOL_CALL_BLOCK_END` only at **ordinal 37**.
- Thus approving the native-preview-safe prefix would still permit six inputs
  beyond the standard application's tool-completion boundary. This is a
  runtime/application semantic mismatch discovered while implementing the
  authorization primitive, not a missing M51/M52 acceptance mechanism.

The minimum next work is a native clone-based preview that evaluates the same
EOF-sensitive completion decision for every candidate prefix, binds exact
ordinal/tool progression/provenance, and leaves the canonical processor and
response generator untouched. It must not publish tool IDs/effects in preview.
Do not weaken/remove M11's current boundary, copy DSML grammar, reparse model
history, add a second recipe truth, or adopt candidate scheduler authority.
Once it exists, the already implemented adapter can admit tools using that
canonical permission, then qualify tool-result continuation and effect re-entry.
Current tool-enabled requests are rejected before reservation or target mutation;
ordinary requests have `parse_tool_calls=False` from canonical recipe conversion.

## Qualification and capability limits

[Evidence and commands](../artifacts/m54-continuation/README.md).
Affected core/runtime/application regressions: **247 passed + 8 subtests**,
46.69 s, exit 0. Historical naked-cycle/report-queue counterexamples remain;
authorized-cycle tests prove the terminal-crossing tail cannot be submitted.
Reduced real-MLX tests cover high acceptance across five boundary positions,
protected terminal proposal/verify prohibition, cancellation after authorization,
inexact preview, duplicate batch rejection and failures after settlement,
adoption, partial observation and reporting. M52's existing mid-cycle/RNG/burn
matrix is rerun, not redesigned.

Official-checkpoint, actual loopback uvicorn/HTTP-client standard-route probe:

| First-party development case | Actual result |
|---|---|
| JSON ordinary turn | 9 consumed inputs, 2 real M52 cycles, 6/7 draft accepts; EOS |
| Multi-turn SSE continuation | 9 consumed inputs, exact-prefix same-cache continuation; short-ring protected steps; EOS |
| Max-token SSE | 8 consumed inputs, 2 cycles, 6/6 accepts; length terminal |
| Socket disconnect / partial SSE | 12 committed inputs, 2 cycles, 10/10 accepts; coherent cancellation |
| Exact retry | JSON and all SSE cases return identical frozen bytes; no extra target calls or request-count increments |
| Disconnect certificate | representable=true, exact_prefix=true; re-entry witness exists, but post-disconnect new generation was not exercised |
| Tool-enabled request | HTTP 400 before reservation/frontier/target-call mutation |
| Idle/terminal | all 40 offsets equal committed history; F=251/275/249/253 |
| Target forwards | 43 = four bootstraps + 10/9/8/12 planned decode inputs; no extra target execution |
| Replay / full-cache repack | 0 / 0 |
| Donor / diagnostic execution | execution guard observed 0 forbidden entries |
| Retirement | child receipts=0, producers=0 at idle; parent/child inactive after close |

OFF control also passes the same real socket probe (322.24 s, exit 0):
9/9/8 inputs for dialogue/continuation/length, three inputs at socket disconnect,
33 target forwards, replay/repack zero and parent retired. The first three cases
have exact generated-token, message and frontier parity with first-party MTP.
Disconnect stopping points differ because each drains one protected operation,
not because of a matched performance workload. Both yield exact frozen retries
and representable reconstruction witnesses. No all-slot byte-hash or stochastic
application parity measurement was performed.

This is **partial standard application qualification**, not a supported Chat/tool
operational envelope. Greedy only, text Chat strings only, explicitly
`reasoning_effort='none'`, one live session, explicit output <=64,
prompt+output reservation <=512. Those conservative gates
are not a general 512-position MTP qualification and do not copy candidate 8K.
The actually exercised frontiers are above; new contexts/capabilities are not
inherited from OFF. Vision, structured output, tools, persistence/restore,
Responses/Messages, stateless execution, automatic capacity, concurrency/batching
and long-context expansion fail closed for this strategy. Stateful arbitrary
stop strings remain rejected by the existing standard policy; native stop
preview is tested separately, not claimed as a supported stop-string endpoint.
Stochastic application RNG is unqualified; reduced M52/M53 RNG regressions pass.

## Performance and remaining promotion boundary

Full tool/application correctness is NOT PASS, so **no matched operational
performance benchmark** or native candidate run was conducted. OFF control uses
the same correctness probe, not a candidate-performance comparison. OFF vs
first-party vs candidate decode tok/s and separate verify/settlement/application
phase comparison remain **N/A**. No M53 speed or candidate history is relabeled
as M54 performance evidence.

The probe retains coarse diagnostic proposal, authorization, execution and report
wall times. `execution_s` still combines target verification/sampling/settlement;
app/prefill/transport and hidden receipt/ring transport are not fully isolated.
Native preview adds CPU work, not an explicit `mx.synchronize` in the adapter;
scalar `.item` reads and M53 receipt/ring evaluations remain. Additional physical
synchronization cost has not been isolated and no zero-sync claim is made.

Full R1 was not run: the required tool envelope is not complete. No default MTP,
candidate removal, normal-runtime or ds41f-runtime promotion, release/packaging,
Vision/restore expansion, kernel/MMA/prefill optimization or zero-copy research.
The next boundary is the **canonical EOF-sensitive tool permission primitive**,
then official-checkpoint tool/effect/cancel/re-entry qualification. Only after
that correctness PASS should matched OFF/first-party/candidate phase measurement
and any promotion-blocker reassessment begin.
