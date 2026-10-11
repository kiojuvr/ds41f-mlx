# M37 — bounded internal local-client integration

Source base: clean master after M36R, `40e2303`. Canonical evidence:
`artifacts/m37/qualification.json`; exact requests, observations, certified
messages, effects, transcripts and native counters: `workflow.json`. Final commit
resolves with `git log -1 --format=%H -- artifacts/m37/qualification.json`.

Decision: **QUALIFIED_BOUNDED_INTERNAL_LOCAL_CLIENT** for the eight-request,
two-session living-client workflow below; not release promotion.

## Boundary and design

`ds41f_mlx.internal_local_client.InternalLocalClient` is a reusable, explicitly
experimental capability above the existing **local/browser Python RuntimeClient**
HTTP boundary. It is not a qualification-only controller or another recipe
implementation. The existing browser `StatefulToolChatClient`, browser persistence,
public routes, Rust `ds41f_api` raw methods and release selectors are unchanged.
This choice reuses the ordinary message/tool HTTP architecture and strict Python
JSON handling without adding a dependency/parser to the deliberately raw Rust
transport. The current-source Rust tests and real M35 Rust socket-drop gate remain
part of the qualification. Neither browser nor Rust callers implicitly opt in.
Only an explicitly injected internal singleton backend accepts this workflow.

One thread, one living client, one live session, one frozen request and one copied
latest outcome. No workflow framework, autonomous substitute requests or history
normalization. The API makes user additions distinct from ledger-owned tool
results and cannot accept a replacement conversation in request options.

| Client state | Action / ownership |
| --- | --- |
| retired | Explicit create; preserve only already available ordinary application messages |
| ready | Submit user additions with next expected sequence; freeze exact bytes before HTTP |
| in_flight / streaming | No second request; stream handle owns socket and closes on close/context exit/drop |
| ambiguous | Discard local fragments; explicit centralized reconcile, active means wait |
| ready after recoverable settlement | Replace assistant from certificate, then allow ordinary next turn |
| tool_pending / tool_completed | Reserve before effect, retain ordinary result, submit it with next sequence |
| tool_ambiguous | Effect result unknown; never automatically execute it again |
| unrecoverable / poisoned | Distinct visible classifications; explicit DELETE then fresh creation |
| expired / stopped | Surface condition, no speculative replacement; explicit reconciliation/restart decision |

All text, non-streaming and tool responses settle through `reconcile`/`_accept`.
Even a successful ordinary JSON response is not recovery permission. SSE events
are display-only and never accumulate into authoritative assistant history.

## Fences and certificates

Sequence starts at 1 only after fresh-server confirmation. Submission does not
advance it. Matching active observation does not POST or advance. Not-admitted
observation is not an ACK of absent delayed work: only the frozen same-body,
same-sequence POST may be retried. A settled slot is re-observed by exact-byte POST
and its JSON wrapper is the original outcome, never a generation. Only a certified
recoverable settlement enables the next sequence; negative/poison stops ordinary
operation. Changed body, sequence disagreement, malformed certificates/outcomes,
changed settled result, ledger mismatch and failed DELETE/create stop the client.
No failure heuristic silently generates a substitute request.

The client validates version/positive predicates, completed-tool permission,
response identity and ordinary witness/message binding, trusting the **server's**
official-recipe certificate. It never re-encodes tokens, implements DSML, guesses
parser state, or treats syntactic JSON as proof. The actual next ordinary request
still passes the unchanged server exact-prefix check. Raw canonical IDs, native
caches, model objects, DSpark and recipe internals are not client recovery APIs.
GET's internal qualification diagnostics are discarded; only the fenced outcome
wrapper is retained/exposed. Native diagnostics remain qualification evidence.

`observe_identity` distinguishes current re-observation from a provably expired
older identity. Expired returns an explicit `outcome_state=expired` and action:
use the already current certified conversation or explicitly restart. It never
POSTs old work. Wrong/retired identities cannot resume. A current request displaced
by a newer server slot stops advancement as expired; no shared-client support is
claimed. The real workload also directly verifies old fenced POST rejection.

## Four separate authorities

1. **Server canonical native history:** sole model/cache history, including canonical
   output never delivered and protocol source that may be unrepresentable.
2. **Certified ordinary protocol history:** the official interpreted assistant
   result with a positive exact-prefix reconstruction certificate. Not every
   native boundary has such a representation.
3. **Local transcript:** application-visible messages; repaired by replacement
   from (2), not concatenated received bytes and never uploaded as native truth.
4. **Execution ledger:** living-client effect ownership keyed by
   `(session_id, sequence, call_index, call_id)`, also binding the entire call.
   Reservation precedes execution. Completed ordinary results are reused; a
   reserved/failed result never permits silent effect repetition.

Ledger capacity is 128 effects with **no eviction**, result content at most
64 KiB per effect. Capacity/mismatch failures do not execute. Current request
body is bounded to 1 MiB, outcomes remain one latest copy, and timing history is
bounded to 128 samples. These are payload bounds, not an indefinite-service RSS
guarantee. Server prompt/response/context and singleton limits remain unchanged.
Tool callbacks are synchronous and may have application-specific latency. No
process/crash-safe or distributed exactly-once claim follows from the ledger.

## Coherent real agent workload

One retained session, then one explicit fresh session:

- normal streamed text;
- completed Paris call, four observations, one effect, stored-result continuation;
- text stream interruption through the actual owned stream abstraction, certified
  assistant replacement and subsequent ordinary admission;
- complete Berlin call lost in a **controlled terminal delivery window**, zero
  consumed events; four observations, one effect, stored-result continuation;
- expired first request: explicit client result plus old fenced POST rejection,
  with no generation;
- real argument-delta interruption leaving unfinished DSML: negative certificate,
  zero execution, explicit DELETE, old ID/request cannot resume, fresh creation;
- ordinary **non-streaming** fresh request, populated only from legitimate visible
  application conversation, followed by the same certified reconciliation path.

Across DELETE, preserved messages are prior certified assistants, completed tool
calls with their real stored results, prior user inputs and the last submitted
user input. The final unfinished native assistant/tool fragment is **omitted**;
there is no ordinary representation to preserve. No invented tool result or
placeholder enters the application transcript. Fresh creation is explicit and
can intentionally prefill those ordinary messages anew. This is new-session
prefill, not replay or cache recovery of the retired native session. Preservation
is not promised for arbitrary incomplete/ambiguous application tool exchanges;
those still require an explicit application decision, not invented history.

## Qualification / observations

Measured latencies/resources are recorded in `qualification.json`:

- Warm normal text/tool-result request walls: **0.67–1.14 s** (the first streamed
  greeting additionally loads the model for **67.48 s**, with **4.43 s** initial
  prefill/shape work; its 72.91 s total is not a steady-client latency).
- Settled lookup plus same-identity observation: about **2.0–2.9 ms**. Text-loss
  reconciliation including waiting for cleanup: **66.2 ms**; partial-tool
  settlement/reconciliation: **37.8 ms**. These are wall observations, not pure
  CPU overhead. Model decode and server transport wait remain separate in traces.
- Duplicate tool ledger calls without effect: **7.0–10.5 µs**; initial reservation,
  local deterministic stub and result storage: **35.3/43.8 µs**. No remote tool
  latency or universal overhead bound is claimed.
- Explicit negative-session DELETE: **1.05 ms**, fresh create **0.57 ms**. Fresh
  ordinary prefill/handoff: **3.64 s**, decode **163 ms**, request wall **3.86 s**.
  Six retained-session suffix turns are separate from the two initial prefills.
- Client serialized payload peak **6,948 bytes**, final **6,414 bytes**, two ledger
  entries. Estimates exclude Python allocator overhead. Combined qualification
  process max RSS **20,006,109,184 bytes**, server MLX active **309,217,529,760 bytes**,
  cache **0** at measurement; these include the checkpoint, not client-only RSS.

The Berlin zero-event loss uses controlled 1-second per-frame delivery delay:
its **14.25 s** wall time is not a normal-request performance sample. No optimization
or indefinite retention qualification follows from these bounded measurements.

Retained-session traces require zero model-history replay and zero full-cache
repack, coherent forty target/three DSpark offsets and no settlement proposal or
verification. Fresh target allocation/prefill is reported separately. Transport
waiting, lookup/reconciliation, model decode, prefill and ledger-only repeated
observations are not conflated.

Synthetic fault regressions supplement, not replace, real checkpoint evidence:
never-started transport loss, active identity, changed bytes/sequence, malformed or
changed certificate, ambiguous callback effect, ledger bounds/mismatch, expired
current/old identities, negative versus poison, and DELETE/create failure.
The real integrated partial-tool and completed-tool workflows preserve exact
requests and effects. Unstarted header-send failure and internal poison remain
controlled M36R lifecycle regressions, not claimed naturally sampled failures.

No server/runtime authority defect was exposed. The initial workload expected a
fresh tool-enabled model to obey a text-only instruction; it validly requested a
new certified tool instead. This harness expectation failure is preserved in
`workflow-initial-fresh-tool.*`, not repaired by bypassing the client or executing
unwanted effects. The final fresh text request explicitly disables tools at the
new-session protocol envelope. Superseded in-development evidence-serialization
attempts and the passing pre-hardening exploratory workflow are also preserved /
excluded in `commands.json`; none is current-source authority. Strict certificate
field/hash validation and explicit client-state classification of HTTP rejections
are covered by the final client rerun and regressions.

Current-source gates and hashes accompany the artifact: **38 local/browser tests**,
**110 runtime/horizon/OFF tests + 32 subtests**, **46 prior semantic/evidence tests
+ 137 subtests** (three explicit historical deselections), **5 preserved release-OFF
tests**, **7 Rust transport/lifecycle tests**, **28 native official-recipe fixtures**,
**18 real M35 HTTP turns**, the real M36 byte/tool/socket-pressure probe, all four
M36R fenced HTTP sessions and **16 checkpoint native interruption/fault cases**.
The final M37 source/evidence gate checks the coherent workflow and hashes separately. Historical M35/M36 HEAD
hash assertions and M36R's final artifact hash gate are explicitly deselected
when checking old evidence; new source hashes and fresh native/HTTP reruns provide
current authority. Historical evidence is not rewritten.

## Unsupported / next milestone

Still unsupported: public/default/release MTP, persistence/process restart,
client crash-safe tool exactly-once, distributed idempotency, concurrent/shared
MTP, batching, token-exact immediate abort, 200K HTTP MTP, unbounded backpressure,
portable native wheels and universal partial DSML recovery. M33 horizon, M34 cache
transfer, M35 response lease, M36 representability blocker and M36R certificate /
fence / exact-prefix admission are unchanged.

Recommended **M38: bounded internal recovery-client operational soak and fault
matrix**: repeated complete/partial/negative transitions, ledger exhaustion,
DELETE/create transport ambiguity and conservative ambiguous application effects
through this reusable client. Measure retained diagnostic/resource bounds over a
longer but finite living-client workload. This is the remaining operational gap,
not automatic release promotion. A later release/admission evaluation must be a
separate milestone with explicit supported-runtime ownership and packaging gates.
