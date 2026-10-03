# M36R — design before runtime changes

Authority: clean M36 `df788d5`; pinned M33–M36 recipe candidate
`29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`, oMLX candidate
`fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`. M36's partial DSML
counterexample remains valid. No change to the M33 strong semantic horizon.

## Proposed bounded contract

A certificate is existential, not a handwritten parser-state whitelist. Let H
be the sole retained native canonical IDs after settlement. Interpret emitted
canonical inputs with the existing official processor (including official EOS
suppression). Accumulate its official Chat Completion response. Append that
ordinary assistant message to the original ordinary request, followed by ordinary
next-turn messages. Use official request conversion/rendering/encoding. Require
`encode(witness)[:len(H)] == H` and a strictly longer encoding. This is equality
at the retained admission boundary, not equality of the whole extended request.
No raw IDs are inserted into a conversation. Tool witnesses need one ordinary
tool result per call to satisfy official conversion; qualification placeholders
are never execution or invented arguments and lie *after* the retained boundary.
A tool outcome additionally requires canonical native DSML block terminal and
recipe `tool_calls` finish. JSON validity alone is irrelevant.

The server constructs a fixed ordinary witness using a next user message (and
non-execution tool-result placeholders when needed). Failure is a negative
certificate for that authoritative reconstruction: no best-effort repair. This
conservative admission policy need not decide whether some different hypothetical
conversation could encode the same IDs. Each real continuation still independently
checks the exact prefix before any mutation. A positive witness does not authorize
arbitrary changed tools, thinking settings, results, or transcript edits.

Session outcome states: not_admitted, active, recoverable, unrecoverable, poisoned.
Busy/native cleanup overrides settlement visibility. Protocol-unrepresentable is
not an internal failure: retire caches, preserve diagnostic outcome, require
DELETE and fresh session. Poisoned means internal mutation/ownership/protocol
failure. Existing legacy `state` labels may remain for old probes; a distinct
`outcome_state` is authoritative for this contract. No idle publication precedes
settlement plus certificate. Outcome observation never feeds the model/parser.

## Local request fence

Only the explicitly injected internal backend supports an opt-in integer
`X-DS41F-Request-Sequence`, starting at 1. Bind it to SHA256 of exact request body
bytes. One latest slot per session, plus monotonically increasing consumed
sequence; no eviction ambiguity because older sequences reject as expired.
Same sequence/same bytes: active rejects without starting another owner; settled
returns the outcome as JSON (even if the original requested SSE), never generates.
Same sequence/different bytes rejects. Gaps reject. Pre-conversion/admission
rejection does not consume sequence; GET exposes the next sequence. Header-send
failure before iteration is not_admitted and retryable; once worker start is
entered it is active and must settle or poison, never retry as unstarted.

GET and POST run on one event loop; reservation has no intervening await after
checks and before lease acquisition. Absence in GET is an observation, not an
ACK that delayed POST cannot arrive. Retrying the *same* sequence closes that
race. A client must not move on with another sequence while an old POST may be
in flight. Outcome slot is replaced only by the next admitted sequence. The
single client copies the settled result before moving on. No persistence/restart,
distributed idempotency, shared clients or unlimited outcome archive is promised.
Legacy requests without the header retain exact-prefix protection but do not get
the ambiguous-request contract; mixing modes on a fenced session is rejected.

## Client / tool workflow

After transport loss discard local UTF-8/SSE fragments, GET until non-active,
then re-observe/retry the same request identity. Recover only a positive
certificate. Replace the local assistant entry; do not concatenate deltas.
For tools, require positive certificate plus canonical completion evidence;
key the local execution ledger by `(session_id, sequence, call_index, call_id)`.
Record/reserve before execution and keep the result for repeated observation.
This prevents duplicate execution in one living client, not crash-safe or
distributed exactly-once side effects. Nonrecoverable calls never execute.
Construct next ordinary request with real results and next sequence; official
conversion and exact-prefix check remain mandatory. No replay/repack or caches
reconstruction is allowed. Unsupported scope from M36 is unchanged.

## Qualification plan

Use actual native recipe/tokenizer fixtures for text, Unicode pending/full,
reasoning transition, marker prefixes, tool block/name/argument prefixes, valid
JSON before canonical block end, complete calls, length/EOS, lossy JSON mode.
Use real checkpoint HTTP for lost bytes, partial tools, complete tools and
idempotent observations/continuations. Preserve and rerun M36 socket pressure,
M35 lifecycle/native transfer tests, M33 horizon and checkpoint interruptions,
OFF policy and Rust gates. Record positive and negative witnesses and counters.

## Qualification decision

**QUALIFIED_BOUNDED_CERTIFIED_RECOVERY**, internal singleton only. Machine-readable
canonical result: `artifacts/m36r/qualification.json`; raw evidence is alongside
it. Final commit resolves with `git log -1 --format=%H -- artifacts/m36r/qualification.json`.
The implementation follows the pre-change design above. M36 remains a valid
counterexample to universal ordinary-protocol reconstruction. No upstream parser,
M33 horizon, target/DSpark math, native cache handoff or release selector changed.

### Precise admission rule and limits

The certified representation is the *official interpreted message*, not any text
that happens to tokenize to H. Certificate version 1 binds its frontier and H hash,
ordinary witness, encoded length, first mismatch and completion predicate. The
hash is evidence identity, never a replacement for exact ID comparison. Existing
exact-prefix admission still independently checks the real next request against
retained H. Witness conversion is protocol tokenization, **not model history replay**.
Both positive and negative certificates retain evidence without reparsing with
another grammar. Pending decoder/parser source is never guessed or flushed by
extra sampling. Cache retirement does not truncate the preserved canonical IDs.

This is a conservative reconstruction policy, not a completeness theorem over
all possible API envelopes. Negative means the authoritative message/witness is
not certified and the runtime cannot continue. In particular, reasoning-only
output with `content:null` is not accepted by official historical assistant
conversion; alternate hypothetical envelopes are not repaired into scope. The
qualified useful subset is text and weather-style function tools with unchanged
ordinary prompt settings/schema, plus exact-form reasoning-transition fixtures.
Different schemas, JSON modes, null/empty-field substitutions, arbitrary raw
special-token transcripts and other untested alternatives have no universal
recovery claim. Even complete tools can be nonrepresentable. A successful witness
does not exempt any actual continuation from exact-prefix admission.

### Native official-recipe matrix

`recipe-matrix.json`: **28 controlled native recipe/tokenizer cases**, not sampled
checkpoint generations. All feed real token IDs through the existing guard and
native processor; ready/finish/control suppression mirror the real owner. These
are representation fixtures, not claims that synthetic caches execute the model.

| Class | Exercised result |
|---|---|
| Ordinary canonical text, full café/emoji, length, ordinary EOS | Positive exact-prefix witnesses |
| Literal DSML marker with tool parsing disabled | Positive text witness; not a tool-parser success |
| Reasoning→text without discarded separator | Positive exact-form witness |
| Completed canonical one/two calls; required and auto tool choice | Positive exact-prefix plus DSML terminal/`tool_calls` certificates |
| Decoder-pending emoji, including pending IDs at EOS | Negative; ordinary message cannot retain pending canonical IDs |
| Leading newline; newline discarded at reasoning transition | Negative first-token/source mismatch |
| Reasoning only / unfinished think-marker prefix | Negative conversion or source mismatch |
| Enabled DSML marker/block prefixes; partial tool name/arguments | Negative; no executable completed canonical block |
| Valid JSON arguments, invoke closed but block unfinished, including length finish | Negative, regardless of JSON validity |
| Four token-exact prefixes of the canonical completed tool source | **Exact prefix true, semantic completion false**; cannot execute/continue |
| Completed alternate `tool_calls` closing spelling | **Semantic completion true, exact prefix false**; cannot execute/continue |
| JSON fence consumed by official JSON-output parser | Negative source mismatch; not repaired |

Nine positive and nineteen negative certificates establish why neither a parser
state label, valid JSON nor even canonical semantic completion alone is enough.
The fresh evidence gate independently re-encodes witnesses and checks every
negative fixture's continuation rejection/zero tool execution before mutation.
This fixture admission gate uses a synthetic serving lifecycle, clearly separate
from the actual checkpoint HTTP rejection below.

### Real client and fencing evidence

`http.json` runs the real checkpoint, native recipe, retained caches and loopback
uvicorn h11. Four logical sessions exercise:

- socket loss inside a multibyte UTF-8/SSE fragment, discarding bytes and replacing
  the assistant with the certified server outcome, then ordinary continuation;
- socket loss after a real tool-argument delta: coherent canonical H survives,
  negative certificate, `outcome_state=unrecoverable`, zero tool executions, and
  mutation-atomic ordinary continuation rejection;
- a completed canonical Berlin call lost in a **controlled terminal delivery
  window**, with zero body bytes read: positive certificate, three identical
  POST outcome re-observations, ordinary real tool-result continuation;
- active reservation/generation retry rejection without starting a second owner.
  An admitted prompt can settle with zero generated response tokens and a
  representable empty assistant. This is still committed work, not permission to
  resubmit the turn as a new sequence.

GET repeated three times and settled POST repeated three times are identical and
non-mutating; POST returns an explicit sequence/outcome/certificate/response JSON
wrapper, not replayed SSE. Changed body at the same sequence rejects atomically.
Out-of-order sequence before admission leaves the initial not_admitted session
unchanged. After the next sequence is admitted, old identities reject as expired,
not silently evicted into new generation. Unstarted header-send failure/retry and
started protected failure/poison fencing have **synthetic lifecycle regressions**;
these are not claimed as naturally sampled socket-header failures.

The tool ledger checks *settled recoverable state and matching sequence* before
certificate/execution. Four observations of the same canonical completed call
produce **one effect** and reuse the stored ordinary tool result. A reserved but
failed/ambiguous effect is never automatically retried. Partial/unrepresentable,
active, poisoned and wrong-identity observations cannot execute. This is one
living local client/session, not distributed or crash-safe exactly-once effects.
The production browser/Rust client is unchanged; the probe and small ledger
exercise the contract, not a released agent integration.

### Current-source gates and defects

Fresh runtime, native horizon, OFF-policy, preserved-release ABI, historical
semantic/evidence, Rust, M35 real HTTP, M36 byte/tool/socket-pressure and all sixteen
native checkpoint interruption/fault gates are recorded in the artifact. The
historical M35/M36 HEAD-source-hash checks are explicitly deselected: those hashes
cannot describe this milestone. One initial historical-hash test selection error
is preserved, not counted as a passed gate; M36R checks its own current hashes.

No new native ownership mismatch was found. This milestone replaces the M36
heuristic tool-only containment with certificate-based containment for every
ordinary HTTP settlement. Internal review regressions additionally require the
tool ledger to reject active/wrong-sequence outcomes and DELETE retirement errors
to classify as internal poison rather than protocol limitation. These are narrow
new-contract hardenings, not repairs of upstream representability. The original
M36 incomplete-tool evidence is preserved byte-for-byte.

The actual continuation traces check all forty target and three DSpark offsets
at H and consume the retained wrappers. **All instrumented model replay and full
cache repack counts are zero.** No new proposal/verification occurs in settlement.
M35's eighteen HTTP turns and matched direct control still pass. The affected M36
real 8192-byte socket/256-KiB-comment pressure case still exhibits restrictive
send and pull-based production plateau; exact fresh timings/resources are in the
artifact. Certificate construction happens inside shielded bounded settlement,
not an independent production queue. There is no enlarged backpressure benchmark
or indefinite diagnostic-retention claim.

### Remaining scope / next milestone

Still unsupported: public/default/release MTP, persistence/process restart,
token-exact immediate abort, concurrent/shared MTP, batching, distributed
recovery/idempotency, 200K HTTP MTP, unbounded backpressure and portable native
wheels. Partial canonical DSML/tool recovery remains intentionally unsupported;
DELETE/fresh session is a valid protocol outcome, not a reason to generate more.

Recommended **M37: bounded internal local-client integration** of sequence-fenced
recovery and the execution ledger, including user-visible DELETE/restart and
expired-outcome handling. Do not couple that integration to release promotion,
persistence, distributed effects or longer-context claims.
