# M36 — client/agent canonical recovery and admission

## Decision

**BLOCKED_CANONICAL_PROTOCOL_REPRESENTABILITY**. Machine-readable authority:
`artifacts/m36/qualification.json`. This is a completed blocker investigation,
**not successful M36 qualification** and not release promotion. Source base is
`9de5750` (clean master at M35). Final commit resolves with
`git log -1 --format=%H -- artifacts/m36/qualification.json`.

The [M33 horizon](milestone-33-semantic-horizon.md), M34 native cache transfer,
and M35 transport ownership remain intact. There is still exactly one canonical
model/session history. The investigation stops rather than repairing an agent
transcript by guessing, regenerating, replaying, or weakening exact-prefix admission.

## Blocking evidence

`pre-pressure-harness-fix.json` preserves the initial real checkpoint/official
recipe/loopback HTTP counterexample. The client consumes a tool-argument delta,
closes its actual socket, waits for settlement, and observes:

- a coherent target/DSpark/canonical idle frontier;
- an assistant tool call with arguments **`{"city"`** and no finish reason;
- an `idle` session that does **not** have a complete executable recipe result.

This is not a mere partial SSE frame. The canonical DSML source itself stopped
before completion. Socket loss is not permission to complete its JSON or generate
its missing DSML suffix. An event accumulator is not a protocol completion owner.

The corrected real probe in `probe.json` independently attempts ordinary recipe
conversion with that recovered message and a **non-execution sentinel** tool
result. Conversion succeeds, but its encoding is **not an exact canonical prefix**:
first mismatch at token index **308**, canonical length **310**. No tool is executed.
The HTTP continuation is rejected without mutation. Thus even making the history
structurally acceptable to Chat Completions does not solve canonical admission.
Changing the sentinel cannot repair the earlier assistant-source mismatch.

The inspected pinned recipe sources explain the boundary:

- `stream/decoder.rs` retains IDs until they decode without a trailing replacement
  character; IDs still pending at finish do not contribute text or completion usage.
- `stream/state_machine.rs` retains marker/parser source until sufficient input.
- `stream/processor.rs` stashes source/actions and emits tool argument deltas before
  a backend/semantic finish; only its finish owner selects the final reason.
- Chat Completions `response/schema.rs` accumulates deltas without validating that
  a tool block completed. Request conversion and rendering encode an ordinary
  historical assistant/tool exchange, not an arbitrary raw DSML continuation.

Consequently a cache-quiescence certificate is necessary but not sufficient for
an ordinary client reconstruction certificate. No second parser, raw-token splice,
cache reconstruction or server-side tool execution is introduced.

## Narrow corrections and regressions

1. **Unfinished canonical tool advertised idle:** internal settlement now consults
   the existing authoritative recipe guard and its DSML completion provenance.
   A response containing tool deltas without a canonical DSML block terminal and
   recipe `tool_calls` finish poisons the session, even if its current arguments
   happen to be valid JSON or the backend has finished by length/EOS.
   Shielded cleanup retires native owners/caches using the existing lease boundary;
   DELETE is required. The diagnostic response remains evidence, **not execution
   permission**. Completed recipe tools are unchanged. This is fail-closed
   containment, not a solution that continues an interrupted tool turn.
2. **Poison visible before cleanup settled:** the first containment rerun exposed
   `poisoned` while the response lease was still busy, so repeated GETs could differ
   after the client thought it had reached a boundary. `busy` now takes precedence
   until cleanup/retirement releases that exact lease. `poison-get-defect.log` and
   `early-poison-observation.json` retain the real failure and explicitly labelled
   synthetic/source reproduction. The final real probe observes stable settlement.

`tests/test_m36_recovery_boundary.py` covers incomplete arguments, valid JSON
without canonical completion, backend finish without DSML completion, completed
tools, text prefixes and poison/lease ordering. The artifact records fresh affected runtime, native horizon, checkpoint
interruption, OFF-policy, Rust and full M35 real HTTP gates. Historical M33–M35
artifacts are not rewritten or relabelled as current-source evidence. The historical
M35 HEAD-source-hash test is explicitly deselected in the prior-evidence gate;
its old hashes cannot describe this correction. Fresh M36 source/evidence checks
and the real M35 rerun supply current-source gates instead.

## Positive bounded observations (not full qualification)

The real client additionally closes after receiving the **first byte of a
multibyte UTF-8 character inside an unfinished JSON/SSE event**. Its byte prefix
cannot decode as UTF-8. It discards that local transport fragment and replaces the
assistant transcript entry with the settled official recipe message **`café 🙂 `**.
Repeated GETs are idempotent. Ordinary recipe conversion is an exact prefix and
ordinary HTTP continuation succeeds, frontier **21 → 42**, with zero replay/repack.
The client never concatenates received bytes to guess the missing protocol.

Resubmitting the original request after settlement is rejected mutation-atomically
in both the positive text and poisoned tool cases. This demonstrates existing
exact-prefix/poison protection, **not a revision-fenced ambiguous-retry contract**.
There is no application ACK, token-exact delivery knowledge, distributed idempotency,
or new public recovery API. GET remains the internal diagnostic mechanism. A fresh
M35 rerun also exercises the existing Rust iterator-drop and completed canonical
tool recovery workflows; neither becomes a new M36-qualified agent contract.

## Actual socket pressure

The real uvicorn **h11** server uses an inherited **8192-byte SO_SNDBUF**. The
qualification-only ASGI wrapper adds a **256 KiB valid SSE comment** per streamed
frame. It never changes recipe events, tokens, or the normal backend. Padding is
restricted to SSE responses (an initial harness mistakenly padded JSON GETs with
Content-Length; that failure is preserved and excluded).

A client opens the 768-response-budget guide and reads no body for 30 samples at
100 ms intervals. Three 262-KiB writes are observed; one send is restrictive for
**3.276 seconds**. Canonical production plateaus at two responses during sustained
pressure, then settles at three after disconnect (one protected in-flight phase).
The pull-based iterator does not create an independent unbounded generation queue.
The final canonical frontier is **51**, response/session lease is released, and
DELETE succeeds. Disconnect-to-observed-settlement is **1.970 s**, not misreported
as a pure cancellation-check or cleanup timer; native cleanup itself is **1.154 ms**.
Protected model work/shape compilation accounts for much of this case's wall time.

Response growth is bounded by the existing **768 response / 8192 total token**
admission, retained official events, the currently serialized frames and the
transport write buffer. Expanded comments are created per send, not retained in
canonical history. The wrapper's fixed padding is a controlled transport workload,
not evidence of naturally sampled hundreds of megabytes of model text. It establishes
actual finite write restriction, not unbounded-backpressure or long-context safety.
Qualification diagnostics (`session_traces` and closed-session records) are retained
across requests; their total lifetime retention is not qualified for indefinite
service. The artifact separately records current RSS and MLX active/cache bytes.

Text recovery observation takes **2.968 ms**; next-turn prefill/handoff **223 ms**,
first canonical from admission **284 ms**, request elapsed **664 ms**. These are
single local measurements, not latency guarantees. Decode, serialization, send
restriction, cleanup and lookup/poll timings remain separate in raw evidence.
All instrumented model-history replay and full-cache repack counts are **zero**.

## Remaining contract and next milestone

The requested four-way, bounded, client-visible outcome contract is **not
established**. In particular, never-started/ambiguous request identity fencing,
arbitrary parser-pending reconstruction, and a fresh complete M36 tool execution
ledger remain unqualified. Poisoned partial tool responses must not be executed;
zero executions in the negative probe is not proof of general exactly-once tools.
Partial-byte success is limited to the exercised canonical text case. Native pending
UTF-8/marker cases from M33 are not relabelled new client recovery successes.

Recommended next work: **M36R — canonical protocol representability and recovery
admission design**, before any release/admission promotion or longer-context gate.
Define a bounded protocol reconstruction certificate and an explicit local-session
request/outcome fence. Determine which parser-pending canonical boundaries can be
represented by ordinary recipe re-encoding and which must fail closed. Do not
silently complete unfinished tools, sample through a cancellation to make them
representable, or change the strong M33 horizon to obtain an agent-looking transcript.
If no such ordinary representation exists, keep those boundaries poisoned/unsupported
and explicitly narrow any future qualification claim.

Still unsupported: public/default/release MTP; persistence/process restart;
immediate token-exact abort; concurrency/shared execution/batching; distributed
recovery; 200K HTTP MTP; portable native wheels; unbounded backpressure. Existing
release policies and public Rust/API surfaces remain unchanged.
