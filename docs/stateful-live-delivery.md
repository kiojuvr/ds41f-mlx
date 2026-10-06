# Standard-OFF stateful live delivery

**PASS in the bounded representative scope.** The separate
[Web application closeout](web-application.md) also passes its current-source gates.
R1 and broader runtime/release promotion remain deferred.
Baseline findings are retained in `web-client-production-gaps.md` and
`artifacts/web-client-investigation/`. This document describes the new delivery
contract; a RUNNING/FAIL receipt is never a qualification pass.

## Ownership and commit-before-visible

M44/M45's `TargetGenerationSession.next_token` consumes a sampled token through
one complete all-layer target transaction, synchronizes/validates the frontier,
then publishes consumed history. Unconsumed lookahead is not committed history.
M8 remains the sole live generation/cache owner. No token execution, rollback,
replay, repack, new sampler or alternate cache is implemented in the transport.

`runtime/live_turn.py:LiveRecipeTurn` is a worker-confined official recipe parser
cursor. It advances only after a complete M8 target step. Therefore role/usage
outputs depend on the committed prompt/bootstrap, and text/reasoning/tool deltas
depend on already consumed canonical tokens. As in M35, parser output need not be
one event per token. Active canonical progress belongs to generation, not an
independent session record or socket-delivered frontier. GET reports busy during
active work. Socket sends/yields are not application acknowledgements.

A semantic terminal calls M8 idle transfer **before** publishing its final recipe
finish events. The same physical cache list transfers back to idle; lookahead is
retired. `last_turn` publication occurs on the same worker before the finishing
batch is returned. Cancellation does the same transfer at the last completed
transaction, not an unsafe GPU abort or natural-max-token drain.

## Live transport vs history / backpressure

`serving/stateful_stream.py:StatefulStream` owns a response reservation, not model
state. It drives one protected worker operation at a time using the existing
single-worker executor. No background producer or event queue runs ahead.
An advance returns a single parser batch. Every event in that batch is delivered
in order before another target transaction can begin. ASGI send/TCP buffers are
bounded downstream; a stalled send prevents subsequent generator demand. Explicit
cancellation can settle a between-demand cursor even when a socket send is blocked.
A batch may contain multiple official parser outputs, bounded by the request/output
admission envelope; it is not an ever-growing accumulated response event list.

The canonical response accumulator necessarily scales with the response. Only
64 historical diagnostic events are retained; they are **never** used to deliver
the stream. The full response, not the diagnostic tail, is used for GET recovery.
An explicit unshielded asyncio checkpoint between events/transactions avoids the
M35 socket-send cancellation-starvation defect.

## Disconnect, explicit Stop and settlement

The response reserves busy state before headers or iterator startup. Response-level
`InferenceStreamingResponse` cleanup closes even an unstarted iterator, so header
send failure cannot strand a reservation. Each response clears only its own
reservation. Admission while another request is busy rejects. Waiting for the
single-flight lock is cancellable without touching another session's owner.

A socket disconnect cancels the iterator. `_call` drains the current protected
operation before cleanup. The transport gate serializes explicit Stop with that
operation, including startup. Cleanup cancels at the completed transaction frontier,
finalizes official parsing, publishes the canonical response, retires generation
and releases the lock. Repeated cleanup is idempotent; settlement is never retried.

`POST /v1/sessions/{id}/cancel` with `{ "request_id": "..." }` requests this same
settlement and returns the settled session record. The request identity comes from
`X-DS41F-Request-ID` on the SSE response, or `active_request_id` on busy GET. Stale
identities reject with 409 and cannot cancel a newer turn. This is not MTP's
sequence/body replay fence and does not implement automatic generation retries.
Non-streaming compatibility execution retains its existing natural-turn behavior;
the new Stop endpoint is specifically for active standard-OFF live streams.

## Reconciliation and tool permission

After disconnect/Stop the client must wait for GET to leave busy, then replace its
provisional assistant with `last_turn.response_json`. That response may include a
small committed but undelivered suffix. `last_turn.cancelled`, its request ID,
request_count and frontier distinguish canonical settlement from local drawing.
The protocol uses the official length finalization on interruption; cancellation
is additionally explicit in the session record, not misrepresented as natural EOS.

A finalized partial response is **not automatically ordinary-prefix reconstructable**.
The existing official recipe reconstruction witness checks the exact canonical
prefix. This check is used for cancelled turns and completed tool-call outcomes,
now accepting the actual checkpoint for Vision validation. The witness does not
execute tools, allocate a second cache, generate tokens or replay the prompt.
Its full transcript/image bytes are not retained. Only the bounded certificate is
published. A completed tool call is executable only if its certificate says so.
Incomplete DSML/tool grammar or undecodable partial token bytes may be
protocol-unrepresentable, as established in M36/M36R. In that case the cache is
explicitly retired and GET says `state=unrecoverable`, `recovery_state=unrecoverable`.
The diagnostic response is retained but conveys no tool execution permission.
DELETE/fresh session is required; no silent rollback/completion/replay occurs.

Protected parser/state failures similarly retire rather than advertise ambiguous
idle authority. Legitimate ordinary exact-prefix continuation still uses existing
M11/P6 admission; historical image bytes/identities must be retained unchanged.

## Evidence and scope exclusions

- `tests/test_stateful_live_delivery.py`: native official parser with lifecycle
  doubles; full >64-event delivery, pre-iteration cleanup/stale Stop, demand
  backpressure and cancellation during a protected transaction.
- `tools/qualify_stateful_live_http.py`: actual checkpoint HTTP stream, canonical
  equality, slow reader, repeated real socket disconnect/explicit Stop, ordinary
  continuation, tool terminal/one client-side result, Vision/new-image/text/cancel.
- `artifacts/stateful-live/http.json`: durable RUNNING/FAIL/PASS receipt and source
  hashes. This is HTTP qualification, **not** the final Web 12-step campaign.

The cancelled/plain-text representability gates pass with the real model. The
current HTTP receipt also verifies application correlation, stale count rejection,
full text/tool reconstruction and one finish event. Its slower reader pauses .8s
per eight events. `tests/test_stateful_live_backpressure.py` additionally holds an
ASGI send blocked and proves no further target demand, explicit Stop settlement,
and unstarted header-failure cleanup. Mid-tool-prefix interruption executes zero
tools and may explicitly retire, not claim universal resumability.
TCP-buffer saturation, arbitrary image containers/Unicode/DSML prefixes and full
R1/200K/1M are not implied by a bounded representative campaign.
