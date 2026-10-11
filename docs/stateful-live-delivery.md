# Standard-OFF stateful live delivery

This page owns **OFF demand-driven transport over canonical native commit**.
[Generation](generation.md) owns consumed history/settlement;
[OFF API](api.md) owns endpoint behavior;
[application state](web-application.md) owns frontend journals/effects.
Ordinary MTP uses its separate Scheduler/pump contract, not this cursor/transport.
Original HTTP PASS and later Web/UI acceptance are distinct historical scopes.

## Ownership and commit-before-visible

First-party target generation consumes a sampled token through the complete
all-layer transaction, synchronizes/validates frontier, then publishes canonical
history. Lookahead is not committed. M8 owns live generation/cache; transport
implements no new token execution, sampler, rollback, replay, repack or cache.

Worker-confined `LiveRecipeTurn` advances the official recipe only after a complete
M8 step. Role/usage follows committed prompt/bootstrap; text/reasoning/tool deltas
follow consumed tokens. Parser event count is not token count. Active canonical
progress belongs to execution, not a socket frontier; GET reports busy and
writes/yields are not client acknowledgements.

A terminal transfers the same cache list back to idle and retires lookahead before
final recipe finish events. Worker `last_turn` publication precedes returning that
finishing batch. Cancellation settles at the last completed transaction, not an
unsafe GPU abort or drain to the natural maximum.

## Live transport vs history / backpressure

`StatefulStream` owns one response reservation, not model state. It requests one
protected operation on the existing worker executor at a time, with no background
producer running ahead. Each returned parser batch is delivered before the next
target transaction. A blocked send stops further demand; explicit cancellation
can settle between demands even while send is blocked.

The pending batch is distinct from accumulated response/history. Canonical response
storage scales with actual output; the bounded 64-event diagnostic tail is never
the delivery source. GET uses the full latest response, not that tail. Explicit
unshielded asyncio checkpoints admit disconnects rather than assuming every socket
send suspends. Downstream buffering is not proof of client observation or a tested
TCP saturation envelope.

## Disconnect, explicit Stop and settlement

Busy is reserved before headers/iterator startup. Response-level cleanup closes an
unstarted iterator too; failed header send cannot strand ownership. Each response
clears only its reservation. Busy admission rejects; cancellation while waiting
for the single-flight lock cannot mutate another owner.

Disconnect closes the iterator. Protected `_call` work drains before cleanup;
the transport gate serializes Stop/start/advance. Settlement finalizes official
parsing, publishes actual canonical outcome, retires generation and releases lock.
Cleanup is idempotent; a failed settlement is not retried as mutable execution.

`POST /v1/sessions/{id}/cancel` takes `request_id` from SSE
`X-DS41F-Request-ID` or busy GET `active_request_id`. Stale IDs reject with conflict,
not cancellation of a later request. Its acknowledgement contains the session
observation; authoritative reconciliation still uses count/identity/GET. This is
not singleton sequence/body replay or production-MTP reserved-UUID identity.
OFF Stop applies to live streams, not legacy nonstreaming natural-turn execution.

## Reconciliation and tool permission

Wait for GET to leave busy, then replace provisional output with
`last_turn.response_json` using request/application identity and expected count.
A committed undelivered suffix may appear. Cancellation is explicit in the session
record; recipe interruption finalizes length, not invented natural EOS.

Finalized partial text is **not automatically ordinary-prefix representable**.
Official reconstruction checks exact canonical IDs and checkpoint/image identity
without another cache, model execution or replay. Only the bounded certificate,
not its full witness transcript/image bytes, is retained. Completed calls grant
effect permission only under a positive executable certificate.

Incomplete tool/DSML grammar or undecodable bytes can retire the native cache and
publish `unrecoverable` state with diagnostic outcome, not continuation/effect
permission. Parser/native failures likewise retire rather than publish ambiguous
idle authority. Explicit DELETE/fresh-session decisions are different from silent
rollback, sampling onward, fabricated results or retained-history replay.
Legitimate continuation keeps exact recipe prefix and old image identities.

## Evidence and scope exclusions

[Original HTTP receipt](../artifacts/stateful-live/http.json) and the
[pre-compilation edition](archive/editorial-baseline/stateful-live-delivery.md.txt)
retain exact source hashes, representative model stream/Stop/tool/image evidence,
slow-reader setup and original PASS wording. RUNNING/FAIL is not PASS; neither an
old HTTP result nor later Web acceptance qualifies every changed payload.

`qualify_stateful_live_http.py` is an existing reproduction driver, not a request
to start a model campaign here. Focused delivery/backpressure/retirement tests
exercise official parser/lifecycle seams, >64-event full delivery, protected cancel,
blocked ASGI send and pre-iteration header failure. These do not prove arbitrary
Unicode/DSML/image prefixes, universal resumability, TCP-buffer saturation,
near-1M/long-context, full Vision, R1 or release scope.
