# Ordinary Native Messages SSE heartbeat — PASS

## Scope and source

Baseline: clean `master`, **87433bc868a2943d854fbbdf3c03e886f3f6eade**.
One Runtime transport change; no Recipe, Harness, generation-policy, tool-gate,
Target/DSpark, batching, admission-policy or Production changes. The client's
already configured **30-minute timeout was not changed**.

Exact tested source/test Git blob IDs and SHA256s are in
[source-revisions.json](source-revisions.json). These identify the working-tree
inputs independently of the enclosing local commit. The wheel manifest records
the baseline commit; it is **not** claimed to represent a clean baseline build.
The changed Runtime file matches the installed candidate byte-for-byte.
[Candidate package receipt](candidate-packages.json): 128 wheel members checked,
zero mismatches, wheel SHA256
`f0a2b2504743fdf0dc8836bcefcc22c5f1e9cbe48b24aaffa07a4a0d4912a455`.

## 1. Failure mechanism and complete execution path

- `serving/server.py:create_app.api_handler`: request body and bounded admission,
  then `run_in_threadpool(prepare_request)` for policy/conversion/render/encoding.
  These occur **before** the SSE response and retain their existing JSON HTTP
  refusal/status behavior. No early 200 response or pre-admission inference was
  introduced.
- `ProductionMTPBackend.ordinary_response`: validates executable capacity,
  shielded `_call(_enqueue)`, creates request-scoped delivery/settlement state,
  starts `_pump`, then returns `OwnedResponse(InferenceStreamingResponse)`.
  Executor admission before this response is returned is not an open SSE body.
- `_pump` awaits `_call(scheduler.step)`; `_call` runs native work on the existing
  single executor and shields mutations through cancellation. The HTTP event
  loop is free while that worker executes.
- `ProductionScheduler._schedule_waiting` performs paired lookup/restore and
  `_start`, including native cold/suffix prefill, paired capture and P5 handoff.
  There may be no semantic events before this finishes.
- `ProductionScheduler.step` advances the same native application. Its existing
  `tool_use` boundary holds the entire tool suffix in `pending_delivery` until
  canonical settlement. It can deliver many **empty internal outputs** while
  tokens continue to be generated. Native `ping`, when present, is also held if
  it is in that suffix. No model-level parser or semantic owner was changed.
- `_retire` settles, captures/publishes eligible paired state, and only then
  releases the remaining tool suffix in the terminal `RequestOutput`.
- Previously `ordinary_response.chunks` waited on `queue.get()` and emitted only
  Recipe events. An empty output produced **zero HTTP body bytes**. Thus healthy
  native execution could leave the wire silent for over 300 seconds.
- Normal terminal output ends the iterator. Error output raises and closes the
  stream. HTTP disconnect/cancellation closes the iterator through
  `InferenceStreamingResponse.__call__` and `OwnedResponse` cleanup, which uses
  the same `_cancel`, watcher-stop and canonical settlement/retirement paths.

The original six approximately 306–313-second aborts are consistent with this
silence boundary. The original Windows exception `cause` remains unavailable:
we do **not** claim direct proof of `UND_ERR_BODY_TIMEOUT` in that client, or infer
that its configured 30-minute timeout is absent. Transport/library/proxy read
inactivity deadlines are separate from application/absolute-duration deadlines.

## 2. Minimal fix and ownership

`serving/production_mtp.py` adds a fixed **15.0-second** transport interval and an
absolute monotonic deadline around **only the delivery queue read**. On timeout,
the SAME HTTP body iterator yields exactly:

```text
: keepalive

```

The yielded comment reaches StreamingResponse/ASGI `send`, with the existing
`text/event-stream`, `Cache-Control: no-store` and `X-Accel-Buffering: no` headers.
The deadline is reset after an actual yielded frame is consumed, **not** after
an internal empty output. This is essential during deferred tool generation.

Only `queue.get()` is cancelled by the timer, never the native operation or its
shielded worker future. No producer task, speculative read-ahead, extra queue,
retry, new Recipe event, parser or accumulator was added. Awaiting ASGI sends
preserves existing backpressure. The timer cannot run ahead of a blocked client.

There is no timed wait inside a terminal output's event sequence: its
`message_stop` is followed by termination/cleanup, never another comment. Errors
raise before another iteration. Disconnect/task cancellation unwinds the same
iterator and native cleanup. JSON uses the untouched nonstream branch.

The broker is shared with ordinary Chat Completions and logical Chat sessions;
it is preferable to fix that existing HTTP ownership point rather than wrap all
streaming endpoints or add a separate Messages transport. Shared Chat framing
and `[DONE]` have focused wire coverage. Singleton/other response owners were
not broadened. `tool_removal` remains unsupported.

## 3. Deterministic real-HTTP regression

`tests/test_sse_heartbeat.py` uses real uvicorn TCP sockets and HTTPX reads, not
buffered ASGI TestClient. Physical inference/retirement is controlled; the actual
ordinary ingress, broker, worker dispatch, **ProductionScheduler.step tool gate**,
and native Recipe guard/events are used.

Timing: 25ms heartbeat, 150ms read-inactivity timeout; a 240ms synchronous worker
prefill pause and repeated empty outputs during partially assembled native tool
arguments. The negative control sets the heartbeat interval to 60 seconds and
**does reproduce HTTPX ReadTimeout** with an 80ms read bound on the same silence.

Seven focused cases establish:

1. Comments arrive before the first semantic frame and during the withheld tool
   suffix despite frequent empty broker outputs.
2. Normal native text and complete exact-ID tools retain their framing/order.
   Removing comment frames yields the native event stream **byte-for-byte**.
3. A tool block is not observed before the controlled settlement boundary; native
   exact-ID Tool Result conversion succeeds.
4. Success ends with `message_stop` and EOF, with no following comment.
5. Disconnect during initial silence and during tool assembly drains/cancels the
   owner; delivery/settlement maps and pump are empty, with no orphaned server.
6. Terminal generation failure closes HTTP (RemoteProtocolError, not read timeout),
   discloses no incomplete tool and emits no fake `message_stop`.
7. JSON contains no comment bytes; shared Chat SSE retains JSON frames and `[DONE]`.

Final [pytest.log](pytest.log): **135 passed**, including all seven new tests plus
native Messages/semantic-guard, paired production and request-failure regressions.
Only existing FastAPI/Starlette deprecation warnings were reported.

[Controlled wire](controlled.sse), [unchanged semantic oracle](controlled-semantic.sse),
[timestamps](controlled-timing.json): 24 comments, 20 semantic events, total
0.790s, maximum inter-read gap 0.0923s (< 0.150s inactivity bound). Total duration
exceeding a read timeout is expected: a read timeout is **not** an absolute limit.

The existing Harness dependency `eventsource-parser` **3.1.1** parsed these actual
captures in 7-character fragments: comments were ignored and semantic events
were identical. It also verified both native wire captures against their canonical
content. [Script](verify-sse.mjs), [PASS result](parser-verification.json).
No Harness files/configuration were modified.

Reproduce the focused regression/dependency tests with the isolated admitted env:

```sh
SSE_HEARTBEAT_WIRE_EVIDENCE="$PWD/artifacts/sse-heartbeat" \
/private/tmp/ds41f-sse-heartbeat-env/bin/python -m pytest -q \
  tests/test_sse_heartbeat.py tests/test_recipe02_messages_serving.py \
  tests/test_recipe02_semantic_guard.py tests/test_production_mtp.py \
  tests/test_request_failure_diagnostics.py
node artifacts/sse-heartbeat/verify-sse.mjs /private/tmp/ds41f-alpha2-e2e/harness
```

## 4. One bounded real-model acceptance

A disposable **copy** of the stopped Production dependency closure was made at
`/private/tmp/ds41f-sse-heartbeat-env`; only the candidate Runtime wheel was
installed there, non-editably. Corrected Recipe 0.2 native SHA256 remained
`93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2`.
Flash-1017, GLM **native**, ordinary `mtp-serving-v1`, paired MTP, canonical
launcher/admission, loopback port 18089. [Build](build.log), [install](install.log),
[health/identity](initial-health.json).

[Acceptance client](acceptance.py) and [test-only launcher](candidate-launch.py):
one safely controlled **16.5-second worker pause immediately before the first
real native `_start`/prefill**, then unmodified native execution. This explicitly
is a controlled silent interval, not a claim that this small model request
naturally required 16.5 seconds of prefill. The observer also reads the SAME
canonical settled response; it does not replace or reparse Recipe semantics.
Neither observer is part of the installed Runtime or Production.

Exactly **two real-model requests**, one successful model lifetime; no huge tool
arguments, full Harness E2E or endurance campaign:

| Request | Wire observation | Native result |
| --- | --- | --- |
| Tool | Comment at **15.184s**, first semantic frame **20.047s**, complete at **20.987s** | 297 prompt + 37 generated tokens; `report_value({"value":7})`, `stop_reason=tool_use`; frontier **334** |
| Exact-ID result continuation | Normal SSE, no heartbeat needed; **0.423s** | 351 prompt, **334 cached**, 2 generated tokens, `ACK`, `end_turn`; frontier **353** |

Tool ID: `call_00_MczIkMd5LV5n9QbNQmkP9071`. The matching Tool Result was sent
unchanged in the next request. Both requests report `canonical_settled=true`,
`cache_published=true`, `aligned=true`, replay/repack **0**; the second restores
the first's complete paired frontier. Native wire content matches the observed
canonical response via the standard SSE parser. Original native `ping` events
are retained, distinct from Runtime heartbeat comments.

Evidence: [tool.sse](tool.sse), [timed HTTP chunks](tool-wire.json),
[continuation.sse](continuation.sse), requests/responses and status snapshots,
[native-settled.jsonl](native-settled.jsonl),
[native-acceptance.json](native-acceptance.json), [runtime log](native-runtime.log).
Final health is ready/no fatal error/no active or queued work. Graceful control
shutdown completed with exit code **0**; no forced termination.

Setup-only issues were preserved in [setup-error/](setup-error/): importing lazy
Transformers before origin admission initially failed startup (no model loaded);
the launcher now performs the unchanged real admission before attaching observers.
An initial test capture variable incorrectly used reserved `DS41F_*` naming and
was rejected as designed; it was renamed to `SSE_HEARTBEAT_WIRE_EVIDENCE` in the
test harness, with no Runtime admission/configuration change. Final runs pass.

## 5. Production and promotion disposition

[Before](production-before.json)/[after](production-after.json) hashes of **487**
Production Runtime/Recipe/package-metadata and entrypoint files are identical
(excluding disposable `.pyc`). No Production package install, configuration edit,
server startup, client timeout adjustment, push or merge was performed.

**Suitable for a separately authorized Production promotion**, within this focused
transport/native acceptance scope. No promotion is authorized or performed here.
Heartbeats fix observed HTTP inactivity; they do not defeat absolute request
limits, arbitrary network failures/proxy buffering, a blocked event loop or all
possible client timeout policies. None of those broader claims is needed for PASS.
