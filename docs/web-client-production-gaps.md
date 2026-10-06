# Historical baseline: Web delivery investigation and partial hardening

This is the retained **baseline investigation**, not the current application status.
The original findings and receipts below are not rewritten as passes. Current
architecture, operations and qualification are tracked in
[Web application](web-application.md) and [live delivery](stateful-live-delivery.md).

**Status at the investigation stage: NOT COMPLETE / NOT QUALIFIED as a production-facing Web application.**
This is not a closeout or authorization to proceed to R1. Baseline is master
`2fe40a9af2da8cbeab8a9543a6925a91c4452e6d`.

## Actual boundary

```
browser HTML/CSS/JS
 -> ds41f_mlx.web (FastAPI, synchronous urllib RuntimeClient, client ToolRegistry)
 -> public standard-OFF HTTP serving backend
 -> first-party single-worker M11/M8/target execution authority
```

Search/fetch remain application-side tools. The model receives ordinary tool
messages in the same session. Fetch's public DNS/address/redirect rejection is
unchanged. No model executor, KV representation, runtime selector, session owner,
authentication, npm/build dependency or persistence authority was added.

The existing browser localStorage transcript is application history, **not proof
of the committed runtime frontier**. The current client does not reconcile it.
The public record contains `request_count`, `last_turn.response_json` and runtime
diagnostics, not an unlimited full ordinary conversation or original image bytes.
GET cannot reconstruct arbitrary lost browser history. Restore loads native idle
state and image identities, not the historical inline image payloads needed in a
subsequent ordinary request. Existing artifact persistence must be retained; a
future UI needs an application-history snapshot associated with each saved
artifact, without treating that snapshot as executable model state.

## Real-model baseline evidence

`tools/probe_web_stateful_delivery.py` exercises an already launched real runtime
and Web server, with no generation retries, tools or prompt rebuilds. Evidence:

- `artifacts/web-client-investigation/baseline.json`: complete observation receipt;
- `summary.json`: small derived metrics;
- `collector-first-fail.json` / `.log`: retained collector error, **not a model
  failure or a qualification pass**. The original collector used `response`
  rather than the actual `last_turn.response_json` key. It cleaned up its test
  sessions. After correcting the collector, a distinct run produced the baseline;
- `client-tests.txt`, `render-tests.txt`, `affected-tests.txt`: bounded checks.

Reproduce using the repository's admitted environment, not a new model selector:

```sh
.venv/bin/python -m ds41f_mlx.serve --port 8000
.venv/bin/python -m ds41f_mlx.web --port 8080
.venv/bin/python tools/probe_web_stateful_delivery.py
node --test tests/web_render.test.cjs  # optional existing Node, no npm dependency
.venv/bin/python -m pytest -q tests/test_m19_web_client.py
```

The actual resident-model run observed:

| Observation | Result |
| --- | --- |
| Real `/api/chat` text through Web surface | 2.960 s; `application/json`, non-streaming |
| Public stateful `stream:true`, 128 output budget | first headers at 6.467 s |
| GET immediately after receiving SSE headers | `idle`, `request_count=1`, completed `last_turn` |
| SSE reconstruction vs canonical response | **different**: canonical starts `1`, delivered text starts around `34` |
| Frames delivered | 64 retained protocol events + `[DONE]` |
| Socket close after GET observed `busy`, before response headers | idle after 6.455 s; natural `length`, all 128 tokens consumed |
| That turn replay / repack counts | 0 / 0 |
| Test session cleanup | explicitly DELETE'd; no replay/retry |

This is an HTTP-level **baseline defect observation**, not a browser operational
qualification. No claim is made for the requested 12-step workflow, tool effects,
Vision history/reload, persistence, fresh server restart, restore or long-session
soak through a completed application. The first probe incurred the actual cold
load; the final timings above are warm. The background servers used for this
investigation were stopped afterwards.

## Integration defects found

1. `serving/server.py:session_chat` awaits `run_stateful_chat_turn` before creating
   a StreamingResponse. `tool_boundary_session.py:run_current_assistant_turn`
   returns only after natural/tool/length stop and idle commit. Thus SSE framing
   is present but **generation is not streamed** on this public stateful route.
   Stateless streaming and internal MTP recovery are not substitutes.
2. `M11AssistantTurn.stream_events` keeps `events[-64:]`; the same diagnostic tail
   is used as the public SSE response. Real output exceeding the tail loses its
   beginning. Browser-side streaming alone would expose silent truncation.
3. `_call` drains an already running worker operation on cancellation. In the
   stateful route there is no active response-body iterator before generation
   completes. A socket close does not mean Stop at the currently visible token.
   Our real pre-header disconnect consumed the entire natural length-limited
   turn. Do not claim the internal MTP fence/certificate contract on this route.
4. Web async routes call blocking urllib and tool functions directly. While the
   model HTTP request is pending, the Web event loop cannot normally service
   status/Stop requests. Merely adding a Stop button does not fix this.
5. Browser reload blindly trusts one localStorage history; restore clears it.
   Ambiguous transport outcomes can lose committed tool/user/assistant history.
6. Tool duplicate checks cover only a single execute_calls list. No cross-request
   effect reservation/reconnect protocol is present. Automatically retrying the
   current Web chat route can duplicate generation and tool effects.
7. Original source-link HTML interpolates provider-controlled URL into a quoted
   attribute with an escaping function that does not escape quotes, and does not
   reject active URL schemes. Tool activity was also immediately erased by render.

Items 1–6 remain unresolved. Item 7 and some rendering shortcomings are fixed in
this partial work; none of these observations invalidate the qualified numerical
text/Vision execution core.

## Partial fixes, NOT a completed application

`web_static/render.js` renders model/provider text with DOM text nodes only. It
uses no `innerHTML`, interprets only fenced code, separates reasoning and tool
calls/results with disclosure controls, and displays sources again from JSON tool
messages after rerender/reload. Source links accept absolute HTTP(S) without
credentials; malicious schemes and quote injection cannot create script/attributes.
Images are displayed only from explicit inline PNG/JPEG/WebP base64 content parts,
never external transcript URLs, without re-encoding the original bytes. This is
**display support, not attachment/admission/Vision workflow implementation**.
Runtime remains image validation authority.

The legacy app now renders the latest 100 messages by default with explicit
100-message expansion. This bounds default DOM construction, **not** total stored
transcript growth or the complete application's long-session lifecycle. Tool
activity is no longer erased immediately. A failed DELETE no longer silently
forgets the session identity; restore reports that ordinary history was not loaded.

Checks: 4 dependency-free renderer tests using the existing Node installation;
11 existing Web/tool tests; 32 affected session/Vision tests plus 8 subtests pass.
Affected tests were M8, M11, stateful request policy, multimodal input/history,
cancellation and restore ownership. No runtime source changed; these are bounded
unit checks, not a new text/Vision real-model qualification or full R1.

## Required next work / decision

A production client must first have a public standard-OFF stateful **delivery**
contract that supports real incremental output and a defined cancellation outcome,
without moving execution/KV ownership out of the worker. Diagnostic tail retention
must be separate from delivery. The current qualified commit-before-protocol-event
policy cannot be silently replaced by speculative incremental events. Define and
qualify the affected transport/commit seam explicitly before advertising streaming
or active Stop. Do not fake streaming with timers, use stateless prompt replay,
borrow MTP sequence fencing, or build a second Web generation authority.

Then implement nonblocking Web transport; a frozen application request/reconcile
flow using runtime counts/last_turn with fail-closed uncertainty; durable tool
reservations with no automatic execution after uncertain restart; multiple session
selection; original-byte image attachment/history snapshots and bounded validation;
artifact-linked save/restore UI; reload/restart handling; and browser plus real-model
12-step operational evidence. Missing/mismatched application history must disable
continuation rather than silently rebuild the prompt. A saved history snapshot
must correspond to the exact saved runtime frontier, not the later live transcript.

No completion commit was made: the user's completion condition is not met.
R1 re-verification remains blocked on this Web application work. No full R1,
runtime promotion, release packaging, MTP/DSpark/speculation, multi-user/auth,
internet serving, frontend framework or 200K/1M rerun was attempted.
