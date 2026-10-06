# Single-user localhost Web application

## Status and scope

The everyday Chat surface and its separate current-source UX gate are documented
in [Chat interface](chat-interface.md). That document supersedes the original
management-oriented UI/model-control descriptions below. The following PASS is
the historical `133adc4` runtime/application semantics baseline, not an automatic
PASS for a changed frontend.

**PASS within the single-user localhost standard-OFF scope (`133adc4` baseline).** Current-source
HTTP, initial browser workflow, matched fresh runtime/model restore and operational
soak receipts all pass. See `artifacts/web-application/qualification.json`. No R1, release/runtime promotion,
MTP/DSpark/speculation, multi-user auth, internet serving or 200K/1M rerun is implied.
The [baseline investigation](web-client-production-gaps.md) and all earlier failed
attempts remain historical evidence, not retroactively successful qualification.

## Architecture and launch

```
browser HTML/CSS/JS + IndexedDB ordinary application history
 -> ds41f_mlx.web: nonblocking HTTP proxy / application-side tools
 -> public standard-OFF stateful HTTP API
 -> first-party worker-confined M11/M8/target execution and native persistence
```

Use the existing admitted runtime environment/checkpoint, as described in
[M19 launch](m19-local-web-client.md). The thin Web process needs FastAPI, uvicorn
and httpx; it does not load the model. No npm, framework or frontend build is used.

```sh
.venv/bin/python -m ds41f_mlx.serve --host 127.0.0.1 --port 8000
.venv/bin/python -m ds41f_mlx.web --host 127.0.0.1 --port 8080 --runtime-url http://127.0.0.1:8000
# Open http://127.0.0.1:8080/
```

HTTP streaming uses HTTPX demand-driven reads/yields. Blocking control/tool RPCs
run in a threadpool, not on the Web event loop. Generation remains exclusively
inside the public runtime; no client tokenization, alternate executor, mirrored
KV state, prompt replay or cache format is introduced. See
[live delivery](stateful-live-delivery.md) for transaction/Stop/backpressure rules.

## Everyday workflow

1. Create a session, enter text and Send. Real runtime deltas update provisional
   assistant content; final content is reconciled from canonical GET, not timers.
2. Continue normally. Full ordinary history is submitted for exact-prefix runtime
   admission, not re-prefilled or interpreted as execution state by the client.
3. Enable **Execute search / URL tools** for search or public-page retrieval.
   Provider, arguments/results, progress and durable source links are visible.
4. Attach original PNG/JPEG/WebP files, review previews, then Send. Later text
   turns keep the originals; add another image when needed.
5. Stop uses the admitted server request ID and safe worker settlement. A small
   committed but undelivered suffix can appear after canonical reconciliation.
   Interrupted messages are explicitly labelled; incomplete grammar may retire
   the session rather than invent a resumable ordinary history.
6. Use Header > New chat to close the current native conversation explicitly;
   failed close does not silently forget identity. There is no ordinary session list.
7. Settings > Recovery holds Save/Restore/Reconnect. Save only while idle and with
   no unresolved tools/request; it pairs native artifact/frontier and application history.
8. Reload reconnects and reconciles the current native session, never automatically
   repeating a user request or tool effect.
9. After a runtime restart, choose a matching saved state in Recovery and Restore.
   This creates a fresh native ID and retains original-byte ordinary history.

See [exact control mappings](chat-interface.md#model-controls-exact-mapping) for
Thinking Off/On, supported effort 50/75/100, direct temperature/top-p values,
Auto/output presets and the 32-round tool ceiling. No custom sampler is implemented.
Protocol reasoning and tool declarations freeze at the first turn; Tools Auto/Ask/Off
controls **application execution**, not historical declarations. Only published
`reasoning_content` appears in the collapsed Reasoning disclosure.

## History, uncertainty and recovery

`store.js` stores original ordinary messages, protocol settings, pending-request
journals, effect reservations and artifact-linked snapshots in IndexedDB. These
are application data, never proof of native execution. localStorage holds only
selection identity. Losing history/original images cannot be repaired from GET's
bounded last-turn record: continuation fails closed, without a fresh prefill.

Before a turn, the client durably freezes its body, base request count and a UUID
application correlation. Public streaming admission checks the optional expected
count, publishes the correlation on `last_turn`, and returns a server request ID.
Canonical reconciliation requires the exact next count and matching identity;
foreign or uncertain outcomes cannot silently enter history. An unadmitted request
can only be discarded explicitly after GET confirms no busy/mutated turn. There
is no automatic generation retry. Unknown/unrepresentable output remains diagnostic
or display-only, never tool permission or an ordinary resumable request history.

Before tool I/O, the browser stores a reservation and the living Web process
reserves `(native session, request_count, request_id)` in a bounded ledger. Calls
come from canonical GET with a positive official reconstruction certificate, not
browser-supplied arguments or partial streamed JSON. Concurrent/duplicate effect
POSTs share the completed result or fail closed on uncertainty. Reload uses the
**observation-only** result endpoint. After a Web restart a missing ledger cannot
prove whether I/O happened: the UI blocks reexecution and offers an explicit
unavailable-result action. This is not a claim of universal exactly-once effects
across process/storage loss or independent clients.

Before restore I/O, a known new standard-OFF native ID and `pendingRestore` journal
are durably stored. Losing the HTTP response does not cause another restore:
Reconcile queries that ID and verifies its frontier. Cold model loading can exceed
300 seconds; the restore transport timeout is 1800 seconds, not a loading-time
SLA. Closing a tab does not abort an unsafe in-flight native restore. Wait and
reconcile; do not guess a new model authority. Failed/missing journals, artifacts
or mismatched frontiers block continuation.

IndexedDB write/quota failure is surfaced and prevents an unjournaled action.
Keep the same browser origin/profile and preserve both native artifacts and their
application snapshots. Browser deletion/another profile does not recover original
image bytes from native KV. Saves are explicit user-retained data, not automatic
unbounded snapshots. The first 100 messages are rendered by default, with explicit
100-message expansion; this bounds default DOM work, not the stored conversation.

## Security and limits

- Loopback bind guard, TrustedHost, same-origin JSON mutations, cross-site rejection
  and restrictive CSP. This is not an authenticated LAN/internet/multi-user service.
- No `innerHTML`: safe DOM/text-node Markdown with raw HTML literal, bounded
  inline parsing, fenced code Copy, collapsed reasoning/tools and HTTP(S) links
  without credentials. Original inline images only; no network image loading or
  image recompression by the client.
- Existing `web_tools.py` guards are unchanged: public addresses/DNS only, pinned
  retrieval, every redirect revalidated, bounded redirects/timeout/size/content
  type/excerpt. Search/fetch are outside the runtime; no filesystem/shell tools.
- Runtime is final image/request/sampling authority. UI preflight mirrors the
  qualified envelope: PNG/JPEG/WebP, 16 MiB/file, 4 MP, aspect 1:2..2:1, at most
  four historical images and 8192 consumed multimodal tokens including reserved
  output. Animated/unsupported containers are rejected by runtime validation.
- Web bodies are capped at 100 MiB. Output UI budget no longer has a 4096 cap;
  Auto/4K/8K/16K/32K/Custom retain runtime admission authority (see Chat interface).
  Ledger retains bounded entries per live session and refuses more than 32 unresolved
  ledger session namespaces rather than evicting uncertainty. Four native live
  sessions remain the default runtime capacity. Successful native DELETE removes
  its closed record payload rather than retaining full history indefinitely.
- Diagnostic delivery history is 64 events, distinct from full live delivery and
  canonical response. Native context/response/application history naturally scale
  with explicitly admitted/user-retained data; no fixed total-memory claim is made.

## Qualification

Baseline collectors: `tools/qualify_stateful_live_http.py` and the historical
`tools/qualify_web_browser.py --phase initial|restored|soak` (real Chrome CDP).
The everyday UI collector is now `tools/qualify_chat_ux.py`; the old collector's
management-UI selectors are not an acceptance test for the new surface.
Receipts/logs are under `artifacts/stateful-live/` and `artifacts/web-application/`.
The final manifest pins production sources, current receipts and affected tests.
RUNNING/FAIL and older-source passes are not used to close the application gate.
Affected tests: **177 passed + 8 subtests**; renderer: **5 passed**. Save/restore
matched the same artifact and frontier **3035**, with both original image SHA256s
retained; restored continuation encoded zero historical images. The fresh restore
response was lost by actual browser reload, then reconciled by its known native ID.
Soak additionally passed eight Balanced turns, active browser reload, completed-tool
result loss/observation-only recovery, continuation and the 100-message DOM bound.

Retained failures include the original 300-second cold-restore timeout, omitted
collector `pending` key, navigation/promise observation races, and a temporary
HTTP iterator that disconnected before explicit cancel. These collector errors
were corrected without generation retries. An earlier initial receipt accepted an
Exa quota notice as a result; `initial-quota-false-pass.json` is **not acceptance**.
Normalization now rejects URL-less provider text and uses the existing bounded
failover/error-result path. Current initial acceptance contains real Python-document
source URLs. The final matched fresh campaign repeated the current save, not an
older snapshot accidentally selected by the first collector.

Representative acceptance includes >64-event full delivery/canonical equality,
slow consumption, protected Stop/disconnect/reuse, certified tool terminals,
Vision old-image encode counts, original-byte preservation, browser tool/search/
fetch workflow, save, actual fresh runtime/model restart, lost restore response
and known-ID observation, continuation, active browser reload and effect-result
loss/reload recovery. Unit gates cover saturated ASGI send/header failure,
retirement, effect dedupe/uncertainty, origin/host/body guards and safe rendering.
This does not certify arbitrary partial Unicode/DSML grammars, provider availability,
actual TCP-buffer saturation, maximum image sizes or full R1/long-context campaigns.
