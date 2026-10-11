# Local operator TUI and live observability

Activate the provisioned canonical environment, with the usual explicit
`DS41F_CHECKPOINT` pointing at the external official model, then run **`ds41f`**.
No extra package is needed: the TUI uses Python's optional-at-import stdlib
`curses` module. A terminal is required; missing curses/non-TTY gives an actionable
error without affecting `ds41f start`, `inspect`, or `accept`. The frozen MTP
package inventory is not expanded for a UI framework.

Keys: **S** Start, **X** graceful Stop, **R** Restart, **C** Chat Start,
**V** Chat Stop, **Q** Quit, **D** toggle Diagnostics, **?** toggle help.
On the dashboard **↑↓ / PgUp / PgDn** scroll the runtime log pane only;
scrolling upward pauses follow at a sequence anchor. **L / End** returns to latest.
The status/metrics above the pane never scroll. Arrows scroll secondary views.
The footer stays at the bottom. Controls remain available in every view.

The dashboard puts NOW/current phase/phase elapsed/request elapsed and queue counts
first, then performance and context, then compact alignment/frontier/replay/repack
health. Color is limited to runtime-state and active-phase labels and actual
warnings/errors. Headers and phase rails are not colored as a whole; healthy
alignment rows are monochrome, with only anomalous fields emphasized in red.
Ordinary labels and values use the default foreground, secondary labels are dim,
and important numeric values are bold. MTP acceptance and
alignment/frontier describe the last settled request; other metrics describe the
current request, or the last request when idle. Idle only shows WAITING FOR CLIENT
when the last settled finish reason was `tool_calls`.

At >=100 columns performance/context share two columns; at 70–99 they stack;
smaller terminals use compact labels without dropping metric fields when space
permits. Content width is bounded to 112 columns; stacked metric columns are
bounded to 56 so labels and values do not drift apart on resize. Available height
is spent on gaps between sections, then on the live runtime log pane below
state integrity. At 132×57 the pane has roughly 30–38 rows (depending on request
timings). It is omitted when fewer than three spare rows remain; it never
removes existing information to make room for logs. Observed request
timings use aligned cells rather than a long clipped line. The phase rail includes
RECEIVED, QUEUED, CHECKPOINT and settlement as well as inference phases; complete
phase tokens move to another row when necessary. It shows **only observed**
completed/current phases, not presumed prior or future execution, and is not an
ordered transition log (the projection retains durations, not transition history).
Metric and phase layouts do not use prose wrapping; long log records wrap within
the pane so settlement diagnostics are not silently clipped. Only physically short viewports drop timings and then
the rail before reducing the core to an instrument strip; all details remain in
Diagnostics. Raw cache, queue reasons, subphase timings and other low-level values
continue to live in Diagnostics.

Request elapsed times refresh locally every 100ms, using the same machine's
monotonic clock; snapshot polling is 400ms. The current authoritative snapshot
has no lifecycle-start timestamp: LOADING MODEL therefore displays elapsed `—`
(including rediscovery), never an observation-age timer presented as load time.
Diagnostics states that lifecycle elapsed is not published. No spinner,
percentage, progress bar or synthetic activity.

## Process and control boundary

```
curses TUI → canonical ds41f start --profile mtp-serving-v1
                 → ds41f_mlx.serve / uvicorn / existing production backend
                 → in-process loopback control thread
           → existing ds41f_mlx.web (independent chat process, when requested)
```

Children use a new process session, no inherited terminal, and the same Python
interpreter/canonical entry points. Quit/crash never signals them. TUI child
stdout/stderr still goes to `/dev/null`, but runtime-owned structured history is
retrieved separately; the TUI owns no child pipe. Full arbitrary exception details
remain available from the headless launcher. TUI reports nonzero launcher exit as
FAILED/CONFLICT. Reopen discovers live status and the recent log tail, not remembered
PIDs or ownership modes. Once the runtime exits its in-memory history is gone;
a TUI already connected retains its last observed tail until another runtime starts.

Kernel listening sockets are the only singleton authority:

* runtime/control: `127.0.0.1:18441` (all canonical runtime profiles)
* TUI guard: `127.0.0.1:18442`
* existing chat/control: `127.0.0.1:18443`

There is no daemon, PID/lock file, registry, database, telemetry persistence,
or multi-instance mode. Service socket is reserved before model loading,
including refusal against a pre-upgrade/unrelated listener. The TUI displays
CONFLICT for a default service port occupied without a control listener; it
cannot control a pre-upgrade server. No ATTACH mechanism is provided.
SIGKILL releases socket ownership automatically; `SO_REUSEADDR` allows immediate
restart after TCP TIME_WAIT, **without** `SO_REUSEPORT`.

`GET /ds41f/status`, runtime `GET /ds41f/logs`, and empty-body
`POST /ds41f/control/shutdown` exist only on
the respective loopback control socket. Shutdown sets the existing uvicorn
Server's `should_exit`, draining requests and invoking the existing shutdown /
shielded retirement path. After execution has retired, runtime control remains
available for a fixed 600ms to expose the final shutdown tail. It never waits
for a TUI/consumer acknowledgement and does not delay model settlement. Restart waits for control socket disappearance then
calls the canonical Start (retaining discovered host/port). Stop does not force
kill; very long requests/loading may take time. Restart times out after 120s
without starting a competing runtime. Model loading is owned by canonical
startup, **not** by telemetry/TUI; cold load takes about 100s on this machine.

The control listener binds literal IPv4 loopback, checks the kernel peer's
loopback address and exact numeric Host, and rejects Origin and nonempty
shutdown bodies. Forwarded headers confer no authority. LAN serving can still
bind `0.0.0.0`; there are no operator-control routes on that serving interface
(LAN shutdown returns 404). No new remote administration surface.

## Runtime logs: `ds41f.logs.v1`

The runtime process owns a **512-record** ring. Each record has occurrence-time
UTC `timestamp` (from Python's LogRecord, not polling time), increasing
`sequence`, `severity`, `component`, and a single-line `message` capped at 1024
characters. Sequence is ordering authority; wall-clock adjustments do not reset
it. A random `runtime_id` distinguishes restart from the same process continuing.
Logging/formatting failures are swallowed. Ring locks cover only bounded scalar
copies, never I/O, model execution or Scheduler/state work.

`GET /ds41f/logs?after=<sequence>&limit=<1..128>` returns detached `records`,
`oldest_sequence`, `latest_sequence`, `next_sequence`, `gap`, and `runtime_id`.
Without `after`, it returns the latest batch. Poll from `next_sequence` to catch
up; `gap` says older entries expired, not that execution failed. The same numeric
Host / loopback-peer / no-Origin restrictions as status apply. There is no log
route on the serving API and no chat history change. Status snapshots contain no
logs. Polling is bounded at 128 records per response; TUI retains at most 512
records, initially the latest 128. A paused anchor stays fixed as new records
arrive; expired anchors visibly report a history gap. INFO stays monochrome,
WARNING yellow, ERROR/CRITICAL red. Timestamp and sequence appear on each record;
wrapped continuations are indented. TUI component names omit the redundant
`ds41f.operator.` prefix.

A consumer may quit or hit its one-second HTTP timeout while control is preparing
an answer. Connection loss / send timeout during **headers or body** closes that
request quietly; it is not a runtime error and must not generate a socketserver
traceback. Serialization and unexpected I/O failures are not swallowed. A shutdown
already requested remains committed even if its HTTP acknowledgement cannot be sent.

Canonical startup/admission, lifecycle/model load, received/finished request and
existing MTP settlement output use one logging path. The existing settlement
message is moved, not copied; its exception value becomes the exception type.
Uvicorn's default handlers are disabled for the runtime so lifecycle messages are
not duplicated. Its audited lifecycle templates enter the ring; access paths,
queries and arbitrary third-party messages do not. Other WARNING/ERROR records
enter as content-free component/severity notices (with exception type when
available), never raw exception text or traceback. Prompt, response, reasoning,
tool arguments/results, tokens and file/source contents are not logged anew.

Headless output retains original third-party diagnostics, access output and
tracebacks with UTC timestamps; safe operator records also show sequence.
It uses a daemon writer and bounded 512-item queue (16 KiB per output item).
A broken/slow stderr cannot block inference: output may drop under saturation,
while the independent ring still retains its recent safe records. There is no
stdout/stderr interception, database, daemon, telemetry dependency or persistent
log history. Process exit gives headless output at most 200ms to drain; a slow
stream is never awaited indefinitely. `--print-config` remains machine-readable JSON.

## Snapshot: `ds41f.live.v1`

Root fields: `schema`, `sampled_at`, `sampled_mono`, `state`, `endpoint`,
`current_request`, `requests`, `active_requests`, `queued_requests`,
`last_request`, `cache`. Lifecycle: STARTING, LOADING, READY, STOPPING, FAILED.
TUI derives BUSY from observed in-flight requests; STOPPED/CONFLICT from discovery.

Each request contains server-generated opaque `request_id`,
`request_received_at`, `request_started_at`, `request_started_mono`,
`current_phase`, `phase_started_at`, `phase_started_mono`, `last_progress_at`,
`completed_phase_durations`, `metrics`. When applicable: `queue_entered_at`,
`queue_entered_mono`, `queue_reason`, `first_token_at`, `last_token_at`,
`finish_reason`, `finished_at`, `finished_mono`.

Projection phases (not an execution state machine):

```
RECEIVED → [QUEUED/http_preparation] → ENCODING
         → QUEUED/scheduler_worker → CACHE_LOOKUP ↔ RESTORING
         → SUFFIX_APPEND ↔ CHECKPOINT_CAPTURE → P5_HANDOFF
         → DECODING → SETTLING → DONE / ERROR
```

ENCODING includes ordinary recipe conversion/render/encode; their completed
subtimings remain separate. RECEIVED includes bounded body receipt. Queue age
includes waiting for the existing HTTP preparation lease or worker/Scheduler;
queue reason identifies which boundary. Counts include admitted preparation
requests, not just Scheduler.running/waiting. The active generation is prioritized
in NOW while other queued requests and oldest queue age remain visible.

Phase transition durations accumulate per request in milliseconds (including
repeated append/capture or lookup/restore segments). Existing authoritative trace
measurements are published separately in seconds: recipe convert/render/encode,
lookup, paired restore, prompt capture, suffix append, P5, first native/canonical
decode call and total TTFT. They need not equal projection durations: projection
also includes boundary and worker overhead. TTFT retains the existing handler
arrival-to-first-canonical definition; request elapsed starts earlier at HTTP
boundary arrival and includes preparation-lease waiting.

IDLE explicitly means no request waiting inside this server. Last `tool_calls`
finish plus no active request shows WAITING FOR CLIENT. This says nothing about
whether a client tool is running, or whether another continuation will arrive.
No client behavior is inferred.

## Metric definitions / privacy

* Context: actual encoded prompt / admitted context envelope.
* Cache hit: `cached_tokens / prompt_tokens`, per request; cached/new counts
  displayed separately. The paged cache's query hit rate is **not** this metric.
* New: prompt minus cached. NEW PREFILL: `new - 1` (nonnegative), excluding P5's
  terminal holdout. Prefill tok/s divides this actual append work by
  `suffix_append_s`, excluding checkpoint capture. A full-prefix hit never
  claims the full prompt was newly prefilled.
* Decode tok/s: generated canonical token count / accumulated native decode-call
  seconds (not end-to-end delivery throughput).
* MTP accept: sum(depth_accepted) / sum(depth_drafted), last **settled** request
  only; no mutable native state is inspected by the TUI/control thread.
* Replay/repack: confirmed request trace counters. Frontier/alignment: last
  settled canonical frontier and equality of all target/DSpark offsets. Unknown
  values are `—`, never optimistic zeros/alignment.
* Cache: scalar retained paged/checkpoint summary, copied on the worker after
  settlement; not a live native inspection.

The worker/HTTP authority publishes explicit allowlisted scalar counts/timings;
control returns a detached deep copy. No prompt, response, reasoning, tool args,
results, tokens, file/source contents, native objects or exception text enters
telemetry. Diagnostic update failures are swallowed, never inference decisions.
Only current requests and one last completion are retained in process memory.

## Runtime log acceptance (installed package)

[Focused receipt](operator-runtime-log-acceptance.json) records the installed
wheel SHA and actual 132×57 curses acceptance. Final-package startup/admission,
model load, READY, a real HTTP request, aligned settlement (frontier 25,
replay/repack 0) and graceful shutdown appeared in the dashboard with occurrence
UTC timestamps and sequences 1–16. Quit/Reopen rediscovered the same runtime ID
and settlement tail; arrows paused follow and L restored it. Resize to 80×28,
40×10 and back preserved the footer/core (40×10 omits logs). Long settlement
records wrapped, and final shutdown/retirement logs remained visible after STOPPED.
Raw ANSI was decoded including ncurses' CSI scroll-region operations and terminal
cells inspected. The prompt sentinel was absent from the log API. Headless
singleton conflict and admission refusal retained timestamped diagnostics and
exit 2. Runtime and TUI were gracefully stopped after acceptance.

**59 affected tests pass**, including bounded retrieval/gaps, privacy, reconnect
and restart identity, paused anchors, layout degradation, broken logging and a
stalled output sink. Model/dependency/native identities and execution semantics
are unchanged. The new wheel was constructed with the existing one-way package
builder and installed non-editably; no checkout execution or validation bypass
was used for runtime acceptance. No OFF/long-context/general-release campaign or
release promotion is claimed. Local raw captures/driver/wheel are retained under
`artifacts/operator-runtime-logs/`; the concise receipt is versioned above.

### Control-disconnect follow-up

The reported `BrokenPipeError` on status response delivery is covered by explicit
transport-loss handling shared by status, logs and shutdown replies. **50** focused
operator tests pass; **11** disconnect cases also pass against the newly installed
non-editable wheel from `/tmp`, using isolated disposable control listeners (no
checkout runtime or model load). Real TCP resets before response delivery produce
no socketserver error, subsequent polling succeeds, and shutdown remains committed.
Header/body BrokenPipe, reset, abort and timeout paths are tested; invalid JSON
scalars and unrelated OSError still fail visibly. Wheel SHA:
`01c0ffab35e32763144ae22ad5d319a4ce4c00a17ac447dcce179815087b4636`.
This narrowly scoped follow-up does not claim a rerun of the model/TUI timeline
above or additional model qualification.

## Dashboard presentation acceptance (historical, before runtime logs)

The presentation redesign leaves the projection schema, runtime, launcher and
lifecycle methods unchanged. Focused tests cover states, queues, honest unknowns,
metric definitions, content exclusion, observed-only rails, layout breakpoints,
height degradation, integrity anomalies, diagnostics and help. A real curses PTY
passes exercised 110×32, 80×28, 60×20/26 and 40×10 dashboard sizes, Diagnostics resize
and scrolling, help and Quit. Captured terminal cells were inspected for hierarchy,
column alignment, clipping and the stable footer. This pass used disposable
content-free Projection fixtures, not model inference; the loaded-runtime/control
acceptance below is historical and is not claimed as rerun for this UI-only change.

## Actual local acceptance (pre-redesign runtime/control)

Verified using the installed `ds41f` executable through a real PTY and real
OpenAI-compatible requests on the provisioned MTP environment:

* TUI/second-TUI refusal; Start; STARTING/LOADING/READY; second runtime refusal
  even on a different service port.
* Received/current phases and locally advancing elapsed; completed timings;
  concurrent-request queue; IDLE; tool_calls/client-wait and paired continuation.
* Repeated prompt: 19/21 cached, 128 generated; replay/repack 0, frontier 149,
  target/DSpark aligned; MTP accepted 94/95 drafts.
* Tool continuation: 330/351 cached (94.0%), 21 new / 20 actual prefill tokens,
  TTFT ~401ms, frontier 354, aligned, replay/repack 0.
* Quit preserves runtime; reopen rediscovers it; graceful Stop and Restart;
  actual loaded-runtime SIGKILL followed immediately by canonical headless Start
  succeeds without cleanup; LAN `192.168.68.56` cannot connect to control and
  gets 404 for serving-port shutdown.
* Existing chat process Start, `/api/status`, graceful Stop.
* Final-source smoke also passed: TUI Start, STARTING/LOADING/READY/STOPPING,
  real repeated request with 25/27 cached tokens, TTFT 183ms, frontier 59,
  aligned, replay/repack 0; sentinel prompt/response absent from snapshot;
  Quit/Reopen and TUI graceful Stop.

Small warmed check (two repeated 128-token requests per condition): no consumer
~43.03 tok/s vs TUI + **additional 50Hz** snapshot sampling ~42.89 tok/s (~0.33%
change). Warm repeat TTFT 165.8ms vs 167.4ms; first repeats 177.2ms vs 190.6ms.
No sustained material decode penalty observed. This is a small operational
check, not a statistically powered benchmark or proof of zero overhead.

Regression sets pass: **219** in the admitted MTP environment plus **16**
standard-OFF capacity tests in `.venv` (their native recipe pins differ). Runtime source admission inventory
and local installation seal were updated only for these reviewed source changes;
model/dependency/native identities, default profile and release promotion remain
unchanged. Remaining scope: no remote/persistent-history/alerting infrastructure; no new
long-context campaign was run, so the earlier ~220K-context qualification is not
claimed as rerun here.
