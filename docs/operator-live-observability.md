# Local operator TUI and live observability

Activate the provisioned canonical environment, with the usual explicit
`DS41F_CHECKPOINT` pointing at the external official model, then run **`ds41f`**.
No extra package is needed: the TUI uses Python's optional-at-import stdlib
`curses` module. A terminal is required; missing curses/non-TTY gives an actionable
error without affecting `ds41f start`, `inspect`, or `accept`. The frozen MTP
package inventory is not expanded for a UI framework.

Keys: **S** Start, **X** graceful Stop, **R** Restart, **C** Chat Start,
**V** Chat Stop, **Q** Quit, arrows scroll. Layout reflows on terminal resize.
NOW/current phase/phase elapsed/request elapsed precede performance and diagnostics.
Elapsed times refresh locally every 100ms, using the same machine's monotonic
clock; snapshot polling is 400ms. No spinner or synthetic activity.

## Process and control boundary

```
curses TUI → canonical ds41f start --profile mtp-serving-v1
                 → ds41f_mlx.serve / uvicorn / existing production backend
                 → in-process loopback control thread
           → existing ds41f_mlx.web (independent chat process, when requested)
```

Children use a new process session, no inherited terminal, and the same Python
interpreter/canonical entry points. Quit/crash never signals them. TUI child
stdout/stderr goes to `/dev/null`; use the headless launcher for full startup
logs if startup fails. TUI reports nonzero launcher exit as FAILED/CONFLICT.
Reopen discovers live status, not remembered PIDs or ownership modes.

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

`GET /ds41f/status` and empty-body `POST /ds41f/control/shutdown` exist only on
the respective loopback control socket. Shutdown sets the existing uvicorn
Server's `should_exit`, draining requests and invoking the existing shutdown /
shielded retirement path. Restart waits for control socket disappearance then
calls the canonical Start (retaining discovered host/port). Stop does not force
kill; very long requests/loading may take time. Restart times out after 120s
without starting a competing runtime. Model loading is owned by canonical
startup, **not** by telemetry/TUI; cold load takes about 100s on this machine.

The control listener binds literal IPv4 loopback, checks the kernel peer's
loopback address and exact numeric Host, and rejects Origin and nonempty
shutdown bodies. Forwarded headers confer no authority. LAN serving can still
bind `0.0.0.0`; there are no operator-control routes on that serving interface
(LAN shutdown returns 404). No new remote administration surface.

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

## Actual local acceptance

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
unchanged. Remaining scope: no remote/history/alerting infrastructure; no new
long-context campaign was run, so the earlier ~220K-context qualification is not
claimed as rerun here.
