# Local operator TUI and live observability

This page owns **local service control, content-free logs/snapshots and metric
interpretation**. [Operations](operations.md) owns installed environment/startup;
[operator console](operator-chat.md) owns current keys/settings. Web Chat is the sole
conversation frontend; TUI has no editor or session ownership.
Observability is disposable projection, never execution, cache, settlement or
application-history authority. Historical acceptance below is not current HEAD
qualification or a deployment SLA.

## Dashboard

Run `ds41f` in the matching installed environment with configured checkpoint.
The stdlib-curses terminal is optional-at-import; non-TTY/missing curses gives an
actionable error without breaking headless start/inspect/accept. No UI framework
or new model executor is added.

**R** Start/Stop LLM Runtime, **W** Start/Stop Web Chat, **O** Open Web Chat,
**S** Settings, **Q** Quit TUI, **D** Diagnostics, **?** Help.
Dashboard arrows/PgUp/PgDn scroll the log pane; **L/End** returns to latest.
Metrics stay fixed and footer remains visible. Scroll-up pauses at a sequence
anchor; expired records show a gap. No F-key or Chat Ctrl-key mapping remains.

NOW/observed phase/elapsed/queues lead, then performance/context and last-settled
integrity. Phase rails show only observed completed/current phases, not a presumed
execution sequence or transition history. Raw cache/queue/subtimings remain in
Diagnostics. Healthy rows stay neutral; state/phase/warnings distinguish anomalies.
Responsive columns, compact labels and bounded widths preserve the core; logs
use spare height and disappear before displacing core information. Long log records
wrap. Short viewports degrade timings/rail honestly, not synthesize metrics.

Snapshot polling is 400ms; local elapsed refresh uses same-host monotonic time.
No lifecycle-start timestamp is published: LOADING elapsed is `—`, not observation
age passed off as load time. No synthetic percentage, spinner or progress bar.
WAITING FOR CLIENT means idle after last `tool_calls`, not proof of tool activity.

## Process and control boundary

```text
TUI -> canonical ops start --profile mtp-serving-v1 -> runtime/backend
                                                   -> loopback control
    -> independent Web Chat launcher when requested
```

Dashboard Runtime Start selects `mtp-serving-v1`, not a profile picker. Explicit
OFF startup remains in operations; observing a ready OFF service does not change
that launcher selection.

TUI Web Chat Start (W) defaults to `0.0.0.0:8080` with
`--allow-private-lan`; `DS41F_CHAT_BIND=loopback` selects localhost-only, while
`lan` is the default. `DS41F_WEB_PORT` selects the Chat port. This does not change
LLM Runtime bind or direct CLI Web defaults. S Settings changes the next Web
launch's bind/port and the next Runtime launch's Default Reasoning (initially Max).
T in Settings cycles and persists the reasoning preference; Web bind/port changes
remain process-local. Explicit client reasoning controls override the default.
For a LAN OpenCode client, set `DS41F_HOST=0.0.0.0` (or a private interface) before
TUI launch; Runtime otherwise binds to loopback. Running services are not
reconfigured. See [operations](operations.md) for configuration precedence.
TUI does not implement
TLS restart or certificate management; use the explicit Web CLI for HTTPS.
The tall dashboard and Diagnostics show Chat Listen, Local and best-effort IPv4
LAN connection URLs, plus `LAN access enabled — trusted local network only`.
`O` requests the local browser URL nonblocking via macOS open; `W` stops Web Chat. LAN discovery uses hostname
addresses and a local UDP route selection without transmitting packets; an
unavailable address is reported, not fabricated. It is not exhaustive interface
or reachability qualification. TLS and authentication are not added by LAN mode.

Child processes use a new process session, no inherited terminal and canonical
entry points. TUI owns no stdout/stderr pipe; Quit/crash does not signal services.
TUI owns no conversation to retire on Quit. Reopen discovers control
state/recent runtime logs, not saved PIDs or execution ownership.

Kernel listening sockets are singleton authority: runtime `127.0.0.1:18441`,
TUI guard `:18442`, Chat control `:18443`. No daemon, PID/lock file, database,
registry or multi-instance/ATTACH mode. Service bind is reserved before model load;
an occupied service port without control is conflict, not permission to take it.
SO_REUSEADDR allows restart after TIME_WAIT; no SO_REUSEPORT. SIGKILL releasing a
socket does not restore native/session/effect state.

Loopback-only `GET /ds41f/status`, runtime `GET /ds41f/logs` and empty-body
`POST /ds41f/control/shutdown` are not serving-interface routes. Kernel peer must
be loopback, Host exact numeric authority, Origin absent; forwarded headers grant
nothing. Trusted-LAN runtime bind does not expose remote operator administration.

Shutdown sets existing uvicorn `should_exit` and drains shielded retirement.
Control keeps final log tail for 600ms **after execution retirement**, without
consumer ACK or delaying settlement. Stop never force-kills. Stop waits for control disappearance; after 120s it
reports pending shutdown rather than launching a competing process. A later
Start uses explicit next-launch settings. Loading remains canonical
startup authority, not telemetry; historical load times are not an SLA.

## Runtime logs: `ds41f.logs.v1`

Runtime owns a **512-record** in-memory ring. Fields are occurrence `timestamp`,
monotonic-increasing `sequence`, severity, component and single-line message capped
at **1024 characters**. Occurrence epoch is rendered in the host's local timezone,
not polling time, as `YYYY-MM-DD HH:MM:SS.mmm ZONE` (for example,
`2026-10-13 00:02:17.381 JST`). Set `TZ=Asia/Tokyo` at process launch to
explicitly select Japan time. The API retains numeric epoch timestamps.
Sequence orders records despite wall-clock changes; `runtime_id` distinguishes
restart. Locks copy only bounded scalar records, never model/native work or I/O.

`GET /ds41f/logs?after=<sequence>&limit=<1..128>` returns schema/runtime ID,
records, oldest/latest/next sequence and `gap`. Without cursor, return latest batch;
continue from next sequence. Gap means expiry, not execution failure. TUI initially
fetches latest 128 and retains at most 512; paused anchors remain fixed.
Snapshots contain no logs; no serving log route or persistent history is added.
Runtime exit loses its ring; a connected TUI may retain its observed tail until
another runtime ID appears.

Safe operator startup/lifecycle/request/settlement and audited uvicorn lifecycle
messages enter one ring path. Access URLs/queries and arbitrary third-party text
do not. Other warning/error entries carry content-free component/severity notices
and exception type, not raw exceptions/tracebacks. Prompts, responses, reasoning,
tools, token IDs and source/file contents are not newly logged there.

The TUI log pane uses a fixed `level → timestamp → source → message` order:

```text
RUNTIME LOG · FOLLOW
INFO    2026-10-13 00:02:17.381 JST  server  Application startup complete.
```

Display-only source aliases are `uvicorn.error` → `server` and
`uvicorn.access` → `access`; `ds41f.operator.` is stripped from operator sources.
`uvicorn.error` is a logger name, not evidence of ERROR severity. Access aliases
apply only to records admitted by the safe-ring policy above, not raw access URLs.
The API retains original component names. TUI hides `#sequence` in both FOLLOW
and PAUSED views; internal sequence cursors, paused anchors and gap detection are
unchanged. Level text is green for INFO, yellow for WARNING, red for ERROR/CRITICAL
and cyan for DEBUG; source text is dim and messages remain neutral. Styling is
reapplied semantically after wrapping, never stored as embedded ANSI in the ring
or log API. Colours require a colour-capable terminal.

Headless output preserves original diagnostics/access/tracebacks with the same
local-time stamps at the start of the message body, after severity/component;
original logger names and runtime sequence numbers remain in safe headless log
lines (`#sequence` at the end), unlike the TUI presentation. Web Chat keeps
Uvicorn's native level, request-line and HTTP-status colours, with the timestamp
after the level prefix: `INFO:     2026-10-13 00:02:17.381 JST  …`.
Colours are enabled for the respective terminal stream, not redirected files.
Traceback continuation lines remain standard Python tracebacks.
Installed TUI packages must also be updated through the normal package delivery
workflow: restarting an older installed wheel does not pick up checkout formatting
changes, even if Web Chat is launched from the checkout.
A daemon writer uses a bounded 512-item queue and 16 KiB/item ceiling; slow/broken
stderr cannot block inference and may drop output. The independent safe ring is
not that queue. Exit drains for at most 200ms; `--print-config` stays JSON.
No stdout interception, telemetry dependency or persistent log service.

Header/body connection loss or timeout closes a control reply quietly. It does
not undo an already committed shutdown or become a runtime failure. Serialization
and unrelated unexpected I/O failures remain visible. Logging/projection failures
never determine execution admission/settlement.

## Snapshot: `ds41f.live.v1`

Root: schema, sampled wall/monotonic times, lifecycle state, endpoint, current
request, requests, active/queued counts, last request and cache summary.
Lifecycle is STARTING/LOADING/READY/STOPPING/FAILED; TUI derives BUSY and discovery
STOPPED/CONFLICT, not a new native state machine.

Request: opaque generated ID, received/started wall/monotonic times, current phase,
phase-start times, last progress, accumulated completed-phase durations and metrics.
When applicable: queue-entered times/reason, first/last token times, finish reason
and finished times. Current requests plus one last completion are retained, not
an unlimited outcome archive.

```text
RECEIVED -> [QUEUED/http_preparation] -> ENCODING
  -> QUEUED/scheduler_worker -> CACHE_LOOKUP <-> RESTORING
  -> SUFFIX_APPEND <-> CHECKPOINT_CAPTURE -> P5_HANDOFF
  -> DECODING -> SETTLING -> DONE / ERROR
```

This is projection, not execution-transition authority. Received includes body
receipt; encoding includes recipe conversion/render/encode. Counts include HTTP
preparation, not only native Scheduler work. NOW prioritizes generation while
other queues/oldest age stay visible. Queue reason distinguishes preparation lease
from worker/Scheduler waiting.

Phase durations accumulate milliseconds across repeated segments. Authoritative
recipe/lookup/restore/capture/append/P5/decode/TTFT trace seconds are separate and
need not equal projection durations; boundary/worker overhead differs. TTFT retains
handler-entry-to-first-canonical definition; request elapsed begins at earlier
HTTP boundary arrival including preparation-lease wait. Do not sum nested timers.
Idle means no waiting server request; it predicts no external client continuation.

## Metric definitions / privacy

| Metric | Interpretation |
| --- | --- |
| Context | Actual encoded prompt / admitted context envelope |
| Cache hit | Actual paired cached/prompt count per request, not paged query hit rate |
| New / new prefill | Prompt minus cached / nonnegative new minus P5 terminal holdout |
| Prefill tok/s | Actual append work / suffix append seconds, excluding checkpoint capture |
| Decode tok/s | Canonical generated count / accumulated native decode-call seconds, not delivery throughput |
| MTP accept | Sum accepted drafts / sum drafted, last settled request only |
| Frontier/alignment | Last settled canonical frontier and all target/DSpark equality |
| Replay/repack | Confirmed trace counters, not optimistic defaults |
| Cache summary | Worker-copied scalar retention stats after settlement, not live native inspection |

Unknown fields are `—`, never guessed zero/alignment. No control/TUI thread reads
mutable native state. Worker/HTTP allowlisted scalars publish; control returns a
detached deep copy. No content, native objects or raw exception text enters the
projection. See [performance](performance.md) for comparison/measurement scope.

## Runtime log acceptance (installed package)

[Runtime log receipt](operator-runtime-log-acceptance.json) records its installed
wheel, actual PTY/model/request/shutdown timeline, privacy and rediscovery checks.
The [earlier observability edition](archive/editorial-baseline/operator-live-observability.md.txt)
retains all exact sequences/frontiers/rates, regression counts, terminal sizes,
source/seal steps, failure handling and prior acceptance narratives. These are
historical inputs, not fresh later-package approval.

### Control-disconnect follow-up

The original edition retains separate installed-wheel reset/header/body/timeout
checks and SHA. Those disposable control-listener tests did not load a model or
rerun the earlier TUI/model timeline. Current transport semantics are defined above;
compilation repeats no installed-wheel campaign.

## Dashboard presentation acceptance (historical, before runtime logs)

Original fixture/PTY layout and scroll tests used content-free projection data,
not inference. They do not turn presentation redesign into model qualification.
Exact geometry and observations remain in the earlier edition.

## Actual local acceptance (pre-redesign runtime/control)

Original loaded-runtime/control/queue/tool/stop/restart/SIGKILL observations,
small warmed overhead comparison and source-seal counts remain unchanged in the
edition. The finite overhead check is not statistically powered or zero-cost proof.
No remote administration, persistence, long-context rerun or release promotion is
inferred. Local-seal refresh wording is historical; installed admission now governs.
