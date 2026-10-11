# Operator Chat mode

Text-first frontend on the existing production Chat Server application contract.
See [acceptance status](operator-chat-status.md). Backend session execution and
physical caches remain owned by the runtime.

Run `ds41f` in the provisioned canonical environment. Dashboard remains the
runtime-metrics/live-log home. **C** enters Chat; **Ctrl-T** returns to Dashboard.
The same conversation/draft survive mode switches and terminal resize. Logs and
status keep polling while the separate asynchronous application action streams.

Chat keys (`^` in the footer means Ctrl):

* **F2 / F3**: canonical Chat Server Start / graceful Stop. F3 retires this TUI's
  native conversation first; it does not close Browser sessions or stop runtime.
* **Ctrl-N / Ctrl-W**: new chat (closes the previous session) / close chat.
* **Ctrl-G**: send; **Enter**: newline. No Shift-Enter assumption. IME composition
  confirmation/ordinary pasted newlines cannot send. Unicode arrives via
  `get_wch`, not byte-wise dashboard `getch`. Bracketed paste inserts text and
  suppresses all action keys until its end marker. Terminal mode is restored on
  exit. Arrow keys edit; Home/End move within the logical line; Ctrl-A goes to
  line start; Ctrl-U deletes to line start. Composer wraps by terminal cells.
* **Ctrl-O**: generation settings. Tab/Shift-Tab selects reasoning, temperature,
  top-p, top-k, min-p, penalties and seed; Ctrl-G applies the edited value and
  Ctrl-O returns without changing the conversation draft. Numbers use the
  advertised runtime type/range; seed `null` means no explicit seed.
* **Ctrl-C**: Stop at the authoritative safe boundary; wait for reconciliation.
* **Ctrl-R**: observe/reconcile canonical session/request/tool outcome. It never
  submits another generation or repeats a tool effect.
* **Ctrl-E**: explicitly continue a pending tool response.
* **PgUp / PgDn**: scroll conversation and pause latest follow. **Ctrl-L**: latest.
  Streaming follows only while follow is enabled. Scroll remains paused after
  settlement; resizing reflows conversation without clearing history/input.
* **Ctrl-Q**: quit TUI, retire its session, leave both services running.

New Chat fetches runtime capability/defaults explicitly through `/api/status`.
Both production MTP and standard-OFF require `generation-policy-v1`: all eight
sampling defaults/ranges and reasoning efforts come from the runtime. There is
no legacy sampler fallback. Unknown versions/capabilities fail closed. Settings
remain application preferences across new chats and Dashboard/Chat changes.
Sampling may change between user turns. Reasoning alone locks at submission
(including unknown outcomes); Ctrl-N starts a new conversation to change it.
A user workflow snapshots its policy for tools and explicit continuation; Stop
and canonical GET do not reset it. Tools Auto executes on Chat Server.
Tool declarations are frozen when the session is created; generic call/argument/result messages and any reasoning_content are
rendered as plain text. Image/PDF acquisition declarations are omitted in
production text scope; no attachment UI exists. Server tools alone execute effects
and fit their results.
Tool errors/budget errors pause continuation; Ctrl-E is explicit continuation,
not automatic effect retry. Tools Auto uses the same 128-round runaway breaker.
No terminal content is interpreted as ANSI/terminal control sequences.

## Authority and interruption

`chat_application.py` uses only Chat Server `/api/status`, `/api/session`,
`/api/budget`, `/api/stream`, `/api/tools`, `/api/tools/result`, and session cancel
and DELETE. It does not use `/api/chat`, `StatefulToolChatClient.run_turn`, or a
runtime chat protocol. Browser and TUI are independent, equal application
frontends. The Python client is deliberately text-only. Browser also consumes
the same `generation-policy-v1` authority and workflow/session ownership. Browser
sampling controls remain request-owned; reasoning locks for the conversation.
Production images and persist/restore are disabled. Unversioned Browser generation
journals cannot silently acquire new defaults. Standard-OFF image/persistence
features remain available under their existing runtime contracts.

Each generation freezes history, expected request count and a fresh application
request UUID. The admitted native request identity is retained. SSE output is
explicitly **UNSETTLED**, never appended as an ordinary assistant message. Every
stream termination, including normal EOF and Stop, uses canonical GET settlement
and verifies both identities/count before replacing the live preview. Busy,
missing session, mismatched count/identity and missing canonical response disable
continuation. Unknown unadmitted requests require further observation or close /
new chat; they are never automatically retried or discarded.

Tool reservation is recorded before POST; uncertain outcomes are recovered by
observation of the server ledger. A missing result leaves the reservation blocked
and is never re-executed. A recovered result is appended once. Only an explicit
continuation may advance after a tool error/uncertainty has been resolved.

## Ownership boundary

One text conversation/history and draft live **only in this TUI process**. No
IndexedDB sharing, persistence, native restore, prompt replay or new store exists.
New/close/normal quit issue native session DELETE through Chat Server. Normal quit
requests Stop, waits for the current action/reconciliation, and retires only the
known TUI session, never either service. Lifecycle work is not an inference lock.

A stopped/restarted Chat Server is not an empty conversation: its state is shown
in the header, and unavailable session/request/effect outcomes remain blocked.
Restarting does not resubmit generation/effects. Reconcile explicitly, or close /
new chat. A server restart may have lost its tool ledger even when runtime still
has the native session: unavailable effects remain blocked.

No frontend can guarantee native retirement while its backend is unavailable,
or after SIGKILL. Quit attempts bounded retirement and reports the known session
ID and **retirement unconfirmed; no retry** on failure. There is no background
retry or persistent recovery owner. Keep the backend available for confirmed
close before quitting; after backend recovery, use the known ID with the existing
session API for explicit retirement if necessary. No direct runtime bypass is
introduced for this failure path.

## Qualification

Focused tests exercise real protocol request shapes, canonical replacement of
truncated streaming output, count/application/native-identity fencing, tool
continuation, uncertain-effect non-reexecution, Stop settlement, Unicode paste,
cell wrapping, paused follow and resize. Installed-package model/TUI acceptance
is recorded separately; physical macOS IME composition requires an interactive
terminal check, not an assertion from injected UTF-8 PTY bytes.
