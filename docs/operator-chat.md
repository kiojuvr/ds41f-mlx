# Operator Chat mode

This page owns **TUI Chat keys, process-local conversation and shutdown behavior**.
Dashboard remains the runtime metrics/log home. Chat uses the existing Chat Server
application contract; [application state](web-application.md) owns effects and
[generation policy](generation-policy-frontends.md) owns shared controls.
[Operations](operations.md) owns environment and service launch.

Run `ds41f` in the matching installed environment. **C** enters Chat;
**Ctrl-T** returns to Dashboard. Draft/conversation survive mode changes and resize;
logs/status continue polling during asynchronous streaming actions.

## Chat keys

| Key | Action |
| --- | --- |
| F2 / F3 | Start / gracefully Stop Chat Server; F3 first retires this TUI's session, not Browser sessions or runtime |
| Ctrl-N / Ctrl-W | New chat (close known old session first) / close chat |
| Ctrl-G / Enter | Send / newline; no Shift-Enter assumption |
| Ctrl-O | Open settings / return without losing the conversation draft |
| Tab / Shift-Tab in settings | Select reasoning/sampling field; Ctrl-G applies validated value |
| Ctrl-C | Stop safely and wait for canonical reconciliation |
| Ctrl-R | Observe/reconcile request/tool outcome; no generation or effect resend |
| Ctrl-E | Explicit pending tool/model-response continuation, not effect retry |
| PgUp / PgDn / Ctrl-L | Scroll/pause latest follow / return to latest |
| Ctrl-Q | Quit, attempt retirement of known TUI session, leave both services running |

Footer `^` means Ctrl. Unicode input uses `get_wch`; Enter/IME confirmation and
ordinary pasted newlines do not send. Bracketed paste suppresses action keys until
its end marker. Arrows edit, Home/End use the logical line, Ctrl-A goes to line
start and Ctrl-U deletes to line start. Wrapping follows terminal cells; resizing
reflows without clearing input/history. Scroll remains paused after settlement.
Terminal modes restore on exit. Content is plain text, never executable ANSI or
terminal controls. Physical macOS IME behavior is not proved by UTF-8 PTY injection.

## Controls and tool scope

New chat negotiates `/api/status`: READY, supported stateful capability and
`generation-policy-v1`. No legacy sampler fallback; numeric fields use runtime
types/ranges, nullable seed and advertised reasoning efforts. Preferences persist
in this TUI process across new chats/modes. Sampling changes between user turns;
reasoning locks at submission, including unknown outcomes. Workflow snapshots
survive tools, Stop and GET. New chat retires the old identity before a new
reasoning choice can be submitted.

TUI is text-first with no attachment picker, persistent transcript or native
Save/Restore. It uses Chat Server Auto tools and the same 128-round breaker;
errors/budget failures pause for explicit Ctrl-E. Generic calls/results and public
reasoning render as text. Tools are selected on New chat and frozen for retained
continuation; existing declarations are not rewritten.

Default PDF tools use [retained `document_pdf`](pdf-document-workflow.md), not
legacy `fetch_pdf` chat rasters. URLs/artifact IDs can be supplied as text. Parser
read/search works with main MTP Chat; fresh page Vision needs an explicitly
available existing OFF runtime. This does not enable MTP images. Server tools
execute effects and fit results, not the TUI or native runtime.

## Authority and interruption

`chat_application.py` uses Chat Server status/session/budget/stream/tools/result,
cancel and DELETE, not `/api/chat`, the legacy run-turn helper or a direct runtime
bypass. Browser and TUI are independent equal frontends, not shared transcripts
or a grant of extra runtime sessions. OFF and production-MTP capabilities differ.

Generation freezes history, expected count, application UUID and admitted native
ID. SSE is **UNSETTLED** preview. Every stream termination—including normal EOF
and Stop—uses canonical GET/count/both identities before replacing it with ordinary
history. Busy/missing/mismatched/unrepresentable outcomes block continuation.
Unknown nonadmission requires observation or explicit close/new chat, never an
automatic resend/discard. No preview becomes effect permission.

Reserve before tool POST. Unknown effects observe the server ledger; missing
results stay blocked, never execute again. A recovered result appends once and
explicit continuation advances after resolved error/uncertainty.

## Ownership boundary

One history/draft lives only in the TUI process. No IndexedDB sharing, new store,
native restore or historical prompt replay. New/close/normal quit use DELETE
through Chat Server. Quit requests Stop, waits for action/reconciliation and
retires only its known session, not either service; lifecycle work is not a
second inference lock.

Chat Server restart is not an empty conversation and cannot authorize resubmission.
Its ledger can be lost even while the runtime retains the native session; unavailable
effects remain blocked. Reconcile explicitly or retire/new chat.

Backend unavailability or SIGKILL prevents guaranteed native retirement. Bounded
quit reports known ID and **retirement unconfirmed; no retry** on failure, with
no persistent/background recovery owner. Keep the backend available for confirmed
close; after recovery use the existing session API for explicit retirement if
needed, not a newly introduced direct-runtime failure-path bypass.

## Qualification

[Operator milestone status](operator-chat-status.md) and
[acceptance receipt](operator-chat-acceptance.json) retain their original installed
TUI/PTY/model scope; later frontend controls and PDFs have separate evidence.
The [earlier operator edition](archive/editorial-baseline/operator-chat.md.txt)
preserves original key/scope wording. Historical exclusions are not assertions
that later generation settings or retained document tools are still absent.

Focused tests cover canonical preview replacement, correlation fences, uncertain
effects, Stop, Unicode/paste, wrapping, follow/resize and settings. PTY acceptance
is not physical macOS IME acceptance; that remains a manual terminal check.
No fresh TUI/model campaign, persistence, concurrency or release approval is added.
