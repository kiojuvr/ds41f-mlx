# Operator Chat milestone acceptance

Base backend: `086adf4` (production sessions). Backend execution/session ownership
and numerical semantics are unchanged. The previous pre-milestone protocol blocker
is superseded by runtime-advertised `production-sessions-v1`.

## Installed-wheel real TUI acceptance

Collector: `tools/accept_operator_chat.py`, run with a non-editable final-wheel
Python from `/tmp`. The actual curses TUI receives UTF-8/key events through a PTY;
HTTP calls in the collector only observe canonical records and inject an external
Chat Server disconnect. No model response or tool result is mocked.

Local raw ANSI, screenshots and canonical receipt:
`artifacts/operator-chat/production-final/`; summarized receipt is retained in
`docs/operator-chat-acceptance.json`. The collector requires `pyte` only for evidence
projection, not for the delivered TUI. Its resize-wide-cell repair affects only
screenshots; raw terminal output is retained unchanged.

Passed: READY and Chat connection; explicit capability/default display; Japanese
multiline user text; live model text; multi-turn; real generic `fetch_url` call to
`https://example.com/`, acquired original-byte hash/result and model continuation;
Stop while number output is streaming; cancelled canonical GET; safe next user
turn; Dashboard/live-log observation during streaming and Chat return; 500-line
Japanese draft paste without submission; scroll pause/latest; 132×40 → 80×28 →
40×10 → 132×40 resize; external Chat Server stop/restart followed by canonical
reconciliation without resend; F3/F2 orderly close/restart; new conversation;
Ctrl-Q session DELETE with both services remaining READY.

Every observed settled turn's session diagnostics were aligned at its canonical
frontier, prompt replay=0 and full-cache repack=0. Stop was actually marked
`cancelled=true`; subsequent user turn settled. External graceful server shutdown
can allow an already accepted generation to finish: that outcome was recovered
through GET, not inferred from the partial stream or retried.

Physical macOS IME composition was **not** driven by PTY injection. Wide-character
committed text and Enter-as-newline were exercised; physical IME composition remains
an explicit manual terminal check, not a fabricated acceptance claim.

## Scope

Runtime `/health` is passed through Chat Server `/api/status`. Both frontends
recognize the advertised production capability and its defaults, independently of
profile naming. Unknown capability/default combinations fail closed. Existing
standard-OFF explicitly retains its existing compatibility contract. Production
images, image/PDF tool acquisition, reasoning/sampling customization and native
persist/restore are not exposed. There is one live logical session, no frontend
session manager, no Browser/TUI shared history or TUI persistent store.

Dependency/native candidate identities are inherited by the canonical package
builder; only the application package inventory changes. Broad production-session
qualification was not rerun. Focused Python and Browser regression tests complement,
but do not substitute for, the installed-model TUI receipt.
