# Operator console / multi-conversation product boundary

## Current result

**TUI = operator console. Web Chat = sole Chat frontend. Multiple conversations
are ordinary tab-lifetime behavior.** This is an operator/application delta, not
runtime-core requalification, performance work or release promotion.

Removed: Python TUI Chat application, editor/transcript/renderer, session ownership,
Ctrl workflow/settings duplication and obsolete TUI collectors/tests. Historical
receipts/editions are unchanged. The former PDF Python Chat collector now contains
only fixtures reused by Browser probes. Current docs describe the new boundary.

Dashboard keys: **R** Start/Stop LLM Runtime, **W** Start/Stop Web Chat, **O** Open
Web Chat, **S** Settings, **Q** Quit TUI. D/? and log navigation remain secondary.
No F2/F3 or TUI Chat shortcuts remain. Example current stopped surface:

```text
LLM Runtime  Stopped · :8000 · mtp-serving-v1
[R] Start Runtime
Web Chat  Stopped · LAN · :8080
[W] Start Web Chat   [O] Open Web Chat
```

Settings changes the next Web launch bind/port; generation controls are Web-only.
Mac O uses nonblocking standard open, with no browser lifecycle ownership. Q leaves
both services running. Source tooling delegates model Start to the already provisioned
installed environment; application-only options/credentials are not forwarded.

## Conversation/resource policy

New adds an independent memory conversation; first Send creates its optional known-ID
runtime correlation. Switch preserves history/draft/attachments and does not interrupt
hidden generation or redirect tools/canonical outcomes. Delete removes an idle UI
conversation and requests internal cleanup; it is not required to free resources.
Tab memory holds uncertainty journals, not an archive. No IndexedDB conversation
history, Save/Restore UI, persistent conversation DB or restart reconstruction remains.
Existing immutable document/artifact source storage is unchanged, not a chat archive.

Ordinary MTP retains up to 16 logical conversations and at most **one native-active**
generation. Other requests use the existing Scheduler queue. The same upstream paired
authority holds per-conversation pins, with two spare publication slots; no second
owner, batching or numerical change. Expected-count/UUID/canonical/tool fences remain.

A living tab renews leases every 30 seconds. Web/runtime independently retire nonbusy
idle/orphan correlations after 600 seconds, with 30-second sweeps; busy/queued work
and in-flight Web effects are protected. Beacon/DELETE success is not correctness
proof. Missing state refuses continuation without replay/resend. Reusable paired
cache entries remain bounded/upstream-evictable after logical pins release.

## Actual-host checks

Current-source `.venv` TUI/Web were checked before installation, then checked again
for the final application deltas. No staging environment was created.

- Real PTY R/W Start and Stop, S, Q and service rediscovery; real macOS open with
  TUI remaining responsive. Names and legacy-shortcut removal confirmed.
- Real Chrome A → B → A, independent drafts/history, additive New, concurrent
  same-tab requests: observed native active=1, queued=1. A continued correctly after B.
- A/B/C each settled two canonical turns; all 40 target offsets and three DSpark
  offsets aligned at frontiers 984 / 864 / 1570. Replay/repack zero for all final
  outcomes; earlier first-turn and private-IP outcomes were also zero.
- B Stop reconciled without resend; C used the actual retained PDF tool loop and
  answered inventory 17. No Browser JavaScript errors.
- Deliberately lost pagehide beacon: orphan absent after the real 600-second idle
  policy plus sweep allowance, without user Close or polling that renewed its lease.
- Real private-IP HTTP Chrome on this Mac: insecure context, randomUUID unavailable,
  cryptographic fallback, two independent conversations and canonical outcomes.
- Final-source Browser switched during a delayed two-PDF upload: both documents
  stayed with A; B remained untouched, with no native session allocation.

Local receipts/screens: `artifacts/operator-console/{source-ui,real-model,final-source-ui}/`,
`lan.json`, `dashboard-final.txt` and raw `dashboard-final.ansi`.
Collector: `tools/accept_operator_console.py`. Installed MTP admission passed for
final wheel SHA256 `df8fbb5f169c7d3433630068775957292ff247b257ace6751d4742dd391e92fb`.
The full model receipt used wheel `ba24ecf18879792ced7a2d0901d21ba66cf86a948da786d9ecb14eaf2b96c1fa`;
the final cut changes only application file-batch ownership/environment filtering,
verified by final-source UI and focused tests. Scheduler/session/server resource
code is identical across those two cuts. No closed numerical implementation changed.
Failed attempts retain separate local
folders: initial observation race, CSP-incompatible polling and wrong queue projection
observer. These were collector faults, not successful product receipts. The collector
now uses direct DevTools polling, native /health queue observation and standard
terminal SU/SD handling; production output is not repaired for evidence.

Focused regression result: **195 Python tests, 23 Node tests passed**.
Documentation authority: **46 tests passed**, repository self-containment checker
passed. Existing unregistered source-scoped performance references were reviewed
in the contextual index only; no performance text/receipt or frozen inventory changed. Includes actual
curses key routing, context ownership, idle pin release, busy retirement protection,
count/UUID fences, paired publication/no-replay, tool effect dedupe/uncertainty,
private-LAN boundary and existing PDF/Vision application admission. Deprecation
warnings concern existing FastAPI on_event APIs, not failed checks.

## Change size

Approximately 49 source/test/document files, 1.5K lines added and 2.5K removed
(excluding evidence data). The TUI Chat application/editor/renderer/settings alone
removed 612 lines; its key routing/ownership and obsolete collectors/tests were also
removed. New state is limited to per-conversation contexts and bounded idle leases,
not another chat implementation, persistence layer or executor.

## Limits / remaining verification gaps

Retention is bounded (16 logical conversations, existing 4096 lifetime identity budget),
not an unlimited/multi-user service. A suspended/disconnected tab beyond its lease
cannot restore native state; history stays readable but continuation refuses safely.
No blocker was observed in the tested product flow. Separate physical LAN-terminal
routing/firewall behavior and non-Chrome browsers were not newly qualified. No account,
authentication/TLS, Internet, persistence, Vision-quality or core performance scope is added.
