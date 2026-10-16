# Generation policy — frontend / real-client integration

This page owns **shared frontend generation-control ownership and workflow
snapshots**. Runtime values/defaults remain in [capabilities](capabilities.md)
and `generation-policy-v1`, not an application sampler. Browser placement/labels
live in [Chat interface](chat-interface.md); [operator console](operator-chat.md) has no generation frontend;
journals/effects in [application state](web-application.md).

## Contract and controls

Web Chat requires READY and supported `generation-policy-v1`/stateful
capability. Missing/unknown versions, parameters, ownership or inconsistent
defaults reject before generation; no legacy sampler fallback. Chat Server
`/api/status` projects runtime health/policy. Python `chat_capability.py` and
Browser `policy.js` consume the same authority, not independent numeric defaults.

The eight fields are temperature, top_p, top_k, min_p, repetition_penalty,
presence_penalty, frequency_penalty and seed. Validate runtime type/bounds,
including exclusive bounds, finite values and nullable best-effort seed.
Thinking uses runtime effort enums with Off=`none`; no percent/alias translation
or thinking-token budget. Invalid/oversized settings do not submit a request.

Browser Generation/Advanced expose those settings without changing draft/history.
Browser Reset uses advertised defaults while preserving locked reasoning. Each
conversation retains its tab-local preferences/workflow; TUI does not duplicate them. Effective continuation and
next-user choices must remain distinguishable when they differ.

Reasoning freezes at first submission, **including unknown outcomes**. New chat
adds an independently configured conversation without retiring the old one. Tool declarations remain frozen
for retained continuation. Sampling alone is request-owned, editable between user
turns. Stateless API reasoning is request-owned; frontend retained ownership does
not change that contract.

## Application policy versus runtime omission

A user turn snapshots settings and sends all eight fields explicitly. Tools,
result consumption and explicit continuation retain that workflow snapshot through
Stop, canonical GET, conversation switch and recoverable reconnect.
A later user turn snapshots current preferences. Omitted non-UI fields still get
runtime defaults **for that request**, not previous UI/server-turn inheritance.
The legacy `/api/chat` helper does not silently force greedy sampling either.

Snapshotting grants no execution or retry permission. Canonical count/native/
application identities, completed-call certificates and effect reservations remain
required. Unknown generation/effects are not resent; cancellation cannot reset a
workflow into guessed defaults. Missing/unversioned Browser generation journals
fail closed rather than silently acquiring a new policy. Journals are tab-memory
only, not persistent conversation archives or a shared Browser/TUI store.

Auto output is a complete-request runtime reservation, not a frontend estimate or
unlimited generation. Selected profile context and output ceilings apply; preview
is nonbinding observation and actual admission rechecks. This page implements no
sampler, output allocation, speculative graph or native persistence behavior.

## Installed real-model evidence

The [frontend acceptance receipt](generation-policy-frontend-acceptance.json)
records exact installed wheel/source scope, TUI/Chrome/OpenCode executions and
canonical policy observations. The
[original frontend edition](archive/editorial-baseline/generation-policy-frontends.md.txt)
retains base `28a7cee`, wheel SHA, all execution/effect/test counts, f32 values,
raw evidence paths and failed attempts. Its unchanged-backend assertion applies
to that integration delta, not later HEAD.

The removed `accept_generation_frontends.py` collector belongs to those historical
receipts. `accept_generation_opencode.py` remains a source-scoped tool; current Web/
operator product checks use `accept_operator_console.py`. Their observers delegate to original
telemetry or forward unchanged client wire bytes, not new production sampler/APIs.
Recorded scenarios include runtime defaults, all controls, workflow versus next
turn, reasoning lock, real effects, Stop/GET, server restart and explicit recovery.
OpenCode helper omission resolves runtime defaults rather than another request's
policy. Those observations are not universal client/agent or deterministic-seed
proof; no new real-model campaign is requested by this compilation.

## Safety boundaries and remaining unsupported capabilities

Early Stop inside incomplete thinking/tool grammar may be unrecoverable. Do not
weaken runtime settlement to make every partial response resumable. Successful
cancel/next-turn acceptance used a representable boundary; unknown/unrepresentable
history remains blocked with an explicit New Chat option and internal resource cleanup.

PTY bytes/collector screen projection do not prove physical macOS IME composition.
Browser handler events likewise do not certify every OS input implementation.
Collectors' terminal-model repairs are evidence tooling, not delivered TUI changes.

Unknown contracts, thinking budgets, unversioned journals and native MTP restore
remain unsupported. TUI Chat has been removed. Browser images and isolated
stateless PDF page observations now use advertised bounded Chat Vision, including
ordinary MTP: see [application Vision integration](web-mtp-vision-integration.md)
and [document workflow](pdf-document-workflow.md). The old integration's image/PDF
exclusions are historical, not current application restrictions. Seed stays best-effort.
Missing history and uncertain effects cannot be repaired by automatic resend.
Unit/renderer gates supplement source-scoped model receipts; neither compilation
nor test counts broaden runtime core, R1 or release qualification.
