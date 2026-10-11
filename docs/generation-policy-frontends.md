# Generation policy — frontend / real-client integration

Base: `28a7cee02d2d28fe24d5f26b9f9c45b7b696b5b8`. This is application integration,
not sampler/default/speculation requalification. Serving, runtime and model
execution payload bytes are unchanged from that base.

## Contract and controls

Both frontends require READY + `generation-policy-v1`. There is no legacy
sampler fallback. Unknown version/capability/ownership, missing parameters and
inconsistent defaults reject before generation. Runtime `/api/status` supplies
all eight defaults/bounds/types and reasoning efforts; `chat_capability.py` and
`web_static/policy.js` consume the same contract and application ownership rules.
Focused fixtures exercise both implementations against the runtime contract.

- **TUI:** Ctrl-O opens generation settings; Tab/Shift-Tab selects reasoning,
  temperature, top-p, top-k, min-p, penalties and seed; Ctrl-G applies; Ctrl-O
  returns. The existing independent composer, Enter/newline, Ctrl-G/send,
  bracketed paste and Dashboard layout remain intact. The Chat header shows
  current choices; the settings view shows all values/defaults/bounds and lock.
  Invalid/oversized numeric input is rejected without changing settings or
  submitting a request. Composer text/cursor survive settings and mode changes.
- **Browser:** Generation contains temperature/top-p/top-k/min-p; Advanced
  contains penalties/seed. Think uses advertised enum names directly, not
  percentages or effort aliases. Reset uses runtime defaults, preserving locked
  reasoning. Chat distinguishes next-user-turn choices from the effective
  workflow/continuation snapshot when those differ.

Reasoning freezes at first submission, including unknown outcomes, for these
retained stateful frontends. It cannot change in that conversation. New chat
retires the old native session and permits a new reasoning choice. Sampling is
request-owned, editable between user turns, not session-locked. Stateless API
reasoning ownership is unchanged.

## Application policy versus runtime omission

A user turn snapshots application settings and sends all eight fields explicitly.
Tools, tool-result generation and explicit continuation use that workflow's
snapshot. Cancellation and canonical GET never replace it with defaults. A
subsequent user turn snapshots current choices. TUI retains preferences in its
existing process; Browser uses its existing ordinary application journal. No new
native persistence/restore, shared Browser/TUI history or session manager exists.

This explicit application policy is **not** runtime inheritance. Ordinary clients
that omit fields still get production defaults for that request. The legacy
`/api/chat` helper no longer inserts implicit greedy temperature either.
Tool reservations, canonical identity/count checks and unknown-outcome fences
are unchanged: no effect retry or automatic generation resend.

## Installed real-model evidence

Final installed non-editable wheel SHA256:
`6b226c36a5390f99a8c8f6ab77055b59311ce32a03a025bf317504c695fccd66`.
Existing MTP dependencies/checkpoint/numerical identities were inherited and
strict launcher admission passed, with no gate bypass or dependency refresh.

Collectors: `tools/accept_generation_frontends.py`,
`tools/accept_generation_opencode.py`. The acceptance-only
`observe_generation_runtime.py` delegates to the original telemetry method and
records settled **execution** policy, upstream greedy/stochastic classification,
canonical offsets and replay/repack. `observe_client_wire.py` forwards original
OpenCode request bytes/response chunks without defaults, rewriting or retries.
Neither adds a production runtime API or sampler implementation.

- **Actual curses TUI / PTY: PASS, 11 executions.** Runtime defaults; temperature/
  top-p changes; top-k/min-p and all advanced controls; oversized seed rejection;
  multi-turn; Dashboard return; real Chat Server disconnect/restart and canonical
  GET; new reasoning-on conversation and rejected mode change; real fetch_url,
  acquired original-byte result and continuation; new reasoning-off conversation,
  public streaming Stop, canonical cancelled settlement, next turn and sampling
  change followed by multi-turn.
- **Actual Chrome Browser Chat: PASS, 11 executions.** Same eight controls and
  defaults/bounds; output length boundary and explicit continuation; next-turn
  temperature differs from the original continuation snapshot; reasoning-on and
  disabled mode/effort controls; real fetch_url/result/continuation; fresh
  reasoning-off streaming Stop/GET, Chat Server restart/reconnect, next turn and
  subsequent sampling change. CSP stayed enabled. Actual Browser wire bodies and
  screenshots accompany native execution evidence.
- **Real OpenCode 1.18.35: PASS, 10 HTTP-200 executions / 8 tool effects.**
  Unmodified built-in Build agent implemented a CSV/Decimal utility in an isolated
  scratch project, created/ran tests and completed a practical answer. Independent
  checks passed quoted commas, Unicode headers, blanks, empty data, missing column
  and invalid/nonfinite cells. SIGINT during actual decode cancelled the runtime;
  no automatic request followed Stop. An explicit new user turn in the same
  OpenCode session ran the tests and completed recovery. No duplicate tool-effect
  identity or exact request body was observed. CLI launch explicitly aligns PWD,
  cwd and --dir; earlier launch-context/provider failures occurred before any
  model request and are not runtime admission failures.
  Main agent requests specified temperature=.65/top-p=.8; the title helper
  specified temperature=.5 and omitted top-p, which resolved to runtime default
  ≈.95, not previous UI .8. Omitted filters/penalties/seed resolved to production
  defaults. Every policy matched actual execution; all were stochastic and all
  target/DSpark offsets aligned, replay=0/repack=0.

Runtime observed: defaults temperature=1, top-p≈.95, top-k/min-p=0, neutral
penalties, seed=null; selected .65/≈.8/40/.03, penalties 1.03/.1/.1, seed=123;
subsequent temperature=.75. Recipe f32 conversions are preserved (e.g.
.65→.6499999761581421, .8→.800000011920929). All TUI/Browser executions are
upstream-classified stochastic, with replay=0, repack=0 and every target/DSpark
offset equal to the canonical frontier. Tool call identities are not executed
again after result settlement/reconciliation.

Raw local evidence: `artifacts/generation-policy/frontend-{tui,browser,opencode}/`,
`frontend-execution.jsonl`, and `opencode-wire*.jsonl`. The compact retained
receipt is [generation-policy frontend acceptance](generation-policy-frontend-acceptance.json).
Earlier collector/prompt failures remain separate, never retroactively PASS.

## Safety boundaries and remaining unsupported capabilities

Early Stop inside thinking or incomplete tool-call grammar can correctly yield
an unrecoverable session. Such attempts were observed: the UI blocked continuation
and never resent or executed the incomplete call; explicit close/new chat was
required. No runtime contract was weakened to make that interval resumable.
Successful cancel→next-turn acceptance uses a representable public text boundary.

The PTY collector preserves raw ANSI unchanged. Its evidence projection adds
ECMA-48 scroll-region SU/SD handling omitted by pyte 0.8; this changes only the
collector, not curses or terminal output. Physical macOS IME composition is not
a PTY claim; committed wide-character editor/paste safety is regression-tested.

Unsupported: unknown generation versions/capabilities, thinking token budgets,
unversioned generation journals, TUI attachments/native persist/restore, production
MTP image/PDF acquisition and native persist/restore. Missing ordinary history or
unrepresentable/unknown outcomes cannot be repaired by automatic resend/restore.
Seed remains the runtime's best-effort capability, not a deterministic promise.
Focused application/editor/cancel/legacy-proxy tests: **43 passed**; Browser
contract/platform/render tests: **18 passed**. Separate affected backend seams
(generation policy, production MTP/sessions): **51 passed**. These are separate
sets, not a summed distinct/full-suite claim. Mock/unit results are supplementary
to the real-model receipts above. No universal loop-free agent guarantee is made.
This milestone does not promote a release or broaden core qualification.
