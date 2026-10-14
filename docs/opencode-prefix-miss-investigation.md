# OpenCode final → next-turn prefix miss investigation

## Status — Provisionally closed (non-blocking)

Provisionally closed at the user's direction on 2026-10-14. This investigation
is **not a blocker** for ordinary runtime operation or release readiness, and
is **not a declaration that prefix/cache continuity is fully resolved**.
Retain the attributed Plan-path evidence and the unresolved boundaries below;
stop further investigation here and return to ordinary runtime operation.
Existing independent release/qualification gates are unchanged.

The continuation investigation on 2026-10-14 identifies a first divergence and
owner on a real OpenCode Plan final → human follow-up path (below). That miss is
legitimate: the client moves a historical instruction to a different message.
No runtime workaround was introduced. **This does not establish the owner of
all clients' misses, or retrospectively identify the unrecorded messages in the
2026-10-13 112K-token incident.** The older token-only evidence remains limited
as described below; its previous administrative closeout was not ownership proof.

## 2026-10-14 reproduction and first divergence

Normal admitted `mtp-serving-v1`, official external checkpoint, default reasoning
`none`, unmodified OpenCode 1.18.35 built-in Plan agent/provider/tool set.
Only the existing `tools/observe_prefix_frontier.py` observation launcher was
extended; no serving/model/template/cache implementation was edited.
Session: `ses_ec702acd8ffeUKqgEK6T9wdsOJ`.

```sh
opencode run --agent plan -m ds41f-mlx/deepseek-v4.1-flash --format json \
  'README.md を一度だけ読み、このプロジェクトの目的を一文で答えてください。ほかの調査や変更は不要です。'
opencode run --session ses_ec702acd8ffeUKqgEK6T9wdsOJ --agent plan \
  -m ds41f-mlx/deepseek-v4.1-flash --format json \
  'その説明を、専門外の人向けに一文で言い換えてください。追加の調査やツール実行は不要です。'
```

The first human turn reads README and returns final text; the second returns
final text without additional tools. External turn order is confirmed by the
same session ID in both CLI streams, not inferred from adjacent server calls.
The automatic title request is separate and is not the transition of interest.

Evidence: `artifacts/opencode-prefix-miss/continuation-structure.jsonl`,
`direct-public-control.json`, and `reproduction-summary.json` in that directory.
The observer records no conversation text or field fingerprints. Fingerprints
are keyed per process, retained only in memory, and compare parsed incoming JSON
**before** request conversion with recipe-converted messages. Message attribution
uses two truncated renders from the actual recipe and their stable LCP with the
full prompt, not a locally invented template or additive token-length estimate.

| Boundary | Finding |
| --- | --- |
| Previous request/final | Request 9306 tokens; canonical 9357; prior tool continuation reused 7725 |
| Next request | 9389 tokens; cached=0; request LCP and canonical LCP both **7361** |
| First token mismatch | Zero-based index 7361: 67487 → 41272 |
| API/public history | First changed field is **messages[1].content**, the original human user message |
| Recipe-converted history | Same first changed message/field; role=user; stable before/through boundaries 7331/7364 contain the mismatch |
| Specific input change | Old content 1543 chars, new 57; old Plan reminder removed; non-reminder text equal; identical reminder on a later message |
| Retention/lookup | Old 9357 frontier remains live; zero lookup removals; no compatible live/observed publication |

The first divergence is at the original user's text ending/reminder separator
(BPE can merge the separator with the preceding text), **not** at the new
assistant final or its settlement. System/tools/config fingerprints have not
changed. The recipe faithfully renders the changed historical user content;
there is already divergence at the incoming OpenAI-compatible API boundary.
The public-history comparison (prior request messages + returned assistant
message) finds that same earlier user-field change before reaching the returned
assistant message. Raw optional-field differences alone are not classified as
token divergence; the independently measured token LCP is the authority.

### Root cause and owning layer

The installed OpenCode bundle contains `SessionReminders.apply`, which selects
`e.messages.findLast((y)=>y.info.role==="user")` and, for non-experimental Plan
mode, pushes the synthetic Plan reminder onto that user's `parts`. The reminder
is not persisted in the original user's parts. On the next human turn it is
therefore absent from that historical message and added to the new latest user.
This rule, plus identical-reminder relocation in actual incoming requests,
identifies **client conversation/history reconstruction** as owner on this path.
The bundle is `/opt/homebrew/Cellar/opencode/1.18.35/bin/opencode`, SHA256
`8c3c351b138cfe35905ab11846a1373f1beea590aee7eda412fb765b72c79d82`.
No reasoning round-trip is needed for this reproduction.

These are instructions, not cache-ignorable decoration. Moving them changes the
canonical token history and can change their semantic scope. Restoring the old
9357-row state into the new request would be incorrect. Removing the reminder,
ignoring the mismatch, or rewriting the old canonical state on the server is
not a repair. A client-side change would need to keep previously supplied
instructions stably represented without changing its intended instruction
semantics; no external client source/config was patched in this investigation.

### Affected-path acceptance and operational disposition

- Actual Plan path: tool continuation hits canonical 7725; final→new human miss
  is explained by the exact input change; both final responses settle normally;
  **replay=0, repack=0**, no cancellation. No false cache hit is manufactured.
- Independent direct Chat Completions control retains a reminder in its original
  message, appends the actual returned public assistant message and a human
  follow-up, and hits **cached_tokens=36**, the entire prior canonical frontier.
  Canonical LCP=36, no wire/converted/public historical change; replay/repack=0.
  Responses are the requested `OK` then `DONE`, both stop. Usage total=35 is not
  the canonical frontier: settlement also retains the terminal token.
- Small observer regressions: `python -m pytest -q tests/test_prefix_frontier_observer.py`.
  These validate observation helpers, not a new release qualification campaign.

**Repair decision: no ds41f runtime repair for the attributed Plan path.** Its
human-follow-up latency remains until the owning client stabilizes history;
longer sessions amplify the same loss. This evidence does not certify every
ordinary long-session/client path or close the exact old 112K incident without
its missing structural input. No new model execution/replay/repack or release
admission change is made; no broad release-readiness claim follows from this
bounded investigation. Stop here rather than expand qualification or redesign
settlement. The temporary observation server is shut down after capture; the
pre-existing wire observer and external client configuration are untouched.

## Earlier 2026-10-13 token-only capture

Captured on 2026-10-13 in `artifacts/opencode-prefix-miss/events.jsonl`.
The user confirms sending a new implementation request after the final answer
in OpenCode. Two stop(hit) → next-request miss transitions are present. The
latest lookup has sufficient evidence even though its generation had not yet
settled at inspection.

| Evidence | First transition (line 125) | Latest transition (line 141) |
| --- | ---: | ---: |
| Previous request tokens | 106136 | 110679 |
| Previous settled canonical tokens | 108043 | 112482 |
| Next encoded request tokens | 108117 | 112262 |
| Request LCP = canonical LCP (zero-based frontier) | 8108 | 6399 |
| Divergence token IDs (previous → next) | 1004 → 223 | 66910 → 10276 |
| Previous cached tokens | 105313 | 109991 |
| Next cached tokens | 0 | 0 |
| Compatible live pair frontiers | none | none |
| Compatible previously observed publications | none | none |
| Lookup-removed payloads | 0 | 0 |
| Previous final pair still live before lookup | 108043 | 112482 |

Requests: first `2b516ab237ef46ae84376c07a7cab6a7` →
`c8dd1f2a6cf7427f8fbb5dffcc15e2ce`; latest
`8853f0d586e84fdcb85d19849e81e5e3` →
`e6ea7549e3644ad5abfd632ee1a6c548`.

**Classification: historical request render/encode divergence, not a compatible
paired checkpoint being missed by lookup or lost to eviction.** The next prompt
changes before the end of the previous *request*, not merely in re-rendered final
assistant output. The old final pair remains physically retained but is no longer
a matching prefix. No compatible pair exists at or below either LCP frontier;
none was observed in the complete publication history (56/62 records, below the
256-record window). Evictions occurred during the workload, but no previously
published pair matches these next requests. Both lookups remove zero payloads.
The two tool continuations after the first miss resume exact canonical hits at
108362 and 109991 tokens, respectively.

The precise changed message/field or OpenCode rendering rule is not identified:
no request body was recorded. The evidence establishes token-prefix divergence
in the OpenCode request path, not a specific client implementation bug.

Earlier logs lacked IDs/inventory and were not used to establish this cause.
No runtime/cache/render fix was made; only observation launcher import order was
adjusted to install hooks after successful admission.

## Capture

Start the normally admitted runtime using its usual interpreter and server
arguments, substituting this observation launcher for `-m ds41f_mlx.serve`:

```sh
<admitted-python> -I /absolute/path/tools/observe_prefix_frontier.py \
  /absolute/path/artifacts/opencode-prefix-miss/events.jsonl \
  <usual-server-arguments>
```

Use OpenCode directly against port 8000, not the old port-8001 wire observer:
that old observer records full request bodies. Keep the normal provider/model
settings. In one practical session, complete a task that actually uses tools,
wait for the final assistant answer, then send a new user message in the same
session. Record the external session identity/turn order separately without
prompt text. Avoid concurrent unrelated clients for this single capture.

The observer now additionally logs parsed API/public and recipe-converted
first changed field paths, Plan-reminder relocation booleans, stable recipe
message brackets for text mismatches, and settlement replay/repack counters.
Its keyed field fingerprints are never written. Reminder stripping is solely
a diagnostic equality comparison, never a serving normalization or cache rule.
Text message attribution is not claimed for image-expanded prompts. Truncated
renders occur only for a changed converted message, not on every token/turn.

The original observer logs:

- previous request and settled canonical history vs next encoded request LCP;
- token counts and only the two token IDs at each divergence (no decoded text);
- compatible live pair frontiers before lookup and actual restored frontier;
- prior compatible publications (bounded 256-record token-only history);
- pair drops, distinguishing lookup invalidation from publication/external
  eviction, and explicit authority clears;
- a final→next miss candidate after the next request settles.

The observer does not retain arrays, caches, or refcount leases, alter admission,
change lookup, or normalize OpenCode requests. It requires an interpreter in
which the production imports are already admitted; it does not bypass sealing
or dependency validation. Observe ordinary OpenCode calls, not the separate
logical-session retained-lease path. Helper LCP checks and syntax compilation
passed. The extended launcher was exercised with the admitted loaded production
model on the 2026-10-14 path above.

## Interpretation

Check that the external client evidence confirms same-session final assistant
→ new user turn and preceding tool-loop hits; the observer's candidate label
alone cannot establish that semantic transition.

- `request_lcp` below the previous prompt length: historical request rendering
  changed, not just generated assistant re-rendering.
- `canonical_lcp` below previous canonical length: the next rendering is not an
  exact canonical extension. Locate the divergence with counts/frontier IDs;
  do not infer that all lower paired frontiers are incompatible.
- Nonempty `compatible_live_frontiers` with cached=0: compatible payload existed
  but did not restore. Check lookup drop events to distinguish indexed lookup
  failure from validation/acquisition failure; do not call this eviction.
- `previously_published_now_absent`: a compatible publication was observed but
  has disappeared. Correlate drop/clear records to establish eviction/clear.
- `no_compatible_checkpoint_observed`: no matching pair in live inventory or
  bounded observed publication history. This is not proof of prior eviction.

Eligibility reserves two prompt tokens (`C <= len(next_prompt)-2`), model identity
and historical image identity, matching the ordinary acquire contract. No
checkpoint at the exact LCP frontier is assumed to exist: only actual immutable
published pairs count. Render divergence and paired retention can both contribute;
report both rather than forcing a false either/or diagnosis.
