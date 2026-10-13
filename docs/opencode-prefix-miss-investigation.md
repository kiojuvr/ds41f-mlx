# OpenCode final → next-turn prefix miss investigation

## Status — CLOSED

Closed at the user's direction: the captured OpenCode next-turn requests change
an existing historical token prefix. No server paired-retention/lookup defect
was demonstrated, and no runtime fix is warranted for this investigation.
Identifying the exact OpenCode field/rendering rule is outside this closed scope.

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

The observer logs:

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
passed; the launcher has not yet been exercised with a real loaded model.

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
