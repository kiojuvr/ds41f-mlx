# Phase 3 / Step 3B-R1 — automatic headless compaction feasibility

**FEASIBLE, model-free only. C4 remains unqualified.** The installed pressure
listener reaches the official range transaction and summarization request through
an ordinary Agent pre-step. No human-command seam is needed for this path.
No weights, native inference, successful summary, replacement, deployment or push.
Step 3B's manual-command blocker remains true; it does not block automatic pressure.

## Authority and reproducibility

Started clean on `Multimodal-Runtime`, actual/expected HEAD
`2cb8b6c301f79a1a00c788740c2dac3127479984`. Read documentation policy,
implementation/multimodal plans, Phase 1 and historical Steps 3A/3B. Those historical
reports are unchanged. Reused the reviewed official source checkout
`/private/tmp/ds41f-w1-harness-source` at
`687f282e8625c6b8d11ff58211a90524be2e4317`; no renewed general upstream audit.

Evidence: [phase3-step3br1-evidence/](phase3-step3br1-evidence/).
`integrity.py` rechecks Step 3B's installed identity receipts and the original
operational inventory. Harness/headless/compaction-basic/compaction/session/
token-meter/DeepSeek packages remain **0.2.1-rc.2**, Node **v24.5.0**. Exact
per-file hashes and versions are reused from
`phase3-step3b-evidence/installed-identities.json`; new integrity receipts bind that
receipt's SHA256, verified file counts and package versions rather than duplicate it. Package manifests do not establish a build Git SHA or whole-tree equality
with the reference revision.

Reproduce from the repository, with the existing official installation:

```sh
node artifacts/multimodal-runtime/phase3-step3br1-evidence/feasibility.mjs
python3 artifacts/multimodal-runtime/phase3-step3br1-evidence/integrity.py check2
git diff --check
python tools/check_repository_self_containment.py
```

The test reads the retained normally composed Step 3B YAML, not production
configuration. Its YAML reader warns on unrelated official `!!js` tags; it does
not evaluate them or mount the headless task runner. Normal CLI composition was
already established in Step 3B. No installed package/configuration is edited.
An initial diagnostic incorrectly passed already-resolved adapter options back
through its raw-config validator; corrected to use the retained composed raw
provider configuration. This was a model-free setup error, not a provider request.

## Installed call chain and lifecycle

Paths here are installed `node_modules/@deepseek-ai/<package>/lib/index.js` under
`~/.local/share/deepseek-harness`. Bundled regions correspond to the requested
pinned `compaction-basic/src/{index,config,region,summarizer}.ts` implementations.

| Owner / installed lines | Verified behavior |
| --- | --- |
| `dsh-headless:209–278, 293–382` | Fresh `agents.create`; adoption requires persistence/query, matching cwd, no preset/subagent/fork or live owner, then `agents.resume`. Both routes wait idle and call ordinary `agent.followup`; final `sessions.flush` precedes exit. |
| `dsh-agent-loop:903–1027` | `turn/start` precedes `preStep`; claim inbox and assemble system prompt, then waterfall `agent/pre-step`, **before** step/start and admission of claimed input. |
| `dsh-agent-loop:1040–1272` | Step prepares/routs request, admits messages and logs resolved `request/header` in `buildRequest` **before** provider streaming. Assistant settlement, tool execution/results, then step/end precede the next step hook. |
| `dsh-session:590–641, 1603–1615` | Full canonical request envelopes fold from durable request/header events; snapshot/resume seeds reconstruct the latest header. |
| `dsh-compaction-basic:743–758, 834–867` | Constructor resolves config; auto=true independently registers pre-step → `compactIfNeeded(agent, 'pressure', signal)`. Command dispatch is unrelated. |
| `dsh-compaction-basic:922–991` | Routed header target → exact policy → `tokenMeter.measure` → `llm.resolveModelInfo` → `resolveCompactSpec`; threshold check, optional official pruner/recheck, balanced range, official `compactRegion`. |
| `dsh-token-meter:694–779` | Current durable surface plus header/tool envelope; route-owned image pricing, heuristic text, matching successful usage anchor when no lower than estimated anchor, plus surface deltas. Not Runtime's exact serialized token count. |
| `dsh-llm:2111+`; `dsh-llm-deepseek:2107+` | LLM resolves registered adapter model information; this configured model has local declared context/default output. No network/model discovery is needed for this lookup. |
| `dsh-compaction-basic:290–365, 399–736` | Official transaction validates range/open turn/lock, opens marker, replays protected system + selected messages/header tools; streams purpose=compaction, frames summary, requires strict shrink, appends summary + user checkpoint `surfaceOp: replace` + end. |

**First usable routed header:** at the first request's `buildRequest`, not at Agent
creation or its first pre-step. A completed request is sufficient, but not necessary:
even a subsequently failed provider attempt may already have logged its route.
AgentOptions alone do not satisfy pressure's `routedTarget()`.

1. **Fresh session before first request:** hook runs, normally no header → null,
   no measure/model lookup/compaction. Pending initial user text is not yet surface.
2. **Between tool steps:** after completed assistant/tool results and step/end,
   hook runs inside the same open turn with the earlier routed header.
3. **New headless invocation `--session-id`:** persisted completed session is adopted,
   opening a new turn emits pre-step with the reconstructed **previous** routed
   header. The newly claimed followup is not yet measured. New route changes are
   resolved later during request preparation; keep route fixed for acceptance.
4. **Completed tool-result exchange:** next non-concluded tool step reaches the
   hook; no hook between individual parallel results. If a tool concludes the
   turn, or ordinary assistant completion ends it, no unconditional post-completion
   hook occurs: another legitimate followup is required. Same-step provider retries
   rebuild requests; they do not themselves emit pre-step.

## Supported proposed policy and effective budgets

Only supported requested fields are changed, in a **task-local future profile**:

```yaml
auto: true
modelPolicies:
  - provider: deepseek-official
    model: deepseek-flash
    thresholdRatio: 0.00390625  # 1/256; absolute 4096, not a smaller model capacity
    headroomTokens: 2048
    retainTokens: 1024
    maxTokens: 512
    compactionRetries: 0
```

`feasibility.mjs` executes the **verbatim installed pure config region** with the
official `deepFreeze`, including validation/target merge/spec resolution. It also
instantiates the real DeepSeek adapter with the retained raw provider config,
resolves its model info without HTTP, and runs the real installed engine under
this policy. `result.json` retains defaults, exact override, adapter info and spec,
not just intended YAML. No provider/summary target override is used.

| Quantity | Actual effective value / calculation |
| --- | --- |
| Header/provider/model | `deepseek-official/deepseek-flash` |
| Adapter context | **1,048,576**, unchanged |
| Ordinary output reservation | **1,200**, materialized in the real controlled request header; adapter default also 1,200 |
| Message budget | 1,048,576 − 1,200 = **1,047,376** |
| Pressure budget after headroom | 1,047,376 − 2,048 = **1,045,328** |
| Pressure threshold | floor(min(1,048,576 / 256, 1,045,328)) = **4,096** |
| Verbatim retention | explicit **1,024**; `0 <= 1024 < 4096` |
| Summary output cap | explicit **512**, independently bounded, not 65,536 or ordinary 1,200 |
| Pressure additional retries | **0** → at most **one** successful-reduction attempt per hook |
| Overflow retry default | **1**, inherited, not changed or exercised |
| Other targets | untouched defaults: .8 / 65,536 / .16 / 65,536 / 1 |

The low *ratio* reflects the real million-token declaration; the absolute 4K
threshold allows useful exchanges rather than an immediate tiny-history trigger.
One quarter is retained, with boundary rounding; retaining an indivisible large
message/tool group may exceed 1,024 substantially. Headroom is not a fake output
reservation. A 512-token concise checkpoint can replace a multi-thousand-token
selected range; strict shrink still decides actual success, including framing.

This is an acceptance-specific policy, not a recommended production default.
Runtime context admission, numerical profile, images/source accounting, paired
limits, ordinary provider cap and capacity are unchanged. For a fresh native run,
measure the real configured prompt/tool baseline; require it comfortably below
4,096 and a useful balanced selected region before claiming a meaningful trigger.
Do not lower the ratio again to compensate for unsuccessful native progress.

## Model-free integration result and range boundaries

`feasibility.mjs` follows the official `agent-loop-testkit` service-mount recipe
with installed Cordis, LLM, SessionStore/projections, system prompt, tools, Agent
registry, Token Meter, BasicCompaction and AgentLoop. The testkit package itself
is not installed; no simulator or alternate Agent/history implementation is added.
The normal headless profile's engine/meter/runner registration is checked against
Step 3B's composition; the official engine listener is mounted and invoked by the
**real installed Agent driver**, not manually dispatched.

The only substituted execution is an explicitly controlled LlmAdapter and a
no-effect test tool. Model information delegates to the real configured DeepSeek
adapter. Ordinary control responses/tool-call chunks are **test doubles, not real
model history or native Harness session evidence**. Three bounded control user
sections distribute pressure across useful selectable history rather than putting
all text in an uncompactable last message. No live persisted headless session is
created; the in-memory ID is explicitly `model-free-control`.

Results (`tests.log`, `result.json`), **13 check groups pass**:

- Initial pre-step: no header/zero pressure; no compaction.
- Completed tool boundary: routed header, 1,200 cap, pressure **75**, no compaction.
- Later completed ordinary turns: pressure **94 → 1,780 → 3,466 → 5,152**.
  Claimed input is absent from that hook's measurement until step admission.
- A legitimate subsequent followup emits pressure hook at **5,152 ≥ 4,096**.
  The unchanged engine selects and validates the actual official region, opens
  compaction, assembles replay input, and reaches `ctx.llm.stream` with cap **512**.
- Selected route-priced history **3,435 tokens**, includes both original controlled
  tool-call/result seqs **8/10**. System seq **4** is excluded. Recent tail survives
  (exact selected/retained seqs are in the receipt). Validation acceptance is real;
  there is no direct call to compactRegion or force-success override.
- At the LLM summary boundary the controlled adapter throws
  `MODEL_FREE_STOP_BEFORE_SUMMARY`. End records failure; no compaction/summary or
  replacement exists. Ordinary turn completes. A later followup retries pressure:
  **two failed summary attempts, seven ordinary requests, seven hooks** total.
- Completed header survives official Session snapshot reconstruction. Invalid
  ratio retention, effective retain==threshold, and exhausted headroom reject.

Range implementation: exclude only a system/message at surface position zero;
accumulate tail node prices backward until retention reached (always keep at least
one node), then walk back to a balanced tool boundary. Null if no useful earlier
range. Transaction independently validates both boundaries, rejecting split/open
tool groups. Retention is **not** manual `/compact`'s literal zero. The control's
selected content does not predict which native images will survive.

## Failure and retry contract

| Condition | Installed behavior / operator consequence |
| --- | --- |
| Missing usable routed header | Silent null. Wait for legitimate request progress; AgentOptions are not a pressure fallback. |
| No model context | TargetPressureConfigError; warn once per target per engine instance, continue. Configure actual adapter metadata; do not invent smaller capacity. |
| Invalid config types/keys, contradictory ratio retention, duplicate policies | Load-time rejection. Fix policy before invocation. |
| Capacity/headroom/output/absolute retention conflict | Runtime target config error, warning suppression as above, continues. Integer rounding may itself make tiny thresholds invalid. Fix policy. |
| No compactable range / retained tool group consumes history | Null; pressure persists. More bounded genuine completed history or a separate operator decision is needed, not synthetic events. |
| Incomplete tool pairing | Boundary walk retains whole group; transaction rejects unsafe endpoints. Repair/resolve protocol through supported ownership, never fabricate missing results or retry uncertain effects. |
| Summary not smaller, empty, image output, token-cap truncation | Reject; no successful checkpoint. `max-tokens` becomes MAX_TOKENS incomplete-checkpoint error, not an accepted partial summary. |
| Summary generation error | Close attempt with error; propagate to pre-step warning and continue. Recognized summary-error recovery can offload/reprepare selected sources; arbitrary errors are not recovery triggers. |
| Surface changes | Automatic path requires whole measured surface unchanged during summarization; fail/close rather than commit stale summary. |
| Append/commit failure | Attempt one close; if close fails, unmatched compaction/start remains detectable and blocks later attempts until supported lifecycle recovery. Partial commit must be inspected; never infer rollback. |
| Persistence failure | Automatic compactRegion has **no per-transaction flush callback**, unlike manual compactNow. JSONL live buffer writes best-effort; background failure warns/retains buffer, session/flush drains/retries and rejects loudly. Headless final flush failure fails run. In-memory end/replacement alone is not durable success. |
| Still above threshold after reduction | At most retries+1 reductions per hook, then throw warning and continue; failed summary itself is not retried by this success-loop. Later hooks may try again without pressure cooldown. |

The pre-step handler catches pressure errors and logs **`step compaction failed:
…; continuing the turn`**, then `next()`. Target config warnings alone are suppressed
per target; other failures may recur every eligible step. Cancellation is still
checked by the Agent after the hook. The test demonstrates summary-failure turn
continuation and retry on a later hook, not log transport or persistence durability.

**Separate overflow recovery:** agent/request-error handles only provider-confirmed
`CONTEXT_WINDOW_EXCEEDED`; it bypasses pressure threshold, uses retain=0, optionally
prunes, requires replacement-generation progress, and returns retry subject to its
own limit (default one; reset at idle/successful assistant settlement). It can retry
from already replaced state after an error. Arbitrary Runtime resource rejection
is neither pressure compaction nor confirmed overflow. No such error was induced.

For C4, a warning is not success: stop a future bounded gate if summary/replacement
fails, persistence cannot checkpoint, resource limits reject, lock remains open,
or pressure causes repeated compaction. Inspect exact durable state before any
operator-authorized continuation. Changes to prompt/policy, missing attachments,
failed/truncated summaries, unsafe tool history, storage repair and unresolved
resource refusal require operator intervention, not an automatic local workaround.

## Smallest credible separately authorized native gate (design only)

Use a fresh task-local official headless session with exactly this routed override,
only official `read_image`, disabled session-log as before, same authenticated
small 384×384 fixture/source identities and unchanged Runtime limits. Do not reuse
control text/responses or seed old assistant/tool/image events. Keep the task useful:
inspect two real images, compare observed facts, then retain a concise working
record against a small supplied reference/checklist. No supplied visual answers.

1. Acquire two authenticated images through actual model-directed read_image and
   complete their tool results/ordinary Messages (normally three requests). Retain
   source/tool/occurrence IDs and at least one C>0 compatible paired continuation.
2. Complete one short ordinary comparison followup if needed to establish reuse.
   Inspect passive meter/current-surface evidence after each completion. Do not
   assume raw canonical N equals meter pressure or that new pending task text counts.
3. If below 4,096, use at most three useful, genuinely user-authored reference/
   checklist followups, each at most ~1,000 meter tokens of new text, ordinary
   responses still capped at 1,200. Distribute content across completed exchanges;
   avoid one enormous final message consuming the entire retained tail. No filler
   history, fabricated tools or manufactured image events. Selected balanced old
   history should exceed ~1,500 route tokens for a credible 512-token replacement.
4. **Cheapest trigger:** if pressure crosses immediately after a normal completed
   read_image exchange, its next in-turn pre-step costs no extra ordinary request.
   Otherwise a minimal ordinary `--session-id` continuation opens the next turn;
   it prices the already-completed history, with the new followup still pending.
   Same cwd/provider/profile, persisted session/query and exclusive ownership are
   prerequisites. Native adoption remains a live acceptance boundary.
5. Observe pressure trigger/actual selected seqs and retained tail; real summarizer
   generation ≤512, strict framed shrink, genuine summary + replace + end and
   successful persistence. Count the summarizer in generation/resource budgets.
6. Capture the next ordinary request under the changed projection, genuine prefix
   divergence and stale-pair exclusion; allow natural C=0 or compatible C>0 without
   cache clearing. Complete native prefill/generation/settlement/publication, then
   one ordinary compatible continuation. A text-only retained projection qualifies
   only scoped C4, not fresh historical Vision. Inspect actual selection before
   asserting image survival/fresh historical reconstruction.

Planning ceiling: **nine ordinary generations plus one auxiliary summary**, two
read_image effects, 11,312 output tokens maximum (9×1,200+512), 16,384 prepared
positions per request; reuse prior 900 s session / 180 s generation / host memory,
pressure, swap and shutdown stop rules. This is a candidate envelope to freeze and
model-free admit before any separately authorized load, not permission to execute.
If real output/pressure/range/summary fails within that finite envelope, stop and
report the unmet boundary; do not extend it or tune capacity/admission mid-run.
Two small images need only 3,042 fresh patches, far below 36,864. A few thousand
text tokens fit the unchanged context/body limits, including summary instruction
and reservation; actual serialized requests must still be independently checked.

Reuse Step 3B `LifecycleObservation` **unchanged** for live pre-shutdown Scheduler,
model/backend disposal, executor, weakref and MLX post-cleanup observations. No new
lifecycle owner/intrusive P6/P5 taps. Its seven regression results are reused, not
rerun as native qualification. Missing intermediate N−2/N−1 observations remain a
separate limitation; neither final frontier nor this model-free test fills it.

## Disposition, unverified boundary and integrity

**FEASIBLE:** no upstream feature is missing for pressure-triggered compaction.
Real installed event registration, routed header progress, exact policy/pressure
and balanced transaction selection are demonstrated model-free. Full headless CLI
process/resume and file durability are source/composition proofs here, not a new
executed CLI session. The controlled test deliberately stops at summary generation;
real summary/shrink/durable replace, post-change native state, Vision, resource
cleanup and publication remain unverified. These are the separate native gate,
not fabricated compaction evidence. Manual dispatch is still absent.

No Production code changes, official package edits, native Runtime imports,
operational config modification, installation/provisioning, weights, effects,
production history/attachments, cache clearing, push or deployment. Both new
integrity checks match all **817** original operational Runtime hash/size/mtime/
inode records at `/Volumes/SDXC-512/ds41f-1017` and Step 3B official package hashes.

C3/C5 retain their previous session-scoped PASS; C2/C6/C7/C9 retain previous
qualifications/limitations; C4 and C8 remain unqualified. No numerical kernel work,
Step 3B-R2 or Phase 4 begins. Validation: focused model-free control passes,
`git diff --check` and repository self-containment pass. **Stop after Step 3B-R1.**
