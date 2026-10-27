# Phase 3 / Step 3B-R2 — native automatic compaction attempt

**C4 BLOCKED: the single genuine summary attempt hit the fixed 512-token cap.**
Automatic pressure triggering, two real image acquisitions, compatible reuse and
six native settled/published completions were observed. Harness rejected the
incomplete summary; **no summary record, replacement or changed-projection
continuation exists**. The safe pre-step fence stopped without a second attempt.
Overall evidence is PARTIAL; native cleanup weakrefs also remain unqualified.
No Production changes, rescue run, policy adjustment, push, deployment or Phase 4.

## Authority and installed identity

Clean `Multimodal-Runtime`, actual/expected starting HEAD
`18a289700953a4152541f4bc98912f3a5320f680`. Read the five requested CURRENT owners
and Steps 3A/3B/3B-R1 plus R1's machine result. Those historical files are unchanged.
Reused the official Harness reference
`687f282e8625c6b8d11ff58211a90524be2e4317` and Step 3B installed file identities;
no broad source audit or manual-command investigation.

Evidence: [phase3-step3br2-evidence/](phase3-step3br2-evidence/).
`integrity-before.json` / `integrity-after.json` bind the reused Step 3B identity
receipt and verify all official package files. Harness/headless/compaction-basic/
Session/Token Meter/DeepSeek remain **0.2.1-rc.2**, Node **v24.5.0**; manifests do
not establish an exact build Git SHA. No official package or global profile edit.

Fresh non-editable `/tmp/ds41f-step3br2` was provisioned using normal `mtp_setup`
from the starting source, with native GLM and Flash-1017. Runtime **0.23.0**,
Recipe **0.2.0**, oMLX **0.7.0**, MLX **0.32.2**, Python **3.13.15**.
`provision.log`, `runtime-package.json`, `runtime-manifest.json`, `source-match.json`
and `admission-preflight.json` retain wheel/dependency/payload/source identities.
Original Flash-1017 revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, profile
`flash-1017-pre-woa-fp8-v1`, execution identity
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.

## Frozen preflight and stop control

`prepare.py` created a fresh isolated home/profile/workspace/attachment/session
root. The actual installed normal CLI `--dump-config` and `--help` both succeed;
help exposes `--session-id`. Native Messages, official Files/read_image, read-only
sandbox, Thinking OFF, session-log disabled, unrelated tools/retry plugins disabled.
The real provider requests independently confirm **only read_image** is exposed.
No old conversation, synthetic result, summary fixture or image event was seeded.

The normal composed `compaction-basic` configuration is exactly:

```yaml
auto: true
modelPolicies:
  - provider: deepseek-official
    model: deepseek-flash
    thresholdRatio: 0.00390625
    headroomTokens: 2048
    retainTokens: 1024
    maxTokens: 512
    compactionRetries: 0
```

`config-preflight.mjs` resolves the actual raw provider config through the installed
validator/DeepSeek model-info resolver and executes the verbatim installed pure
policy region, as in R1. `effective-configuration.json` records:

| Quantity | Effective value |
| --- | --- |
| Provider/model | `deepseek-official/deepseek-flash` |
| Declared context | **1,048,576**, unchanged |
| Ordinary reservation | **1,200**, actual routed headers agree |
| Message budget | **1,047,376** |
| Headroom / pressure budget | **2,048 / 1,045,328** |
| Threshold | floor(min(1,048,576 / 256, 1,045,328)) = **4,096** |
| Retention | **1,024**; `0 <= 1024 < 4096` |
| Summary cap / additional reductions per hook | **512 / 0** |
| Overflow retry default | 1, untouched and not exercised |

R1's real installed registration/event tests are reused unchanged, not simulated
again. A narrow **five-case model-free Cordis waterfall test** verifies the new
task stop fence: supported `prepend` ordering, first attempt allowed, failure
rejected after the engine listener returns, failed/high-pressure later hooks
vetoed before entering it, successful low-pressure continuation permitted.

`guard.mjs` only observes official events/LLM streams and bounds tool effects/paths.
Its pre-step middleware wraps `next()`: after a failed compaction/end it returns the
supported `{kind: 'reject'}` **before step admission or another provider request**.
It neither calls/patches compaction nor appends session records, cancels native
mutation, rewrites model output, or owns shutdown. Later high-pressure hooks after
an attempt would also be rejected before the engine. The passive fetch observer
has a pre-network request-budget backstop; it did **not** fire. Native failure
stopped at the intended completed-transaction/pre-step boundary, not mid-request.

Normal installed payload/checkpoint admission precedes serving imports. A fresh
installation's full checkpoint byte rehash exceeded an initial 120 s tool deadline
(no weights or inference); the model-free check was completed with sufficient
inspection time, not a bypass. `prove_installed_admission(checkpoint=True)` then
materialized actual BF16 `pre_woa`, digest
`185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99`.
Authenticated modules/function objects/profile survive subsequent serving imports.
The native worker repeated normal admission before waiting for the Supervisor's
one-load authorization. No reload, late admission or checkout numerical overlay.

Two immutable 384×384 Step 3A fixtures were copied into the new workspace and
hash-verified. Direct viewing independently confirms A **red circle / A17**, B
**blue square / B28**; expected answers were not supplied to the model. Exact
bytes/digests/dimensions/facts are in `images.json`, original PNGs remain in Step 3A.
Model-free worst-case control body: **66,372 bytes, 9,144 canonical positions,
3,042 patches, 294,912 pixels**, current sources validated at C=0; output 1,200.
This is a preflight envelope, not a submitted conversation or proof of arbitrary
future text. Every real request also obeyed the frozen 16,384-position cap and
unchanged Runtime body/image/source/context/queue limits.

`frozen-preflight.json` binds executable diagnostic, policy, task, image, source and
prior-observer/test hashes **before loading**. All bound bytes remain unchanged.
The worker imports Step 3B `LifecycleObservation` directly, unchanged, with the
original app lifespan as sole shutdown owner. Step 3A resource sampling/retirement
is reused by import; no second Runtime/cache/lifecycle owner or intrusive tap.

## Budget and genuine execution

Frozen: one supervised Runtime/load/session; nine ordinary generations, **one
summary attempt**, two read_image effects; caps 1,200 / 512, total authorized
output allocation **11,312**; 16,384 prepared positions/request; 900 s session,
180 s/generation, 1,200 s startup, 2,100 s total, 600 s no progress. All request
attempts are counted, including the rejected summary. No arbitrary Runtime error,
provider-confirmed context overflow or forced cache miss was induced.

Official session: **`session-d046b8fc-c410-4507-a926-351bb81bf363`**.
Four sequential normal headless invocations, same workspace/provider/profile/root,
exclusive owners; invocations 01–03 use supported `--session-id`. Exact argv/prompt
hashes/PIDs/exits appear in `resources.json`; frozen actual text is in `tasks.json`.
Three useful review/checklist followups were predetermined (<~1,000 newly metered
text tokens each); **only the first two were needed/submitted**. They concern
visual evidence, identification and accessible reviewer handoff, not repeated
padding. Responses are actual unedited model output. The third was not submitted.

| Invocation | Outcome / completed pressure |
| --- | --- |
| 00 fresh | Two model-chosen image reads, comparison; exit 0; **2,116** meter tokens |
| 01 adopted | Evidence/checklist followup; exit 0; **3,334** |
| 02 adopted | Handoff/reference followup; exit 0; **5,068** |
| 03 adopted | Short handoff task opens turn 4; pre-step sees **5,068 ≥ 4,096**; summary truncates; boundary rejects; exit 1 |

Eligible pre-step pressures: **0, 1,023, 1,667, 2,116, 3,334, 5,068**.
The short trigger is still pending, not in the measured surface or auxiliary input.
All native admission/execution stayed within budget; **five ordinary requests plus
one summary**, **2,209 actual output tokens**, max real body **21,150 bytes**, max
real canonical input **4,014**. The blocked trigger emits no ordinary generation.
No post-failure followup or second summary was launched.

`lifecycle.jsonl` is a passive capture of official IDs/events, not another session
ledger; `wire.jsonl` captures native requests/full SSE without credential headers
or base64 bodies. `prepared.jsonl` records exact Recipe token IDs/digests/spans.
Six complete compressed raw Runtime traces include the capped summary generation;
`trace-retention.json` reports zero capture errors and retains incomplete outcomes.

## Image and native paired execution

Actual call IDs:

- A: `call_00_ssEfkqEs7x3CyVxY0Cka9336`, result **seq 15**.
- B: `call_00_i7hS37oIgIk9ZbrisMC05865`, result **seq 20**.

Two real Files uploads, exact attachment SHA256 identities, two matching completed
results, no errors/extra effects. The first image is described before B is
requested; final comparison correctly identifies both shape/color/markers.
No exact pixel/colorimetry/positional accuracy claim follows from that review.
Upload/preparation alone is not the Vision evidence: traces prove actual encodes.

| Request | Purpose | N | Restored C | Fresh / covered | Settled frontier | Output / finish |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 00:1 | ordinary | 606 | 0 | none / none | 653 | 47 / tool_use |
| 00:3 | ordinary | 1,153 | 653 | A / none | 1,297 | 144 / tool_use |
| 00:5 | ordinary | 1,780 | 1,297 | B / A | 2,117 | 336 / end_turn |
| 01:1 | ordinary | 2,771 | 2,117 | none / A,B | 3,328 | 556 / end_turn |
| 02:1 | ordinary | 4,014 | 3,328 | none / A,B | 4,629 | 614 / end_turn |
| 03:1 | **summary, not changed continuation** | 3,709 | 3,328 | none / A,B | 4,221 | **512 / max_tokens** |

`sequence.json` binds wire/preparation/raw trace IDs, exact image positions, native
MTP stats, P5/suffix/capture timings, Target40/DSpark3 common frontiers, canonical
settlement and publication. All six emit `message_stop`, match canonical content,
retire prediction/UID/request/native owners, and have empty queues, zero replay/
repack. Twelve paired roots were visible before shutdown; no caches were cleared.
The summary's legitimate C=3,328 is **not** post-replacement reconstruction evidence.
N−2/N−1 schedules follow unchanged code; intermediate Target/DSpark offsets were
not separately sampled. No bitwise cross-partition/full-forward/numerical claim.

## Automatic selection, actual failure and durable unchanged projection

The independent official pre-step pressure listener initiates **compaction/start
seq 50**, ID `bb9f1268-5ade-448f-a583-43a4f4175c7b`, inside turn 4. The routed
header is the intended provider/model with maxTokens=1,200. Actual auxiliary
`llm/stream` purpose is **compaction**, cap **512**; the native adapter marks the
HTTP call as compaction. This is not manual maintenance or overflow recovery.

Actual selected messages are identified by their original IDs in the real replay
input, matched against the official surface/events (no hypothetical selection):

- Selected seqs **[8,9,13,15,18,20,23,31,33]**, inclusive endpoints **8–33**,
  **3,200 route-priced tokens**.
- Protected system head **seq 7**, 32 meter tokens, excluded from replacement
  selection but replayed to the summarizer as upstream specifies.
- Both complete tool-call/result groups **13/15 and 18/20** are selected together.
- Retained recent tail **41/43**, **1,734 tokens** after node/boundary rounding,
  exceeds the minimum 1,024. It is text-only. Retained full surface would include
  protected seq 7. Both prior image occurrences are in the selected older region.
  This is automatic retention, not manual retain=0.

The actual model emits 512 tokens and ends at the unfinished heading
`## Errors and Fixes`. Native finish is **max_tokens**. Installed summarizer maps
that finish to **MAX_TOKENS / incomplete checkpoint**, before returning a summary
or running the framed shrink comparison. **Do not treat a short partial response
as a smaller successful summary.** Full partial text/usage is retained uncorrected.
Usage: **381 uncached input + 3,328 cache-read input + 512 output**.

Official **compaction/end seq 51** records:
`summary truncated` detail is exactly
`"summarization truncated at the token cap (incomplete checkpoint)"`.
No compaction/summary or `surfaceOp: replace` event exists. The listener warns and
would normally continue; the outer task fence instead rejects at this safe hook
boundary. **turn/end seq 52** is `blocked`. There is one start/end pair and one
actual auxiliary request, no retry/backstop activation or second attempt.

After every CLI exit, `persisted.mjs` opens the installed official persistence
backend's **read-only handle**, reads durable events, and derives the projection
with official Session machinery. `persisted-03.json` proves the failed start/end
are durable, every observed original event is intact, and old/new projected
messages/nodes are **identical** to `persisted-02.json`. All original tool identities,
system head, images and tail remain projected; nothing was shadowed. Actual
physical official storage is retained in `session-storage/` before root deletion.
This verifies durable **failure**, not successful compaction durability.

No changed-projection request, stale-prefix exclusion, reconstruction after
summary, or later compatible usability followup was attempted. Historical Vision
after compaction remains unqualified. A hypothetical successful selection here
would remove both old images, but no such replacement happened; do not report
that hypothetical text-only path as native C4 evidence.

## Resources, shutdown and reference limitation

Supervisor span **254.18 s**, one load; initial/preload available
**495.046 / 494.232 GiB** ≥455.226, minimum reserve **198.959 GiB** >64.
Pressure **1**, swap used/swapouts **0**, no competing inference workers.
Sampled lifetime physical footprint **300.620 GiB** (current peak **298.277 GiB**),
below 425 stop / 430 absolute ceiling. No resource stop or uncertain-effect retry.
Host and MLX overlap and must not be added; sampling is not every transient.

MLX loaded active/cache/peak bytes:
**309151331768 / 5590977820 / 311557023684**.
`cleanup-lifecycle.json` observes:

- Before original shutdown: live Scheduler, requests/waiting/running/UID map empty,
  no native owners/source occurrences/deliveries/sessions; 12 paired roots.
- After original shutdown: Scheduler absent, executor shut down, deliveries and
  sessions empty; backend still has a model (close does not erase that attribute).
- After diagnostic app/server/backend cells are set to None, GC and allocator
  cache clearing: **backend/model weakrefs still live**, MLX active
  **309151331760**, cache **0**, peak **311557023684**. No observation exceptions.
- Original server/application lifecycle runs normally, Runtime exit **0**;
  all four Harness processes and Supervisor exit, loopback port 18083 is free.

**Cleanup remains PARTIAL**, not PASS and not a demonstrated Runtime leak. Source
review confirms the reused observer stores only weak backend/model refs and
string errors; worker clears its explicit strong cells, including the load closure.
No retained traceback exception is in the diagnostic receipt. Nevertheless, no
live `gc.get_referrers`/framework/native reference graph was captured at final
observation, so an additional diagnostic/framework/native owner cannot be located
postmortem. Process exit does not supply the missing in-process weakref death or
model-allocation reclamation proof. Do not alter Production ownership to force a
pass; locating that reference needs separate bounded diagnostic work, not another
native load in this task. Step 3B's seven model-free observer tests remain reused
with their original scope and do not supersede this native limitation.

All task-owned installation/build/session roots were removed **after** retaining
physical session storage, traces, package/source identities and authenticated
normalized attachment hashes (`cleanup.json`). Original source fixtures remain.
Before/after all **817** operational Runtime inventory records match retained
SHA256/size/mtime/inode baseline. `/Volumes/SDXC-512/ds41f-1017` was never modified,
imported for execution, deployed to or loaded.

## Dispositions, validation and stop

| Required classification | Actual result |
| --- | --- |
| AUTO_PRESSURE_TRIGGER_VERIFIED | **PASS** — 5,068 ≥ 4,096, official pre-step |
| OFFICIAL_SUMMARY_GENERATED | **PARTIAL** — genuine output, incomplete at 512 |
| SUMMARY_SHRINK_VERIFIED | **BLOCKED** — official check not reached |
| OFFICIAL_COMPACTION_DURABLY_COMMITTED | **BLOCKED** — durable failed attempt only |
| CANONICAL_PROJECTION_CHANGED | **BLOCKED** — projection identical |
| STALE_PAIRED_PREFIX_EXCLUDED | **BLOCKED** for C4 — no changed input |
| NATIVE_RECONSTRUCTION_COMPLETED | **BLOCKED** for C4 |
| HISTORICAL_VISION_AFTER_COMPACTION | **BLOCKED / not exercised** |
| CANONICAL_SETTLEMENT_VERIFIED | **PASS** for six actual requests, not C4 |
| PAIRED_PUBLICATION_VERIFIED | **PASS** for six actual requests, not C4 |
| CLEANUP_VERIFIED | **PARTIAL** — final weakrefs/weight-sized MLX remain live |

**C4 is BLOCKED/unqualified, never PASS.** C3/C5 retain Step 3A's bounded PASS;
C2/C6/C7/C9 retain prior broader limitations. This adds scoped compatible reuse,
normal native protocol/settlement/publication, genuine capped-summary failure,
durable unchanged-history containment and safe no-retry observations, not general
reconstruction/resource/recovery qualification. C8 and Phase 4 remain unqualified/
not started.

Smallest remaining Phase 3 blocker: the official summary did not complete under
the fixed authorized cap. A separate design/authorization must address that finite
summary constraint before another C4 attempt; no threshold/cap/prompt adjustment
or regeneration was made here. Final native cleanup reference ownership also
needs diagnosis before claiming full reclamation. No missing manual-command
feature is implicated by this automatic failure.

Validation: five focused stop-control checks and configuration resolution PASS;
normal installed admission/actual zero-weight numerical gate PASS; offline
`audit.py` PASS for the **observed failed** sequence; package/817-path integrity,
Python/Node syntax, `git diff --check` and repository self-containment PASS. No
unrelated/full Phase 2 test campaign. **Zero Production changes.** Local evidence
commit only; no push, deployment, additional experiment or automatic Phase 4.
**Stop after Step 3B-R2.**
