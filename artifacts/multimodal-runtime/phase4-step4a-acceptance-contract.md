# Phase 4 / Step 4A — integrated acceptance contract

**Disposition: PARTIAL. Frozen proposed workload; Step 4B is NOT authorized.**
The installed mechanisms support the intended sequence in controlled model-free
execution. Nine authenticated sources and conservative Recipe/admission controls
pass. The existing native task guard is not a ten-effect/8K guard; exact composed
profile pressure/headroom through Round 3 and a layout-bound integrated paired-byte
envelope remain unverified. Do not call these prerequisites native failures or an
upstream architectural blocker. No weights were loaded, native session started,
Production/official package changed, operational Runtime modified, deployed or pushed.

## 1. Authority and evidence scope

Started clean on `Multimodal-Runtime`, expected/actual HEAD
`ee483a56305ef313ec01885c9fb8d2496e7bcabd`.
Read the six requested CURRENT documents and six retained Phase 1–3 reports.
Reuse their source audit: Harness `687f282e8625c6b8d11ff58211a90524be2e4317`,
DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, Recipe review
`935817c66f5d6e64838f419692545f06787e485f`; delivered patched Recipe 0.2.0
is distinct. No broad source investigation was repeated.

New evidence is [phase4-step4a-evidence/](phase4-step4a-evidence/).
`integrity-{before,after}.json` reuse the Step 3B installed identity receipt and
Step 2B-R2 original inventory, rather than duplicating source hashes/logs.
Official packages remain **0.2.1-rc.2**, Node v24.5.0. Package versions are not
claims of exact build-SHA equality. All **817** operational paths match retained
SHA256/size/mtime/inode at `/Volumes/SDXC-512/ds41f-1017`.

Ownership is unchanged: Harness owns history, attachments/effects and projection;
Recipe owns Messages/canonical image order and Tool Call IDs; DeepInfer is the
cache-aware reference; ds41f-mlx owns normal MLX admission, complete paired
Target40/DSpark3 execution, Canonical Settlement and publication. No second
Harness, history/KV owner, image cache, replay or Runtime lifecycle controller.

## 2. Frozen supported policy

One policy for the entire future session, not round-dependent tuning:

```yaml
# DeepSeek adapter: native Messages, image-capable exact model
maxImagesPerRequest: 3
imageOffloadCountQuantum: 1
maxTokens: 1200
retryPolicy: {mode: normal, maxRetries: 0}
# compaction-basic exact deepseek-official/deepseek-flash policy
# Full fragment: phase4-step4a-evidence/policy.patch.yml
auto: true
modelPolicies:
  - provider: deepseek-official
    model: deepseek-flash
    thresholdRatio: 0.0078125
    headroomTokens: 2048
    retainTokens: 1024
    maxTokens: 2048
    compactionRetries: 0
```

Declared model capacity remains **1,048,576**, ordinary reservation 1,200;
message budget 1,047,376; pressure budget after headroom **1,045,328**;
`floor(min(1048576 / 128, 1045328)) = 8192`. No Runtime context/resource change.
Raw/inline image-byte budgets remain 128/20 MiB; byte quanta 64/10 MiB.
Keep `llm-retry` disabled, session-log disabled, native tools, thinking disabled,
addition-only tools, read-only sandbox, only official `read_image`, parallelism 1,
no title model/subagents/fallback model. Overflow default is still one, **not**
authorized recovery: any context overflow stops this acceptance before a retry.

`config.py` executed normal installed headless `--dump-config` in an isolated
throwaway profile, without following up or creating an Agent. `configuration.json`
retains selected actual rows and full composed-output digest. `ordering.json`
contains real installed adapter/pure policy resolution and model information.
The policy fragment is not a runnable acceptance profile; its task guard has not
been extended. Temporary composition roots were removed.

### Why 8,192 rather than 4,096

`ordering.mjs` reuses Step 3B-R1's official service mounting (Agent, Session,
projections, Token Meter, BasicCompaction and image-offload). Only the adapter's
responses and no-effect tool are controlled. It delegates image pricing/model info
to the installed DeepSeek adapter and uses official `requiredImageOffload` to
raise a recognized budget error. The unmodified official plugin records omissions;
no `image/offload`, summary or replacement event is injected. These are **controlled
histories, not ten genuine reads or native session evidence**. Inputs use the
frozen future tasks; outputs deliberately carry no visual answers.

| Actual controlled observation | 4,096 | 8,192 (chosen) |
| --- | ---: | ---: |
| Round 1 completed pressure | 1,213 | 1,213 |
| Round 2 completed pressure | 1,824 | 1,824 |
| Official offloads before first summary boundary | 3 | 3 |
| Pressure at first summary boundary | 5,101 | 8,258 |
| Completed controlled image results by that boundary | 6 | 6 |
| Official selected route-priced tokens | 3,390 | 6,695 |
| Summary cap | 2,048 | 2,048 |

8K provides **4,096 additional tokens** before premature compaction and a larger
shrink denominator, without a massive endurance task. Previous native R3 pressure
was already 4,820 after just two reads and two review exchanges. That history is
not an integrated forecast; it establishes why 4K is risky, not why every 8K run
will work. Real usage anchors, full system/tool declarations and descriptive output
can price above these controls. The control's small tool declaration/empty system
head means **full-profile ordering is still a prerequisite**, not an established
native ordering guarantee. Do not tune threshold mid-session if it fails.

### Timing, range and summary contract

Pre-step runs before admitting newly claimed user text; it measures already
completed surface history. The first fresh pre-step has no routed header.
`request/header` is logged while building the first actual request, before stream;
completed headers survive official snapshot reconstruction (checked). Adoption
requires persisted exclusive session, same cwd/profile/route and supported
`--session-id`. Header/config changes between rounds are not part of this contract.

For 8K, actual selected seqs are
`[5,8,10,13,15,18,20,23,30,31,33,39,41,47,49,55,62,63,70,71,78,79]`,
endpoints **5–79**, 6,695 route tokens. Protected system seq 4 survives. Tail
**86/87**, **1,520 tokens** (1,272+248), survives although retention is 1,024.
All six call/result pairs **8/10, 13/15, 18/20, 31/33, 39/41, 47/49** are selected
whole. Official backward tail selection and tool-boundary balancing, not manual
selection, determine this. Reaching the summarizer proves transaction validation
accepted the range. The controlled adapter then throws
`MODEL_FREE_STOP_BEFORE_SUMMARY`: **no summary success or replacement was produced**.
The engine catches this and can continue an ordinary request; the future task
must use the existing *post-listener* stop fence, not assume engine failure stops it.

Retain **2,048** as the sole summary cap. Prior native success (1,053 output,
1,241 framed meter tokens <3,182 selected) is supporting evidence, not a completion
prediction. Prior truncated-density estimate at 2,047 output was **2,205 framed
meter tokens**; that is plausibly smaller than 6,695 but is **not a universal upper
bound on tokenizer/meter ratios**. More content may need more than 2,048 output
or have poor shrink. No second attempt, raised cap or rewritten summary prompt.

Four separate observations must pass: (1) normal complete summary finish, (2)
official **framed** estimate strictly less than actual selected range, (3) summary /
checkpoint `surfaceOp: replace` / successful end persisted and read back after flush,
(4) changed-projection native reconstruction/settlement/publication. Each failure
stops; preserve partial text, usage, attempt/end/error and any partial persistence.
Do not infer rollback from a failed flush. A real replacement may shadow every
image; post-compaction historical Vision is stronger optional evidence, not required.

The control tail plus prior real framed summary prices approximately **2,761**;
using the density extrapolation gives **3,725**, excluding any additional native
system/tool envelope. Before Round 3 require actual pressure **≤4,608** and no
failed/open transaction. This is an acceptance stop screen, not a product limit or
permission to alter retention. A full-profile model-free headroom qualification
including Round 3 remains open; no fake successful compaction was constructed here.

## 3. Nine-image authenticated manifest

`images.json` records source paths, SHA256, exact sizes, dimensions, MIME, visible
facts, expected patch geometry and independent admission. A–D reuse the exact
Step 3A bytes; E–I are the only new sources. Total **31,800 bytes**. `manifest.py`
verifies all bytes/geometry; Recipe admission independently verifies every source.
All nine are immutable-by-digest PNG **384×384**, 147,456 pixels, **39×39 = 1,521
patches / 184 expanded positions** each. Modified bytes fail authentication.
`contact-sheet.png` was directly inspected; all nine are independently distinguishable:

| Label | Visible audit facts (operator-only) |
| --- | --- |
| A | red circle, A17 |
| B | blue square, B28 |
| C | green triangle, C39 |
| D | purple cross, D40 |
| E | orange diamond, E51 |
| F | black five-point star, F62 |
| G | teal horizontal ellipse, G73 |
| H | magenta crescent, H84 |
| I | brown two vertical bars, I95 |

**Never send this table, manifest facts or contact sheet to the future model.**
`tasks.json` contains only acquisition requests and meaningful descriptive review
standards, no expected facts. Copy each source into an authenticated read-only
workspace before execution; hash normalized attachments/request bytes as well.
Keep all original sources, especially A, accessible until final audit. Nine is
cumulative, not concurrent or a new cumulative image ceiling. The visual audit is
small shape/color/text identification, not a general vision benchmark.

## 4. Expected causal timeline (one official session)

1. **Round 1:** user requests A→B→C, one model-selected official call per response.
   Three authenticated effects/results, independent visual identifications, fresh
   Vision, compatible paired reuse (covered images skip encoding), complete native
   Messages settlement and paired publication. Normally four ordinary generations.
   No compaction may precede this completed round.
2. **Round 2:** D→E→F. D's completed result makes four represented occurrences.
   At the next adapter request, count refusal causes the official oldest-occurrence
   offload, then a supported projection retry, **not another model generation or
   historical tool execution**. Require changed wire/canonical projection,
   shorter compatible C or miss, and fresh **surviving historical** Vision. A
   full old hit or text-only offload continuation cannot qualify this case.
   E/F may cause additional offloads naturally. Normally four generations.
3. If needed, submit at most four ordered, useful review followups from `tasks.json`,
   stopping as soon as pressure/compaction is reached. They are real future user
   tasks with real native replies, not seeded assistant/tool/summary history.
   A short trigger followup may be needed because ordinary end-turn does not
   unconditionally invoke another pre-step. Do not invoke manual `/compact`.
4. Exactly one real automatic pressure summary must complete/shrink/commit/persist.
   Inspect actual official selection; never require particular native seq numbers
   or retained images from the control. The next ordinary request uses that changed
   projection, excludes incompatible stale paired roots, reconstructs and settles/
   publishes. Require compatible continuation; no cache clearing to force a miss.
5. **Round 3:** G→H→I, then new read of A. Four reads, normally five generations.
   Bind the new A Tool Call ID/result/position to the original source digest;
   old A's omission is not undone. Actual fresh Vision must consume new A.
   Offload may happen inside this round if count exceeds three. No second compaction.
6. One short no-tool comparison/compatibility followup, supported adoption if needed,
   then final durable audit and normal shutdown. All ten effects remain uniquely
   accounted for; no historical effects are replayed during reconstruction.

If official policies cannot produce this order within the budget, record the
feasibility failure. No manual projection edits, synthetic lifecycle controls,
mid-run policy changes or replacement session.

## 5. Frozen proposed finite budgets

Machine authority for proposed numbers: `budgets.json`. These are acceptance
ceilings, **not changes to Runtime limits**.

| Item | Ceiling / derivation |
| --- | --- |
| Ordinary completed generations | **19** = R1 4 + R2 4 + review 4 + trigger 1 + R3 5 + final compatible 1 |
| Auxiliary summary attempts | **1** (complete or not); no rescue |
| All completed generations | **20**, including summary |
| Runtime inference HTTP requests | **20** max, including incomplete/pre-generation refusals; ordinary ≤19, auxiliary ≤1 |
| Adapter generation attempts | **27** = 19+1+7 legitimate count-budget retries; pre-HTTP refusals are not completed generations |
| Generic/uncertain-output retry attempts | **0**; unexpected transport/Files refresh attempt is retained and stops, not extra allowance |
| Official `read_image` effects | **10**, ordered A…I then new A; new call ID each time |
| Official image-offload transitions | **7** max =10−3; compaction can reduce this number, never requires exactly seven |
| Automatic compaction transitions | **1** maximum, including failed starts; no later hook rescue |
| Output per ordinary / summary | **1,200 / 2,048** |
| Total output allocation | **24,848** =19×1200+2048, also charge truncated/incomplete generations |
| Prepared input + reserved output | **32,768 positions/request**, including summary; Runtime context unchanged 1,048,576 |
| Wire body | **16 MiB**, existing 30 s body timeout; actual candidate maximum 186,588 bytes |
| Concurrent represented images | Adapter surface can transiently reach **4**, then budget offload before HTTP; Runtime current projection ≤**3**; no cumulative limit |
| Ordinary/summary duration | **180 s** each; includes native preparation/settlement, not first-token-only |
| Session / startup / overall | **4,200 / 1,200 / 5,400 s**; 20×180=3,600 +600 session overhead |
| Maximum no-progress interval | **600 s**; logs alone are not progress; per-generation 180 s is stricter while executing |
| Sequential headless invocations | **9** =2 rounds +4 review +trigger +R3 +final; exclusive adoption, never parallel |
| Official session storage / attachments | **64 MiB / 8 MiB**, including durable history/attachment store; authenticated fixtures only 31,800 bytes |
| Runtime Files | Unchanged **128 files / 64 MiB retained /16 MiB per file**; expected ≤9 distinct uploads, tenth effect may reuse A's mapping |

One call per response ensures the planned ten effects require tool-request and
tool-result continuation generations; ten fabricated results or uploads do not
count. Tool-call generations that also describe prior results are shared, not
double counted. Final comparison is in R3's last continuation plus one compatible
followup. Summary consumes its own request/output/time allowance. Adapter budget
retries consume attempt/time ceilings but no Runtime output allocation. Any real
HTTP retry consumes an actual-request slot, never a free generation.

## 6. Model-free admission and resource envelope

`admission_test.py` uses the existing separately authenticated dependency fixture
and actual source Recipe/preparation/admission. **Not installed serving admission**,
not fabricated native paired state. Nine independent source controls and eight
projection envelopes pass. Source-test suites separately exercise genuine Recipe
paired-authority selection, C=0 and whole-image-safe C>0 continuation, stale-root
exclusion, missing/digest/source/workload refusal and burn/cancellation fences.
No fractional-image restoration is asserted.

| Conservative control | Input N | N + reserved output | Body bytes |
| --- | ---: | ---: | ---: |
| A/B/C initial | 4,166 | 5,366 | 41,876 |
| First offload-causing four-image candidate | 6,751 | 7,951 | 63,656 |
| Changed B/C/D with surviving historical sources | 12,566 | 13,766 | 103,952 |
| All-current D/E/F after miss | 23,566 | 24,766 | 186,588 |
| Auxiliary selected-history envelope | 23,566 | **25,614** | 186,588 |
| Post-compaction text-only cold | 10,011 | 11,211 | 75,156 |
| Post-compaction surviving E/F alternative | 10,381 | 11,581 | 85,308 |
| Final H/I/new A | 23,566 | 24,766 | 186,216 |

These are deliberately conservative **admission input controls**, not model-visible
synthetic conversation or official projected events. Four-image control is never
expected to reach Runtime under budget three. All-current cold source/workload
checks pass. Each whole-image boundary also validates identity/covered prefix and
fresh suffix work; existing real-authority source tests establish actual selection,
not this source-only frontier check pretending to be a paired hit.

Three images charge **4,563 patches /442,368 pixels /552 image positions** at C=0;
covered-one/fresh-two charges **3,042** fresh patches. Four-image conservative
control charges **6,084 patches /589,824 pixels**. Every current source is decoded,
digest checked and charged, including historical images; offloaded/shadowed
occurrences are not prerequisites. Nine cumulative images do not cost 13,689
fresh patches on every request. All sources fit unchanged per-source 16 MiB /
4,194,304 pixels, aggregate 64 MiB /16,777,216 pixels, 1:2..2:1 and body limits.

**Conditional paired guards remain 12 MiB/projected complete pair and 192 MiB
retained logical roots**, with output+5 speculative positions and active/restored
plus two pending captures accounted. Conditional logical coexistence envelope is
192+3×12 = **228 MiB**, not a total physical-memory estimate. Complete compatible
paired historical coverage is mandatory for any patch discount. This workload
never needs the discount (4,563 <36,864); it must not use it to bypass C=0 checks.
`projected_pair_bytes` requires an actual qualified packed layout, not a cold
per-token estimator. The existing packed-growth/output projection and refusal
tests pass. **No layout-bound byte upper bound for ordinary 32K captures/retained
roots was produced here**; the conditional 228 MiB envelope cannot be advertised
as that bound. This is an explicit remaining resource qualification prerequisite.
Model weights, cold/Vision arenas, full embeddings, transport trees and allocator
cache are additional overlapping physical costs, not added to MLX/host counters.

`zero-weight.json` runs normal installed dependency/payload admission *before*
serving imports, then real BF16 `pre_woa`, matching digest
`185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99`.
It uses the retained Phase 1 test installation with checkpoint=False; it neither
loads weights nor qualifies a new Step 4B installed payload/full checkpoint.
Before eventual loading, normal fresh non-editable installation, exact source
match, full checkpoint admission and real zero-weight proof must pass in the
worker, then authenticated module/function identities must survive serving imports.
Never repeat late payload admission after Transformers metadata drift, patch
origins, overlay checkout numerics or bypass ordering.

## 7. Resource Supervisor and stop rules

Reuse Step 3A/3B-R3 official-client launcher, Resource Supervisor, passive
wire/preparation/native trace collector and persistence inspector. No new general
Harness, lifecycle manager, cache clearing or tool dispatcher. Required minimal
extensions: manifest A–I/path allowlist, exactly ordered ten read permissions and
new-A identity, new ceilings/timeouts, purpose-aware ordinary/summary/attempt
counters, 8K pressure/fence threshold, 4,608 Round-3 entry screen, attachment/
session-byte checks and per-transition evidence predicates. Test those before
any native authorization. The **retained R3 guard still allows only A/B, two reads
and hardcodes 4,096**; it must not be launched unchanged or patched in place in
historical evidence. Seven existing waterfall fence checks pass, not a claim that
the proposed extended guard is qualified.

On macOS M3 Ultra 512 GB: before load available ≥**455.226 GiB**, no competing
worker; host reserve **strictly >64 GiB**, normal pressure only, zero increase in
swap used/swapouts, physical-footprint stop **425 GiB**, absolute ceiling **430 GiB**.
One model load; no automatic reload/restart/replacement session. Measure MLX
active/cache/peak and host/lifetime physical footprint separately throughout
startup, Vision, cold reconstruction, capture/restore and shutdown. Do not add
overlapping counters or assert sampling captures every transient.

| Condition | Required safe outcome; no policy rescue |
| --- | --- |
| Round 1 compacts early / offload never occurs | Stop on observed wrong order or missing required offload predicate after D continuation; preserve attempted projection/error, no further reads to manufacture it |
| Offload continuation has unexpected full hit / no fresh historical Vision | Stop positive case; inspect exact IDs/spans/C, never clear a cache or discard an eligible pair |
| Insufficient pressure after four genuine review turns + trigger | Stop unexercised compaction; no fifth review, filler or threshold reduction |
| No compactable range at pressure | Stop at post-listener fence; no extra history to rescue |
| Empty/incomplete/oversized/not-smaller summary | Stop after first attempt, retain partial output/finish/strict-shrink error; no second summary or larger cap |
| Commit/flush/persistence failure | Stop; archive buffers/storage and partial transactions; no replacement assumption, rollback claim or automatic adoption |
| Missing attachment/source/digest disagreement | Pre-native refusal where applicable; stop, no substitute bytes/history or automatic re-read |
| Resource refusal / arbitrary Runtime HTTP error | Preserve exact guard/values/request and lack of publication; stop, do not translate into offload or retry |
| Cancellation / incomplete SSE / timeout | Shield in-flight native mutation via existing owner; settle exact request or burn/fail closed, retain incomplete trace; no resubmit/effects |
| Uncertain external tool effect | Stop before another effect; preserve exact ID and uncertainty, no replay or speculative retry |
| Repeated compaction / pressure still ≥8192 after first replacement | Pre-listener fence prevents another attempt; stop if second start nevertheless observed |
| Round-3 entry pressure >4608 / frozen budget exhausted | Stop unexercised remainder, not a PASS and not a new product limit |
| Supervisor threshold violation / no progress | Stop new invocations/effects immediately; existing cancellation/shutdown, bounded process termination if needed, record exit and all traces; never reload |

Any failed compaction must stop **after** the official listener but before ordinary
inference proceeds, since official warning behavior otherwise continues. An
incomplete/native outcome is not a settled successful request. After refusal,
archive traces and inspect state before any separately authorized operator action.
No failure silently funds more generations or another session.

## 8. C1–C9 acceptance/qualification matrix

PASS below is always finite-case scope. Missing required positive evidence gives
PARTIAL if mechanism/prerequisite remains unverified, BLOCKED if a demonstrated
limitation/refusal prevents that required case. Safe rejection passes only its
separately declared containment case; it cannot turn missing C8 into PASS.

| Criterion | Existing evidence | Integrated required positive observation | Separate containment / independent qualification | Classification rule |
| --- | --- | --- | --- | --- |
| C1 reuse | Steps 2D-R1/3A/R3 compatible complete pairs | Actual compatible C>0, exact canonical/image prefix, Target40/DSpark3 common frontier, wholly covered images skip Vision, settle/publish and continue | Damaged/target-only pair safely excluded; broader partitions separate | Scoped PASS only with full joins; false hit BLOCKED; missing observations PARTIAL |
| C2 numerical reconstruction | 2E model-free cases; 2D-R1 native C=203; 3A/R3 changed projections | Same-schedule native C=0 or whole-safe C>0 reconstruction with actual current sources, fresh/covered accounting and valid settlement | Same-Schedule vs Different-Partition vs Independent-Reference remain distinct; no universal cross-partition floating equivalence; true intersections/broader discount unresolved; no unjustified full-forward repetition | Overall PARTIAL remains even if integrated transitions pass; mismatched state BLOCKED. Intermediate N−2/N−1 snapshots remain a gap unless existing nonintrusive observation supplies them |
| C3 offload | 3A two official offloads, C=711 historical Vision | Official error→offload→changed wire/canonical→shorter/miss→surviving historical Vision→settle/publish | Missing source/refusal separately tested; arbitrary Runtime 400 is not recovery | Finite PASS with fresh historical Vision; text-only offload is insufficient; omitted evidence PARTIAL |
| C4 compaction | R3 real 1053-token summary, strict shrink, persisted replace, C=0 then C=3120 | Auto pressure, complete real summary, strict framed shrink, durable replace, stale-root exclusion, changed reconstruction and compatible continuation | Incomplete/oversized/flush failures stop; post-compaction historical Vision is optional stronger case | Finite PASS possible text-only; failed required summary BLOCKED for run, not safe-refusal PASS |
| C5 reacquisition | 3A new A occurrence consumed by Vision | Tenth effect/new Tool Call ID, same source bytes, new absolute canonical occurrence, obsolete occurrence not restored, actual fresh Vision and continuation | Missing source separate safe refusal; Files transport alone insufficient | Finite PASS only new inference/effect joins; absent source BLOCKED; missing inference evidence PARTIAL |
| C6 protocol/effects | Prior normal native settlement and exact effect IDs | Ten uniquely owned completed effects; no historical tool execution during prefill; full SSE/message_stop matches Canonical Settlement/publication | Separate small incomplete SSE/cancel/uncertain-effect boundary cases; no deliberate uncertain effects in positive session | Normal scope PASS possible; broader criterion PARTIAL pending negatives; replay/uncertain retry BLOCKED |
| C7 resources/lifetime | R3 safe native footprint/retirement, framework retention diagnosis; 2D-R1 narrower destruction evidence | Loaded-session memory/cache churn and per-request/per-session native retirement; normal shutdown/process exit within frozen limits | Separately authorize same-process Backend/Model weakref death and MLX reclamation; queue/running cancellation/boundary refusals not all covered by positive session | Overall PARTIAL until independent obligations pass; exit alone never full PASS; resource violation fails tested case |
| C8 main finite usability | Separate C3/C4/C5 sessions, not integrated proof | One official session, nine distinct sources, ten reads including new A; reuse, offload, automatic compaction, reconstruction after each, surviving historical Vision after offload, reacquisition Vision, valid native settlement/publication/continuation, no replay, limits respected | Post-compaction historical Vision stronger optional; no unlimited images/session claim | PASS iff every listed positive predicate joins; absent required event/evidence PARTIAL or concrete failed route BLOCKED; Step4A grants no C8 PASS |
| C9 recovery/failure | Official count-budget retries in 3A; normal continuation R3; retained model-free negatives | Ordinary protocol/effect continuity with exact recognized official retries and current-projection reconstruction | Separate small missing-input/resource-refusal/incomplete-output/uncertain-effect cases, no tool replay/proprietary protocol; manual operator intervention documented | Normal bounded route PASS possible; general recovery remains PARTIAL; unsupported arbitrary HTTP recovery must not be claimed |

### C7 explicitly separate obligations

1. Long-running loaded-model memory and cache residency: planned integrated native
   observations, **not measured by this task**.
2. Per-request/per-session ownership retirement: Scheduler queues/UID/native
   predictions/delivery/source owners empty after settlement/burn, before shutdown.
3. Normal application lifespan shutdown, executor stop, process termination and
   port release: retain distinct receipts, not weakref proof.
4. **Separate C7 in-process reclamation qualification required.** R3 identified
   FastAPI **0.142.2** process-global callable-classification LRU keys (maxsize 4096)
   → `_CallIdentity.call` endpoint closures → app/Backend → Model. Task/app root
   release alone leaves Backend/Model weakrefs live. Do not indiscriminately clear
   framework caches, erase `_model` or change Production ownership for a PASS.
   Separately authorize an affected framework/reference-ownership decision and
   model-free regression before any new native destruction qualification.
5. MLX active/cache/peak after actual model destruction and GC/allocator release
   must be observed **in that same Python process**, separate from OS reclamation.

The integrated session can pass C8 while C2/C7 and general C6/C9 remain PARTIAL.
This does not waive those project completion criteria.

## 9. Required passive evidence and audit joins

Reuse existing collectors/archives rather than another test Harness. Join:

`official event/session/tool ID → serialized Messages/request purpose/attempt →
Recipe canonical digest/order/image spans+source SHA → all candidate paired roots
and selected whole-safe C → fresh/covered Vision counts+patches → P6/P5/native
response → Canonical Settlement Target40/DSpark3 → publication/retirement`.

Retain original exact Tool Call IDs, effect source/result seqs, attachment ID,
provider File ID, source digest and canonical occurrence separately. Reacquisition
must preserve original A's effect identity while creating a new one. Archive
successful, failed and rejected native traces synchronously, including final flush;
no HTTP-200-only acceptance. Request audit classifications must distinguish:

- settled complete request with semantic finish, `message_stop`, publication;
- native generation with incomplete outcome, cancellation or missing SSE terminal;
- pre-generation Runtime refusal with exact guard and resource values;
- adapter pre-HTTP offload-budget refusal/retry (not a completed generation);
- real complete compaction commit/persisted replace vs incomplete/failed attempt;
- new real read effect vs repeated mention/source mapping reuse.

Inspect physical official persistence using existing read-only official inspector
following each CLI flush/adoption. Source events must remain durable, selected
nodes shadowed, retained nodes/tool groups balanced, header reconstructed. Keep
native rejection traces as carefully as successes; no massive duplicate source
hash inventories or logs. No new intrusive numerical snapshots/MLX synchronizations.

## 10. Validation, remaining prerequisites and readiness decision

Reproduction (no model load):

```sh
E=artifacts/multimodal-runtime/phase4-step4a-evidence
/tmp/ds41f-phase1-1017-tests/bin/python "$E/manifest.py"
python3 "$E/config.py"
node "$E/ordering.mjs"
node "$E/stop-checks.mjs"
PYTHONPATH="$PWD:$PWD/tests" /tmp/ds41f-phase1-1017-tests/bin/python -m pytest -q \
  "$E/admission_test.py" tests/test_partial_messages_reconstruction.py \
  tests/test_cold_messages_reconstruction.py
# Admission test writes admission.json; redirect logs if reproducing retained receipts.
```

Validation: **41 passed** admission/reconstruction controls; cached-image admission/
packed accounting suite **12 passed**; seven existing stop-fence checks; normal
CLI composition; actual zero-weight BF16 proof; manifest/contact-sheet audit;
proposal arithmetic/consistency; official package/817-path integrity checks;
`git diff --check`; `python tools/check_repository_self_containment.py`.
Initial diagnostic errors (plugin namespace import, snapshot projection registry,
validate_prefix argument) are retained separately and corrected, not hidden native
failures. YAML parsing warns on unrelated `!!js` tags, unchanged from prior work.

**PARTIAL, not BLOCKED:** no supported-mechanism architectural limitation was found.
The chosen supported 8K policy is more defensible than 4K and the admission controls
pass, but the following important qualification prerequisites are still missing:

1. Extend only the existing task-owned guard/observers to nine sources/ten effects,
   nineteen ordinary requests and one summary, 8K/4,608 fences and all frozen
   counters. Model-free prove no failed-attempt continuation/second attempt, exact
   effect order and all stop paths. Do not run the old A/B/two-effect guard.
2. Model-free qualify the **full composed native prompt/tool envelope**, with
   realistic retained usage-anchor sensitivity, range/shrink and Round-3 pressure
   headroom under this one policy. Current small-tool controlled ordering is
   positive but not this exact-profile check. Do not seed a live native session or
   fabricate successful compaction to pass it.
3. Bind ordinary complete-pair/retained/capture accounting to the intended admitted
   packed layout at the prepared/output envelope. The passing conditional-byte
   tests and low fresh patches must not masquerade as a cold physical peak or an
   ordinary 32K pair-byte bound. Preserve all guards; reduce workload only by a
   separately justified contract revision, never silently at execution.
4. Only after those model-free gates and separate native authorization: fresh
   normal installed payload/source/checkpoint identities, actual worker zero-weight
   `pre_woa`, exact frozen artifact hashes and sampled host/competing-worker gates
   before the **single** load. Current test installation is not that payload.

Smallest next decision: authorize a **model-free Step4A qualification closeout**
for items 1–3, not a native retry/campaign or architectural change. No Step 4B
execution until a READY decision explicitly closes them. C7 reclamation is a
separately authorized obligation, not a reason to modify Production in this task.

No Production code changes, official package modification, operational execution,
installation change, push or deployment. Local planning commit only.
**Stop after Phase 4 / Step 4A.**
