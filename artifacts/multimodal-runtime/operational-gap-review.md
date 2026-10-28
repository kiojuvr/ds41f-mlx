# Multimodal Runtime — operational gap review

**Recommendation: proceed to normal official DeepSeek Harness Production operation
within the envelope below. No concrete remaining implementation blocker was found.**
This is a source/evidence deployment recommendation, **not overall C1–C9 PASS,
branch-completion approval, installed-package qualification or deployment**.

## Basis and scope

Reviewed clean branch `Multimodal-Runtime`, expected/actual HEAD
`fac6181229ac0d8b48af006ce0ef8b80989f131f`. Read the six requested CURRENT
owners first. Reviewed Phase 1, Phase 2 reconstruction, Phase 3 lifecycle and
Phase 4 evidence, and inspected the actual execution, cache, resource, Files and
protocol owners cited below. No model load, native campaign, installation,
Production change or upstream-package modification was performed.

Official-source conclusions are revision-bound: inspected the existing clean
Harness source at `687f282e8625c6b8d11ff58211a90524be2e4317`, not today's remote
HEAD. [Phase 1](phase1-gap-audit.md#2-source-map-and-verified-caller-paths)
retains the newer-source comparison, DeepInfer/Recipe revisions and delivered
Recipe distinctions. The executed installed Harness is **0.2.1-rc.2**; its
actual configuration and package identities, rather than assumed source/package
equality, are in the [E2E evidence](phase4-e2e-native.md).

The finite C8 PASS stands: one load/Session, nine sources, ten effects, four
offloads, durable strict-shrink automatic compaction, reconstruction,
reacquisition and final comparison; all nineteen native requests settle, publish
and retire with wire/native event equality. No rerun is needed. Historical failed
attempts and PARTIAL dispositions remain unchanged.

## Operational disposition matrix

“Blocks?” concerns normal operation in the declared envelope, not the original
broad completion contract. Evidence does not establish arbitrary numerical
accuracy, maximum capacity or indefinite endurance.

| Criterion / concrete operation | Implementation and official owner | Existing evidence | Actually remaining / classification / minimum remedy | Blocks? |
| --- | --- | --- | --- | --- |
| **C2: complete miss; surviving historical images after offload** | Harness supplies current projection/bytes; Scheduler admits independent Messages C=0 and uses existing MTP cold path [S1–S3]. | [2B-R2](phase2-step2br2-continuation.md) native cold lifecycle; E2E requests 01:3 and later offloads encode three current images at C=0, settle/publish. | **Supported with explicit limitation:** all current inputs and fresh-work admission required. No implementation remedy. | No |
| **C2: earlier text/whole-image pair plus uncovered historical suffix** | Same authority preserves actual compatible C; covered images skip Vision; uncovered images encode wholly at absolute positions [S1–S3]. | [2D-R1](phase2-step2dr1-continuation.md) native C=203 covered corn/fresh historical carrots; [3A](phase3-step3a-offload-reacquisition.md) C=711; E2E 06:8 C=3394 surviving H/I + new A. | **Supported with explicit limitation:** whole-image-safe prefix, normal aggregate envelope. No second cache or alternative algorithm needed. | No |
| **C2: changed bytes/order/position, shortened or rewritten projection** | Recipe owns canonical order; image-aware keys plus exact restore exclude stale roots, select earlier pair or miss [S2]. Removed occurrences are not inputs. | [2E](phase2-step2e-projection-reconstruction.md): seven actual Recipe/model-free controlled lifecycle cases; native offload rewrites and native compaction C=0 in E2E. | **Additional assurance:** arbitrary byte/order edits and compaction retaining an image tail lack dedicated native acceptance. Implemented through same path; no demonstrated failing admitted case. Retain this uncertainty, not a blanket numerical PASS. No required repair identified. | No demonstrated blocker |
| **C2: partial image coverage / aggregate historical discount** | Capture refuses image intersections; keys/restore prevent fractional reuse. >36,864 aggregate patches retains complete historical-coverage and pre-clone paired-byte requirements [S1–S3]. | 2E shortening excludes intersecting stale candidate; model-free refusal/fallback; [cached admission](../cached-image-admission/report.md) native conditional-discount boundaries. | **Supported with explicit limitation:** no execution from a bisected image; earlier safe root/miss is the supported route. Mixed uncovered-history reconstruction above aggregate limit can refuse even if fresh suffix alone is small. No generalization required inside envelope; do not claim it supported. | No, with bounded current projection |
| **C2: numerical comparisons** | Same existing MTP P6/N−2/N−1/P5 route, no tool replay [S3]. | [Numerical reference](phase2-numerical-reference.md), 2B-R2 first-token/state differences; 2D-R1/E2E lifecycle and identity joins. | **Additional assurance:** unresolved cross-partition/full-forward differences and unsampled intermediate native boundaries remain. No general floating tolerance or bitwise equivalence is established; neither is a reconstruction-specific defect established. Preserve safeguards/evidence, not another numerical algorithm. | Not by itself |
| **C7: persistent requests/cache/queue/source retirement** | One native lane; upstream bounded queue/LRU, 18 pairs, request-local sources, bounded trace deques, shielded cancellation, settlement/burn and map retirement [S1–S5]. | E2E empty request/native/source/delivery owners and all resource ceilings; cached admission native cancellation, byte refusals, queued-abort and conservative maximum-queue model-free evidence. | **Supported with explicit limitation:** bounded resources, not every possible peak qualified. No demonstrated accumulating request/cache ownership or exhaustion defect. No repair required. | No |
| **C7: surviving Backend/Model weakrefs** | Live server intentionally owns one loaded model. Framework callable-classification LRU keys retain endpoint/app/backend graph after app-root release; close stops executor but does not erase `_model` [S4–S5]. | [3B-R3 ownership diagnosis](phase3-step3br3-compaction-cleanup.md#native-shutdown-actual-retained-owner-and-resources); E2E post-cleanup active bytes essentially loaded baseline, allocator cache zero. | **Supported with explicit limitation:** one application/model lifetime per process; stop/restart process to replace it. Same-process app/model destruction is **Additional assurance**, not demonstrated per-request leakage. No framework-cache clearing or `_model` erasure needed for server operation. | No; in-process reload not supported |
| **C7: graceful exit and wider stress/cancellation** | Original app shutdown clears Files, retires Scheduler/cache and waits executor [S4–S6]. | E2E exit 0/port release; MLX/host/physical high-water separately recorded; prior native running cancellation. | **Additional assurance:** no complete changed-projection cancellation/negative/maximum/endurance matrix. Process exit is not proof of weakref death; weakref survival is not failure to exit. No new native campaign required for this recommendation. | No demonstrated blocker |
| **C6: completion, exact tool identity, stream/JSON settlement** | Recipe semantic owner and native accumulator; tool suffix withheld until settlement, JSON waits final output; no Runtime tool executor [S1,S3,S6]. | E2E nineteen `message_stop` terminals, full wire equality, exact ten effect identities and no replay; prior ordinary JSON/SSE evidence. | **Supported with explicit limitation:** completion is not HTTP 200, argument-end or `tool_use` stop reason alone. Existing successful paths need no repetition. | No |
| **C6: interruption, malformed boundaries, uncertain effects** | Shielded same-ID cancellation settles or burns; native mutation/retirement failure fails engine closed. Harness finishes model stream before tool execution [S1,S4,U3]. | Model-free Scheduler/Recipe negatives in 2E; [4B-R1](phase4-step4br1-observer-terminal.md) installed-translator malformed/incomplete/race controls; prior native cancellation. | **Additional assurance** for full native failure matrix. **Supported with explicit limitation:** no automatic reconciliation/resend of uncertain effects, no Messages outcome-GET contract. Task observer correction is not a new Production safety layer. No recovery protocol to add. | No within fail-closed boundary |
| **C9: count/byte offload, compaction, stale File IDs** | Official adapter budgets → `IMAGE_OFFLOAD_REQUIRED` → durable offload/new projection; official pressure/recognized context overflow compaction; bounded stale-ID refresh and upload-to-inline fallback [U1–U4]. Runtime reconstructs the resulting canonical request. | E2E four recognized pre-HTTP offloads, automatic summary/replacement/adoption; [Gate F0](../harness-files/gate-f0-report.md) Files transport; source/model-free coverage for wider Files recovery. | **Supported with explicit limitation:** exact trigger/policy and bytes available. Stale-ID/overflow recovery not all exercised natively in E2E. No arbitrary-HTTP-error offload. | No |
| **C9: resource refusal, missing attachment, native fatality, uncertain effect** | Runtime admission/fatal fence; Harness owns history, attachments, tools and operator lifecycle [S1,S6,U1–U4]. | Existing refusals/cancellation; Phase 3 capped-summary rejection; E2E positive recovery only. | **Supported with explicit limitation:** operator may reattach, use supported client history/compaction/configuration, or retire failed process after effect reconciliation. No guaranteed automatic recovery. Durable Files, crash-safe effects, lost exact Chat lease recovery are **Upstream/excluded scope**. | No normal-path defect demonstrated |

### Source anchors checked

Local links resolve in this repository; symbol names disambiguate line drift.

- **S1:** [`production_mtp.py`](../../ds41f_mlx/serving/production_mtp.py),
  `_schedule_waiting` (52–142), `_retire`/abort/step (144–320): pre-clone discount,
  independent Messages reconstruction, source validation before native start,
  paired publication/fatal containment and settled tool suffix. A failed discount
  admission propagates refusal; it does not invent an earlier state or force C=0.
- **S2:** [`paired_checkpoint.py`](../../ds41f_mlx/serving/paired_checkpoint.py),
  `image_keys/image_ranges`, `PairedCheckpoint.capture/restore`, authority
  `acquire/publish`: identity includes digest/span/grid and exact prefix; damaged
  ordinary candidates invalidate to earlier root/miss; no Target-only restoration.
- **S3:** [`runtime/multimodal.py`](../../ds41f_mlx/runtime/multimodal.py),
  `validate_prefix/workload/sources`, `encode_new`, `prepare_multimodal`;
  [`internal_mtp.py`](../../ds41f_mlx/serving/internal_mtp.py) `_start` (196–337):
  same existing append/tap/N−2/N−1/P5 path, sparse absolute embeddings, restored
  pair or new caches. There is no historical-tool invocation in reconstruction.
- **S4:** `production_mtp.py` `_pump/_cancel/ordinary_response` (426–571),
  `close_serving/shutdown_serving`; `internal_mtp.py` `_retire/close`:
  same-ID shielding, source/native/maps cleanup, executor wait.
- **S5:** [`deepseek_recipe_backend.py`](../../ds41f_mlx/serving/deepseek_recipe_backend.py)
  constructor (203–214), `_call` (226–250): single worker and default 32-entry
  trace histories (serving aliases `session_traces`), not unbounded transcripts.
- **S6:** [`server.py`](../../ds41f_mlx/serving/server.py) preparation/API and
  startup/shutdown hooks (262–305);
  [`image_files.py`](../../ds41f_mlx/serving/image_files.py) `ImageFiles`,
  `resolve_messages`: 128 files/64 MiB, expiry sweep, explicit not-found code,
  immutable request snapshots; Files shutdown and inference shutdown are distinct.

Official source links below pin **U** to the inspected Harness revision:
[U tree](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317).
Paths are relative to that tree; the [Phase 1 source map](phase1-gap-audit.md#2-source-map-and-verified-caller-paths)
provides the full caller chain and Recipe/DeepInfer comparison.

- **U1:** `packages/compaction/compaction-image-offload/src/index.ts:25–49`,
  `llm/llm-deepseek/src/images.ts`, `llm/llm/src/content.ts`: only recognized
  adapter count/byte policy refusal selects occurrences. Offload retains attachment
  references and substitutes text, not tool replay or remote KV invalidation.
- **U2:** `packages/compaction/compaction-basic/src/index.ts:144–345`,
  `region.ts`: automatic pressure and recognized context-overflow recovery,
  balanced selection, complete summary and strict shrink before durable replacement.
  Headless rc.2 lacks supported human `/compact` dispatch; automatic pressure is
  the demonstrated route, not `/compact` submitted as user text.
- **U3:** `packages/llm/llm-deepseek/src/translate.ts:102–165`,
  `packages/core/agent-loop/src/agent.ts:450–560`: closed blocks/stop reason and
  terminal required; failures settle attempts/interrupted content, not executable
  invented calls; tools follow successful stream completion. `llm-retry/src/index.ts`
  is a provider-policy request retry owner, not an effect reconciliation service.
- **U4:** `packages/llm/llm-deepseek/src/adapter.ts:90–170`,
  `request-files.ts` (`retry`), `file-store.ts:198–316`, `transport.ts:22–40`:
  one recognized stale-ID retry, upload failure permits request-wide inline
  fallback, quota reclaim requires list/delete. Runtime lacks list, so full remote
  quota cleanup is not compatible; fallback remains available within inline/body
  limits. Provider INVALID_REQUEST does not become IMAGE_OFFLOAD_REQUIRED.

## Supported Production envelope and meaningful limits

- Single operator, macOS/M3 Ultra **512 GB**, one persistent application/model per
  process; Flash-1017 original checkpoint, native GLM/paired MTP, delivered Recipe
  0.2.0 and pinned oMLX/package identities matching the retained evidence. Thinking
  OFF is the integrated evidence scope. No other host/variant/profile inherits it.
  Loopback or explicitly trusted private LAN only; no multi-user/Internet claim.
- Official Harness rc.2 native independent Messages, attachments and tools;
  `session-log-deepseek.enabled=false`, addition-only tool updates, no unsupported
  Messages features. Harness remains sole history/offload/compaction/tool owner.
- Use the [executed supported configuration](phase4-e2e-evidence/effective-configuration.json)
  as the initial operational policy: count **3**, count quantum **1**, automatic
  pressure **8192**, retention **1024**, summary cap **2048**, ordinary output
  **1200**, generic provider retries **0**. Do not use an unbounded retry policy
  to conceal failed mutation or ambiguous delivery. Official pre-HTTP offload and
  bounded stale-ID handling are distinct from resending uncertain generations.
- Size current images/projection so reconstruction fits **36,864 aggregate
  patches** (thus no dependency on the broader discount route). Count three alone
  does not prove that. Preserve 16 MiB body, 64 MiB encoded request bytes,
  16,777,216 source pixels, per-image geometry/format, context/output, queue and
  physical-headroom limits. The E2E sample used small 384×384 sources and at most
  9353 prepared+reserved positions; this is measured evidence, not a new product
  maximum or a guarantee for every admitted input. Maintain reserve/pressure/swap
  monitoring; native peaks are not inferred from logical array accounting.
- No fixed cumulative image-inspection ceiling follows from bounded current
  residency. **Files storage is separately capped**, not automatically purged by
  offload: IDs expire/sweep; uploads may hit quota before expiry. Official fallback
  can continue using current attachment bytes if the entire inline request fits
  Runtime's smaller body limit. Do not assume Harness's 20 MiB inline/128 MiB raw
  budgets equal Runtime admission. Saturation/expiry recovery over long sessions
  has not been natively qualified; no inevitable exhaustion blocker follows from
  capped storage plus available bounded fallback. Reattachment/operator action is
  required if attachment bytes or normalized paths are actually gone. No permanent
  attachment or restart-persistent File-ID guarantee is made.
- Compaction summaries can truncate or fail to shrink; Harness rejects without
  replacing history (observed in [3B-R2](phase3-step3br2-native-compaction.md)).
  The successful 2048-cap policy is not a promise every summary completes. Ordinary
  user/operator intervention on failed compaction is acceptable; never fabricate
  a replacement or automatically repeat an uncertain effect. A legitimate new
  canonical projection with available admitted inputs can reconstruct; that is
  different from replaying an unsettled generation.
- On interruption, absent valid terminal or uncertain external effect, stop and
  reconcile using Harness/tool/operator knowledge. Runtime cannot certify whether
  a client received a settled response and has no Messages outcome lookup. On a
  native fatal fence, retire/restart the process only after dealing with effect
  uncertainty; reconstruction of known canonical history is possible, native
  persistence or automatic crash recovery is not promised.

## Remaining blockers, deployment versus closure, and smallest next decision

**Concrete implementation blockers, ranked: none demonstrated within this envelope.**
In particular, there is no evidenced legitimate admitted normal Harness projection
that fails due to a missing reconstruction capability. This is not a universal
proof: arbitrary projection variants have model-free rather than native coverage,
and no endurance claim follows from nineteen successful requests.

The broad original contract still has open C2 case/numerical qualifications, C7
changed-path failure/peak and same-process reclamation obligations, and broader
C6/C9 negative/recovery evidence. Therefore **do not mark these criteria PASS or
close the branch as fully C1–C9 complete**. PARTIAL is not BLOCKED. Equally, an
unexercised image-retaining compaction, unavailable native fractional-image path,
universal schedule comparison or weakref-death requirement does not demonstrate
an ordinary-use blocker. The framework-retained single loaded model is not growing
per request; repeated in-process app/model recreation is outside this envelope.

No next implementation or native acceptance task is necessary to resolve an
identified functional blocker; do not repeat C8 or begin Phase 4C. Actual installed
promotion remains a **separate authorized operations decision** under
[operations](../../docs/operations.md#source-qualification--promotion): current
Production installations were never updated by this branch's isolated evidence.
Do not treat this recommendation as permission to overwrite or as fresh acceptance
of an unidentified wheel. Any later package delta needs its ordinary affected
qualification, not another whole-session confidence campaign by default.

**Proposed minimal CURRENT clarification (not applied by this review):** add a
short separate practical-deployment disposition linked to this report, keeping
C1–C9 completion and all historical statuses unchanged. State that (1) C2 does not
require universal equivalence across MTP schedules or fractional-image execution
when valid safe-pair/miss selection is used, (2) C7 ordinary server readiness
requires bounded request/native retirement and graceful process exit, with
same-process unload/reload explicitly separate, and (3) C6/C9 allow documented
operator intervention and fail-closed unsupported recovery rather than automatic
retries for every refusal. Existing numerical distinctions already support (1);
no relaxation of input admission, identity checks, settlement or uncertain-effect
fences is justified. Adopting this clarification is an explicit contract-owner
choice, **not retrospective PASS or silent replacement of the original contract**.

Lightweight validation: `git diff --check`, repository self-containment and
single-file authority-label checks passed. The latter checks wording/links, not
runtime qualification. No tests importing/loading the model were run.

**Final decision: yes—normal DeepSeek Harness Production operation can proceed
within the declared supported envelope; no concrete operational capability is
shown missing. Full broad-contract branch closure remains a separate decision.**
