# DeepSeek Harness 0.2.2-rc.1 compatibility audit and bounded E2E

## Decision

**PARTIAL overall; compatible within the tested, restricted macOS Web / Native
Messages scope. Not unconditional rc.1 or Windows Desktop compatibility.**

Official rc.1 Web → official Native provider → admitted installed Flash-1017
Runtime completed **41 real-model Native requests**, **15 exact-ID Tool
Call/Result pairs** (14 successful, one intentionally aborted), multi-image
continuation, official offload, manual compaction, one automatic Goal round,
three Harness restarts, two model cancellations and one running-tool cancellation.
Two deliberately invalidated provider File IDs recovered by official reupload,
including one after a Harness restart. Canonical/pair/wire checks pass.

No new integration blocker was reproduced in this envelope. `tool_removal` still
fails closed and remains unsupported. General Cordis/Plugin-manager lifecycle
coverage is partial. **Every Windows Desktop execution test is NOT TESTED.**
No implementation fixes, admission changes, production configuration changes,
credential changes, custom Harness fork, push or deployment promotion occurred.

### Compact compatibility matrix

Statuses describe the bounded paths below, not universal feature parity.
**Totals for these ten rows only: PASS 7 / FAIL 1 / PARTIAL 1 / NOT TESTED 1 /
NOT APPLICABLE 0.** The FAIL is a preserved unsupported feature, not a new regression.

| Capability | alpha.2 Baseline | rc.1 Result | Evidence |
|---|---|---|---|
| Native Messages | PASS, 26 real requests | **PASS**, 41 admitted requests; canonical streaming | `audit.json`, `wire.jsonl`, native traces |
| Tool execution | PASS; text cancellation only | **PASS**, scoped sequential execution, exact IDs, running harmless-tool cancellation and recovery | requests 1–9, 16–17, 24–30, 38–40; UI 41–43; fixture log |
| Deferred tools | PASS after corrected Recipe | **PASS**, 15 all-deferred requests, live activation/withdrawal and restore | requests 6–20; retained tool additions; no dummy schema |
| Skills / Plugins | PASS scoped registry/Skills; broader management unqualified | **PARTIAL**, explicit/model Skills, catalog replacement, registry lifecycle and restart pass; new loader/journal failure paths not executed | UI 12–14, fixture log, persistence; §6 |
| Slash commands | PASS manual compact + one Goal round | **PASS**, actual browser `/compact` and `/goal`, one round completed | command seq 191/221, compaction 192/195/197, Goal 222/242 |
| Vision / Files | PASS bounded A/B, offload and reacquisition | **PASS**, four composer attachments + one image-tool occurrence; up to three retained occurrences; two stale-ID recoveries | requests 15–26, 36–37; recovery stimuli; source hashes |
| Session recovery | PASS Harness-only restart, text cancellation | **PASS**, three Harness restarts, browser reload, two model cancellations, tool cancellation, retained-image continuation | UI 10–11, 30–43; 336 persisted events |
| Paired KV invariants | PASS 40 Target / 3 DSpark, replay/repack zero | **PASS**, all 41 canonical frontiers and complete compatible roots verified | `prepared.jsonl.gz`, `trace-*.json.gz`, `audit.py` |
| Windows Desktop | NOT TESTED | **NOT TESTED**, no Windows machine/session available | §7 manual procedure; no binary execution/provenance claim |
| `tool_removal` | Unsupported / rejected | **FAIL**, unchanged unsupported semantic capability, excluded by `addition-only` | HTTP 400 probe; unchanged Harness serializer; unchanged official Recipe HEAD |

## 1. Exact identities and environment

Execution: **2026-11-06 UTC / 2026-11-07 JST**. Apple M3 Ultra, arm64,
512 GiB RAM; actual OS `macOS-26.5.2-arm64-arm-64bit-Mach-O`, build 25F84, Darwin 25.5.0
(see `environment.json`; checked directly rather than inheriting baseline identity).
Python 3.13.15; isolated Node **24.22.2**, Playwright **1.65.0** /
Chromium **158.0.8127.18**. System Node 24.5.0 does **not** meet
Harness >=24.9; it was not used to run Harness. This minimum already applied to
the alpha.2 baseline, not a newly discovered rc.1 protocol requirement.

| Component | Exact identity |
|---|---|
| Repository examined | `c7b7e16` at task start; executable Python files all match the tested installed package (`source-comparison.json`) |
| alpha.2 source anchor | `dsh-v0.2.2-alpha.2`, `f3803c96974ad4a94195fb97e5c637e9c2a3afe3` |
| rc.1 source anchor | Official `dsh-v0.2.2-rc.1`, `66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db`; remote tag resolved before checkout |
| Harness executed | Official npm `@deepseek-ai/dsh@0.2.2-rc.1`, **320 dsh packages all rc.1**, unpatched. Lockfile URLs/SRI and installed package-tree SHA256 in `package-lock.json` / `official-packages.json` |
| Cordis executed | `@deepseek-ai/cordis` 4.1.0; Loader/Include/Group 1.1.0; Cosmokit 1.9.0; Schemastery 3.19.0 |
| Runtime executed | 0.23.0, installed source provenance **`960036e9d831f25d1fa948e3d932cbce302427f1`**, native GLM, `mtp-serving-v1`; isolated copy of `/Volumes/SDXC-512/ds41f-1017`, not execution in the production environment |
| Recipe | 0.2.0; source `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus existing consolidated native patch including all-deferred correction. Wheel provenance `be5299f4649675ce2e6159a87bcc4a89c05bf24aa3b0bc7ba3f58d5124df38bb`; native `93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2` |
| Official Recipe re-audit | Current official HEAD **`935817c66f5d6e64838f419692545f06787e485f`**, unchanged from previous removal audit; no `tool_removal` representation/encoding found |
| DeepInfer-aligned dependency lane | Admitted retained oMLX 0.7.0 payload, MLX/Metal 0.32.2, mlx-lm `0.31.4.dev132+g94cdcae13`; exact dependency versions, payload hashes/origins in `installed-admission.json`. No newer DeepInfer checkout substituted |
| Installed execution identity | **`1c2c5c385e9a5812880a9f5007a9b704edfecc6bf77b27d7a72c68dcabde1545`**; admitted dependency identity, not a substitute for source provenance |
| Model | Official DeepSeek-V4.1-Flash-1017, revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, `flash-1017-pre-woa-fp8-v1`; read-only checkpoint `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Session | **`session-1066f8e2-5b35-4dac-b7d8-08bd0d7e2be1`**, one accepted conversation; unused draft is not inference evidence |
| Windows Desktop | No installed/executed binary; no installer hash or Windows build equivalence asserted |

Native/dependency/checkpoint-byte admission passed before model load; the copied
checkpoint hash cache is isolated. There was one model load, no runtime reload.
The oMLX installation metadata records a local build-directory origin, not a new
independently verified DeepInfer Git commit. The admitted installed payload and
its retained hashes are the execution authority; no latest-upstream DeepInfer
qualification is inferred. The npm binaries are identified separately from the source tag: neither a
reproducible source-build claim nor Windows-binary equivalence follows from npm.
The official GitHub release metadata (`release.json`, published November 4) has
no attached assets; it is not an installer verification receipt.

### Isolation and configuration

Disposable root `/tmp/ds41f-rc1-e2e`; Runtime `127.0.0.1:18097`, Web
`127.0.0.1:18197`. `DSH_HOME`, `DSH_AGENTS_HOME`, npm, interpreter prefix,
checkpoint-check cache, KV root, fixture workspace and browser context are isolated.
The provider base URL has **no `/v1` suffix**. Requests go to `/v1/messages`.

Reused alpha.2 overlay: `addition-only`, Session-log upload disabled, plugin
inventory enabled, thinking disabled / effort off, explicit 32768 context and
1024 output budget; retries, telemetry, title requests and account authentication
disabled. The Read Only UI policy is selected before any Tool execution. Only
harmless receipts and official read/image/Skill/Goal tools are exposed. No shell,
write, plugin-install or arbitrary-network tools. The first text turn has the
Web default Workspace Write policy but executes no Tool.

The initial image watermark is the baseline's **1**; before any image input the
first Harness-only restart raises it to **3**, removal quantum 1, to test retained
multiple-image occurrences and then deliberately cross the official offload
threshold. These are isolated **Harness orchestration** settings, not Runtime
admission caps or unsafe overrides. Runtime admission is unchanged. No image
failure was hidden by shrinking Runtime policy. `acceptance-initial.patch.yml`
and `acceptance.patch.yml` retain both settings.

The copied baseline diagnostic plugins are not official product modifications:
`observer.mjs` is passive; `fixture-initial.mjs` supplies registry receipts and
mounts official plugins. The last Harness restart uses `fixture.mjs`, adding one
abort-aware delayed **read-only test Tool** solely to observe running-tool
cancellation. It logs begin/abort and never reads the receipt if aborted. The
official npm files are not patched (`official-package-verification.json` rechecks
all 320 package trees unchanged after execution). Existing production settings were not read
for credentials, rewritten, or used as the test profile.

## 2. Upstream alpha.2 → rc.1 audit

Sources: official [comparison][compare], [rc.1 tree][rc1], retained
`upstream-commits.txt`, `upstream-diff-stat.txt`, `source-delta.json`,
`native-provider.diff`, `orchestration.diff`, `projection.diff`, `desktop.diff`,
`extensions.diff`, and `vendor.diff`.

**Important:** rc.1 release notes summarize changes since **0.2.1-rc.5**, not since
alpha.2. Their image-upload, tool-executor split, conditional Tools, default model
capacity, Skills fallback and PTC headlines cannot all be called new alpha.2→rc.1
changes. Final source diff, not commit-message presence on merged side branches,
is the authority for the delta below. Tests named here were inspected as upstream
coverage; **the upstream full test suite was not executed**.

| Area / confirmed delta or continuity | Source in rc.1 / change anchor | Owner and ds41f impact | Existing coverage / new E2E need |
|---|---|---|---|
| Native registration, step activation, admission and settlement: core implementations **unchanged** | `packages/core/{tools,agent-loop,tool-executor-native,tool-executor-ptc}/src`; source-delta receipt | Harness orchestrates, Recipe IDs/streaming, Runtime canonical authority; no new runtime feature indicated | alpha.2 E2E and runtime tests remain relevant but not sufficient. Fresh exact-ID and sequential calls executed here |
| Deferred/all-deferred history projection and tool identity/order **unchanged** | core Session/tool history, `llm-deepseek/src/serialize.ts` (generic type annotation only), wire types | Existing corrected Recipe still required; withdrawal in addition-only reprojects history, not removal encoding | Existing all-deferred tests + alpha.2 baseline; rc.1 requests 6–20 and exact Result continuation pass |
| Native provider implementation: typed purpose/route generics, request identity factored for extensions; compact header preserved | `llm-deepseek/src/{adapter,serialize}.ts`; `a36ddf4c40`, request-purpose refactor | Harness/Recipe HTTP boundary; no new Native body or settlement contract found | Upstream serialize/stream/loader tests; real text, Tools and native summary request needed and executed |
| PTC vs Native: executor implementations unchanged in final delta; Trajectory now retains actual failed Tool text / subcall display | `core/tool-executor-*`, `client/ui-trajectory` and `client/ui-tool`; `86bf2734c6`, `c47d9a3868` | Harness presentation/execution ownership. Native Tool output does not authorize PTC execution. PTC is outside this declared execution lane | Existing executor/Trajectory tests; no PTC E2E claimed. Native aborted result observed, not PTC ABORTED parity |
| Cordis lifecycle/configuration materially changes | `vendor/{cordis,loader,include}`, `a91a827546`, `bf1235541c`, `60d811de43` | Harness framework owns raw/effective config, awaited update/commit, dependency teardown; no parallel Runtime plugin owner | Upstream `scripts/cordis-{config-access,core-lifecycle,include-journal,entry-commit}.spec.ts`, boot/HMR tests. Scoped mount/dispose/restore executed; exhaustive failure/journal/HMR permutations still need E2E |
| Include/Loader cross-tree moves, colon IDs and refresh rejection containment | `f45c48dce4`, `7a5df74f31`, `9a26b453c3`; vendor log V25–V27; boot recompose awaits old fibers, avoids duplicate retries | Harness; potential plugin enable/disable/persistence correctness. Not a Recipe/MLX fix | Existing journal/entry tests; current fixture uses no cross-tree move or failed config commit. NOT VERIFIED here |
| Plugin discovery/delivery/installation diagnostics change | `boot/plugin-manager/src`, `dbd881ab46`, `90838b790a`, `9005026ea6`; official bundles / optional CUA delivery | Harness package management; optional platform plugin is not proof of model/tool interoperability | Upstream manager/operations, desktop CUA packaging tests. Catalog/mounted fixture observed; real installation failure and third-party update rollback not tested |
| Skills discovery/activation implementations **unchanged** | `skill/{skill,skill-filesystem,tool-skill}/src`; only tests/manifests change | Harness instruction/history owner; no runtime Skill mechanism | Filesystem and catalog tests + alpha.2. Explicit/model Skill and watched catalog replacement executed; inaccessible-root permutations not repeated |
| `/goal`: disposal abort now disarms continuation without durable pause, keeping active phase | `goal/goal-round-driver/src/index.ts`, `f26f308e5e` | Harness Goal/session shutdown owner; ordinary completion unaffected | Upstream `goal-round-driver.spec.ts`; completed round executed. **Active Goal during shutdown/disposal NOT TESTED**; completion does not prove that new branch |
| `/compact`: command/runtime summary code unchanged; purpose typing added | `compaction-basic/src/types.ts`; command/summary implementation comparison | Harness compacts history, Recipe creates Native request; changed history legitimately misses prefix cache | Upstream manual-compaction tests; genuine summary, durable replacement and follow-up executed. General summary fidelity and automatic compaction not qualified |
| Session projection: one event-level frame, selected-unit snapshot materialization, released units retained in checkpoints | `945006c984`, `eb1713f856`; `session/session-projection/src/index.ts`, controller listener/list changes | Harness history/client/persistence owner; Runtime must only see resulting canonical request | Upstream registry/registration-reader/summary/cache and session-controller tests. Restarts/reload/compaction exercised; released-unit unload + failed persistence not exhaustively tested |
| Legacy Session history migration validation changes for old subagent descriptors | `dffda31215`, `e51cb3959f`; `session-format-v0-to-v1/src` | Harness; no alternate ds41f history owner | Upstream legacy/relationships/validation tests. New-format Session resume executed; legacy migration NOT TESTED |
| Files reuse, expiry recognition/reupload, normalized image prep and immediate composer upload **unchanged** | `llm-deepseek/src/{request-files,file-store,files-api,upload-index,images}.ts`, `client/file-upload/src` | Harness retains original attachments/index; Runtime Files remain process-local. New UI uploading does not confer persistent provider IDs | Upstream files/file-store tests + prior Files reports. Actual rc.1 uploads, stale 404 recovery twice, offload/reacquisition and quota diagnostics executed; TTL expiry/runtime restart not tested |
| Model catalog/limits Native config and default capacity implementations **unchanged** | `llm-deepseek/src/{models,model-info,config,defaults}.ts`; model-row/select UI layout changed | Harness selection/capacity policy; explicit limits still required for smaller endpoints. No new MLX capacity implied | Existing catalog/model settings tests; pinned route executed only. Model switching/smaller-context failure NOT TESTED |
| Desktop: trusted original-file download / clipboard fallback IPC and DevTools menus; CUA macOS helper delivery/onboarding | `apps/desktop/src/{file-actions,ipc,main,preload-app}.ts`, `1de7f17cc0`, `4214f86da3`, `dbd881ab46`; Windows build-root cleanup `b2a7d2ec95` | Desktop/Host owns transport and native permissions; no evidence of changed Native Files recovery contract. CUA source for common Computer Use unchanged, macOS helper changes are not Windows qualification | Upstream file-actions/preload/startup/CUA tests; official Windows installer/Host/CUA execution **required, unavailable** |

Session-log request-upload exclusion refactoring (`a36ddf4c40`) does not add
Runtime ingestion of `dsh_session_log`. That extension remains disabled/rejected.
No delta in the selected Native provider introduces a required Runtime header,
new Tool settlement authority or a persistent Files contract.

### `tool_removal` gate

rc.1 still serializes the official wire block:

```json
{"type":"tool_removal","tool":{"type":"tool_reference","name":"baseline_receipt"}}
```

This wire shape and its in-history position already existed in alpha.2. A wire
schema is **not** the model's authoritative deactivation encoding. Official
[Recipe HEAD][recipe] is still `935817c...`; its request converter/semantic representation
and V4.1 encoder provide no removal implementation. The retained search found no
removal encoding. No private specification or authenticated global code-search
claim is made.

`tool-removal.request.json` uses that official shape against the actual isolated
Native endpoint: **HTTP 400**, native deserialization `unknown variant
tool_removal`, before inference (`tool-removal.response.json`). This negative
capability test does not require model sampling. It is deliberately separate
from actual addition-only Harness requests, which send **zero** removal blocks.
Harness's durable internal `tool-removal` event after withdrawal is projected
away; it does not establish wire removal support. Prior unsupported classification
is unchanged; no guessed encoding, dummy activation or removal emulation added.

## 3. Baseline evidence reused and evaluated

[alpha.2 E2E][alpha] supplies the previous 26 requests / 12 successful Tools,
installed identities, observer/fixture/browser automation, read-only canonical
joins, persistence inspector and acceptance overlay. This investigation reuses
those scripts and image bytes, **not its PASS label for rc.1**. Current
capabilities, operations, implementation plan, runtime completeness,
[alpha.2 audit][alpha-audit], [all-deferred correction][deferred], production
promotion/heartbeat reports and [Files contract][files] were consulted.

The old `/tmp/ds41f-alpha2-e2e` install was stripped by prior cleanup (directories
survived but files did not); it was not an executable baseline. The task instead
copies the admitted current installed production package to a new disposable
prefix. Its executable Python files match current source, and the existing
corrected Recipe/native identity is admitted. No build or fix was necessary.

Every alpha.2 accepted scenario family is evaluated against actual rc.1 here:
Native text/stream, tool registry additions/withdrawal, all-deferred, sequential
Tools, explicit/model Skills/catalog replacement, images/offload/reacquisition,
manual compact, Goal completion, cancellation/fresh continuation, Harness
restore, persistence and canonical paired roots. Added coverage: composer image
uploads, simultaneous retained images, two stale-ID recoveries, browser reload,
image-bearing Harness restart and running harmless-tool cancellation.

Supplementary model-free checks:

- `source-model-free-tests.log`: **158 passed**, using the existing source-tree
  tests and authenticated installed dependency subprocess. Covers Recipe Messages,
  all-deferred/removal rejection, image Files, quota and multimodal cancellation
  doubles. These are not real-model acceptance by themselves.
- `rc1-quota-tests.log`: **22 passed**, the overlapping quota tests repeated with
  `HARNESS_NODE_MODULES=/tmp/ds41f-rc1-e2e/harness/node_modules` and Node 24.22.2.
  The existing diagnostic invokes unmodified actual rc.1 Files client/store over
  loopback TestClient-served HTTP. This qualifies bounded quota handling, not
  Vision inference or Desktop.
- Initial `-I -m pytest` attempts failed at collection/origin assertions because
  these tests explicitly require source-tree imports; no guard was removed.
  Rerun without `-I` satisfies their intended contract. Logs retained. The first
  standalone inspect also lacked required `DS41F_CHECKPOINT`; the actual worker
  sets it and passes full admission. A redundant inspect was time-bounded before
  completion; its empty output is not acceptance evidence.

## 4. Actual E2E evidence

All request indices below are zero-based **admitted** requests in `audit.json`;
expected rejected requests are separately retained in `wire.jsonl`.
Browser actions/screenshots and `client-actions.jsonl` record actual UI dispatch,
not slash text sent to a headless substitute or replayed request bodies.

| Scenario | Observed result / receipt |
|---|---|
| Initial text | Request 0 returns `RC1_TEXT_OK`. Earlier UI 1 was deliberately not accepted: sent before server startup and failed TRANSPORT; no Runtime HTTP request or Tool effect occurred |
| Tool registration, ordering, withdrawal | Requests 1–9: BOOT-438, late registration LATE-927, two separate ordered calls in one turn, baseline disposer withdrawal, late-only all-deferred continuation; first Harness restart restores same Session |
| Skills | Requests 10–14: explicit `/receipt` source `skill-invocation`, actual `skill` Tool, watched replacement catalog `receipt-next`, `SKILL-ALPHA-614` then `SKILL-BETA-827` |
| Composer Vision and interleaved Tool | UI 15–19 / requests 15–18: A blue triangle AX-31; B orange square BX-72; both compared, one actual late receipt, subsequent retained-image request with no upload or Tool |
| Image-tool occurrence and multiple attachment offload | Requests 19–21: genuine read_image(A), then another composer B. Three retained image occurrences before threshold crossing; official pre-HTTP `IMAGE_OFFLOAD_REQUIRED` → `image/offload` seq **178**, targeting image event 134. No image-tool replay; request 21 reconstructs with three actual image occurrences |
| Stale B File ID | Diagnostic DELETE of isolated B ID; next official request receives **404 file_not_found**, official code uploads B again and retries once; request 22 succeeds with orange square BX-72. No history/index edits or manual body retry |
| Manual compact and continuation | Browser `/compact` seq **191**; Native summary request **23**; start/summary/end **192/195/197**, durable replacement and successful command completion. UI reports **53 history items (~1993 tokens)**. Queued continuation runs afterward, request 24 C=0, late receipt once in 24→25; no prior effect replay |
| Fresh image after compact | Composer A, request **26**, authenticated original bytes, one new Vision occurrence, correct AX-31. Old images need not survive in the compacted prompt to answer historical text |
| Goal | Browser `/goal` seq **221**; create seq **222**, one automatic round, actual late_receipt/get_goal/update_goal, complete seq **242**, revision **1→2**, active→complete, roundsStarted=1; driver stops |
| Model cancellation before wire publication | UI 29/30; request **31** settles after one canonical token but **zero wire SSE events**. Follow-up **32** returns `CANCEL_RECOVERY_OK`; incompatible unseen output is not selected as a prefix |
| Browser reload | UI 32–33, request **33**, newest image AX-31 and `RELOAD_OK`; retained pixels still represented in prepared request |
| Mid-stream cancellation | UI 34/35, request **34**; model starts a refusal/explanation rather than the requested number list. Stop interrupts real streamed text, not a successful counting task. Follow-up **35** returns `STREAM_CANCEL_RECOVERY_OK`; no Tool certificates/effects in either cancelled generation |
| Image-bearing Harness restart | Third application run restores the same Session and existing provider IDs; request **36** returns AX-31 / `HOST_RESTART_OK` with retained image and no reupload. Runtime stays loaded |
| Stale A after Harness restart | Diagnostic DELETE of A ID after that successful restore. Official request gets 404, reuploads A and retries once; request **37** returns AX-31 / `POST_RESTART_RECOVERY_OK`. This is **not** a Runtime restart or Windows test |
| Running Native Tool cancellation | Last Harness restart mounts diagnostic slow Tool. Request **38** canonically authorizes `slow_receipt`; fixture begins wait at 16:43:26.810Z. Browser Stop at UI 42 aborts it at **16:43:36.115Z**; **no read effect**. Exact-ID error Result seq **320**, `Error: The operation was aborted`. Requests **39–40** carry that same Result, execute a new late receipt once, and return `TOOL_CANCEL_RECOVERY_OK`. No automatic slow retry |

The cancelled Tool ID is **`call_00_BNTAsGngBBrsk94tGfhW2119`**; recovery Tool
ID is **`call_00_THsvaU6P0A21IYo58ZyN3322`**. Every call, arguments, result sequence
and next-request association is retained in `audit.json`. Interrupted Tool
execution happens **after** its model request's canonical settlement; it is not
an interrupted, uncertified native tool suffix.

The compact summary contains a model-authored factual imprecision suggesting the
withdrawn baseline Tool remained callable. Actual calls show **no baseline
execution after withdrawal**, and no replay. Treat summary prose as model output,
not a diagnostic source of registry truth. Summary generation and continuation
pass; perfect summarization fidelity is not claimed. Smallest action: inspect
original Session/Tool log when operational decisions depend on summaries; no
Runtime fix is justified by this sample (model-quality/orchestration boundary).

## 5. Canonical, paired, resource and persistence invariants

`audit.py` is the rc.1 adaptation of the alpha.2 auditor, retaining the same
canonical/wire/paired checks and adding final scenario counts, prompt accounting,
image digests and cancelled-Tool assertions. It does not fabricate sessions,
requests or cache roots. **PASS: 41 serial requests, 15 exact-ID calls/results**.

- **41 HTTP 200** admitted real-model requests. The same passive wire log also
  has **two intentional 404s**, both followed by official upload/retry recovery,
  and **one 400** from the unsupported-removal probe. No unhandled runtime fault.
- For 39 normally completed requests, complete wire SSE equals native Recipe
  events and ends in `message_stop`. Each cancelled wire is an exact prefix;
  zero events for request 31 is explicitly permitted only because its serial
  trace confirms cancellation. Traces are joined in single-worker start order,
  response IDs checked wherever visible; every wire body SHA joins its prepared
  canonical prompt. This is not a claim that cancellation delivers a completion.
- Prompt usage input + cache-read + cache-create equals **canonical prompt ID
  length**, including images. Every settled frontier equals prompt IDs + native
  canonical generated IDs; **40 Target offsets and 3 DSpark offsets agree**,
  prediction retired, queue empty, paired state published.
- Every C>0 selects an actually available complete canonical-prefix pair, with
  **exact covered image identities** and no frontier through a partial image.
  Fresh Vision count equals uncovered occurrences. All image digests are exactly
  the retained A/B fixture hashes, not a filename-only association.
- **Prompt/history replay and full-cache repack = 0** throughout. Offload and
  compaction legitimately reconstruct changed histories at C=0. Repeated A/B
  uploads/occurrences are explicit new user/tool acquisitions, not replayed
  effects. No cache-hit promise is inferred from addition-only.

Representative N/C/fresh-Vision boundaries:

| Request | Boundary | N / C | Fresh images |
|---|---|---|---|
| 6 | all-deferred withdrawal reprojection | 588 / 0 | 0 |
| 15 | first composer A | 2567 / 2191 | 1 |
| 16 | composer B with retained A | 2978 / 2589 | 1 |
| 18 | retained A/B, no new image | 3084 / 3058 | 0 |
| 20 | new Tool-supplied A occurrence | 3617 / 3198 | 1 |
| 21 | official offload / changed projection | 3808 / 0 | 3 |
| 23 | native compaction summary | 4215 / 3812 | 0 |
| 24 | post-compaction projection | 2230 / 0 | 0 |
| 26 | fresh post-compact A | 2683 / 2308 | 1 |
| 31 → 32 | cancellation / safe continuation | 3539 / 3509 → 3565 / 3509 | 0 / 0 |
| 34 → 35 | streamed cancellation / continuation | 3660 / 3619 → 3734 / 3658 | 0 / 0 |
| 36 → 37 | Harness image restore / stale-ID recovery | 3778 / 3745 → 3833 / 3800 | 0 / 0 |

No numerical oracle campaign was run. These are paired/canonical consistency and
bounded Vision correctness checks, not proof of numerical equivalence at every
context size or memory-pressure endurance.

`resource-summary.json`: **437 samples**, minimum available **131.46 GiB**,
maximum MLX active **291.79 GiB**, maximum MLX cache **8.82 GiB**, swap **0 bytes
throughout**. Existing diagnostic stop conditions: 64 GiB available floor, no
swap growth, 2700-second worker limit; none fired. No admission override or
runtime soft-cap change. Final health: ready, fatal_error=null, active=queued=0.
Normal Runtime lifespan shutdown recorded; intended post-uvicorn `runtime-exit`
receipt is absent (same SIGTERM behavior as baseline), so no in-process memory
reclamation claim. Both test ports bindable after cleanup.

`persistence.mjs`: **333 passive emitted events exactly equal official persisted
bytes**, **336 total**, including three official restore `session/end-seed`
sentinels not emitted live. `session.v4.jsonl.zstd` and the official-profile
archive retain original Harness-owned history, attachments and upload index.
No competing persistence owner was introduced.

## 6. Findings, unsupported scope and smallest actions

### F1 — FAIL: in-history `tool_removal` (pre-existing, high impact if enabled)

- Reproduce: use retained removal JSON against `/v1/messages`, or configure an
  isolated Harness route for in-history removals and withdraw a Tool. The former
  was executed; the latter is the unchanged source-derived path, not another E2E.
- Observed: HTTP 400 unknown native request variant, before model execution.
  Expected for full capability: authoritative Recipe conversion/encoding with
  correct deactivation semantics, ordering and continuation. Harness wire shape
  alone is insufficient.
- Ownership: Recipe semantic contract/encoder first; Runtime admission must
  continue failing closed until that contract exists.
- Operational impact: a route emitting removal blocks cannot continue. The
  tested `addition-only` route avoids this unsupported request, with potential
  history reprojection/cache costs; avoidance is **not support**.
- Smallest action: retain `addition-only`. Any implementation requires explicit
  approval and upstream authoritative encoding evidence plus dedicated tests.
  **Further upstream evidence necessary; no fix implemented.**

### F2 — PARTIAL: complete Skills/Plugin/Cordis lifecycle (qualification gap)

- Reproduce verified part: use retained registry baseline→add→drop, mount official
  Skill/Goal/file plugins, replace Skill folder, restart official Harness and
  continue the same Session. Logs show discovery, execution and restoration.
- Unverified part: failed Include journal writes/flush, cross-tree entry moves,
  plugin installation/update rollback, active Goal disposal and arbitrary HMR or
  dependency replacement. Expected for a general claim: no lost durable state,
  duplicate execution, pending dependency deadlock or stale capabilities across
  those operations. No failure is asserted merely from missing tests.
- Diagnostics: official rc.1 boots with only the expected pending account-controller
  (disabled account service; API-key route works). No unexpected activation failure.
- Ownership: Harness/Cordis, not Recipe or MLX. Severity **medium operational
  qualification risk**, high if depending on live failure recovery without proof.
- Smallest action: keep tested preset/registry operations; collect focused
  official loader/journal and active-Goal-shutdown evidence before claiming that
  broader scope. No speculative fix. Further upstream test/execution evidence
  necessary for expansion, not to close the bounded Native lane.

Other boundaries preserved:

- Windows and Windows Computer Use: **NOT TESTED**, not inferred from Web, shared
  sources, macOS helper changes or successful receipt Tools.
- Automatic retrieval of an offloaded historical image (rather than explicit new
  acquisition), provider TTL expiry, Runtime restart/Files loss, permanently missing original
  attachment, repeated stale-ID rejection, crash during unknown effect, true
  effectful cancellation/rollback, high-volume concurrent clients and long-term
  endurance: **NOT TESTED** here. Deleted-ID recovery exercises stale recognition,
  not every expiration or crash lifecycle.
- PTC mode, VM state persistence and SSH bootstrap: **NOT APPLICABLE** to the
  declared Native executor lane; audited for ownership, not accepted by this run.
- Thinking ON, model switching, automatic compaction and Goal multi-round/exhausted
  limits: no new qualification. `/goal` new disposal branch remains untested.
- `dsh_session_log`, URL/path/document image sources, server tools, constrained
  output and unsupported Files download/retrieval endpoints remain unsupported
  as documented. Official Harness recovers retained attachment bytes; it does
  not require the Runtime to become an attachment-history service.
- The 50 MiB composer UI limit does not expand Runtime source-byte/pixel/token
  admission. No near-limit images were tested.

Setup diagnostics are not product regressions: wrong initial CLI argument order
(`--patch` after `web`), early request before Runtime readiness, source-test
import mode, and incomplete inspect are retained and corrected only in test
invocation. Actual Runtime/Recipe admission and accepted rc.1 requests pass.
No MLX/DeepInfer lifecycle/admission failure was observed after successful load.

## 7. Official Windows Desktop — NOT TESTED and reproduction procedure

No Windows host, VM, remote Desktop session or official installer execution was
available. Consequently **connection, Native requests, attachments/multiple-image
continuation, commands, Skills/Plugins, Windows Computer Use, Session restore,
Desktop restart and post-restart File IDs are all NOT TESTED**. No local modified
Harness or Web browser is substituted for these results.

The historical Windows File-ID failure remains **unresolved/unqualified** by this
assessment. Prior repository investigation explicitly lacked the exact failing
Windows install and wire identity; it did not prove universal Files causality.
The rc.1 Native Files recovery implementation is unchanged. Web's two successful
recoveries show the bounded existing contract works with this Runtime, not that
the official Windows failure is fixed or merely bypassed. Immediate composer
upload was already present in alpha.2's final source and still does not persist
provider File IDs across Runtime loss.

Reproducible manual acceptance (use an isolated user profile and approved trusted
LAN endpoint; do not expose unauthenticated Runtime publicly):

1. Obtain **official** Windows Desktop rc.1 from official release distribution.
   Record download URL/time, installer SHA256, Authenticode publisher/verification,
   installed About version, OS/build/architecture and bundled Host/package identity.
   Do not build or patch a substitute. Reject an alpha/other rc as target evidence.
2. Back up existing Harness profile; create a disposable profile/workspace. Point
   its Native DeepSeek provider to the existing runtime base URL without `/v1`,
   explicit endpoint-appropriate context/output limits, `addition-only`, Session-log
   upload off. Preserve production admission/credentials. Record effective config
   with secrets redacted and runtime identity/health before tests.
3. Text turn → genuine harmless sequential receipt reads → exact call/result-ID
   continuation. Exercise deferred activation with only deferred declarations.
   Observe canonical `message_stop` and certificate-before-effect boundary.
4. Attach retained A PNG, confirm blue triangle AX-31; attach B, confirm orange
   square BX-72; compare both, interleave one harmless Tool and continue without
   reattachment. Track composer upload receipt, Harness attachment identity,
   provider File ID, raw/normalized digest and Runtime prepared spans separately.
5. Activate one disposable Skill explicitly and via Tool; check plugin discovery,
   harmless enable/disable/restore. Enable the **official Windows Computer Use**
   plugin through its supported installation flow. Record plugin/version/platform
   availability. On a disposable desktop, perform screenshot/inspection only,
   verify Vision, then one explicitly approved reversible action. No arbitrary
   computer-control permission is implied by Web receipt acceptance.
6. Invoke actual UI `/goal` for one bounded receipt/complete round; invoke
   `/compact`, require real summary + durable replacement + successful follow-up.
   Stop text generation and a harmless cancellable Tool separately; recover with
   new instructions and check IDs and effect log for no automatic replay.
7. Wait for idle and **Quit via tray/application menu** (closing X can merely
   hide the app). Confirm Host exits, reopen the same official Desktop and Session,
   continue retained images without reattaching. Record whether provider IDs are
   reused, preemptively reuploaded, or rejected/recovered. This distinguishes a
   genuine recovery from bypass by fresh upload.
8. Separately, on an explicitly approved **isolated** endpoint, invalidate one
   owned provider ID (or let its ordinary TTL expire). Do not edit Harness index,
   history or image identity. Require recognizable missing-ID response, exact
   mapping invalidation, bounded reupload from retained bytes, one successful
   retry, unchanged pixels, and no Tool replay. Then, if authorized, test actual
   Runtime restart separately: provider Files are process-local. Never delete
   production/user IDs just to provoke a test.
9. Test absent original attachment / repeated rejection as a negative case:
   expect visible bounded failure, never silent mismatched image continuation,
   guessed file identity or unlimited retry. Preserve network status/error body,
   app/Host logs and original Session; redact secrets, not identity/digest joins.
10. Collect sampled memory/MLX use and swap, canonical prompt/Target/DSpark roots,
    effect counts, final empty queues/health. Classify each Windows scenario by
    PASS/FAIL/PARTIAL/NOT TESTED; do not replace this report's Windows status until
    actual official-binary evidence exists.

## 8. Reproduction, retention and final operational decision

Reuse `runtime.py`, `client.mjs`, `observer.mjs`, `fixture-initial.mjs` then
`fixture.mjs`, `acceptance-*.patch.yml` and baseline A/B images. The diagnostic
worker must run **`python -I` from an admitted copied installation**, not source
or an editable package; the source-tree pytest checks intentionally use the
opposite import mode. Require full installed/checkpoint admission; never bypass
an origin, ABI, resource or identity failure to reproduce these results.

Install official npm rc.1 via the retained lock (`npm ci --ignore-scripts`) with
Node >=24.9. `client.mjs` uses Playwright Chromium. Set isolated `DSH_HOME`,
`DSH_AGENTS_HOME`, `ALPHA2_EVIDENCE` (legacy diagnostic variable name pointing to
this artifact directory), and a disposable `DS41F_LOCAL_API_KEY`. Start worker
with the explicit checkpoint variable and the official app with:

```sh
dsh --profile web --patch /absolute/path/acceptance.patch.yml --no-open --port 18197
```

Use temporary fixture files BOOT-438/LATE-927. Skill contents are described in UI
12–14 / retained profile: `receipt` returns SKILL-ALPHA-614, replace it with
`receipt-next` returning SKILL-BETA-827. Replay **UI action instructions**, not
wire bodies or uncertain effects. Registry stages are baseline→add→drop; first
restore uses add→drop→full, next image restore the same; final restore adds slow
after full. Initial/updated overlay corresponds to the first restart. UI actions
25 and 27 were queued while compaction was running; official orchestration
executes them after compaction, not by mutating history externally. Stop the slow
Tool only after fixture `slow_begin`, before its 30-second wait finishes.
Retained stimuli specify the exact isolated IDs deleted during this run; in a
new run select its own IDs by authenticated image association, never reuse those
opaque IDs blindly. Model wording/timing is not deterministic.

Offline verification:

```sh
python3 artifacts/harness-rc1-compatibility/audit.py
node artifacts/harness-rc1-compatibility/persistence.mjs \
  artifacts/harness-rc1-compatibility/session.v4.jsonl.zstd
```

Wire bodies/Files responses, prepared IDs/available paired roots, native traces,
passive app events, diagnostic effects, UI JSON/screenshots, source diffs,
identities, official Session/profile archive and resource samples are retained.
`prepared.jsonl` is compressed losslessly after shutdown; auditor reads either
form. Neither model weights nor installed application/dependency trees are
committed. Temporary processes have exited; test ports are free. Disposable
installations may remain for reproduction and are not production launches.

**Existing Runtime production configuration can remain unchanged.** No tested
Native protocol delta requires a Runtime or Recipe correction. This does not
qualify arbitrary existing Harness presets: rc.1 still needs the declared
protocol restrictions, explicit capacities and a supported Node interpreter.
Do not remove `addition-only` / disabled Session-log upload or change admission
on the strength of this report. The isolated client image watermark is not a
production recommendation.

**rc.1 may be declared compatible only within the tested Native Web envelope**
with the stated partial Plugin lifecycle boundary. Windows Desktop, full plugin
management, effectful unknown-result recovery and unsupported semantics remain
unqualified. No code fix is proposed as necessary for this scope. Decisions
remaining with the user: whether/when to adopt official rc.1, provide Windows
access for the manual validation, and separately authorize any future unsupported
semantic implementation only after upstream authority is available. None is a
prerequisite to close this bounded audit. Documentation/evidence are committed
locally; completion response records the local hash and working-tree state.

[compare]: https://github.com/deepseek-ai/deepseek-harness/compare/f3803c96974ad4a94195fb97e5c637e9c2a3afe3...66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db
[rc1]: https://github.com/deepseek-ai/deepseek-harness/tree/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db
[alpha]: ../harness-alpha2-e2e/report.md
[alpha-audit]: ../harness-alpha2-compatibility/report.md
[deferred]: ../harness-all-deferred/report.md
[files]: ../../docs/harness-image-files.md
[recipe]: https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f
