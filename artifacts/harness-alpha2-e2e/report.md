# Official Harness 0.2.2-alpha.2 — Native E2E acceptance

## Disposition: PASS, bounded configuration and workflows

Actual official Harness Web application → `deepseek-official` Native Messages →
isolated corrected ds41f-mlx → real Flash-1017 completed **26 native requests**,
**12 genuine tool effects**, browser-dispatched commands, Skills, images and one
intentional cancellation. This is not universal alpha.2 feature parity or a
Production promotion. No Runtime/Recipe/Harness implementation was changed.

## 1. Exact source and execution identities

Execution: **2026-10-31 UTC / 2026-11-01 JST**, Apple M3 Ultra / 512 GiB,
macOS 26.5.2 (25F84), Python 3.13.15, Node **24.22.2** (meets Harness >=24.9).

| Component | Identity |
| --- | --- |
| Runtime source | Current-master baseline **`2798732e2fa33e1a74bdecf9e70a69a951946eca`**, Runtime 0.23.0; task branch `acceptance/harness-alpha2-native`. Installed Runtime Python files all match source. |
| Runtime wheel | Fresh non-editable native-GLM / Flash-1017 package; SHA256 **`f9233aa3b3eec1939b7c3fe86ef078aac369dc6e9111f36e5f58df1eb46cb0c3`**. |
| Harness source anchor | Exact official tag **`dsh-v0.2.2-alpha.2`**, **`f3803c96974ad4a94195fb97e5c637e9c2a3afe3`**, rechecked against official remote; reused archive SHA256 `028c9331b0c48161d623ed4c611cdb40f93fbefde96f46fa6242d35215324d82`. No source audit repeated. |
| Harness actually executed | Official published npm **0.2.2-alpha.2** application and all **321 `dsh*` packages**, installed only under the disposable root. Exact resolved URLs, SRI and installed package-tree hashes retained. npm supplies no `gitHead`/build SHA: release-artifact execution is identified separately from the source tag, **not a claim of reproducible source-build byte equivalence**. No official package/source files were patched. |
| Recipe | 0.2.0, official source `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus retained consolidated patch including all-deferred correction. Wheel SHA256 **`be5299f4649675ce2e6159a87bcc4a89c05bf24aa3b0bc7ba3f58d5124df38bb`**; native **`93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2`**. Not the older Production native. |
| Model | Official Flash-1017 revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, numerical profile `flash-1017-pre-woa-fp8-v1`, read-only checkpoint `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`. Fresh installation verified all 48 shard byte hashes before loading; subsequent unchanged-file cache checks retained. |
| Installed execution identity | **`1c2c5c385e9a5812880a9f5007a9b704edfecc6bf77b27d7a72c68dcabde1545`**. Exact native/dependency/checkpoint admission and ABI guards passed. |
| Application Session | **`session-1ace165c-372e-4fe2-9315-6633f4446123`**, one accepted conversation, restored once after a Harness-only configuration restart. One successful Runtime/model load; no reload during the accepted work. |

Machine-readable owners: [identities.json](identities.json),
[official-packages.json](official-packages.json), [package-lock.json](package-lock.json),
[installed manifest](installed-execution-manifest.json),
[installed admission](installed-admission.json), [source match](source-match.json).

## 2. Isolation, configuration and actual client

Disposable root: `/tmp/ds41f-alpha2-e2e`. New Runtime wheel/install, npm installation,
`DSH_HOME`, `DSH_AGENTS_HOME`, browser context, fixture files and KV root are isolated.
Runtime uses `127.0.0.1:18097`; Harness Web uses `127.0.0.1:18197`. The provider's
base URL has **no `/v1` suffix**; actual requests reach `/v1/messages`, not Chat.

[Initial composition](composed-config.yml), [final composition](composed-config-final.yml)
and [profile overlay](acceptance.patch.yml) retain the actual settings:
`addition-only`, thinking disabled, effort off, 1024 output-token cap, 32768 route
context, image high-watermark **1** / removal quantum **1**, Goal cap **2 rounds**
after the restart. The image watermark deliberately exercises official offload;
it does not change Runtime image admission. Session-log upload is disabled;
plugin inventory remains enabled. Retry middleware, auxiliary title generation,
account authentication and telemetry are disabled for a local API-key fixture.
The unused account-controller reports pending while its account service is disabled;
the tested API-key route remains available.

The selected temporary `acceptance` preset offers harmless fixture readers, then
mounts official read/read_image, Skill filesystem/loader and Goal tools through
Cordis. It does not expose shell, writes, Plugin-installation or arbitrary network
tools. Other shipped presets are not accepted by this test.

**Actual browser UI dispatch**, automated with Playwright/Chromium, sends the user
turns and invokes `/compact`, `/goal` and Stop. This is not headless slash-command
text or a synthetic provider. [client.mjs](client.mjs),
[client-actions.jsonl](client-actions.jsonl) and numbered UI JSON/screenshots retain
those actions. Initial UI locator attempts failed before any model request; the
corrected contenteditable selector is used for acceptance. A workspace chooser
created a second unused draft; no model work ran in that draft.

The first baseline turn used the Web default Workspace Write policy but offered
only reader tools. Subsequent accepted turns use the genuine UI-selected **Read
Only** policy. The app's default workspace path appears in the Session header;
fixture readers/image acquisition use explicit disposable paths and perform no
workspace writes. Existing user configuration and Production were not changed.

## 3. Connected workflows and results

Request indices below are zero-based [audit.json](audit.json) rows.

| Area | Actual execution / linked evidence | Assessment |
| --- | --- | --- |
| Agent and exact tool lifecycle | Browser asks for `baseline_receipt`; genuine plugin reads `baseline.txt` → `BOOT-438`. Live registration adds `late_receipt` → `LATE-927`; disposal withdraws baseline. Each real effect has a native certificate, Harness call/result and exact-ID next-request continuation. Requests 0–7; [fixture log](fixture.jsonl). | **PASS** |
| All-deferred | The Web base also offered immediate `working_directory`, so the first withdrawal alone did **not** produce all-deferred. Explicitly disabled that unused tool in the isolated profile and restarted **only Harness**, restoring the same official Session. The fixture re-registers late, then disposes baseline before the next turn. Requests **6–7** declare only `late_receipt`, deferred, retaining its historical activation; real call/result/final response succeed. Later official tool mounts also remain deferred: **ten all-deferred requests, 6–15**, succeed. No initial/dummy tool or manual protocol fixture. | **PASS** |
| Representative Plugin behavior | [fixture.mjs](fixture.mjs) uses real scoped `ctx.tools.register`/disposers and `ctx.plugin` for official plugins. External stage-file changes are explicit test stimuli, not Session or request mutation. Live add/withdraw plus configured unload/reload/official restore are observed. No MCP or arbitrary Plugin-manager lifecycle claimed. | **PASS, scoped** |
| Explicit Skill | User says `Please apply /receipt ...`; official Skill instructions enter history with `source.kind=skill-invocation`; model returns `SKILL-ALPHA-614`. No fabricated Skill Tool Result. Request 8. | **PASS** |
| Model-invoked Skill and catalog change | Actual `skill` calls load `receipt`, then filesystem watcher sees its removal/replacement by `receipt-next`. A durable replacement catalog lists only the new skill; new call loads it and final response is `SKILL-BETA-827`. Requests 9–12; sources and exact call IDs in audit. | **PASS** |
| Image acquisition/interpretation | Genuine `read_image` reads A (blue triangle, AX-31) and B (orange square, BX-72); model correctly identifies/compares them. Official attachments join original PNG SHA256 → Files transport → Runtime spans → native Vision. Requests 13–16; original [A](image-A.png), [B](image-B.png). | **PASS, bounded visual sample** |
| Official offload and reconstruction | Second image exceeds configured route watermark. Official `IMAGE_OFFLOAD_REQUIRED` occurs **before HTTP**, followed by `image/offload` **seq 134** targeting A result **117**. Official projection retries without replaying either tool effect; request 16 reconstructs at C=0 with B's actual pixels. No Runtime recovery workaround. | **PASS** |
| Manual `/compact` | Browser dispatch produces `command/run` **139**, compaction start/summary/end **140/141/143**, genuine native summary request **17**, durable replacement **142** (`replace` seq 8–130), and successful `command/done` **144**. UI reports 41 history items compacted (~1821 tokens). This is real compaction, not merely handler success. | **PASS** |
| Post-compaction/new acquisition | New ordinary request **18** reconstructs changed history at C=0 and calls read_image on A again; new ID/result **155**, same authenticated source bytes as original A. Request **19** restores C=2368 and encodes the fresh image; model confirms AX-31 and contrasts B. Not historical effect replay. | **PASS** |
| `/goal` and automatic rounds | Browser dispatch `command/run` **161**, official Goal creation **162**, genuine Goal-driver user source `kind=goal`, round **1/2**, then actual late_receipt/get_goal/update_goal effects. Goal **`goal-09ece226-93c8-42b0-bcbd-88ed5385dedc`** changes revision **1→2**, phase **active→complete**, `roundsStarted=1`, and driver disarms. Requests 20–23. No claimed initial READY response: the command schedules the automatic round directly. | **PASS, one complete automatic round** |
| Cancellation and continuation | Browser starts text-only counting, then clicks **Stop generating**; no tool effects. Request **24** settles cancellation. A new, non-retry instruction receives `CANCEL_RECOVERY_OK` in request **25**, health healthy/queues empty. | **PASS, text-only cancellation** |

All **12** tool IDs, names, arguments, result seqs and immediate continuation
request indices are in `audit.json`. Example corrected all-deferred call:
**`call_00_fDGdc7c41d0nkJRtzUcr3488`**, call/result **64/65**, requests **6→7**.
Original/new A IDs are **`call_00_yHruwX9SmRuNWh0kcmAD3624`** and
**`call_00_3vvoT23cjJ0qM6wtotF08027`**; both bind SHA256
`3c56ebabdb297c9ee4bf451a9cb698e40ea3f7ef88f46731e1ca4f3344cdb90a`.

## 4. Canonical, paired and failure observations

[Read-only audit](audit.py) joins original wire bodies by SHA256 to captured Recipe
canonical IDs and available paired roots, then native response IDs to complete
settlement traces. **26/26** HTTP requests return 200. For **25 normally completed
requests**, full SSE events equal native events including `message_stop`.
Every request settles all **40 Target layers / 3 DSpark rings** at the recorded
canonical frontier, exactly prompt IDs + canonical generated IDs; prediction
retired, queue empty, paired state published, **prompt replay/repack = 0/0**.
Every C>0 is an actually available complete canonical-prefix pair with exact covered
image identities and whole-image-safe frontier. Fresh Vision counts match uncovered
occurrences. Cache misses are accepted, not disguised as hits.

| Boundary | Request | N / C | Fresh Vision |
| --- | ---: | ---: | ---: |
| Initial baseline withdrawal reprojection | 4 | 603 / 0 | 0 |
| Corrected all-deferred after Harness restore | 6 | 598 / 0 | 0 |
| Exact-ID all-deferred continuation | 7 | 637 / 621 | 0 |
| First A delivered | 14 | 2707 / 2290 | 1 |
| Covered A continuation | 15 | 2790 / 2744 | 0 |
| Official A offload / B reconstruction | 16 | 3064 / 0 | 1 |
| Native manual summary | 17 | 3442 / 0 | 1 |
| Post-compaction changed projection | 18 | 2314 / 0 | 0 |
| New A occurrence | 19 | 2785 / 2368 | 1 |
| Intentional cancellation | 24 | 3946 / 3910 | 0 |
| Safe fresh follow-up after cancellation | 25 | 4162 / 3944 | 0 |

Cancellation is intentionally not labeled protocol completion: wire captures
**188** events, an exact prefix of **192** canonical native events. Browser saw
through 93; in-flight native output reached 95 before quiescence. Target/DSpark
settle at **4135**, and consistent state is published under existing policy.
The follow-up selects the actually compatible **3944** input-prefix pair, **not**
the incompatible 4135 root containing unseen output. It does not resume the count,
invent a completion or dispatch uncertain tools. This is relevant bounded failure
coverage, not cancellation of arbitrary effectful tools.

The only agent request error is the expected official pre-HTTP image-budget
refusal; it recovers through official offload. No transport error, malformed
Tool Result, stale pair, model request retry after unknown effect, or actual
compatibility blocker was observed.

An early worker exited **before model load** because the diagnostic script repeated
installed-origin inspection after lazy Transformers import. The worker now retains
its successful original admission result, as the existing evidence worker does,
and uses the canonical `/private/tmp` interpreter path. No admission guard or
installed package was altered. UI locator/setup and read-only audit refinements
are test-harness corrections, not fixes to product incompatibilities.

## 5. Retention, cleanup and reproducibility

- Passive application log and original official compressed Session agree:
  **207** emitted events exactly match persisted bytes; **208** durable events
  include one official restore `session/end-seed` sentinel not emitted live.
  [Persistence inspector/result](persistence.json), [script](persistence.mjs),
  [official Session](session.v4.jsonl.zstd). The [isolated profile archive](official-profile.tar.gz)
  also retains original attachment objects and official storage. No competing
  Session owner was added.
- [wire.jsonl](wire.jsonl) retains Native Messages bodies/responses and Files
  responses; [prepared.jsonl.gz](prepared.jsonl.gz) retains canonical IDs/images
  and actual available paired roots; `trace-*.json.gz` retains every native trace.
  Two genuine Files uploads serve three image-read occurrences; reacquisition
  can reuse the existing file object but has a new Tool Call/Result ID.
  Application sources, commands, goal changes, compaction and final texts are in
  [harness-events.jsonl](harness-events.jsonl). UI 15–27 captures accepted workflows.
- Sampled minimum available memory **196.1 GiB**, maximum MLX active **292.0 GiB**,
  **zero swap used**, [resource summary](resource-summary.json). Finite sample, not
  resource/performance qualification. Worker bounds were 64 GiB available-memory
  floor, no swap growth and 2700 seconds; no bound fired.
- Final health has no fatal error and zero active/queued requests. Harness,
  browser and Runtime exit; both ports are independently bindable. Runtime log
  records normal application shutdown. Uvicorn re-raises SIGTERM after its lifespan
  shutdown, so the worker's intended post-`run()` `runtime-exit.json` is **not**
  produced; no in-process model-reclamation claim is made. [Cleanup](cleanup.json).
- Reproduction: install exact npm dependencies from the retained lock with a
  >=24.9 Node; build the native-GLM Flash-1017 Runtime wheel from the baseline
  using `tools/build_execution_package.py`, provision its admitted dependencies
  and corrected retained Recipe, then use `runtime.py` and the official Web CLI
  with the retained overlay. Copy `fixture.mjs`/`observer.mjs` beside the isolated
  npm installation for package resolution. Create the two receipt files, the
  retained image fixtures and the Skill files described by the action/fixture logs.
  Drive the official browser UI with `client.mjs` and the logged actions. Registry
  stages are baseline→add→drop, then add before the one Harness restart→drop→full;
  the Skill folder changes receipt→receipt-next before action 18. Do not inject
  stored request bodies or rerun uncertain effects. Model wording is not deterministic.
- Offline checks rerun `audit.py`, and run Node `persistence.mjs` against the retained
  compressed Session. No model load or inference is required to verify these joins.
  Final audit/persistence checks and repository documentation self-containment
  pass ([documentation check](documentation-check.log)); `git diff --check` passes.

## 6. Limits, changes and decisions

**`tool_removal` remains explicitly unsupported and deferred.** No removal block
was sent or emulated. Addition-only can rewrite history and has no cache-hit
promise. Full Plugin management/MCP/schema-change permutations, automatic Goal
multi-round limits, general cancellation/unknown-effect recovery, Thinking ON,
Windows, multimodal endurance and source-build byte reproducibility are unverified.
Earlier rc.2 automatic-compaction/numerical evidence retains its historical scope;
this run supplies alpha.2 **manual** compaction and bounded visual continuity only.

Changes are confined to retained acceptance scripts/evidence plus materially updated
`docs/capabilities.md` and `docs/operations.md`. No Runtime/Recipe implementation,
retained native artifact, official Harness source/package, Production installation,
persistent user configuration or README changed. No push, merge or promotion.

**No actionable integration blocker or additional user decision is required to
close this bounded acceptance.** Production installation/promotion remains a
separate decision; it still has the older all-deferred restriction. The local
acceptance commit is supplied in the completion response, not embedded here.
