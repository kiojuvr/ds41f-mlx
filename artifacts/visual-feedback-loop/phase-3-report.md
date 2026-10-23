# Phase 3 — Official Harness visual loop

**BLOCKED — renderer evidence-path configuration violated the workspace sandbox.**
One official Harness conversation ran on macOS, created the supplied starter,
attempted rendering, and ended naturally explaining the blocker. **Zero task
screenshots, zero `read_image` calls, zero screenshot-guided corrections.** No
replacement conversation, custom evaluation-driver run, permission widening or
corrective follow-up was used to obtain a favorable result.

This is an attempted-deployment failure, **not proof that official Harness lacks
workspace-local rendering capabilities**, and not a Runtime Vision regression.
The failed wrapper was archived unchanged rather than repaired during the run.

## Four independent conclusions

| Criterion | Outcome | Basis |
| --- | --- | --- |
| Transport | **BLOCKED / not established** | Nine native Messages requests succeeded, but all were text-only. No Files request, inline image, attachment or image interpretation occurred. Phase 0's earlier transport PASS is not reused as Phase 3 evidence. |
| Loop | **BLOCKED — 0 corrections** | Two renderer invocations failed before Chromium launch. The model's later specification-driven rewrite is not image-guided and is not counted. |
| Quality — bounded | **NOT EVALUATED** | Unchanged nine-check rubric retained, but neither initial nor final artifact was rendered in the Harness task. No rubric score or visual-improvement claim is assigned. |
| Paired Cache — observed scope | **Image scope NOT ESTABLISHED** | All nine text-only requests show canonical settlement, compatible prefix restoration, Target/DSpark agreement and zero replay/repack. There were no historical/new images to test continuity or fresh-only encoding. |

[Offline audit](phase-3-evidence/audit-results.json), [official events](phase-3-evidence/harness-events.jsonl),
[sanitized wire](phase-3-evidence/wire.jsonl), [cache summary](phase-3-evidence/cache-summary.json).

## Starting state and identities

- Requested branch **`Visual-Feedback-Loop`**, initial clean HEAD
  **`ee552d14d0acc9349abfa364b580a6e48656f8a5`**. No competing Harness or Runtime
  inference process observed; dedicated `127.0.0.1:18083` was free. The later
  [preflight](phase-3-evidence/preflight.txt) records the two then-new helpers;
  unrelated background-process arguments were removed from retained evidence.
- Official installed CLI, headless/base bundles, DeepSeek provider/API-key
  adapter, filesystem tools and attachment packages **0.2.1-rc.2**, Node
  **v24.5.0**. All 17 inspected package versions and entrypoint SHA-256 values:
  [identities](phase-3-evidence/identities.json). Seven overlapping Phase 0
  implementation hashes match exactly. An upstream build Git revision is not
  recorded in the manifests. `dsh` was not on PATH; the official CLI was invoked
  as `node ~/.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js`.
- Reused only disposable **`/tmp/ds41f-phase1-1017-tests`**, installed Runtime
  **0.23.0**, provenance **`2fdea4fe7367e540300d7c81d4fed402d716a274`**. The three
  request/backend files checked in-process against this checkout match byte-for-byte.
  A further offline comparison found **all 113 installed Runtime Python sources**
  identical to the checkout ([source check](phase-3-evidence/source-package-match.json)).
  Full installed execution identity equals Phase 2:
  **`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`**.
- Original **DeepSeek-V4.1-Flash-1017**, checkpoint
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
  **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, 48 official shards,
  **`flash-1017-pre-woa-fp8-v1`**. Existing full-byte checkpoint authority and
  unchanged-file identity cache were used without bypass or enrollment.
- Python **3.13.15**, packaged Recipe **0.2.0**, repaired Recipe wheel SHA
  `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`,
  MLX **0.32.2**, MLX-LM **0.31.4.dev132+g94cdcae13**, oMLX **0.7.0**.
  [Full admission identity](phase-3-evidence/runtime-identity.json),
  [execution-process admission](phase-3-evidence/runtime-admission.json).
- Actual serving process PID **59531** admitted/bound identity **before numerical
  backend imports**, then applied Runtime config, constructed the official
  backend, loaded through its executor during startup and reached
  `status=ready`, `model_ready=true`, GLM **native**, **`mtp-serving-v1`**, paired
  Target/DSpark ON. Thinking OFF. No Runtime/Recipe/cache code was changed.

There were two pre-generation bootstrap corrections: the first startup lacked
explicit `DS41F_CHECKPOINT` and failed admission; a subsequent admitted bootstrap
was alive but had not registered canonical eager model loading. It was stopped
with **zero generation requests/traces**, then the supported startup loading
sequence was added. The [missing-checkpoint error](phase-3-evidence/startup-missing-checkpoint.log)
and [zero-trace exit](phase-3-evidence/preload-bootstrap-exit.json) are retained.
No real-model conversation was restarted. These setup mistakes are not hidden
or counted as model attempts.

## Harness architecture and permission boundary

Fresh temporary root **`/tmp/ds41f-phase3-harness.xQVspE`**, separate `DSH_HOME`,
`DSH_AGENTS_HOME`, profile, session, HTML workspace and Runtime KV workspace.
No operator profile, session or shared instructions were copied. Used official
headless `--json`; [help](phase-3-evidence/headless-help.txt),
[profile](phase-3-evidence/profile.patch.yml), [composed tree](phase-3-evidence/composed-config.yml),
[installed-interface inspection](phase-3-evidence/inspection.md).

Official provider/auth, native Messages, write/edit/read tools, `read_image`,
attachments, session/result management and Files-to-inline fallback remained
owners. There was no replacement provider, converter, attachment manager, tool
dispatcher or second agent framework. Session-log upload, generation retries,
web/subagents and unrelated tools were disabled as in Phase 0.

Inspection found that macOS Seatbelt is a **file-write** sandbox, not a Bash
command allowlist or read/network fence. A small deployment extension used the
verified official **monotonic `tools.guard()`** API solely to deny non-admitted
operations. It registered/dispatched no tools, changed no arguments/results,
and made no LLM calls. The actual model-facing tools were official `write`,
`edit`, `read`, `bash`, `read_image` (plus inherited, unused `exit_plan_mode`,
which the guard would deny).

- `workspace-write`, approval `never`; no escalation/full access.
- Filesystem calls limited by the guard to workspace `index.html`; screenshot
  reads limited to produced `render-00.png` through `render-05.png` there.
- Bash limited to the **exact** foreground command:
  `/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render.py`.
  No background jobs, alternative workdir, shell suffix or escalation allowed;
  executor timeout capped at 25 s.
- The wrapper reused Phase 2 static validation, bounded process-tree cleanup and
  unchanged `tools/visual_feedback_renderer.py`. Intended browser controls stayed
  1280×720/DPR 1, local Arial, network-aborted requests/DNS denial, CSP, page JS
  disabled, Chromium sandbox, fixed locale/timezone/light/reduced-motion/sRGB,
  source-size/blank/dimension checks. Chromium **155.0.8059.85**, Playwright
  **1.64.1**, renderer Python **3.13.15**, Pillow **13.0.0**. Browser/framework/font
  hashes still equal Phase 2; [renderer identities](phase-3-evidence/renderer-identity.json),
  [packages](phase-3-evidence/renderer-packages.json).

**Exact blocker:** the wrapper's `PHASE3_EVIDENCE` pointed at the repository's
`artifacts/visual-feedback-loop/phase-3-evidence/`. Its first
`artifact.write_text(html)` was outside the workspace/temp writable roots and
raised **`PermissionError: [Errno 1] Operation not permitted`**. No browser was
launched and no task screenshot existed. The host-only smoke render had passed,
but did **not** test this invocation inside the Harness sandbox. This is our
preflight/configuration oversight, not a reason to grant wider permissions.

The policy's 16 offline denial/allow checks passed. Real Harness denied both
escalation requests, an out-of-scope source read and an arbitrary `ls` command.
No safety policy was weakened. A future separately authorized attempt needs
workspace/temp-local evidence staging with external archival, and a sandboxed
preflight **before generation**. That correction was not applied here.

## Task, budgets and actual trajectory

The [task](phase-3-evidence/task.json) retains Phase 2's exact visual goal, starter,
constraints and rubric reference. Only instructions for official tool names and
separate render/`read_image` calls were adapted. No hidden DOM measurements,
rubric results, expected patch, discrepancy description or human follow-up
reached the model. Its initial write preserves the starter plus a trailing newline.

[Budgets](phase-3-evidence/budgets.json) were recorded before the conversation:
**5400 s** total, **1200 s** startup, **600 s** generation wall and stream-idle
bounds, **2600 output tokens** per request, **24 generation requests**, **30 tool
admissions**, **five correction renders / six total renders**, **65,536 source
bytes**, **25 s** shell/render bound (22 s inner renderer), **16 MiB** whole body.
Request/tool limits allow the extra official `read_image` step and bounded file
reads/edits relative to Phase 2's two-tool driver. Sampling used official
Harness/native defaults, observed stochastic, **not** the Phase 2 driver's
explicit temperature 0. This is disclosed; no model quality comparison is made.

Session **`session-aa080951-57bf-4e7f-a2b1-4e289a2425d1`**, one turn, Harness PID
**71614**, **89.51 s**, nine completed model responses, eight tool calls/results,
**2653 output tokens** total. Original IDs and effects:

| Step | Exact Tool Call ID | Official operation / outcome |
| --- | --- | --- |
| 1 | `call_00_Bb4r0xeOxEdLuZo8ci5A1897` | `write` starter, completed |
| 2 | `call_00_4wBvpqDo0A0t0yEqEu023449` | approved `bash` renderer; process exit 1, sandbox denial |
| 3 | `call_00_aRaZgh4Qclmjxs0fyOG35245` | renderer escalation denied before execution |
| 4 | `call_00_rsTbRPDBWJFoNX8hCAmC5552` | renderer escalation denied before execution |
| 5 | `call_00_8zFKD3YHv5g1nPPzI6at9203` | out-of-scope `read` denied |
| 6 | `call_00_bl4rG03s2dVmNEVdqufE3499` | arbitrary Bash inspection denied |
| 7 | `call_00_EOTquGXafxZ7RE6d9sL24223` | completed text/spec-driven rewrite, **not a correction** |
| 8 | `call_00_McvyITTjItKyqyZmlXra4508` | model requested exact renderer again; same explicit failure |
| 9 | — | model naturally ends acknowledging no visual pass |

The observer/human sent no follow-up and automatically retried no POST or effect.
The model itself requested the second known-failing render in the same session;
that is retained, not presented as a successful iteration. Official Bash status
`completed` means the process settled, **not exit 0 or screenshot success**.
Harness emitted `turn_end.reason.kind=completed` and exited 0; that does not mean
the task passed. An attempted stop signal found it had already exited naturally.

[Two artifact revisions](phase-3-evidence/artifact-01.html),
[final revision](phase-3-evidence/artifact-02.html), [diff](phase-3-evidence/artifact-02.diff)
and actual final workspace bytes are retained and verified against completed
canonical write arguments. Final source still has a light-slate subtitle
(`#cbd5e1`), not literal white; no source review substitutes for rendered scoring.
The unchanged nine-check rubric does not establish literal subtitle-color
compliance or general design quality. **No post-hoc render/evaluation is used to
replace the absent official loop.**

## Native wire, history and settlement

Actual request order: **nine `POST /v1/messages`, each HTTP 200**. Complete JSON
body byte counts in order:
**14,224; 15,880; 17,958; 18,576; 19,171; 19,555; 20,310; 23,712; 25,842**.
Each whole-body hash, exact tool-result order/ID and content type is retained.
There were **no `/v1/files` requests and no images**. Therefore image MIME,
normalized variant, transmitted hashes/dimensions, historical image identities,
new-image wire origins and fresh-only encoding are **not applicable/untested**,
not inferred from Phase 0 or from the preflight PNG.

The process-local passive observer passed original fetch inputs/responses
unchanged and retained no credential headers or full base64. Unlike Phase 0,
it persisted `message_stop` **immediately on that event**, not only after cloned
EOF. All **nine direct native `message_stop` events** were retained and their
complete assistant blocks equal the Runtime's canonical responses. Requests
2–9 still show cloned-reader `AbortError`, **after the stop was captured**; this
limitation is retained. Request 1 reached observer EOF. Completion claims here
are supported both directly on native wire and independently by official events.

All eight tool admissions follow their corresponding captured native completion;
IDs/arguments match canonical responses and official tool events. Successive
requests preserve prior tool-result ordering. All nine Runtime traces show:

- Restored frontier equals the preceding canonical frontier; final frontier
  **6370**, all **40 Target** and **three DSpark** offsets agree on every request.
- Authority `settled`; consumed/emitted/quiescence frontiers agree, predictions
  and queues retired, cancellation false, paired cache published.
- `prompt_replay`, `full_cache_repack`, quiescence history replay/repack all zero.

These are **text-only observations**. Zero images cannot prove historical-image
coverage, stable normalization, new-span classification or no historical
re-encoding. No cache instrumentation, eviction, image-history deletion,
compaction workaround or replacement conversation was added.

## Evidence, checks and cleanup

Compact evidence is under [phase-3-evidence/](phase-3-evidence/). Raw canonical
traces/protocol events are losslessly gzip-compressed; `cache-summary.json`
records decompressed hashes. `executed/` archives the exact failed helper bytes
from their original `tools/` locations; [executed hashes](phase-3-evidence/executed-source-hashes.json)
map those locations. They are **not installed production tools**. Bootstrap,
profile setup, observer and single-run supervisor are retained as execution
records, not general agent infrastructure or automatic continuation scripts.

- `policy-test.json`: **16 offline policy checks passed**, no tool execution.
- `audit.py`: offline-only audit passed for all nine canonical responses/stops,
  eight call/result IDs and ordering, completed write arguments, prefix/frontier
  settlement, zero replay/repack, source archive hashes, Phase 0 package matches
  and exact Phase 2 Runtime execution identity. Audit PASS is **not task PASS**.
- `preflight-render.png` / its metrics are explicitly a **host smoke test** of
  the supplied starter, never a Harness-produced/delivered task observation.
  Its successful rendering did not validate the later sandboxed archive path.
- No broad regressions, extra real-model attempt, second renderer, Files API,
  Windows, Thinking ON, 3D, endurance or Phase 3B/4 work was performed.

[Final state](phase-3-evidence/final-state.json): Harness/supervisor exited;
Runtime shut down through the existing backend retirement/close sequence,
PID 59531 retired, nine traces saved, port **18083 free**. Owned temporary
home/profile/agents/session/workspace/KV tree removed after exact final-artifact
archival. No task browser was launched or left behind. Persistent operator
installations/runtimes and checkpoint files were not changed, reinstalled or
stopped. No unrelated user session or credential data was retained.

No README/capability claim was expanded because acceptance did not pass.
**Stop at this BLOCKED report; no push or automatic next phase.**

---

## Continuation 1 — sandbox-local staging (separate execution record)

**BLOCKED at mandatory sandbox preflight; no model loading or conversation.**
The original report, receipt, archived failed helpers and evidence above remain
unchanged. Starting branch `Visual-Feedback-Loop`, clean HEAD
`70ae00b58ee04e87f0e216b927bbc4e2d4f300b0`; no competing Harness, Runtime or
renderer process was observed. The unrelated remote-desktop service was left alone.

### Focused correction and actual sandbox test

Restored the archived helpers to their original `tools/phase3_render.py` and
`tools/phase3_harness_guard.mjs` paths. The guard is byte-identical to the original:
it still only denies operations through official `tools.guard()`, never dispatches
or changes results. Offline checks additionally deny private staging reads,
writes, edits and image reads. No model can access staging through the admitted
tool surface. Screenshots would remain workspace-local for official `read_image`.

The renderer now requires `<workspace>/phase3-stage`, verifies contiguous metadata
IDs, and uses exclusive evidence creation. Partial artifact reservation prevents
retrying an uncertain render or overwriting evidence. The external supervisor
created staging **before** execution and set `PHASE3_EVIDENCE` to it. Permanent
repository retention remained outside the sandbox and outside model tools.

`tools/phase3_sandbox_preflight.mjs` instantiated the **installed official
`LocalSandboxProvider`** in a Cordis context, called its public `confine()` with
`mode=workspace-write`, then executed its **unaltered returned argv**. This is the
same provider/policy and inner `bash -c` invocation used by official
`SandboxBashExecutor.confine()`, not a hand-built substitute profile. It does not
exercise the complete agent/tool scheduler. Effective Seatbelt argv, writable
roots and complete stdout/stderr are in
[preflight](phase-3-evidence/continuation-1/sandbox-preflight.json).
The exact approved command remained:

```text
/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render.py
```

| Preflight requirement | Observed result |
| --- | --- |
| Read existing workspace `index.html` | PASS; supplied starter validated and archived identically |
| Write staged artifact | PASS; `phase3-stage/artifact-00.html` created under actual Seatbelt restrictions |
| Launch Chromium and render | BLOCKED; execution reached `BrowserContext.new_page`, then `Target crashed` |
| Valid 1280×720 screenshot | NOT PRODUCED |
| Corresponding render metadata | NOT PRODUCED |
| Repository evidence write remains denied | PASS; separate confined probe raised `PermissionError: [Errno 1] Operation not permitted`; no probe file exists |
| No remaining renderer/browser processes | PASS after failure and before workspace removal |

Only one renderer invocation ran. The original repository-path failure is resolved,
but the exact sandboxed render operation is **not verified**. The browser crash's
root cause is not established; no permission widening, browser-sandbox disabling,
renderer changes, host-only substitute or retry was used to evade this gate.
The official policy grants only the canonical workspace and its standard platform
temporary roots (plus `/dev/null`), exactly as generated by the installed provider.
The denial probe confirms an outside-root write is refused; it is not a complete
filesystem-write audit of Chromium.

After both results had been saved, the initial preflight helper called nonexistent
`ctx.dispose()` and exited with a teardown `TypeError`. The exact executed helper
and launcher error are retained separately. The checked-in helper now uses the
provider's own grant cleanup method and returns failure when the gate fails;
that correction was syntax-checked, **not re-executed**. It does not alter the
already captured browser failure. The external retention process initially
stopped conservatively on an overbroad process filter (unrelated LM Studio
crashpad and supervisor shell text); the filter was narrowed without stopping
those processes. Archive copies were hash-verified again before removal.

### Identities and deliberately unexecuted acceptance

All **17 official Harness package versions/entrypoint hashes** were rechecked
against the original qualified identities: **0.2.1-rc.2**, Node **v24.5.0**, all
matching ([identities](phase-3-evidence/continuation-1/identities.json)). No
installed package changed. `visual_feedback_renderer.py` and Phase 2 validation
remain byte-identical; [source hashes](phase-3-evidence/continuation-1/source-hashes.json).
The browser's version was not returned by the failed render; prior qualified
browser identity is historical, not a new successful-render observation.

Offline inspection of the **disposable** `/tmp/ds41f-phase1-1017-tests` confirms
Runtime **0.23.0**, Recipe **0.2.0**, and all **113 Python source files** match
this checkout ([offline check](phase-3-evidence/continuation-1/runtime-offline-check.json)).
Required serving identity remains provenance
`2fdea4fe7367e540300d7c81d4fed402d716a274`, original Flash-1017 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, native paired MTP/GLM, Thinking OFF.
Those admission facts were established in the original attempt; **no fresh
admission, numerical backend import, model loading, Runtime listener or KV
workspace was started** here because the preflight hard gate failed. Thus no
new admission or paired-cache PASS is claimed. The retained admission-before-
backend-import/eager-loading bootstrap was inspected and not changed.

The retained official headless profile, native Messages provider, official
filesystem/image attachment pipeline, Files-to-inline fallback, passive observer,
exact Phase 2 Field Notes starter/goal/nine-check rubric and original Phase 3
budgets were inspected and left unchanged. No new profile/session/conversation
was started; no instructions, expected corrections or scores reached a model.
There are **zero tool-call IDs, attachment events, native `tool_result` images,
Files/inline requests, image-order records, canonical native `message_stop`
records, or Runtime image/cache traces** for this continuation. Preflight source
is supervisor-supplied, not a model revision or screenshot-guided correction.

| Conclusion | Continuation result |
| --- | --- |
| Transport | **BLOCKED / not tested**; no screenshot or official image observation |
| Loop | **BLOCKED — 0 screenshot-guided corrections**, no conversation |
| Quality | **NOT EVALUATED**; unchanged nine-check rubric, no rendered score |
| Paired Cache | **NOT TESTED**; no inference or image-bearing continuation |

### Retention, checks and cleanup

All new evidence is isolated under
[continuation-1/](phase-3-evidence/continuation-1/). The external supervisor copied
both workspace files (`index.html`, staged `artifact-00.html`) to
`preflight-workspace/` and compared **every SHA-256 against the live source before
removing the temporary root**. See [manifest](phase-3-evidence/continuation-1/retention-manifest.json)
and [cleanup](phase-3-evidence/continuation-1/cleanup.json). Partial evidence is
retained; no real-model evidence is mixed with preflight output. With no next
operation authorized by the gate, the workspace/render reservation was removed,
not reused as render 00 of a model task.

Python AST/Node syntax checks passed; **21 offline guard checks passed**, including
private staging denial, exact command, escalation/background/timeout denial and
out-of-scope/image-read denial. These are policy checks, not successful rendering.
No generation request or uncertain tool effect was automatically retried. No
credential headers or base64 request bodies were collected. Final process check
found no renderer/Chrome/Playwright process; temporary workspace removed only
after verified retention. Persistent operator installations, Runtime/Recipe/MTP,
checkpoint, permissions and official packages were unchanged. No capability or
README claim is expanded. **Stop: BLOCKED; no push, replacement conversation,
Phase 3B/4 or 3D follow-up.**

---

## Gate P — official Harness multi-image Messages continuation

**Protocol/history PASS; overall qualification BLOCKED by incomplete Runtime
trace retention.** One official session completed the requested three sequential
image observations. No Harness/Runtime protocol incompatibility was observed.
The final Runtime trace was not archived, so complete three-image paired-cache
acceptance is **not established**. This is an evidence-collection limitation,
not model task failure or evidence of a cache defect. No second conversation
was run to replace the incomplete evidence.

| Acceptance | Result | Direct basis |
| --- | --- | --- |
| A. Official Harness transport | **PASS** | Three completed official `read_image` results arrived as native nested inline images; subsequent responses identify undisclosed visual details. |
| B. Messages history continuity | **PASS** | Successive wire requests retain A, A+B, A+B+C in the same locations/order with unchanged PNG bytes/dimensions and original opaque call IDs. All four native responses completed. |
| C. Runtime paired-cache continuation | **NOT ESTABLISHED — partial evidence passes** | First three traces establish fresh A/B encoding, A historical reuse, exact prefix restoration, Target/DSpark agreement and zero replay/repack. Final A+B+C trace missing. |
| D. Model visual interpretation | **PASS** | Correct marker/shapes for A and meaningful dashboard layout/palette/footer differences for B/C; minor geometric/color-description imprecision retained. |
| E. Phase 3 full Visual Feedback Loop | **NOT TESTED** | No HTML writes, rendering, browser execution or autonomous correction. Earlier BLOCKED attempts above remain unchanged. |

### Isolation, identities and fixed budgets

Starting clean branch `Visual-Feedback-Loop`, HEAD
`92f5f762bbacda74530b5510ad4f9017cd83387e`. Evidence:
[protocol-gate/](phase-3-evidence/protocol-gate/).

- Installed official CLI/base/headless/provider/filesystem/attachment packages
  **0.2.1-rc.2**, Node **v24.5.0**. All 17 retained package versions/entrypoint
  hashes match the previously qualified installation. Upstream build Git SHA
  remains unavailable in installed manifests. [Identity receipt](phase-3-evidence/protocol-gate/identities.json).
- Disposable `/tmp/ds41f-phase1-1017-tests`: Runtime **0.23.0**, provenance
  **`2fdea4fe7367e540300d7c81d4fed402d716a274`**, Recipe **0.2.0**;
  execution identity **`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`**
  equals Phase 2/3. All **113** installed Runtime Python files match this checkout.
- Original Flash-1017 checkpoint, revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`, numerics
  `flash-1017-pre-woa-fp8-v1`, native paired MTP/GLM, Thinking **OFF**.
  The byte-identical Phase 3 Runtime bootstrap preserves in-process admission
  **before backend imports**, environment application, executor-owned eager
  loading, and ready health. [Admission](phase-3-evidence/protocol-gate/runtime-admission.json),
  [full identity](phase-3-evidence/protocol-gate/runtime-identity.json).
- Fresh isolated Harness home/profile/agents/session/workspace; official `dsh`
  headless/native DeepSeek provider, official attachments/preparation and
  Files-first fallback. Only model-facing tool: **`read_image`**. Read/write/edit,
  Bash/PowerShell, web, subagents, generation retries and plan-mode disabled.
  Official **read-only** sandbox retained. No installed package modified.
- The previous Bash/rendering guard was **not loaded**. A minimal monotonic
  `tools.guard()` fence admits at most three image effects on the supplied
  paths, solely to enforce the required effect budget. It registers/dispatches
  no tools, enforces no image ordering, and changes no arguments, results or
  conversation history. Normal Harness scheduling selected the observed order.
- Budgets saved before generation: **one conversation, four generation requests,
  three image effects, 1200 output tokens/request, 180 s generation deadline,
  900 s conversation deadline, 1200 s startup, 16 MiB body**. The passive Phase 3
  observer adds only a pre-fetch request/token/body budget fence. No prompt
  correction, generation retry, synthetic tool result or direct generation POST.
- One pre-generation bootstrap was rejected because the Harness-only
  `DS41F_LOCAL_API_KEY` environment-variable **name** reached Runtime's strict
  profile configuration. Its log/zero-conversation receipt are retained in
  [startup-rejected/](phase-3-evidence/protocol-gate/startup-rejected/).
  Removing that Harness-only variable from the Runtime child environment
  resolved startup without changing admission. No model was loaded or request
  generated by that rejected bootstrap.

### Immutable images and actual sequence

Copied all three supplied PNGs before the conversation, chmod 0444; original,
copy, transmitted and final-copy hashes agree. No image transformation or new
image generation was performed by the supervisor. Full hashes and paths:
[images.json](phase-3-evidence/protocol-gate/images.json). All were **1280×720 PNG**.

| Image | Retained source | Bytes | Original = copy = transmitted SHA-256 |
| --- | --- | ---: | --- |
| A | Phase 0 `probe.png` | 24,147 | `7f594ec2e888cf9777a8a9bddc1baacc03df39459e30a43f37eaa9d7dd41f08a` |
| B | Phase 2 continuation `render-00.png` | 35,329 | `fc6dd2f551cb196b84eb64d1bd0c91b01633ce881403d4bc4f0a1d602c1176d9` |
| C | Phase 2 continuation `render-01.png` | 32,710 | `0c03c9f7a75a5a3a653b38a2df900f4ad533abba9dbc6d4a98af0116d7ac359f` |

Single headless turn, **49.39 s**, **1095 output tokens**, Harness exit **0**.
[Exact prompt](phase-3-evidence/protocol-gate/task.json) contains filenames and
observation/sequence instructions, not expected visual content or source descriptions.
[Official events](phase-3-evidence/protocol-gate/harness-events.jsonl),
[canonical observed sequence](phase-3-evidence/protocol-gate/sequence.json):

1. `call_00_ZcIiwOa0qAg79gyq4jIk1628`: official `read_image(A)`, completed.
2. After receiving A, model describes its heading, **red circle → blue square →
   green triangle**, **K7M4** marker; requests B with
   `call_00_2ISyN0qhwm2ZzKm11Hb63652`, completed.
3. After receiving B, model describes the orange-red header, stacked yellow
   statistic panels and **12 / 84 / 24**, distinguishes A; requests C with
   `call_00_kYn7jJHefVNkvz1d28sE1848`, completed.
4. After C, model describes inset navy rounded header, horizontal white cards,
   pale background and visible status footer, then compares all three and ends.

Independent direct PNG review:
[independent-review.json](phase-3-evidence/protocol-gate/independent-review.json).
The model overstates B's header as flush to the edge and tiles as touching;
there are narrow margins/gaps. Its guessed hex color is not an acceptance fact.
These do not invalidate the meaningful, accurately observed distinctions.

### Actual native transport/history contract

[Sanitized wire events](phase-3-evidence/protocol-gate/wire.jsonl) retain complete
serialized body sizes/hashes, native SSE completion, image facts and call IDs;
no credential headers or full base64 bodies are retained.

| Generation / wire request | Images | Complete body bytes | Native outcome |
| --- | --- | ---: | --- |
| 1 / 1 | none | 6,509 | 200, `message_stop`, request A |
| 2 / 3 | A | 39,947 | 200, `message_stop`, interpret A/request B |
| 3 / 5 | A, B | 88,681 | 200, `message_stop`, interpret B/request C |
| 4 / 7 | A, B, C | 134,604 | 200, `message_stop`, interpret C/final comparison |

Every image is `source.type=base64`, MIME `image/png`, **inside** official
`tool_result.content`, not an attachment string or offloaded reference.
Nested locations are respectively
`messages.2.content.0.tool_result.2`,
`messages.4.content.0.tool_result.2`,
`messages.6.content.0.tool_result.2`; each corresponding result has content
`[text,text,image]`, `is_error=false`, and its original model-generated ID.
Historical image locations, order, hashes, lengths and dimensions remain stable
on every subsequent request. No removal, replacement or normalization drift
was observed. Each tool admission follows its corresponding direct native
`message_stop`; results complete before the next image-bearing request.

Three actual `/v1/files` requests (wire 2/4/6) returned **404 `not_found`**.
Each uploaded A's prepared bytes: the unresolved first historical attachment
was retried before request-wide inline fallback. B/C therefore were not
individually uploaded via Files; they were successfully delivered inline with
all preceding images. This is observed official behavior, not an implemented
Files API or transport override. All four native `message_stop` events were
captured. Three cloned-reader `AbortError` events occur **after** persisted
`message_stop`; wire request 5 reached observer EOF normally.

### Paired-cache coverage and exact remaining limitation

[Cache summary](phase-3-evidence/protocol-gate/cache-summary.json):

| Generation | Images | Restored → settled frontier | Fresh encodes | Coverage |
| --- | --- | --- | ---: | --- |
| 1 | none | 0 → 741 | 0 | Full retained trace |
| 2 | A | 741 → 1805 | 1 | Full retained trace |
| 3 | A+B | 1805 → 3017 | 1 | Full retained trace; A historical reuse |
| 4 | A+B+C | **not retained** | **unknown** | Native completion exists; final Runtime trace missing |

For all **three retained** real-model traces, all 40 Target and three DSpark
frontiers agree, preceding canonical prefixes restore exactly, authority settles,
queues/predictions retire and cache publishes; prompt replay, full-cache repack
and quiescence replay/repack counters are zero. Two fresh image encodes and one
historical reuse are directly established. **Do not report aggregate fresh=3**:
that is expected, not measured by the retained evidence.

The reused watcher polls `backend.serving_traces` every 250 ms only while
`not server.should_exit`; the supervisor retires Runtime immediately after
Harness exits. [Runtime exit](phase-3-evidence/protocol-gate/runtime-exit.json)
records **four traces in memory**, but disk retains only `trace-00` through
`trace-02`. A missed final polling window is consistent with this shutdown
sequence; exact thread timing was not instrumented. The final trace cannot be
recovered from the removed owned workspace. No replacement conversation was run.

**Boundary: external trace retention/owned-process shutdown**, not an observed
Harness attachment, Messages serialization, Runtime admission, multimodal
preparation or paired-cache incompatibility. Exact Recipe-expanded span identities
and historical flags were also not separately recorded by this reused trace
projection; the wire occurrence identities/order are direct evidence, but no
complete Recipe-span identity audit is claimed. Native 200/completed final text
cannot substitute for the missing Target/DSpark/cache-publication evidence.
No Runtime, Recipe, Vision, Scheduler or cache authority was changed, and no
speculative correction is implemented.

### Audit, retention and cleanup

[Offline audit](phase-3-evidence/protocol-gate/audit-results.json) passes checks of
retained wire/history/call IDs/settlement and all three available canonical
traces, while explicitly reporting the coverage gap rather than a gate PASS.
It adapts the existing Phase 3 audit, makes no HTTP/model/tool requests, and does
not reconstruct or alter model history. Raw traces are losslessly compressed;
[retention manifest](phase-3-evidence/protocol-gate/retention-manifest.json)
records retained file hashes. No new general testing framework was introduced.

Harness PID **84265** and owned Runtime PID **72175** both exited **0**; startup
rejection also retired. Final process verification found owned PIDs gone,
**18083 free**, both temporary roots removed, all copied image bytes unchanged
before removal. [Cleanup](phase-3-evidence/protocol-gate/cleanup.json).
Persistent operator Runtime/installations/checkpoint and official Harness packages
were not changed. No browser/renderer was started or debugged, no push, and no
full-loop/Windows/Phase 3B work. **Stop: protocol/history demonstrated; overall
Gate P remains BLOCKED on missing paired-cache acceptance evidence.**
