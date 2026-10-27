# Phase 3 / Step 3B — official compaction continuation

**BLOCKED at the installed human-command adapter gate, before provisioning or
loading weights. No compaction transaction or native continuation was attempted.**
This is a feasibility finding, not C4 qualification or a Runtime failure.

## Starting authority and evidence

Actual starting HEAD, on clean `Multimodal-Runtime`:
`74f8b1b066196c993fb5e1e80bc5576ef07c2b3e` (matches expected).
Read the requested policy, plans, serving/Vision/Files owners, Phase 1 and
Steps 2E/3A reports; inspected retained Step 3A runtime, supervisor, audit,
shutdown, resource receipts and traceback. Prior evidence is unchanged.
No Production or official Harness source changes.

Supporting evidence: [phase3-step3b-evidence/](phase3-step3b-evidence/).
`preflight.py` reproduces isolated normal configuration composition and help;
`installed-identities.json` hashes the installed relevant manifests and JS/YAML;
`composed-config.yml`, `effective-configuration.json`, `headless-help.txt` and
`contract-result.json` retain the blocker. No credentials, base64 image bodies or
weights are retained. Placeholder API-key configuration is not a credential.

Reused the existing official source checkout at
`687f282e8625c6b8d11ff58211a90524be2e4317`; no general upstream audit repeated.
Installed Harness, headless, command-compact, commands, compaction-basic,
compaction, session, token-meter, DeepSeek provider, tool-fs and attachment-local
are **0.2.1-rc.2**; Node **v24.5.0**. Installed package manifests alone do not
establish an exact build Git SHA. `postflight.py` rechecks the retained installed
file hashes without changing them.

## Precise installed blocker

The ordinary headless composition contains the **command registry**, official
`command-compact`, `compaction-basic` and token-meter. This is **not** a
headless human-command invocation adapter:

- Installed `dsh-headless/lib/startup.js:34–41` exposes only task text, `--json`,
  `--session-id`, and help. Stdin is a single task read through EOF, not an
  interactive command stream.
- Installed `dsh-headless/lib/index.js:290–369` obtains that task, creates/adopts
  an Agent, waits idle, then unconditionally calls
  `agent.followup(createUserMessage({content: [{type: "text", text: task}],
  source: {kind: "user"}}))`. There is no command dispatch in this runner.
- After quiescence it flushes and exits. `--session-id` supports ordinary session
  continuation, but routes its new task through exactly the same user-message
  path; it does not supply an idle maintenance command channel.
- Installed `dsh-command-compact/lib/index.js` registers `compact` with
  `ctx.commands.register` and delegates to `ctx.compaction.compactNow`. Merely
  mounting it does not change the headless runner's input interpretation.

Thus `dsh --profile headless "/compact"`, stdin `/compact`, or adoption with that
text would submit an ordinary model-facing message, expressly prohibited by this
task. **None was submitted.** No synthetic events, direct service calls,
replacement records, command adapter plugin, or alternate mutation route was used.
The only CLI invocations were model-free `--dump-config` and `--help`; both exited
0 through supported interfaces. Normal composition succeeds, but the required
supported headless invocation route does not exist in this installed adapter.

**Owner: official Harness headless input/adapter capability, not Recipe, Files,
Runtime paired identity, numerical execution or cache invalidation.** A subsequent
execution needs an explicitly agreed official human-command-capable surface, or
upstream supported headless command dispatch. Changing to a GUI/server workflow
or adding a local dispatcher is not silently authorized here. No new architectural
component or proprietary recovery mechanism was implemented.

## Installed engine contract and range feasibility (source only)

Relevant pinned owners remain `packages/compaction/command-compact/src/index.ts`,
`compaction-basic/src/index.ts`, `region.ts`, `summarizer.ts`,
`packages/core/agent-loop/src/agent.ts`, `packages/core/session/src/surface.ts`,
and `packages/llm/token-meter/src/index.ts`. Installed command/engine behavior
inspected agrees with this scoped source path:

1. `/compact` handler → `compactNow` → `agent.runMaintenance`. Maintenance requires
   idle state; waking work starts the driver or is latched behind maintenance.
   It is not a model-facing command or assistant tool invocation.
2. Manual `compactNow` passes the **actual current session** and
   `tokenMeter.measure(session)` to `selectCompactableRange`, with literal
   **retainTokens=0**. This is independent of automatic retention settings.
   The meter replays the durable surface and prices its current routed image
   representation; it is not a Runtime token-count substitute.
3. A system head at surface node 0 is excluded. Selection accumulates from the
   last node even at zero retention, so at least one recent node remains.
   It walks the retention boundary backward to balance tool calls/results;
   validation checks both range boundaries. No live range was selected here.
4. The official summarizer replays the selected conversation prefix with protected
   system/tools and appends its own instruction. It calls `ctx.llm.stream` with
   `purpose: "compaction"`. The summary is model-generated; truncation, empty
   text or image output fails. The framed text-only checkpoint must price strictly
   smaller than the selected route-priced span.
5. Successful replacement appends `compaction/start`, `compaction/summary`
   (summary, provider/model, usage and shadowed source range), a text-only user
   checkpoint with **`surfaceOp: {op: "replace", startSeq, endSeq}`**, source-event
   relationships, and `compaction/end`, then flushes. Source events remain durable;
   the surface fold shadows their selected nodes. Summary prose is not an
   executable tool-effect ledger. Maintenance returns to idle for continuation.

The composed compaction plugin has **no explicit configuration overrides**.
Installed `resolveConfig` defaults summarizer maxTokens to headroomTokens,
**65,536**, not Step 3A's ordinary provider budget of 1,200. The summarizer
explicitly supplies that maxTokens; Step 3A's nine-request/10,800-output guard
must not be blindly reused for a future compaction run. No session budget was
frozen or execution authorized after this earlier adapter gate failed. Automatic
compaction also defaults on; a future exactly-one-manual-command design must
account for that through supported policy.

Multiple genuine completed image/tool/text exchanges could plausibly supply a
smaller range, but a short conversation can return null or fail shrink validation.
Manual zero retention does **not** promise survival of any historical image.
No fabricated history, expected image answers, precomputed summary, live range,
or size-reduction claim is supplied by this source feasibility analysis.

## Corrected diagnostic shutdown observation

`lifecycle.py` is a passive observer of the **existing ASGI router lifespan**, not
another Runtime lifecycle controller. `runtime.py` retains the Step 3A collector
with that adaptation, but is explicitly execution-disabled by this blocker; it
is not a runnable approved workload and has no disposable installation/session
configuration. Step 3A files and its PARTIAL disposition are untouched.

The observer wraps the original lifespan before server startup:

- Immediately before leaving its original lifespan body (before original
  shutdown), snapshot Scheduler requests/waiting/running/UID mapping, native
  owners, source counts, paired roots, backend state and MLX; establish only a
  **weak** model reference while the model still exists.
- After original shutdown, observe expected Scheduler absence, remaining safely
  visible backend deliveries/sessions, model presence and executor shutdown.
  Scheduler absence is expected, not dereferenced as if its owner survived.
- After the caller releases app/server/backend diagnostic references, collect
  garbage/allocator-cache observations and weakref death, then persist the final
  receipt. The observer never closes the backend or registers a second closing
  handler. The original app remains the sole shutdown authority.

Observations are best-effort with separately retained string errors; no exception
or traceback objects are retained as hidden model owners. Original serving or
shutdown exceptions propagate unchanged. Receipt persistence failure is reported
to stderr, never substituted for the original failure. External process/socket
checks remain distinct from in-process model/allocator evidence.

**Zero-weight lifecycle regression: 7 passed** (`lifecycle-tests.log`). Controlled
objects through real Starlette lifespan handling cover live Scheduler observation,
normal disposal to None, repeated absence observation, original body/shutdown
exception preservation, optional observation failures, final receipt persistence,
failed persistence without exception masking, weakref death after reference
release, and exactly one execution of an existing event shutdown handler.
This verifies the diagnostic seam, **not actual model cleanup**. Two intermediate
runs of the added event-handler test used removed application-level registration
methods; the fixture was corrected to the existing router's supported
`add_event_handler`, as used by the retained runtime. Both development failures
are retained separately; neither involved Production execution.

Focused unchanged-boundary checks: **25 passed**, 25 existing deprecation warnings
(`focused-tests.log.gz`): genuine Recipe changed-history selection, paired fallback,
Messages/Files exact nested tool IDs/current sources, canonical settlement,
publication and burn. These reuse the existing installed test environment, not a
new disposable Runtime admission or native compaction projection. No full Phase 2
or Step 3A campaign was repeated.

## Execution, resources, integrity and missing evidence

**Zero model loads; zero inference/summarizer requests; zero image acquisitions;
zero tool effects; zero compaction invocations; no new Harness session ID.**
Only an isolated temporary home/profile/workspace was created for config/help;
it was removed in `finally` (`preflight-cleanup.json`). No task-owned Runtime
installation, model process, supervisor, session history or listener was started.
The current Runtime remains source-only/undeployed.

The requested later model-free execution gates—fresh disposable `mtp_setup`,
payload/checkpoint admission, actual numerical `pre_woa` proof and authenticated
post-serving module graph, fixture preparation and frozen session/Supervisor
budgets—were **not attempted after the necessary command-route prerequisite
failed**. Neither old receipts nor existing test admission substitute for them.
The ordering and physical gates remain unchanged for any separately authorized
future attempt; no Transformers late admission or load fallback was introduced.
No native physical peak, reserve-under-load, swap-growth, MLX cleanup or numerical
frontier claim can follow from this model-free task.

`integrity-before.json` and `integrity-after.json`: all **817** original operational
Runtime inventory paths match retained SHA256/size/mtime/inode baseline.
`/Volumes/SDXC-512/ds41f-1017` was not modified or imported for execution.
`postflight.json` records unchanged official package files, removed temporary
root, zero task-owned workers/listeners, external free loopback port 18083 and a
point-in-time host memory/pressure/swap sample. This is not a sampled inference
resource peak or actual model reclamation result.

| Required receipt | Actual Step 3B evidence |
| --- | --- |
| Session/image/tool manifest and original IDs/effects | Not instantiated; no effects to replay |
| Pre-compaction surface, Messages, Recipe IDs/images, resident pairs | None: no session or inference |
| `/compact` result and durable start/summary/replacement/end | Not invoked; blocked before weights |
| Real summarizer request/output/provider usage and smaller replacement | None; source contract only |
| Shadowed sources, system head, retained tail/tool balance | Source contract only; no live projection |
| Post-compaction Messages/Recipe projection and prefix divergence | Not exercised |
| Cache C, stale-pair exclusion and compatible earlier fallback | **Not measured**, not forced C=0 |
| Fresh/covered historical Vision; C/N−2/N−1/N/P5/native/settled frontier | Not exercised; no intermediate offsets |
| Settlement/publication/native ownership retirement | Focused controlled tests only, no native session |
| Cleanup/resource reclamation | Diagnostic regression passes; native evidence absent |

## Dispositions and remaining gaps

| Classification | Disposition |
| --- | --- |
| OFFICIAL_COMPACTION_COMMITTED | **BLOCKED** |
| SUMMARY_REQUEST_EXECUTED | **BLOCKED** |
| CANONICAL_PROJECTION_CHANGED | **BLOCKED** |
| STALE_PAIRED_PREFIX_EXCLUDED | **BLOCKED** for official compaction |
| NATIVE_RECONSTRUCTION_COMPLETED | **BLOCKED** |
| FRESH_HISTORICAL_VISION_AFTER_COMPACTION | **BLOCKED / not exercised** |
| CANONICAL_SETTLEMENT_VERIFIED | **BLOCKED** for native Step 3B |
| PAIRED_PUBLICATION_VERIFIED | **BLOCKED** for native Step 3B |
| CLEANUP_VERIFIED | **BLOCKED** for actual native session; model-free regression PASS |

C4 remains unqualified: no genuine compaction transition, text-only or multimodal,
was executed. C2 remains PARTIAL; no new numerical accuracy, cross-partition
identity or universal floating tolerance is claimed. C6/C7/C9 retain their prior
bounded evidence and remaining incomplete/failure/resource/recovery gaps; this task
adds no native effect or recovery qualification. A future text-only compacted
continuation would be narrower than fresh historical Vision and must be labeled
accordingly. C8's complete session and resource workload remains Phase 4,
**not started**. No MTP/full-forward comparison, FP8/BF16 or `wkv` investigation.

Validation: focused tests above, Python/Node syntax, `git diff --check`, and
`python tools/check_repository_self_containment.py` pass (retained validation log).
No push, deployment, operational modification, retry or automatic follow-on task.
**Stop after Phase 3 / Step 3B's installed-adapter feasibility finding.**
