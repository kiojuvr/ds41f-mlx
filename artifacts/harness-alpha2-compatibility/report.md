# Official DeepSeek Harness 0.2.2-alpha.2 Native Messages compatibility audit

## Disposition and scope

**Substantial, working, bounded compatibility; not full alpha.2 parity.** Retained
execution evidence establishes official-Harness Native Messages tool execution,
exact-ID results, paired-KV continuation, visual workflows, automatic compaction,
and changed-history reconstruction. It does not establish unrestricted plugin or
dynamic-tool compatibility, or alpha.2 end-to-end acceptance.

This report preserves the preceding independent, read-only investigation. Its
creation is documentation-only: no investigation rerun, compatibility test,
inference, installation, plugin operation, configuration change, or deployment.
The findings and recommendations below are evidence, **not implementation
approval**, a new roadmap, or operational instructions.

The assessed path is exclusively:

**Official Harness → `deepseek-official` / Native Messages → ds41f-mlx
`POST /v1/messages`.**

`llm-pi-ai`, OpenAI Chat Completions, OpenCode, and unrelated provider defaults are
outside scope. No result through those routes is used as Native Messages proof.
Harness alpha.2 was **not executed against the Runtime** during the audit.

## 1. Examined identities and provenance

| Component | Identity examined | Qualification |
|---|---|---|
| Official Harness | Tag `dsh-v0.2.2-alpha.2`, commit **`f3803c96974ad4a94195fb97e5c637e9c2a3afe3`** | Verified against the official remote tag and GitHub commit object before inspecting the archive of that commit; primary upstream authority |
| Official commit tree | `457c6e9983dd1af8683453f871ce84a4eff1247d` | Commit message: `release(dsh): 0.2.2-alpha.2 (#7437)` |
| ds41f-mlx repository | `master`, **`62def62553552a266e4571943cda48217eb87c59`** | Clean working tree during audit; same branch/HEAD and clean tree before preserving this report |
| Operational installation | `/Volumes/SDXC-512/ds41f-1017`, Runtime distribution **0.23.0** | Installed manifest `source_commit`: **`69689d514fc5acb2cbe229f8c863da1e0a855e4a`**, not repository HEAD |
| Recipe | **0.2.0**, source base **`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`**, retained native semantic patch | Source archive, patch, retained wheel and installed native bytes inspected; not an unpatched-version claim |
| Installed Harness | `/Users/kioju/.local/share/deepseek-harness/node_modules/@deepseek-ai/` | Inspected `dsh`, `dsh-llm-deepseek`, `dsh-goal`, and `dsh-compaction-basic` package manifests: **0.2.1-rc.2**, not alpha.2; manifests do not prove a build Git SHA |
| Historical Harness executions | Official **0.2.1-rc.2** | Version/source limitations belong to each retained report; no inference that rc.2 execution qualifies alpha.2 |
| Repository `.venv` | Recipe **0.1.1** | Not the operational Native Messages installation |

All **114** installed Runtime Python files matched `69689d5` in the audit's
read-only comparison. Against audited HEAD, **109** matched and these five differed:

- `ds41f_mlx/serving/server.py`
- `ds41f_mlx/serving/capacity.py`
- `ds41f_mlx/serving/mtp_public.py`
- `ds41f_mlx/serving/image_files.py`
- `ds41f_mlx/runtime/multimodal.py`

Installed `messages_admission.py`, `ordinary_admission.py`, and `production_mtp.py`
matched HEAD. Thus the removal admission finding applies to both identities;
current source resource/Files improvements must not be attributed automatically
to the installed package. The [69689d5 installation report][installed-report]
retains installation and rollback evidence.

Recipe artifacts:

- [Source archive][recipe-source]: archive-relative paths below identify actual
  Recipe source, rather than substituting a newer Recipe version.
- [Native semantic patch][recipe-patch]: retained semantic/streaming implementation.
- [Artifact/build identity][recipe-identity] and [retained wheel][recipe-wheel].
- Installed `_native.abi3.so` SHA256:
  `eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2`.
- Retained wheel SHA256:
  `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`.

The [development evidence][development] records the patch, installed-native
identity, and earlier direct-Harness acceptance. Installed version labels alone
are insufficient provenance.

## 2. Ownership and interpretation

- **Harness:** canonical Session history and persistence, orchestration, human
  commands, Skills, plugin lifecycle, tool selection, permissions and execution.
- **Recipe:** Messages grammar/conversion, semantic representation and canonical
  model-prompt encoding; native response/stream semantics.
- **ds41f-mlx:** execution capability admission, inference, settlement and
  paired-KV correctness. It is not a plugin host or slash-command parser.

These boundaries follow the [Harness agent request builder][h-agent],
[Session tool-history projection][h-history], [Messages serializer][h-serialize],
Recipe sources in §5, and Runtime [admission][runtime-admission],
[execution][runtime-execution] and [paired checkpoint][runtime-pairs] owners.

`dsh_session_log` ingestion remains **intentionally unsupported for this
integration**, preserving the established Session ownership decision. Disabling
it loses server-side event upload/acceptance-watermark behavior, not Harness's
local canonical Session persistence, Goal domain, compaction or request derivation.
Do not acknowledge ingestion that did not occur or create a duplicate Session
history subsystem. This exclusion does **not** excuse ignoring removal or other
independent unsupported controls. See [contributor implementation][h-session-log]
and Runtime `convert_messages` / unsupported-wire tests.

### Status and evidence vocabulary

- **SUPPORTED:** the stated bounded behavior is implemented and supported by the
  identified evidence. A model-free conversion result is labeled as such, not E2E.
- **PARTIAL:** constituent protocol support exists, but full feature/lifecycle
  acceptance is missing or scope is intentionally narrower.
- **UNSUPPORTED:** a concrete required shape/control is rejected or unimplemented;
  configuration avoidance is not support for that capability.
- **UNKNOWN:** evidence is insufficient for the specified plugin/extension.

Source inspection, existing unit/model-free tests, actual Native Messages
requests, and actual official-Harness execution are distinguished below. The audit
read retained tests and results; it did not run a new test campaign.

## 3. Retained evidence and applicability

| Evidence | Established result | Scope and limitation |
|---|---|---|
| [Messages development/verification report][development]; [native Messages tests][messages-tests] | Conversion/encoding, thinking/signature transport, fragmented/multiple calls, exact IDs, system updates, deferred declarations and `tool_addition`; controlled HTTP/SSE and failure handling | Source and retained unit/model-free results are not full application acceptance |
| [Actual Messages results][messages-results], [model log][messages-log], [driver][messages-driver] | Actual `/v1/messages` text, streamed `read_file`, synthetic exact-ID result and paired continuation: **330 cached +28 suffix** | Tool ID **`call_00_YW0srlFqeLBXVg2Bctck4347`** joins the response/certificate/continuation. Result was synthetic; no Harness tool effect |
| [Development report: Direct official Harness acceptance][development] | Official rc.2 actually read `workspace/acceptance.txt`; exact call/result ID **`call_00_81AYjlZranyky3VGQpne1140`**; continuation restored frontier **754**; final `HARNESS_DIRECT_MTP_OK` | Bounded text/read-only filesystem case; stable/addition-only registry; extensions disabled. Current audit relied on the retained review: referenced original `/private/tmp/ds41f-direct-harness-e2e/` directory was unavailable |
| [Official Harness visual feedback report][visual-report] and linked evidence | Actual Native Messages image-guided edit/render operation | Bounded workflow, not arbitrary plugin acceptance or general visual-quality qualification |
| [Manual-compaction investigation][manual-report] | Source/model-free diagnosis: installed headless input is task text, not `/compact` dispatch; manual maintenance/summarization ownership traced | **No manual `/compact` execution**. Missing headless adapter is not a Runtime wire defect |
| [Completed automatic-compaction report][compaction-report] and linked evidence | Genuine summary, strict shrink, durable replacement; changed projection **C=0**, adopted followup **C=3120** | Official rc.2; immediate post-compaction projection was text-only |
| [Final finite E2E report][e2e-report], [audit][e2e-audit], [acceptance][e2e-acceptance], [wire evidence][e2e-wire] | **C8 PASS:** 19 Native Messages requests, ten genuine reads/nine sources, four image offloads, automatic compaction, changed-history reconstruction, reacquisition and paired publication | rc.2, bounded workload; all 19 requests joined native/wire completion. Not all negative/recovery/numerical cases or alpha.2 acceptance |
| Same final E2E evidence | Reacquired A: result **134**, new call ID **`call_00_YsREeVjJm2R3aaQgtHKz3190`**; compatible text prefix **C=3394**, surviving H/I and new A encoded; final comparison reused **C=5996** | New occurrence/effect, not replay of original A. Historical Vision immediately after compaction was not exercised |
| [Runtime native tests][messages-tests], `test_in_history_system_and_tool_addition` | Official conversion and encoding of system update plus deferred tool activation | Model-free; prior Harness `addition-only` setting was explicitly **not** dynamic-registry acceptance |
| [Upstream projection tests][h-content-tests], [Session history tests][h-history-tests], [agent tool-update tests][h-agent-tests], [serializer tests][h-serialize-tests] | Tagged source specifies removal/addition projection, deferred activation, schema-series reset and serialization | Inspected tests, not newly executed tests or Runtime E2E evidence |

Earlier separate API/frontend tool results are not promoted into Native Messages
proof. For example, the [earlier tool-audit receipt][other-route-receipt] retains
its own CPU/frontend scope, not acceptance of the integration assessed here.

### Goal evidence boundary

Goal packages appear in retained compositions, but no located retained artifact
establishes the complete **interactive `/goal` → automatic Goal rounds → Goal-tool
completion** lifecycle. Generic tasks called goals, ordinary followups, and package
presence are not that proof. This is a gap in the located evidence, not a claim
that a separately held original Goal execution result would be invalid.

## 4. Official command, Skill and plugin semantics

### Human commands

[CommandRuntime][h-commands] is a plugin-owned human-command registry:
`register`/`execute` records `command/run` and `command/done`; command text and
human result are not automatically model messages. The [interactive client][h-command-ui]
resolves `leadingInput`, `popupSelect`, `action`, or direct `execute` dispatch.
The [headless runner][h-headless] instead submits its single task using
`agent.followup(createUserMessage(...))`. No new headless command requirement is
implied: use a composed interactive Harness client for human-command qualification.

| Command/class | Owner and consequence |
|---|---|
| `/goal` | [Host handler `executeGoalCommand`][h-command-goal]: show/create/edit/pause/resume/clear Goal state. Create/edit attachments become an ordinary user followup; subsequent automatic rounds are separate orchestration |
| `/compact` | [Host handler `executeCompact`][h-command-compact] calls `compaction.compactNow` under idle maintenance; may issue an auxiliary summary and replace history. Manual dispatch is distinct from demonstrated automatic pressure compaction |
| `/model` | [Interactive popup][h-model-ui]; [Session `selectModel`][h-session-commands] controls subsequent provider/model/reasoning selection. Directory loading is not itself a generation request. Runtime only admits its [Flash aliases][runtime-aliases] |
| `/plan` | [Plan service][h-plan] changes guidance; optional text/attachments steer the agent. First entry progressively admits `exit_plan_mode`; later transitions generally retain that tool |
| `/permission` | [Harness sandbox/approval selection][h-permission] changes local execution policy and potentially guidance; no Runtime permission/execution authority is transferred |
| `/file` | [Client attachment action][h-file-ui]; later text/image projection uses existing content mechanisms |
| `/export`, `/feedback` | [Export][h-export] and [feedback][h-feedback] operations are Harness-owned; command lifecycle does not inherently become model-visible conversation |
| Skill `/name` gesture | A separate direct-user-message mechanism, not ordinary host-command registration; see below |

### Goal lifecycle, not just a command name

The [Goal domain][h-goal] persists `goal/change`; activation is process-local and
restoration does not silently inherit armed authority. The [round driver][h-goal-driver]
waits for idle/quiescent ownership and persistence checkpoints, queues numbered
objective-bearing user messages, fences stale revisions/competing input, imposes
round limits, and pauses/disarms on relevant cancellation/errors. Admitted automatic
rounds set `startsRequestSeries: true`.

[Goal tools][h-goal-tools] are `get_goal`, `create_goal`, `update_goal`. Harness
checks direct-human/automatic-round authority, renders JSON as text Tool Results,
and can defer ordinary wrapup context. No Goal-specific Messages extension is
required. Request-series admission can change prompt/tool-history projection and
reuse eligibility; it does not require Runtime-owned Goal persistence.
[Command tests][h-goal-command-tests] and [round-driver tests][h-goal-driver-tests]
are source-level evidence, not Runtime lifecycle execution acceptance.

### Skills

[Tool-skill `apply` / `invokedSkillNames`][h-tool-skill] implements two distinct paths:

- **Explicit user invocation:** whitespace-bounded `/skill-name` gestures in direct
  user text, including mid-sentence, inject loaded instruction messages after
  background/catalog context. Unknown or user-disabled names remain prose.
- **Model invocation:** ordinary `skill` Tool Call, scoped registry lookup, and
  text Tool Result containing canonical `<skill_content>` and resource hints.

The durable catalog is **user text**, not a new Messages block. Changes publish
replacement catalogs, including empty tombstones; incomplete discovery differs
from confirmed removal. Compaction can cause re-publication. Body-only edits load
on demand without necessarily changing catalog text. Registry provider disposal
invalidates discovery; [Skill registry and `renderSkillContent`][h-skill] own these
semantics. [Skill tests][h-skill-tests] cover catalog replacement, restrictions,
compaction re-publication and explicit gestures.

Removing a skill entry does not itself remove a model tool. Unloading/restricting
the `skill` plugin can change the tool registry and encounter dynamic-tool limits.
Ordinary Tool Call success alone does not qualify this full behavior.

### Plugins and representative categories

[Plugin manager][h-plugin-manager] and [package operations][h-plugin-operations]
separate installation/discovery from activation. Version/build-script checks,
profile persistence and supported reload/HMR own changes; a saved change can
require a subsequent step/restart. Cordis effect disposal unregisters contributions.
[ToolRuntime `register`/`select`][h-tools] evaluates scoped restrictions/conditions
per step. Conditions, reloads and disconnections can change offered tools.

- Filesystem/editor/shell plugins cross the boundary as ordinary declarations,
  calls and text results; image tools additionally return images. See [filesystem][h-fs]
  and [shell][h-shell] sources; real read/image execution is bounded by §3 evidence.
- [Web search][h-web-search] and [fetch][h-web-fetch] are client tools with rendered
  text, not automatically Messages server-side web-search blocks.
- Skills/Goal/plan use ordinary context and Tool Results as described above.
- [MCP tool bridge][h-mcp-tools] normalizes schemas and text/image output;
  generation swaps and [connection lifecycle][h-mcp-connection] can unregister tools.
- [`plugin_manager`][h-plugin-tool] is an ordinary model tool whose approved
  operations can change future prompts/tools. Runtime does not perform installation.
- [Request contributors][h-extensions] can add top-level fields. Shipped inventory
  and Session-log contributors are distinct from ordinary plugin tool transport.

Normal plugin operation can therefore encounter rejected `tool_removal` in
`in-history` mode, or default-enabled Session-log upload. Not every plugin has the
same wire effects, and no unrestricted plugin E2E claim follows from a file read.

## 5. Dynamic-tool semantics and consequential gaps

### 5.1 Upstream declaration/history pipeline

[Agent `buildRequest`][h-agent] logs request headers and name additions/removals.
[ToolHistoryProjection.apply/snapshot][h-history] resolves definitions from
referenced headers. [Tool declaration/projection functions][h-content]
`toolDeclarations` and `projectToolUpdates` then adapt history to the route.
[Messages `serialize`][h-serialize] emits deferred declarations and system-role
update blocks after the preceding user/tool-result turn. [Adapter][h-adapter]
adds `mid-conversation-tool-changes-2026-07-01` beta opt-in for update blocks.

The [tagged default Flash catalog][h-models] declares `toolUpdate: addition-only`;
[configuration][h-config] also supports `in-history`. Neither current deployment
configuration nor this default settles the desired long-term Runtime capability.

| Event | `in-history` | `addition-only` |
|---|---|---|
| Initial tools | Baseline declarations | Baseline declarations, filtered to currently active names |
| New tool | Retained deferred definition, activated at recorded addition | Same while currently active |
| Deactivation | Keep historical definition; emit removal at recorded position | Omit inactive definition; filter its activation blocks; omit removal block |
| Unchanged reactivation | Preserve removal and subsequent re-addition positions | Removal interval disappears from tool-update projection; retain only needed activation |
| Historical calls/results | Retained | Retained even when declaration is absent |
| Same-name definition changes | Restart declaration series | Restart declaration series |
| Missing/incomplete history or auxiliary prefix omitting needed updates | Fall back to current declarations without developer updates | Same fallback |

`projectToolUpdates` also omits earlier declaration-series updates. Session restore
reconstructs tool history; compaction/replacement or other series changes can
establish a new baseline. A same-name changed definition is **not** an in-place
schema-update block: `ToolHistoryProjection.apply` detects changed JSON and resets
its series. Unchanged restored names can use recorded re-additions instead.

Source tests: [projection tests][h-content-tests], particularly “omits removed
definitions and removal blocks for addition-only routes,” “re-offers an unchanged
restored tool through its recorded blocks,” and “declares a deferred tool once
across removal and re-addition”; [history tests][h-history-tests], particularly
“restarts declarations when a retained name returns with a changed definition”
and “reconstructs inherited history on restore and continues from live events.”

### 5.2 `tool_removal`: unsupported, not merely untested

The official [serializer][h-serialize] produces:

```json
{"type":"tool_removal","tool":{"type":"tool_reference","name":"name"}}
```

Recipe source locations inside the [retained archive][recipe-source]:

- `deepseek-recipe/src/protocol/anthropic/messages/request/schema.rs`:
  `MessagesContentBlock` has `ToolAddition`, no `ToolRemoval`.
- `deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs`:
  `convert_system_message_blocks` resolves additions; no removal conversion.
- `deepseek-recipe-core/src/messages.rs`: system semantic representation carries
  tool definitions, not a removal operation.
- `deepseek-recipe-encoding/src/v4/dsv41.rs`: `render_message` renders additions as
  `### Updated Tool Schemas`, not deactivation.

Runtime [Messages `blocks`/`convert_messages`][runtime-admission] also excludes
removal. Changing its allowlist alone cannot supply the missing Recipe semantics.

Addition-only **does deactivate current tools**: inactive declarations disappear,
and Harness's selected execution tool set independently governs dispatch. It is
not a workaround that leaves every disabled tool callable. Nevertheless it loses
the model-visible removal/reactivation timeline and historical declaration context
for disabled tools. It is not support for the `tool_removal` capability.

Historical calls/results survive both modes. Recipe `transform_messages` preserves
assistant call names/inputs and pairs result IDs without requiring each historical
name in the current declarations. Omitted declarations therefore do not inherently
make old call/result history malformed; the model's declaration context differs.

### 5.3 Canonical prompt identity and paired-KV implications

Recipe V4.1 `render_tool_prompt` excludes deferred schemas from the initial prompt;
`render_message` renders activated schemas at their system-history location.
Therefore, based on source semantics:

- Adding a deferred declaration need not rewrite already rendered tokens.
- Removing an initially rendered declaration can change the initial tool prompt.
- Removing a later-added declaration can erase an earlier schema-addition rendering.
- In-history removal is designed to preserve earlier declarations/updates and
  append deactivation instead, improving opportunities for prefix continuity.

**No mode guarantees a cache hit.** System consolidation, schema-series resets,
compaction, image projection, retained roots, and terminal holdout still matter.
[Paired checkpoint `acquire`][runtime-pairs] and [Runtime execution][runtime-execution]
require an actually compatible complete pair, not an arbitrary common-prefix
fragment or target-only state. The [automatic-compaction evidence][compaction-report]
shows old roots sharing only a short prefix were not reused: genuine changed
projection reconstruction selected C=0, then its compatible followup reused C>0.

In-history removal has a credible efficiency advantage over affected addition-only
reprojections, but **there is no measured removal-specific KV-hit, TTFT or throughput
evidence**. Do not claim every removal causes a cold miss, or every in-history
removal retains a hit. Schema changes still reset the upstream declaration series.

### 5.4 Independent removal recommendation — not implementation approval

**Recommend Recipe-owned removal support**, while retaining addition-only as an
alternative. It is not an emergency Production patch. Reasons are upstream-mode
compatibility, historical availability fidelity, fewer avoidable projection rewrites,
and practical registry lifecycle operation. Addition-only remains viable for a
bounded deployment, but is a compatibility/performance trade-off, not full parity.

Before implementation, establish **authoritative DeepSeek prompt-encoding semantics
for deactivation**. Harness defines the wire block, not the exact model-rendering
bytes. Do not invent Runtime-only deletion prose or drop the block silently.

The minimum sound change would require Recipe schema/role/reference validation,
conversion, ordered semantic representation and authoritative encoding; correct
historical definitions/reactivation/final availability and schema-series handling;
then Runtime admission enablement. Model-free projection/canonical-token checks
must precede a narrowly authorized Harness lifecycle acceptance. No new parser,
Session store, plugin executor or cache authority is justified.

### 5.5 Separate gap: all-deferred declaration sets

Recipe `partition_tools` in
`deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs` rejects a
nonempty tool list when every tool has `defer_loading=true`:

> tools: at least one tool must have defer_loading=false. All tools cannot be deferred.

Harness [tool projection][h-content] imposes no equivalent requirement and can
produce such a set. Example: addition-only removes the last immediate baseline
tool while currently active later-added tools remain declared deferred and have
retained activation history. Recipe refuses the declaration set before that
activation can make it useful.

This is an **independent source-established incompatibility**, not a consequence
of merely lacking `tool_removal`. The [Runtime development report][development]
records all-deferred native admission failure during fixture correction. The
competing source rules are established; occurrence and impact in a real tagged
plugin lifecycle remain **unverified**. A focused tagged-projection-to-Recipe
model-free check is the smallest next discriminator, not an inference campaign.

## 6. Compatibility matrix

Status is limited to the behavior and evidence stated in each row. “Avoidable”
means configuration can avoid the feature, not that the feature is supported.

| Capability | Official behavior / model-visible or wire effect | Runtime/Recipe behavior | Evidence | Status | Practical impact / priority | Remaining uncertainty |
|---|---|---|---|---|---|---|
| Human slash-command dispatch | Host registry and interactive client dispatch; command lifecycle not inherently model history | Harness-owned; ordinary resulting content supported, no Runtime command parser needed | §4; [commands][h-commands], [client][h-command-ui] | PARTIAL overall qualification | No new architecture; medium qualification priority | Intended interactive client E2E absent |
| `/goal` and Goal lifecycle | Persisted control, automatic user rounds/new series, Goal tools/text results and wrapup | Constituent wire shapes supported | §3–4; [Goal driver][h-goal-driver], [tools][h-goal-tools], [native tests][messages-tests] | PARTIAL | Acceptance uncertainty; medium | Complete interactive lifecycle not located in execution evidence |
| Manual `/compact` | Idle maintenance; auxiliary summary and durable surface replacement | Required transport/reconstruction implemented; manual dispatch not executed | [Manual report][manual-report]; [handler][h-command-compact] | PARTIAL | Qualification gap, not identified wire defect; medium | Interactive invocation/cancellation/persistence acceptance |
| Automatic compaction | Pressure summary, strict shrink, replacement and continuation | Actually demonstrated over Native Messages | [Completed compaction][compaction-report]; [final E2E][e2e-report] | SUPPORTED, scoped | Functional/correctness; preserve | rc.2 evidence; broader recovery and retained-image cases not all qualified |
| `/model` | Client model/reasoning selection affects later requests | Fixed admitted Flash aliases; no multi-model promise | [UI][h-model-ui], [selectModel][h-session-commands], [aliases][runtime-aliases] | PARTIAL | Intentional deployment scope; low | Tagged interactive selection not executed |
| Explicit Skills | User gesture injects canonical loaded instruction text | Supported text path; feature E2E absent | [Tool-skill][h-tool-skill], [tests][h-skill-tests], native text evidence | PARTIAL | Medium qualification priority | User-only policy, catalog/lifecycle integration not executed |
| Model-invoked Skills | Catalog user text; `skill` call and text result; updates/re-publication | Existing content/tool path; complete Skill lifecycle not accepted E2E | [Registry][h-skill], [tool/tests][h-tool-skill], §3 | PARTIAL | Medium | Ordinary call success does not qualify catalog changes/compaction |
| Ordinary plugin tools | Client schema/call/result; Harness execution/permissions | Implemented and executed within bounded tools/workloads | [Messages results][messages-results], direct acceptance in [development][development], [E2E][e2e-report] | SUPPORTED, bounded | Functional; preserve | Not arbitrary plugins or destructive-tool qualification |
| Plugin management/lifecycle | Profile installation/discovery, activation/reload/disposal; subsequent prompt/tool changes | No Runtime plugin host required; lifecycle may reach unsupported removal/extensions | [Manager][h-plugin-manager], [operations][h-plugin-operations], [management tool][h-plugin-tool] | PARTIAL overall; ownership boundary supported | Conditional functional restriction; high dynamic-lifecycle priority | No live installation/enable/disable campaign; application can be deferred to restart |
| MCP tools | Schema discovery, text/image normalization; swaps/disconnects unregister | Ordinary shapes supported; changing MCP registry not qualified | [Bridge][h-mcp-tools], [connection][h-mcp-connection], image evidence | PARTIAL | Medium, subject to high-priority dynamic gaps | Arbitrary MCP output/lifecycle E2E absent |
| Initial declarations | `tools[].input_schema` | Recipe conversion/encoding; actual bounded runs | [Native tests][messages-tests], [E2E][e2e-report] | SUPPORTED, bounded | Functional; preserve | Not every schema/tool population |
| Deferred loading | `defer_loading`; definitions activate through retained history | Implemented except all-deferred restriction | `test_in_history_system_and_tool_addition` in [tests][messages-tests]; §5 | SUPPORTED at model-free conversion level, bounded | High validation priority | Dynamic Harness E2E absent; all-deferred excluded from this support claim |
| Dynamic additions | Deferred new definitions plus system `tool_addition` | Implemented/encoded; model-free verified | Same Runtime test; [projection tests][h-content-tests], [serializer tests][h-serialize-tests] | SUPPORTED at model-free conversion level | Functional/performance; high validation | Actual changing Harness registry not accepted E2E |
| Dynamic removals/reactivation in-history | Retain definitions; system `tool_removal` then re-addition | Missing Recipe representation/conversion/encoding; Runtime rejects | §5.2; [admission][runtime-admission], Recipe archive | UNSUPPORTED, avoidable by mode selection | Compatibility/fidelity/potential performance; high | Exact authoritative prompt encoding and removal-specific performance evidence absent |
| Addition-only deactivation | Omit inactive declarations and historical activation blocks; preserve calls/results | Supported shapes subject to all-deferred edge | §5.1–3; [projection tests][h-content-tests] | PARTIAL lifecycle qualification | Possible reprojection cost; high validation | Not removal support; dynamic E2E absent |
| Same-name schema changes | Restart declaration series; old update series omitted | New schema can convert; exact-prefix safety remains authoritative | [History tests][h-history-tests], [projection][h-history], Runtime pair owner | PARTIAL | Correctness/reconstruction; high validation | No same-name lifecycle E2E; no guaranteed reuse |
| All-deferred declarations | Upstream projection can produce only deferred declarations | Recipe `partition_tools` rejects nonempty all-deferred set | §5.5; [development][development], Recipe archive | UNSUPPORTED | Conditional functional refusal; high | Real tagged plugin-lifecycle occurrence/impact not verified |
| `dsh_plugin_packages` | Default-enabled active package inventory metadata | Accepted and excluded before Recipe; not prompt/tool authority | [Contributor][h-inventory]; [admission][runtime-admission]; inventory tests in [Messages tests][messages-tests] | SUPPORTED as nonsemantic metadata | Compatibility; preserve | No claim that Runtime consumes inventory semantically |
| `dsh_session_log` | Default-on incremental event contributor; acceptance watermark | Rejected; intentionally outside integration goal | [Contributor][h-session-log], unsupported-wire tests | UNSUPPORTED, intentionally excluded | Explicit disabling required; preserve ownership | No server-side event ingestion/acknowledgment |
| Arbitrary plugin request extensions | Contributor may add other top-level controls | Unknown fields rejected | [Extension registry][h-extensions]; [admission][runtime-admission] | UNKNOWN per extension; unknown fields unsupported | Assess concrete need | No universal plugin-extension compatibility claim |

## 7. Other relevant limitations, without scope inflation

Runtime [admission][runtime-admission] / [capability gate][runtime-capabilities]
rejects server-tool blocks, native documents, constrained-output controls, thinking
budgets, `tool_choice:any`, enforced serial tool selection, and unknown extensions.
The [unsupported-wire tests][messages-tests] preserve explicit refusals rather
than silently claiming enforcement.

The tagged [serializer][h-serialize] does not normally emit those controls for
the audited Skill/Goal/web/plugin-tool paths: web tools are client tools, file
projection can use handle text, normal tool calls do not demand serial decoding,
and reasoning uses thinking/effort fields. These are real protocol limitations,
but do not independently justify implementing every feature for this integration.
The [image resolver][runtime-images] supports bounded Files/inline handling; this
must not be confused with an unsupported native document-block feature.

### Conditional model-discovery interoperability

Runtime [`models`][runtime-server] lacks alpha.2 listing metadata
`api_capabilities.anthropic_messages`. The official [listing parser][h-listing] /
[catalog resolver][h-listed-catalog] rejects such a listed entry. However, the
[API-key provider configuration][h-api-key-config] follows that listing only at
the official API origin with its built-in catalog. A custom local endpoint does
not automatically use that path. This is conditional discovery interoperability,
not proof that ordinary local `/model` is broken; optional metadata work is lower
priority unless a supported deployment needs that listing route.

## 8. Independent engineering judgment and priorities

1. **Recipe-aligned `tool_removal` support:** recommended for upstream-mode
   compatibility, historical fidelity and potential reuse efficiency. Establish
   authoritative encoding before implementation; do not simply relax admission.
2. **All-deferred declaration incompatibility:** independently validate and resolve
   the conflicting source rules. Keep it separate from removal work.
3. **Focused dynamic-tool lifecycle acceptance:** add → call/result → deactivate →
   continue → unchanged reactivate → same-name schema change → resume/compaction.
   Compare both update modes using exact wire/canonical-token/pair evidence;
   neither mode guarantees reuse.
4. **Interactive Goal and manual compaction qualification:** use the intended
   interactive Harness client. No headless dispatcher or Runtime slash parser.
5. **Skill and plugin/MCP lifecycle qualification:** explicit and model-loaded
   Skills, catalog replacement/removal and compaction re-publication; one harmless
   changing registry, not a blanket plugin campaign.
6. **Preserve the established bounded integration:** keep valid Native Messages,
   exact-ID, paired continuation, visual/offload/reacquisition and automatic
   compaction results. Do not relabel them as unverified merely because this audit
   did not rerun them. Do not extend them to tagged alpha.2 E2E or all plugins.

These are investigation recommendations only. **No fix, test campaign,
configuration change or deployment is authorized by this report.** No duplicate
Session history, plugin host, parser or unrelated development campaign is proposed.

### Smallest remaining evidence checks — not performed

- Model-free tagged Harness projection → installed Recipe conversion for
  all-deferred and same-name changes; compare canonical token sequences by mode.
- Once removal semantics exist, validate exact expected removal rendering/prefix
  identity before inference.
- Bounded interactive `/goal` create/pause/resume through automatic completion,
  plus separate manual `/compact` auxiliary request/persistence inspection.
- Explicit user-only Skill, model-loaded Skill, catalog updates and post-compaction
  re-publication.
- Harmless activation/deactivation/reactivation/schema-change registry sequence,
  with exact request capture and actual compatible-pair selection.
- Authorized alpha.2 execution: rc.2 results remain valid within scope but cannot
  substitute for running the specified tagged client.

## 9. Reference verification limits

All local Markdown evidence/source links below are repository-contained targets.
Upstream links are pinned to the verified official commit, never unpinned master.
Recipe paths refer to the retained archive and patch, not a temporary extraction.
The temporary upstream/Recipe extraction directories are not permanent evidence
locations and are not required by this report.

The original `/private/tmp/ds41f-direct-harness-e2e/` evidence directory could not
be independently reread during the audit; its retained review in
`docs/deepinfer-development.md` is cited with that limitation. Installed Harness
manifest versions are not exact build-source proof. The complete interactive Goal
lifecycle and alpha.2 Runtime execution remain evidence gaps, not inferred passes.

<!-- Repository references -->
[development]: ../../docs/deepinfer-development.md
[installed-report]: ../production-multimodal-69689d5/report.md
[recipe-source]: ../../third_party/mtp/recipe-02-source.tar.gz
[recipe-patch]: ../../docs/recipe-02-native-semantic-boundary.patch
[recipe-identity]: ../../third_party/mtp/normal-local.json
[recipe-wheel]: ../../third_party/mtp/deepseek_recipe-0.2.0-cp310-abi3-macosx_11_0_arm64.whl
[runtime-admission]: ../../ds41f_mlx/serving/messages_admission.py
[runtime-capabilities]: ../../ds41f_mlx/serving/ordinary_admission.py
[runtime-execution]: ../../ds41f_mlx/serving/production_mtp.py
[runtime-pairs]: ../../ds41f_mlx/serving/paired_checkpoint.py
[runtime-images]: ../../ds41f_mlx/serving/image_files.py
[runtime-server]: ../../ds41f_mlx/serving/server.py
[runtime-aliases]: ../../ds41f_mlx/mtp_profile.py
[messages-tests]: ../../tests/test_recipe02_messages_serving.py
[messages-results]: ../deepinfer-messages-http/results.json
[messages-log]: ../deepinfer-messages-http/real-model.log
[messages-driver]: ../deepinfer-messages-http/run.py
[visual-report]: ../visual-feedback-loop/phase-3-report.md
[manual-report]: ../multimodal-runtime/phase3-step3b-compaction.md
[compaction-report]: ../multimodal-runtime/phase3-step3br3-compaction-cleanup.md
[e2e-report]: ../multimodal-runtime/phase4-e2e-native.md
[e2e-audit]: ../multimodal-runtime/phase4-e2e-evidence/audit.json
[e2e-acceptance]: ../multimodal-runtime/phase4-e2e-evidence/acceptance.json
[e2e-wire]: ../multimodal-runtime/phase4-e2e-evidence/wire.jsonl
[other-route-receipt]: ../tool-capability-audit/cpu-attempt-1/receipt.json

<!-- Official Harness references: all pinned to dsh-v0.2.2-alpha.2. -->
[h-agent]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/core/agent-loop/src/agent.ts
[h-history]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/core/session/src/tool-history.ts
[h-content]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm/src/content.ts
[h-serialize]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/src/serialize.ts
[h-adapter]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/src/adapter.ts
[h-models]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/src/models.ts
[h-config]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/src/config.ts
[h-content-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm/tests/content.spec.ts
[h-history-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/core/session/tests/tool-history.spec.ts
[h-agent-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/core/agent-loop/tests/tool-updates.spec.ts
[h-serialize-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/tests/serialize.spec.ts
[h-commands]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/interaction/commands/src/index.ts
[h-command-ui]: https://github.com/deepseek-ai/deepseek-harness/tree/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/client/ui-commands/src/client
[h-headless]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/bundle/headless/src/index.ts
[h-command-goal]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/goal/command-goal/src/index.ts
[h-command-compact]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/compaction/command-compact/src/index.ts
[h-model-ui]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/client/ui-model-selection/src/client/index.ts
[h-session-commands]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/api/session-controller/src/commands.ts
[h-plan]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/plan/plan-mode/src/index.ts
[h-permission]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/interaction/permission-presets/src/index.ts
[h-file-ui]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/client/ui-conversation/src/client/apply.ts
[h-export]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/session-query/session-log-export/src/index.ts
[h-feedback]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/feedback/command-feedback/src/index.ts
[h-goal]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/goal/goal/src/index.ts
[h-goal-driver]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/goal/goal-round-driver/src/index.ts
[h-goal-tools]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/goal/tool-goal/src/index.ts
[h-goal-command-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/goal/command-goal/tests/command-goal.spec.ts
[h-goal-driver-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/goal/goal-round-driver/tests/goal-round-driver.spec.ts
[h-tool-skill]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/skill/tool-skill/src/index.ts
[h-skill]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/skill/skill/src/index.ts
[h-skill-tests]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/skill/tool-skill/tests/tool-skill.spec.ts
[h-plugin-manager]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/boot/plugin-manager/src/index.ts
[h-plugin-operations]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/boot/plugin-manager/src/operations.ts
[h-plugin-tool]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/boot/plugin-manager/src/tools.ts
[h-tools]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/core/tools/src/index.ts
[h-fs]: https://github.com/deepseek-ai/deepseek-harness/tree/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/fs/tool-fs
[h-shell]: https://github.com/deepseek-ai/deepseek-harness/tree/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/shell/tool-bash
[h-web-search]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/web/tool-web/src/search.ts
[h-web-fetch]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/web/tool-web/src/fetch.ts
[h-mcp-tools]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/mcp/mcp-client/src/tools.ts
[h-mcp-connection]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/mcp/mcp-client/src/connection.ts
[h-extensions]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/deepseek-llm-api-extensions/src/index.ts
[h-inventory]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/plugin-package-inventory-deepseek/src/index.ts
[h-session-log]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/session/session-log-deepseek/src/index.ts
[h-listing]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/src/listing.ts
[h-listed-catalog]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek/src/listed-catalog.ts
[h-api-key-config]: https://github.com/deepseek-ai/deepseek-harness/blob/f3803c96974ad4a94195fb97e5c637e9c2a3afe3/packages/llm/llm-deepseek-api-key/src/config.ts
