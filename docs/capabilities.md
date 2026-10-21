# Current capabilities and limits

This page owns the current profile comparison and support boundaries. It describes
the checkout's implemented contracts and retained scoped evidence, **not fresh
qualification of every later HEAD or the currently installed package**. Deploy only
an explicitly qualified installed payload; see [operations](operations.md).
Architecture explains ownership; specialized documents explain endpoint semantics.

## DeepInfer current implementation

The unified `merge` branch supports official **original Flash and Flash-1017**
with Recipe **0.2.0**, one consolidated native semantic extension, and native GLM.
`--checkpoint-variant flash|flash-1017` selects the immutable package identity;
1017 remains the default. Wrong-checkpoint admission and cross-variant paired
state fail closed. Standard-OFF is retired from production; all OFF descriptions
and evidence retained later on this page are historical, not active capabilities.
The published original-Flash release and historical Recipe 0.1 installations remain separate;
[operations](operations.md#deepinfer-daily-operation-and-harness) owns selection.
Bounded installed Chat/Messages and direct official Harness acceptance are recorded
in the [development record](deepinfer-development.md); neither renews release scope.

Ordinary `POST /v1/messages` uses official Recipe conversion, native Messages
semantic ownership/accumulation and JSON/SSE, not Messages-to-Chat translation.
It supports text/thinking/signatures, client tools, exact-ID results, in-history
system updates, Recipe-supported additions and inline base64 PNG/JPEG/WebP.
Successful streaming terminates with **`message_stop`**, not Chat's `[DONE]`.
Tool suffix delivery waits for canonical settlement; a `tool_use` stop reason or
argument-end observation alone never authorizes effects. Production sessions
remain **Chat-only**; Messages has no logical-session or persistence contract.

Messages intentionally rejects Files/documents, URL/file image sources, server/
web-search tools/results, tool-reference results, unknown extensions (including
Harness `dsh_session_log` / `dsh_plugin_packages`), tool removal, constrained output,
thinking budgets, `tool_choice:any` and enforced serial selection. Prompt-based
named selection is not constrained decoding. No Responses parity is added.

## Profiles

| Capability | `mtp-serving-v1` | `mtp-singleton-v1` |
| --- | --- | --- |
| Selection | CLI default; either package-fixed checkpoint variant | Explicit; same MTP package |
| Physical execution | Pinned oMLX Target/DSpark, ProductionScheduler-owned | Bounded guarded singleton MTP |
| GLM distribution | Native default; portable explicit alternative | Same installed distribution |
| Ordinary APIs | Recipe 0.2 Chat Completions and native Messages, JSON/SSE; health/models | Specialized session API |
| Independent conversations/edits | Safe paired-prefix reuse or miss, including branches | Exact session extension/frozen fences |
| Active execution | One native-active generation; independent requests queue, not batch | One response lease/live session |
| Optional stateful Chat | 16 logical conversations; streaming only | One process-lifetime session |
| Function tools | Recipe-authoritative calls and exact-ID results | Pinned weather/dedicated-client contract |
| Sampling/reasoning | `generation-policy-v1` | Greedy/thinking-off contract |
| Native persistence/restore | Unsupported | Unsupported |
| Images | Resource-budgeted inline Chat/Messages; paired continuation | Unsupported |
| Network | Loopback default; explicit trusted private LAN | Literal IPv4 loopback/exact Host |

These profiles do not grant each other API, persistence, image, concurrency or
application qualification. Historical retired OFF [API](api.md); current protocols:
[ordinary MTP](mtp-production-serving.md), [production logical sessions](production-sessions.md)
and [bounded singleton](mtp-local-release-candidate.md).
GLM selection is fixed by the installed distribution, not a request option or
performance inference. Native admission failure refuses startup, never switches
to portable. See [exact distribution admission](runtime-admission.md#glm-execution-distributions).
This bounded native promotion does not enlarge API/image/long-context qualification.

## Capacity, resource policy and defaults

| Boundary | Current contract |
| --- | --- |
| Text core | At most **1,048,576 total tokens**, or the smaller checkpoint-configured range |
| Ordinary MTP output | Integer **1..393,216**; omitted: **8192** non-thinking, **65,536** thinking low/high, **131,072** thinking max |
| Ordinary MTP total admission | Actual recipe prompt + requested output must fit the total envelope |
| Ordinary MTP body | **16 MiB** bounded JSON ingress; byte budget is not a token estimate |
| Ordinary paired cache | 18 complete immutable pairs shared with logical-session pins (16 conversations + two publication slots), plus reserved null block; upstream lookup/refcount/LRU, no separate eviction authority. Active/restored handles are additional |
| Production logical-session identities | 16 retained logical conversations, one native-active generation; 600-second nonbusy idle retirement (30-second sweep). 4096 non-reusable lifetimes/process and 65536 reserved application correlations/session; exhaustion fails closed |
| Singleton | **8192 total / 768 output**, 1 MiB body; its specialized limits remain unchanged |

The output ceiling is a capability, not mandatory generation or output-sized KV
preallocation. State grows with actual executed tokens. EOS, recipe completion,
tools and application cancellation can terminate far below it. Auto output in
Chat uses runtime budget calculation; it is not an unlimited-completion promise.
Thought/runaway loops, agent loops, timeouts and operator cost are separate policies.

First-party OFF core evidence actually reached the 1M consumed frontier; ordinary
MTP composes retained long-context execution/state evidence with affected serving
acceptance. This is **not** a new near-1M HTTP/agent endurance campaign or a claim
that arbitrary maximum-length workloads have been tested. The measured OFF
maintenance/document-ingestion class accepts long initial prefill latency.
Source-scoped evidence: [very-long text core](very-long-context-production-qualification.md)
and [ordinary serving](mtp-production-serving.md#long-context-promotion-evidence-and-retention).

## Generation controls

Ordinary MTP advertises **`generation-policy-v1`**. Request defaults:

| Parameter | Default |
| --- | --- |
| temperature / top_p | **1.0 / 0.95** |
| top_k / min_p | 0 / 0.0 |
| repetition / presence / frequency penalty | 1.0 / 0.0 / 0.0 |
| seed | null; best-effort, not a reproducibility guarantee |
| reasoning | Headless: Off / effort `none`; TUI Runtime launch: saved preference, initially `max` |

Explicit greedy uses temperature zero. Advertised reasoning efforts are `none`,
`low`, `high`, `xhigh`, `max`; `xhigh` is an accepted compatibility spelling for
canonical effective `high`, not an independent model effort. Recipe conversion
already maps `minimal` to `low` and `medium` to `high`; ds41f's post-conversion
`generation_policy.reasoning_policy` canonicalizes retained `xhigh` to `high`.
No thinking token-budget control is supported.
For Chat Completions, `--default-reasoning-effort` overrides
`DS41F_DEFAULT_REASONING_EFFORT` (headless fallback `none`). The TUI passes its
saved preference as that CLI option; the environment overrides the saved preference
on TUI startup, and Settings can change it for the next launch. Omitted reasoning
controls use the Runtime default; explicit `reasoning_effort` or `thinking`,
including Off, takes precedence. `/health` advertises the effective default in
`chat_defaults.reasoning_effort` and `generation_policy.reasoning`. This option
does not change Responses/Messages reasoning conversion or the singleton's
thinking-off contract. See [operations](operations.md).
DeepInfer ordinary Recipe 0.2 additionally resolves nonthinking `top_p=1` and
thinking temperature `1` / minimum `top_p=.95`; these are official converted
settings, not a change to the retained original-Flash release's default evidence.

Sampling is request-owned. Retained session reasoning mode/effort is frozen and
changes reject; stateless reasoning is request-owned. Retained production sessions
also freeze tool declarations. Unsupported controls reject rather than silently
changing semantics.

External clients own workflow settings and tool-result continuation. An ordinary
request that omits fields gets defaults for **that request**, not hidden
inheritance of previous UI choices. Browser workflow snapshots are retired; the
TUI retains only Runtime launch default reasoning settings. See
[generation controls](generation-policy-investigation.md).

## Streaming, recovery and effects

- Recipe owns JSON/SSE projection; bytes, yields, `[DONE]` and HTTP success are not
  canonical-token acknowledgements or permission to replay a request.
- OFF stateful streaming uses demand-driven live recipe chunks and a bounded
  pending batch; final canonical `last_turn` publication is separate from delivery.
  Ordinary MTP streaming/outcomes have Scheduler-owned settlement semantics;
  neither is the other's transport or retry contract.
- Production logical sessions use expected-count/application-UUID correlations,
  not singleton sequence headers. Identical POSTs reject; GET observes canonical
  outcomes without execution. No automatic generation resend or native restore.
- Incomplete/unrepresentable tool or reasoning boundaries can be unrecoverable.
  Partial chunks do not authorize effects or invented transcript repair.
- Tools execute outside the runtime. Application reservations/known results do
  not promise crash-safe or distributed exactly-once side effects. Unknown outcomes
  remain fenced; process restart is not effect reconciliation.

## Bounded inline images

The separately qualified OFF Chat Completions core allows up to **four inline
single-frame PNG/JPEG/WebP images**, 16 MiB encoded bytes and 4M source pixels each,
aspect 1:2–2:1, high/default/auto detail, within **8192 expanded multimodal positions
including reserved output**. Remote/file URLs, low detail, animations, multimodal
Responses/Messages and excess context are outside this envelope.

New images are encoded once; retained old image identities/spans must remain exact.
Ordinary MTP removes the four-image and 8192-position development caps: actual
expanded prompt + requested output must fit the checkpoint context. Output remains
1..393,216 (Auto reserves the lesser of remaining context and that ceiling).
Per-image format/detail/aspect, 16 MiB, 4,194,304 pixels and 1014 expanded-position
bounds remain. Aggregate preparation admits at most **64 MiB encoded image bytes,
16,777,216 decoded source pixels and 36,864 planned ViT patches**; images are
validated/processed individually. The existing **16 MiB JSON body** bound also
applies, including base64/history. These are finite resource budgets, not a new
image-count ceiling or memory reservation. All supplied images, including cached
ones, consume preparation budgets. Ordinary text and legacy OFF limits are unchanged.

`/health` and `/v1/models` expose ordinary `vision_admission` resource bounds,
`implemented_total_tokens` and `expanded_capacity_model_execution: unverified`.
The ordinary Vision budget response uses `implemented_total_tokens`, not
`qualified_total_tokens`. Source/synthetic MLX tests cover >4 images and >8192
positions. DeepInfer has bounded real-model evidence for six images / 9,353 expanded
prompt positions with the earlier Recipe 0.1 package, then one small Chat image
and cached follow-up with Recipe 0.2. See the [scoped development evidence](deepinfer-development.md).
This is not maximum-capacity, long-context/memory or visual-quality qualification;
real Harness image/Files workflows have not been exercised. Health's conservative
`unverified` field is not a claim that no bounded model case has ever passed.

Image identity is included in
upstream paired-cache hashes; after prefill only the existing target/DSpark pair
is executable authority. Prior-message images require a compatible retained pair:
eviction or an incompatible historical-image edit rejects without hidden replay
or re-encoding. Fresh last-message images may prefill on a safe miss. MTP requires
two terminal text tokens for the existing N-2 prompt capture/P5 handoff.
See [Vision/MTP seam audit](mtp-vision-integration.md) for delta acceptance scope.
The original Vision delta did not publish an installed package; subsequent local
MTP delivery is recorded in the [runtime closure](runtime-completeness.md).
OFF idle persistence and exact fresh-process restore do not re-run old images.
The original OFF receipt alone is not MTP qualification; the linked MTP delta
acceptance is separate. Neither is general visual-quality proof, larger-image/
context qualification or broadened release/R1 qualification. See [OFF API](api.md#bounded-multimodal-developmentcore-extension)
and [original image qualification](multimodal-production-qualification.md).

## Applications

- **Operator TUI:** Runtime Start/Stop, launch Settings, readiness, metrics/logs
  and diagnostics; [operator console](operator-chat.md),
  [live metrics and controls](operator-live-observability.md). Q leaves Runtime
  running. No conversations, Web control, Browser launcher or tool executor.
- **Browser Chat / acquisition / PDF:** retired, including the application server,
  assets, workflows, leases and effect ledger. These were not moved into Runtime.
  Bounded inline Runtime Vision and production logical sessions remain supported.
- **Ordinary clients/OpenCode:** standard Chat Completions and generic client-owned
  tools; no compulsory singleton session/fence protocol. Client acceptance is
  source/version-scoped, not a universal all-client compatibility certificate.
- **Official Harness 0.2.1-rc.2:** direct native Messages text and actual read-only
  file-tool/result continuation passed with supported option exclusions. No image/
  Files, cancellation or dynamic tool-update workflow acceptance; see
  [connection configuration](operations.md#deepinfer-daily-operation-and-harness).
  This historical observation is not comprehensive Agent/Harness compatibility;
  no replacement Harness is bundled or process-managed by the TUI.
- **Rust:** raw OFF HTTP/SSE/process boundary; no Rust model/KV/parser owner and no
  claim of a supported Rust MTP application seam.

External clients retain native Chat Completions and Messages and generic
client-side function-tool protocols. Model-specific orchestration, history and
tool execution belong to those clients or a future dedicated Harness, not Runtime.
Do not mutate an admitted runtime environment to install application dependencies.

## Completion and unsupported scope

Ordinary MTP's paired restore and Scheduler/P5 settlement boundaries are closed for
the supported path. **READY_FOR_SCOPED_RELEASE** for the declared single-operator
local / explicitly trusted-private-LAN runtime, assessed at implementation
`79e96fdd3d12165caf40a885ef3773a768efbcda`; see the
[release assessment](runtime-completeness.md#general-local-runtime-release-assessment).
No broader scope or package promotion follows from that decision.
**Separate-machine trusted-LAN acceptance PASS**: the
[Windows-to-Mac evidence](remote-lan-windows-acceptance.md) records real LAN
health/models, ordinary Chat JSON/SSE and Mac operator correlation.
This remains a trusted-LAN / single-operator boundary; authentication, TLS,
Internet exposure and multi-user serving are unsupported.

Local default-MTP promotion is approved by the [runtime closure review](runtime-completeness.md).
No native MTP persistence/restart recovery, shared/batched MTP execution, sessionized Responses/Messages, cross-backend KV portability,
server-side arbitrary tools, distributed/crash-safe exactly-once effects,
Internet/multi-tenant security or above-checkpoint context claim is granted.
Ordinary trusted-LAN serving has no authentication or TLS; wildcard bind is not a
security boundary. Unsupported stateful stop-string rollback is not added.

A local seal, build success, inspect success, old receipt or directory name is not
fresh real-model/release acceptance. Full R1 verification, package publication,
and one-way release projection remain separate checkpoints;
see [implementation decisions](implementation-plan.md).
