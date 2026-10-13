# Current capabilities and limits

This page owns the current profile comparison and support boundaries. It describes
the checkout's implemented contracts and retained scoped evidence, **not fresh
qualification of every later HEAD or the currently installed package**. Deploy only
an explicitly qualified installed payload; see [operations](operations.md).
Architecture explains ownership; specialized documents explain endpoint semantics.

## Profiles

| Capability | `standard-off` | `mtp-serving-v1` | `mtp-singleton-v1` |
| --- | --- | --- | --- |
| Selection | Explicit fallback; separate OFF package | Local CLI default; MTP package | Explicit; MTP package |
| Physical execution | First-party target generation/model execution; MTP OFF | Pinned oMLX MTP topology, ProductionScheduler-owned | Bounded guarded singleton MTP |
| Ordinary APIs | Text Chat Completions, Responses, Messages within documented scopes | Chat Completions (text and bounded inline Vision); health/models | Specialized session API, not ordinary stateless Chat Completions |
| Independent conversations/edits | Stateless requests prefill fresh; retained sessions require exact extension | Ordinary requests may safely reuse paired prefixes or miss, including edits/branches | Exact session extension/frozen request fences |
| Active execution | Single-flight; overlap conflicts | One native active generation; independent ordinary requests may queue, no shared native batch | One response lease/live session |
| Optional stateful Chat | Exact-prefix sessions, default maximum 4 | One logical session on the same Scheduler; streaming only | One issued process-lifetime session with sequence/body fence |
| Function tools | Ordinary client function tools | Generic recipe-authoritative function tools/results | Pinned weather declaration and bounded dedicated-client contract |
| Sampling/reasoning | `generation-policy-v1` | `generation-policy-v1` | Separate greedy/thinking-off qualification contract |
| Native idle persistence/restore | Validated same-backend artifacts | Unsupported | Unsupported |
| Images | Separate bounded OFF image extension below | Same bounded inline envelope; paired-state continuation, no persistence/history replay | Unsupported |
| Network | See OFF configuration and application network policy | Loopback default; explicit trusted private LAN | Literal IPv4 loopback and exact Host policy |

These profiles do not grant each other API, persistence, image, concurrency or
application qualification. Protocol detail: OFF [API](api.md),
[ordinary MTP](mtp-production-serving.md), [production logical sessions](production-sessions.md)
and [bounded singleton](mtp-local-release-candidate.md).

## Capacity, resource policy and defaults

| Boundary | Current contract |
| --- | --- |
| Text core | At most **1,048,576 total tokens**, or the smaller checkpoint-configured range |
| Ordinary MTP output | Integer **1..393,216**; missing `max_tokens` resolves to **128** |
| Ordinary MTP total admission | Actual recipe prompt + requested output must fit the total envelope |
| Ordinary MTP body | **16 MiB** bounded JSON ingress; byte budget is not a token estimate |
| Ordinary paired cache | Four complete immutable pairs, upstream lookup/refcount/LRU; no separate eviction authority |
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

OFF and ordinary MTP advertise **`generation-policy-v1`**. Request defaults:

| Parameter | Default |
| --- | --- |
| temperature / top_p | **1.0 / 0.95** |
| top_k / min_p | 0 / 0.0 |
| repetition / presence / frequency penalty | 1.0 / 0.0 / 0.0 |
| seed | null; best-effort, not a reproducibility guarantee |
| reasoning | Headless: Off / effort `none`; TUI Runtime launch: saved preference, initially `max` |

Explicit greedy uses temperature zero. Advertised reasoning efforts are `none`,
`low`, `high`, `xhigh`, `max`; no thinking token-budget control is supported.
For Chat Completions, `--default-reasoning-effort` overrides
`DS41F_DEFAULT_REASONING_EFFORT` (headless fallback `none`). The TUI passes its
saved preference as that CLI option; the environment overrides the saved preference
on TUI startup, and Settings can change it for the next launch. Omitted reasoning
controls use the Runtime default; explicit `reasoning_effort` or `thinking`,
including Off, takes precedence. `/health` advertises the effective default in
`chat_defaults.reasoning_effort` and `generation_policy.reasoning`. This option
does not change Responses/Messages reasoning conversion or the singleton's
thinking-off contract. See [operations](operations.md).
Sampling is request-owned. Retained session reasoning mode/effort is frozen and
changes reject; stateless reasoning is request-owned. Retained production sessions
also freeze tool declarations. Unsupported controls reject rather than silently
changing semantics.

Browser/TUI snapshot all eight sampling fields for a user workflow, retaining that
snapshot through tool results, continuation and cancellation/reconciliation. A new
user turn can choose new sampling settings. An ordinary request that omits fields
gets defaults for **that request**, not hidden inheritance of previous UI choices.
See [frontend contract and scoped evidence](generation-policy-frontends.md).

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
Ordinary MTP uses this same input/output envelope. Image identity is included in
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
context support or broadened release/R1 qualification. See [OFF API](api.md#bounded-multimodal-developmentcore-extension)
and [original image qualification](multimodal-production-qualification.md).

## Applications

- **Operator TUI:** LLM Runtime/Web Chat Start/Stop/Open, Settings, metrics/logs
  and process diagnostics; [operator console](operator-chat.md),
  [live metrics and controls](operator-live-observability.md). Q leaves both
  services running. No TUI Chat frontend or conversation/session ownership.
- **Browser Chat:** separate application server over canonical runtime APIs;
  [Chat interface](chat-interface.md), [application/state contract](web-application.md).
  Multiple independent tab-memory conversations; New adds and Switch preserves.
  Uncertainty journals are not an archive/native restore. Server idle/orphan
  retirement, not user Close, releases internal resources.
- **Acquisition/PDF:** bounded search/HTML/original-image/PDF tools are application
  effects. [PDF workflow](pdf-document-workflow.md) retains documents and uses
  separate fresh OFF page Vision, not main-chat MTP image admission.
- **Ordinary clients/OpenCode:** standard Chat Completions and generic client-owned
  tools; no compulsory singleton session/fence protocol. Client acceptance is
  source/version-scoped, not a universal all-client compatibility certificate.
- **Rust:** raw OFF HTTP/SSE/process boundary; no Rust model/KV/parser owner and no
  claim of a supported Rust MTP application seam.

Application extras belong in a separate development/application environment, not
by mutating an admitted runtime package. See operations for Browser launch and the
[private-LAN application policy](private-lan.md) for opt-in network exposure.

## Completion and unsupported scope

Ordinary MTP's paired restore and Scheduler/P5 settlement boundaries are closed for
the supported path. **General-runtime production release remains incomplete**;
separate-machine direct-LAN acceptance is pending. Same-host LAN-IP requests do
not prove firewall reachability or packets crossing the LAN.

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
