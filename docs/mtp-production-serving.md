# Ordinary MTP serving: state ownership

This page owns **ordinary MTP execution, paired reuse and request retirement**.
[Admission](ordinary-api-boundary.md) owns request authority;
[production sessions](production-sessions.md) owns optional logical continuity;
[capabilities](capabilities.md) owns limits/defaults; [operations](operations.md)
owns installed deployment. Historical diagnostic/acceptance results below qualify
their recorded inputs, not later HEAD or every supported maximum workload.

## Development authority

`mtp-serving-v1` is the production-semantics boundary. The
[serving decision policy](production-serving-parity-closure.md#mtp-production-semantics-boundary)
requires current rationales for restrictions, established capability rather than
fixture bounds, qualified-core reuse and affected real-workload acceptance.
B1/B2 are connected for the supported ordinary path; the original SP1 BLOCK is
history. [Local default promotion](runtime-completeness.md) is now closed separately;
this is not general-runtime release completion, full R1 re-verification or
Internet/multi-user readiness.

## Local default

`mtp-serving-v1` is the normal local default. It exposes ordinary health,
models and Chat Completions, JSON/SSE, without compulsory session IDs, sequence
headers or certificates. Independent conversations, edits, branches and tool
round trips use compatible paired state or safe misses. Optional logical sessions
use the **same Scheduler**, not the singleton qualification execution path.

Use the installed MTP package with [current execution admission](runtime-admission.md).
Current startup instructions and environment paths live in operations; for example:

```sh
export DS41F_CHECKPOINT=/absolute/official/DeepSeek-V4.1-Flash
/path/to/mtp-env/bin/ds41f inspect --profile mtp-serving-v1
/path/to/mtp-env/bin/ds41f start --profile mtp-serving-v1
```

The old source inventory and local-seal architecture is historical qualification/
build evidence, not startup authority. No seal refresh is required. Installed
execution and exact semantic assets remain admitted independently of checkout
identity and unrelated host native-file hashes.

### Direct trusted private-LAN serving

Ordinary MTP permits explicit private-interface, wildcard or IPv6 binding while
loopback remains default. Clients use a reachable bound interface and the ordinary
`/v1` API; local loopback clients still work with wildcard bind. A single valid
IP/IPv6/DNS Host authority with optional port is admitted, not required to equal
the bind address. Missing/duplicate/malformed Host and Origin remain rejected.
No proxy, tunnel, VPN or extra daemon is required.

This is **trusted private LAN / single operator** only: no authentication, TLS,
adversarial multi-user isolation or Internet exposure claim. Wildcard bind is not
a security boundary; choose the private interface/firewall deliberately.
Singleton retains its literal IPv4-loopback/exact-Host contract; OFF is separate.
See [network operations](operations.md) and [application LAN policy](private-lan.md).
Separate-machine acceptance is **CLOSED / PASS**, with real Windows → Mac
transport and operator correlation; see the current disposition under
[direct-LAN evidence](#direct-lan-transport-fix-checks-2026-10-09).
Original same-host LAN-IP checks alone did not prove remote reachability.

## Ordinary function tools and OpenCode

Recipe owns declarations, content/role/tool-result transformation, rendering,
argument parsing, usage and response projection. Generic names, multiple
supported calls and recipe-representable declaration schemas are not reduced to
the singleton weather fixture. Text parts, reminders and empty assistant reasoning
follow authoritative conversion, not client-specific exceptions. Actual body/
encoded-capacity rules apply rather than singleton per-string limits or its
blanket special-token spelling ban. Bounded inline images use the existing multimodal preparation and P6 embedding
merge before the same P5 handoff; see [Vision/MTP integration](mtp-vision-integration.md).

**Tools execute in the client.** Send the completed assistant call message and
matching ordered tool results in the next ordinary conversation request.
Scheduler settlement and paired-prefix rules also apply to changed-result branches.
Partial speculative/tool deltas authorize neither effects nor reusable generated
state. The runtime is not a shell, Web, browser, plugin or MCP executor.

For OpenCode, presentation/provider-local key and wire model ID are distinct.
Configure the canonical `id` separately, use an OpenAI-compatible provider with
local `baseURL`, and restrict `enabled_providers` to avoid Internet fallback.
Client context/output settings must follow advertised capability, not obsolete
singleton limits. Explicit temperature zero is an operator choice, not the
ordinary default. The earlier edition retains the exact OpenCode 1.18.30 example;
that tested version/configuration is not universal client compatibility proof.

## Owners and production boundaries

| Owner | Responsibility |
| --- | --- |
| `ProductionScheduler`, subclass of pinned oMLX Scheduler | Waiting/running requests, IDs/UIDs, queue/abort admission, counters and sole executable BatchGenerator |
| `ProductionMTPBackend` | Shielded worker stepping and asynchronous request-scoped delivery/settlement broker |
| Official recipe HTTP integration | Conversion/encoding and JSON/SSE projection, not another DeepSeek grammar |
| DwarfStar/P6 and P5 | Committed append, same-live producer revocation and terminal transfer into Scheduler's batch |
| `OMLXMTPGenerationSession` | Subordinate physical/semantic adapter; no second queue, scheduling policy or reusable cache |
| `PairedCheckpointAuthority` | Complete paired payload integrity, immutable capture and compatible mutable acquisition |
| Upstream PagedCache | Hash lookup, refcounts, capacity, root-tail hints and LRU eviction |

The HTTP preparation semaphore protects conversion and releases before executable
admission; it is not the execution queue. Independent requests can queue, but do
not join a shared native batch. Model-specific hooks replace generic prefill,
stop parsing and backend-finish storage with qualified P5/recipe settlement.
No generic cache-corruption re-prefill repairs uncertain mutation.

Complete pairs are indivisible root tails; upstream block sizing follows the
context envelope, not token-grid state splitting/repack. Retention is bounded
by the current capability policy plus the reserved null block. Hash-drop/clear
callbacks release corresponding paired payloads. `total_tokens_cached` counts
actual retained frontiers, not block capacity times pair count. Target-only SSD
restoration is absent. Logical pins use the same upstream refcount owner.

The thin recipe HTTP boundary is retained. Importing the full pinned oMLX HTTP
server adds unrelated unadmitted dependencies and separate protocol machinery;
using its Scheduler/PagedCache does not require that extra server surface.

## Paired capture and restore

A reusable pair binds admitted execution/tokenizer/recipe identity, complete
encoded IDs and committed image identities through C, all forty target layers' seven physical slots and layout/
padding metadata, and all three committed DSpark rings at **the same C**.
Capacity-128 rings preserve absolute-modulo physical order, not chronological
rotation. Draft KV, proposal/verify stashes, future predictions, pending semantic
horizon and request RNG are never reusable checkpoints.

Capture freezes metadata and independent array/cache handles. Acquisition builds
fresh target/ring containers and materializes arrays before use. Functional MLX
values do not justify zero-copy memory budgeting: materialized snapshots can
allocate full copies. Mutable containers/rings never alias retained snapshots.
Capture/restore performs no numeric pack/unpack, whole-cache repack, model forward
or prompt replay. Wrapped-ring/native-array regression asserts snapshot stability.

Ordinary lookup considers checkpoints through **N-2**, leaving a genuine suffix
prefill token and P5's held-out terminal. Fresh prefill seals/captures a pending,
undiscoverable pair at N-2, then P6 continues to N-1; P5 consumes the terminal once.
This is neither recurrent N-to-N-1 trim nor a synthetic zero-length-append P5
certificate. Settled generated pairs can serve later compatible append requests.

Publication requires C+3 to fit the context envelope: genuine suffix, P5 terminal
and at least one reserved output. Near the ceiling only an eligible earlier
prompt pair is retained; unusable generated checkpoints are not published.
Missing/damaged/incompatible ordinary pairs invalidate into a shorter upstream hit
or safe miss. A target-only hash hit is not an MTP hit; edits/branches never rewind
live rings. Retained logical sessions instead require their exact leased pair
and fail before native start if it is lost—no historical-session prefill fallback.

## Lifecycle and publication

```text
prepared request -> Scheduler waiting
  -> compatible pair acquisition or fresh prefill
  -> DwarfStar/P6 append + pending immutable prompt capture
  -> P5 revocation / Scheduler's BatchGenerator + DSpark install
  -> qualified generation + consuming recipe guard
  -> canonical quiescence, prediction/native owner removal, semantic settlement
  -> immutable paired publication or burn -> request/UID retirement
  -> final outcome/delivery projection
```

Live SSE may deliver recipe events before final settlement. Backend finish,
partial text, SSE writes and socket success do not authorize paired publication.
JSON completion waits for canonical settlement and final response projection.
Cancellation shields in-flight mutation, then settles that exact request or burns
state; JSON disconnect probing and SSE cleanup target the same Scheduler ID.
The worker can progress independently of a stalled client.

Mutation/settlement/retirement failure burns aliases, clears reuse and fails the
engine closed, never silently replaying or retrying uncertain mutable state.
Eviction frees immutable snapshots, not active work or application history.
Logical count/frontier/outcome publishes only after native retirement. No implicit
HTTP retry identity, native persistence/crash recovery or exactly-once effects is
promised. GET-only logical reconciliation is a separate optional contract.

`/health` exposes bounded queue/activity/cache counters. Recipe usage reports the
**actual paired restored count**, not raw hash hits. Settlement logs distinguish
request ID, reused/canonical frontier, publication and cancellation. Logging and
telemetry are observers, never settlement or delivery authority.

## Long-context promotion evidence and retention

Long-context serving composes existing execution/state evidence with affected
production acceptance; it is not a new numerical qualification of every maximum
HTTP/agent workload. [Very-long OFF evidence](very-long-context-production-qualification.md)
owns its actual near-1M same-list prefill/continuation/resource runs. Retained MTP
core evidence and current paired/Scheduler rules have their own identities/scopes;
OFF evidence does not silently qualify arbitrary near-1M MTP HTTP requests.

The production prefill bridge uses original admitted layers for bulk work, then
ordinary P6 append produces the final bounded context rows for each DSpark tap.
Native absolute-offset initialization installs those rows once in physical ring
order. Passive pre-P5 handles can be replaced; live native rings are not rewound.
Reused short suffixes append directly to restored rings. No token re-execution,
full-prompt late-layer hidden tensor or new prefill executor is introduced.

Bounded reusable-state retention excludes active restored handles/model/prefill
arenas. Its historical snapshot-copy/headroom rationale and measured packed-state
size remain in the earlier edition; do not assume cost-free backing sharing or
interpret that host-specific estimate as multi-user admission. Acquisition clones
before releasing its upstream lease. Eviction/clear/burn use existing callbacks,
not another byte index, cache or replay mechanism.

## Full-hit TTFT diagnostic

`tools/diagnose_mtp_cache_hit_ttft.py` is the existing finite synthetic driver
through real HTTP ingress, recipe preparation, paired restore and native execution.
It is not injected cache/model state or an OpenCode session. Historical runs used
an admitted single-model maintenance window; **do not run another model beside a
serving process** or treat this page as an instruction to start a new campaign.

Traces expose preparation/conversion/render/encode, lookup/restore, actual cached
count/remaining suffix, suffix append, prompt capture, P5, first native decode,
first decode call/canonical output and total TTFT. TTFT starts at HTTP handler
entry including body read/preparation, excluding upstream header ingress and
preparation-semaphore wait. Restore includes validation/materialization and token
list creation. Suffix timing excludes prompt capture; a full hit still captures
that pending pair. P5 includes terminal bootstrap; native decode is the following
BatchGenerator step/synchronization. Decode-call timing includes ownership checks.

Do not sum preparation with nested recipe timings or native decode with its
parent decode-call timing. Other queue/load/resource/metadata costs remain in
total TTFT; do not assign the residual to attention. Encoding can render internally.
Instrumentation adds no extra MLX evaluation/synchronization. See
[performance interpretation](performance.md).

### Measured production result

The [original diagnostic receipt](../artifacts/mtp-cache-hit-ttft/diagnostic.json)
records seed/hit traces, medium recheck, identity and classification. The earlier
edition preserves the complete phase table, exact counts/times, excluded exit-137
attempts, regression counts and investigation decision.

The finite fixture found context-proportional encoding plus an ordinary model/
handoff floor. Its medium restore transient was non-monotonic with unassigned
cause, not established long-state-copy scaling. It did not reproduce the motivating
long delay universally or establish attention-dominated first decode. No performance
fix, before/after optimization or restore-sharing change was justified. No universal
TTFT bound, maximum-workload performance or default-promotion approval follows.

## Acceptance and regression

These subsections retain historical navigation anchors, not current test totals
or commands to rerun model campaigns. Full narratives and old paths/seal steps
are in the [pre-compilation edition](archive/editorial-baseline/mtp-production-serving.md.txt).
Current installation follows operations/admission, not historical resealing.

### Long-context production-path acceptance (2026-10-09)

[Long-serving acceptance](../artifacts/long-mtp-serving/acceptance.json) records
finite maintenance/generic-tool/OpenCode prompts, repeat/append/branch reuse,
disconnect settlement, eviction and zero replay/repack. Its output ceiling was
then narrower; later output promotion is separate. The collector's earlier
mistaken guaranteed-hit assertion and source-scoped regressions remain historical.
This was a production delta, not a near-1M HTTP endurance/context ladder.

### Generic function acceptance (2026-10-09)

The earlier edition retains real declaration/call/result/SSE and OpenCode read
acceptance, exact reused frontiers, client configuration and regression counts.
`tools/accept_generic_mtp_tools.py` is the retained affected-boundary driver;
weather qualification was not the generic schema. No all-client compatibility,
client-effect exactly-once or broader feature approval is inferred.

### Direct-LAN transport fix checks (2026-10-09)

The original same-host loopback/LAN-IP requests demonstrated ordinary Host/bind,
repeat reuse, disconnect/re-entry and unchanged singleton rejection. Recorded
OFF-native-pin failures and parent reproduction remain in the earlier edition.
At that stage separate-machine LAN acceptance was incomplete; no remote-client
run or host-firewall reachability follows from those original checks.
**Current disposition: CLOSED / PASS**, separately established by
[Windows-to-Mac trusted-LAN acceptance](remote-lan-windows-acceptance.md), not a
relabelling of the 2026-10-09 evidence.

### Existing ordinary execution acceptance

The earlier edition retains fresh/repeat/append/edit/branch, wrapped-ring,
JSON/SSE disconnect, independent queued requests and singleton regression results,
including obsolete historical receipt/fixture failures and import-origin caveats.
Their stated source/dependency scopes are unchanged. Already-closed ordinary B1/B2
work is not reopened; general-runtime release needs the current serving-policy
assessment rather than the old paragraph treating all broader controls/Web as
still unimplemented. Default/profile/promotion decisions remain separate.

## Ordinary output-capability promotion

`serving/capacity.py:ORDINARY_OUTPUT_CEILING` is the shared ordinary output authority
for admission, recipe capacity, Scheduler requests, models and inspect. Singleton
limits remain separate. Output allowance is a scalar termination budget, not
output-sized KV/DSpark/recipe preallocation. Target state grows with executed rows,
DSpark with committed bounded rings, token buffers/history with actual tokens.
Actual prompt + reserved output must fit the total envelope independently of the
output ceiling. Current numbers/defaults are in capabilities.

[Output acceptance](../artifacts/output-capability/acceptance.json),
[allocation/settlement sample](../artifacts/output-capability/allocation-settlement.json)
and [long-serving regression](../artifacts/output-capability/long-context-regression.json)
record finite production evidence: huge-budget short EOS, a 1024-token length
completion beyond the old bound, cancellation beyond that bound, paired reuse,
generic tool/result branches, overflow rejection and real OpenCode continuation.
The earlier edition preserves all exact memory samples, frontiers and regressions.
This is not maximum-output endurance, a memory suite, new runaway policy, numerical
core requalification or release promotion. No model/dependency/receipt is refreshed
by this compilation.
