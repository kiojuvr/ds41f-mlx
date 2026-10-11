# Production Serving Parity Closure

This page owns **production-serving decision policy and the general-runtime release
boundary**. It is not a new implementation authorization, qualification campaign
or list of still-unresolved SP1 contracts. [Architecture](architecture.md) owns
current structure; [capabilities](capabilities.md) owns support/limits;
[implementation plan](implementation-plan.md) owns present priorities.

## Current boundary and release interpretation

**ds41f is not production-release-complete as a general LLM runtime server.**
Execution correctness and a bounded application release are necessary assets,
not proof of all ordinary serving, operational and product requirements.

The original SP1 B1/B2 contracts are implemented for explicit `mtp-serving-v1`:
paired target/DSpark restoration and Scheduler/P5 canonical settlement,
publication and retirement. Independent requests, edits/branches, generic tools
and supported prefix reuse are no longer prohibited pending SP1. The original
BLOCK and proposed next investigation are historical, not present authorization
or blockers. Implementation and scoped acceptance belong to
[ordinary MTP serving](mtp-production-serving.md).

M50R–M55R evidence remains valid within its identities and envelopes. M55R approved
the bounded normal-local source-clone `mtp-singleton-v1` release; it did not certify
a general runtime server or later ordinary-serving HEAD. Standard-OFF remains the
default. Default-MTP review, broader release, stable-boundary R1 re-verification
and any future projection are separate decisions. The unused old `ds41f-runtime`
directory is neither current deployment nor accepted release history.

## MTP production-semantics boundary

**`mtp-serving-v1` is the production-semantics boundary for MTP development.**
Production-facing decisions start from established model/runtime capability and
real supported clients, not deliberately narrow qualification fixtures.

The singleton retains its own bounded session/sequence/certificate and weather
tool contract. Its context/output bounds, exact-extension-only workflow and local
transport assumptions must not automatically become ordinary-serving policy.
Internal correctness, canonical settlement and lifecycle safeguards remain assets;
ordinary clients do not acquire obligations to manage DSpark rings, speculative
queues, fences or canonical certificates.

Switching a supported execution strategy may change performance, resource use and
explicitly admitted capability. It must not unnecessarily change ordinary API
semantics. Optional diagnostic or dedicated-client interfaces may expose internal
machinery without making it compulsory for Chat Completions applications.

### Production restriction burden

A retained or new restriction needs a **current concrete rationale**: correctness,
state/lifecycle integrity, resource or hardware/runtime limits, security, protocol
compatibility, measured operational failure or explicit product scope.

“Previously it was so”, “only that range was qualified” and “smaller is more
cautious” are not sufficient rationales. This does not unconditionally publish
every maximum or inherit another profile's capability. Concrete constraints must
be stated and admitted predictably, without silent fallback.

### Capability, policy and defaults

Separate model/runtime capability, application/operator policy and defaults.
Maximum output is neither a default generation budget nor an allocation reservation.
Prefer capability → application policy → runaway/resource/security guards, rather
than shrinking capability as the first response to an operational concern.
Current values, single-flight/cache policies and unsupported features live in
[capabilities](capabilities.md); network and startup rules live in
[operations](operations.md), not in a duplicate policy-era limit table here.

### Ordinary API authority gate

Recipe owns supported request grammar, conversion, rendering, encoding, parsing
and response projection. ds41f owns concrete execution, resource and security
admission. Do not substitute historical fixture grammar, a successful-client list
or oMLX's separate template/tool parser for recipe authority. Schema acceptance
alone does not prove execution support for every parameter.

The [ordinary API boundary audit](ordinary-api-boundary.md) owns the admission
assessment and acceptance. Admission repair does not itself promote MTP or close
LAN, general-release, Internet-security or multi-user gates.

## Decision rule for future agents

For `mtp-serving-v1` and later production-facing work:

1. Start from established model/runtime capability, not historical qualification bounds.
2. Preserve a restriction only with a current concrete production rationale.
3. Separate capability, safety policy and operator defaults where possible.
4. Reuse authoritative core evidence; validate the changed production boundary.
5. Treat real supported client/workload failures as first-class production evidence.
6. Stop confidence-building test/milestone campaigns once the concrete blocker is closed.

The loop is concrete failure → responsible owner → smallest repair → affected
acceptance → return to real operation. Return to core validation only where
regression evidence warrants it. Progress means blockers removed, not tests,
probes, artifact volume, benchmark coverage or milestone count.

## Required serving domains

These are **closure criteria**, not assertions that every row is unimplemented.
Assess the actual supported local-runtime scope and current evidence per row.
Mature servers (vLLM, SGLang, TensorRT-LLM and Apple/MLX-specific oMLX) can expose
missing responsibilities; they are not normative architectures to copy.
Distributed serving, tensor parallelism, high-throughput continuous batching and
multi-tenant authentication are not automatically local-runtime requirements.

| Domain | Required decision/behavior within the supported scope | Current owner |
| --- | --- | --- |
| Public API | Health/models/ordinary Chat Completions, JSON/SSE and predictable errors; no compulsory MTP application protocol | [HTTP/API boundary](ordinary-api-boundary.md) |
| Request independence | New, edited, regenerated, shortened and branched conversations are valid requests, not exact-prefix corruption | [Ordinary serving](mtp-production-serving.md) |
| Prefix cache | Compatible committed prefix or safe miss; explicit identity, capacity, eviction and frontier observability | [State/paired cache](session-state.md) |
| Conversation/cache separation | Application history, request identity, cache pins and executable state are distinct | [Logical sessions](production-sessions.md) |
| Execution transparency | Profile selection does not silently change ordinary semantics or inherit unsupported capabilities | [Capabilities](capabilities.md) |
| Streaming/cancellation | Delivery ambiguity, native mutation, settlement, outcome retention and failure disposition are distinct | [Generation](generation.md) |
| Scheduling/concurrency | Distinguish one active generation, model, conversation, cache entry and connection; define the admitted envelope | [Architecture](architecture.md) |
| Sampling | Supported controls execute correctly; unsupported controls reject predictably, without execution-profile surprises | [Generation policy](generation-policy-frontends.md) |
| Tools | Recipe-authoritative generic tools and ID-bound continuations; client execution and effect scope explicit | [Ordinary tools](mtp-production-serving.md#ordinary-function-tools-and-opencode) |
| Resources | Model/KV/paired/allocator/Engram/artifact lifetimes, ceilings, cleanup and eviction are intentional | [Runtime strategy](runtime-strategy.md) |
| Configuration | Documented setup/defaults, operator checkpoint/cache/host controls, diagnostics and clear unsupported choices | [Operations](operations.md) |
| Observability | Readiness/backend, prompt/generated/reused counts, latency, resource failures and rejection reasons are explainable | [Serving account](mtp-production-serving.md) |
| Failure isolation | Invalid/unsupported request, miss, cancellation, transport ambiguity, poisoned state and fatal runtime failure are distinguished | [Generation/state](generation.md) |

A miss or edited conversation is not architectural corruption. SSE bytes, backend
finish and cache offset equality do not establish canonical reusable state.
For MTP a reusable checkpoint binds complete target and committed DSpark physical
state at the same frontier; speculative rollback is not conversation rewind.
One active generation does not imply one application conversation for the process.
Optional retained sessions and ordinary independent requests have different reuse
and retry contracts; do not transfer guarantees between them silently.

### Closure assessment

Before a general-serving release decision, maintain one concise assessment with:

```text
Capability | Reference expectation | Current behavior | Classification
Production blocker? | Owner/path | Closure evidence
```

Classify as REQUIRED, REQUIRED_BUT_PARTIAL, DEFERRED_BY_SCOPE or NOT_APPLICABLE.
Record only distinctions that affect the real supported scope; do not create a
large compatibility catalog or infer a blocker merely from absent broad testing.
Separate-machine LAN acceptance remains pending; same-host LAN-IP checks are not
remote-client evidence. Internet/multi-user readiness is not established.
Broader release closure needs its own explicit assessment, not a singleton PASS
or B1/B2 closure extrapolated to every row.

## Architecture-change guard

Preserve the qualified execution core, same-live P5 seam, official recipe semantics,
canonical/effect settlement and strict dependency/admission identity. The
[MTP doctrine](mtp-architecture-reset.md) owns physical conformance requirements.

Before structural replacement:

1. name the exact serving blocker;
2. explain why an existing seam cannot satisfy it;
3. choose the smallest viable architectural change;
4. preserve direct correctness/performance comparison;
5. stop if the rewrite merely promises to recover established behavior later.

“Cleaner architecture” is insufficient. No prompt replay, whole-cache repack,
second executable owner or weakened settlement may bridge an ownership gap.
A coherent new boundary may warrant a milestone and explicit PASS/BLOCK; minor
issues do not each require an SP-series milestone. This page authorizes neither
new implementation nor a broad confidence campaign by itself.

## Historical SP1 assessment

The [pre-compilation source edition](archive/editorial-baseline/production-serving-parity-closure.md.txt)
retains the complete original rationale, SP1 baseline
`55d1c0ae4d236313d5ab888a988ffcbc82cb969e`, dependency hashes/source anchors,
B1/B2 findings, conditional responsibility map, M52R guarantee analysis and proposed
next investigation. Its BLOCK and “no implementation/SP2” instructions applied
to that starting assessment, not to the now-connected ordinary production path.
Historical evidence is unchanged; it must not be rewritten as later qualification.
Current normative definitions are in the owners above, not in that source edition.
