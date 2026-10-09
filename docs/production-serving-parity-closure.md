# Production Serving Parity Closure

## Status

This document defines the production-serving completion boundary for ds41f.

It supersedes any interpretation that execution correctness, bounded application qualification, or a specialized local MTP protocol alone is sufficient to declare the ds41f runtime server production-ready.

Historical milestone evidence remains valid within its stated scope.

In particular, M50R–M55R remain authoritative evidence for the qualified MTP execution topology, canonical/lifecycle semantics, bounded application semantics, dependency identity, admission and normal-local operation.

M55R's release decision must therefore be interpreted as:

> the bounded `mtp-singleton-v1` execution/application profile was release-qualified within its explicitly constrained protocol and environment.

It is **not sufficient evidence that ds41f is a production-complete general LLM runtime server**.

Production serving release remains blocked until the serving requirements defined here are closed.

**Implementation update:** B1/B2 are now connected in production code for the
explicit `mtp-serving-v1` ordinary Chat Completions path. Scheduler owns execution
and settlement; immutable paired target/DSpark checkpoints use pinned PagedCache
lookup/capacity/eviction. Fresh/reuse/append/edit/branch and disconnect acceptance
ran on the admitted host, with the existing singleton contract preserved. See
[ordinary MTP serving ownership and operation](mtp-production-serving.md).
The SP1 BLOCK below is retained as the historical starting assessment, not the
current B1/B2 implementation status. Broader general-runtime release requirements
remain open; no new soak/qualification campaign is required for the already
approved bounded execution core.

---

# MTP production-semantics boundary

**`mtp-serving-v1` is the production-semantics boundary for MTP development.**
Before it, bounded qualification primarily asked: “Can correctness be proved
under these deliberately limited conditions?” From this boundary onward,
production-facing work primarily asks: “Are we withholding established
model/runtime capability expected by real users and clients without a concrete
production reason?” This is development policy, not a new qualification milestone.

```text
historical qualification / bounded evidence
  mtp-singleton-v1; specialized session/fence/certificate contracts
  deliberately narrow capability envelopes
                         ↓
        production-semantics boundary: mtp-serving-v1
                         ↓
ordinary production-facing serving
  ordinary OpenAI-compatible API; real applications / OpenCode; generic tools
  model/runtime capability; production cache/lifecycle/resource policy
```

## Historical evidence versus current production authority

Historical evidence states what was proved at that time within that scope.
Current production authority states what the production path should provide now.
Preserve both without confusing them: `mtp-singleton-v1` retains its 8192-total /
768-output, weather, certificate and effect qualification contract unchanged.
Those bounds must not flow back into `mtp-serving-v1` policy merely because they
already exist. Internal correctness and lifecycle invariants remain assets; their
specialized application protocol is not an ordinary client obligation.

## Production restriction burden

The side retaining or adding a production capability restriction bears the burden
of documenting a **current concrete rationale**: correctness, state/lifecycle
integrity, an actual resource bound, hardware/runtime incompatibility, a security
boundary, protocol incompatibility, measured operational failure, or an explicit
product scope decision.

“Previously it was so,” “only that range was qualified,” and “smaller is more
cautious” are not sufficient rationales. Historical 8192 context, 768 output,
weather-only tools, loopback-only transport, special session IDs and
sequence/fence/certificate requirements do not justify production restrictions
by themselves. This is not an unconditional rule to publish every maximum:
concrete implementation reasons may require a capability limit and must be stated.

A few resolved leaks illustrate the distinction, not a new milestone history:

| Qualification-era constraint | Current ordinary production resolution |
| --- | --- |
| Weather-only declarations | Generic recipe-authoritative function tools |
| 8192 total context | 1,048,576 total context (subject to checkpoint range) |
| 768 output | 393,216 output capability |
| Loopback-only transport | Explicit trusted private-LAN serving |
| Specialized MTP application protocol | Ordinary Chat Completions interface |

See [ordinary MTP serving](mtp-production-serving.md) for the supported envelope
and evidence, including the still-pending separate-machine LAN acceptance.

## Capability, policy and defaults

Keep model/runtime capability, operational safety policy and operator defaults
separate where possible. The established ordinary envelope is **1,048,576 total
context / 393,216 maximum output**; the ordinary default generation budget is
**128**, not a capability ceiling. Thought/runaway loops, agent loops, timeouts,
operator cost and network security are separate policy responsibilities.

Prefer capability → application/operator policy → runaway/safety/resource guards,
not artificially shrinking model capability as the first response. When a concrete
implementation reason requires limiting capability itself, document that reason.
Production cache retention, single-flight execution and trusted-network scope
remain intentional policies, not automatic inheritance of qualification bounds.

## Production evidence and validation loop

Failures from real **supported** clients/workloads are first-class development
inputs, not merely gaps in synthetic qualification coverage: normal OpenCode
prompts rejected, generic declarations denied, real edits truncated, ordinary
client semantics changed, long-conversation reuse failing, cancellation corrupting
state, or operational memory retention problems.

The basic loop is: real workload exposes a concrete failure → identify the
responsible production boundary → repair it → small affected acceptance → return
to real operation. Count blockers removed, not scripts, probes, artifacts or
milestones created; do not prolong confidence-building after closure.

Compose **already-qualified core + new production boundary**. Reuse authoritative
execution-core evidence and validate the changed boundary's delta instead of
re-proving the core for every integration. Return to core validation only to the
extent concrete evidence indicates a regression. Task B/C applied this through
existing long-context evidence reuse, no numerical requalification, no staged
64K/128K/256K campaign, and output promotion without 384K endurance generation.

## Decision rule for future agents

For `mtp-serving-v1` and later production-facing work:

1. Start from established model/runtime capability, not historical qualification bounds.
2. Preserve a restriction only with a current concrete production rationale.
3. Separate capability, safety policy and operator defaults where possible.
4. Reuse authoritative qualification evidence; validate the changed production boundary.
5. Treat real supported client/workload failures as first-class production evidence.
6. Stop confidence-building test/milestone campaigns once the concrete blocker is closed.

## Production semantics is not release completion

This boundary starts evaluation of subsequent feature, API, resource and lifecycle
decisions as real-operation decisions. It does **not** mean `ds41f-runtime`
promotion, general runtime release, or all product features are complete, nor
Internet/multi-user production readiness. Release packaging, full R1 regression,
promotion, Internet security and multi-user serving remain separately scoped.
B1/B2 closure for the supported ordinary path does not close those broader gates.

---

# Why this closure exists

Development to date concentrated successfully on difficult model-runtime problems:

- DeepSeek-V4.1-Flash execution correctness
- DwarfStar/MLX prefill
- oMLX MTP execution topology
- canonical frontier and state ownership
- rollback/commit correctness
- request/effect identity
- cancellation and recovery
- native/dependency reproducibility
- long-context and resource qualification

Those are necessary.

They are not sufficient.

A runtime server must also provide the ordinary serving semantics expected by applications using modern local LLM runtimes.

The historical starting inspection exposed several examples (not a current
unresolved-blocker list; see the implementation update above):

- MTP changes the public HTTP contract instead of remaining an internal execution strategy.
- ordinary `/v1/chat/completions` clients cannot use the qualified MTP path.
- ds41f-specific session IDs, request sequence headers, fences and certificates leak into application responsibility.
- stateful continuation requires the entire existing canonical token history to remain an exact prefix.
- ordinary prompt edits, regeneration, branching and prefix reuse are not handled by a general cache manager.
- application/session ownership and executable KV ownership are too tightly coupled.
- development-machine paths remain visible as operational defaults.
- a user must inspect source or milestone documents to understand normal startup.

These are not isolated missing features.

They demonstrate that **production serving was not previously treated as a first-class architecture and release boundary**.

This document adds that missing boundary.

---

# Core principle

The public server contract must be independent of the internal model execution strategy.

Conceptually:

```text
Applications
    |
    |  ordinary OpenAI-compatible API
    v
Production serving layer
    |
    |-- request semantics
    |-- cache management
    |-- lifecycle
    |-- streaming/cancellation
    |-- tool protocol
    |-- sampling
    |-- resource admission
    |-- configuration
    |-- observability
    |
    v
Execution interface
    |
    |-- standard-OFF
    |-- MTP
    |-- future execution strategies
    v
DeepSeek-V4.1-Flash model execution
```

MTP is an execution strategy.

It must not require ordinary applications to understand:

- MTP session identities
- request sequences
- fences
- canonical certificates
- speculative queues
- rollback state
- DSpark rings

unless an explicitly optional diagnostic/debug interface exposes them.

Switching between qualified execution strategies may change performance and resource characteristics.

It must not unnecessarily change ordinary application semantics.

---

# Reference runtimes

Production-serving requirements must no longer be discovered only from ds41f's existing implementation.

Use mature runtime servers as requirements references, including where relevant:

- vLLM
- SGLang
- TensorRT-LLM
- oMLX for Apple/MLX-specific execution behavior

These projects are reference points for identifying expected serving primitives.

They are **not normative implementations** and ds41f must not copy architecture merely for parity.

For every reference capability, classify it according to ds41f's actual local-runtime scope:

- REQUIRED
- REQUIRED_BUT_PARTIAL
- DEFERRED_BY_SCOPE
- NOT_APPLICABLE

"Parity" does not mean implementing every cloud-scale or multi-GPU capability.

ds41f is primarily a high-quality local runtime.

Features such as distributed serving, tensor parallelism, multi-tenant authentication or high-throughput continuous batching are not automatically required.

The purpose of comparison is to prevent ordinary runtime-server responsibilities from being overlooked again.

---

# Production-serving domains

## 1. Public API contract

A supported execution backend must be accessible through the normal production API.

At minimum the production text path requires:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Applications must not require ds41f-specific session or MTP headers for ordinary use.

OpenAI-compatible request, response, streaming and error conventions should be preserved within the explicitly supported subset.

Additional APIs such as `/v1/responses` may be added when justified, but are not prerequisites for the first serving closure unless required by supported applications.

---

## 2. Request independence

An ordinary Chat Completions request contains the conversation it wants evaluated.

The server must not assume that every new request is an append-only continuation of the immediately preceding request.

Normal workloads include:

- new conversations
- regeneration
- edited last messages
- edited earlier messages
- shortened history
- context trimming
- conversation branches
- repeated system prompts
- tool-result differences
- requests from another application

These cases must have defined semantics.

A prefix mismatch must not automatically mean protocol corruption.

---

## 3. KV / prefix cache management

Executable KV state must not be conceptually identical to an application session.

The serving layer requires an explicit cache-management model capable of reasoning about reusable prefixes.

The minimum production behavior is:

```text
incoming encoded prompt
        |
        v
find reusable committed prefix
        |
        +-- full reusable prefix -> append suffix
        |
        +-- partial reusable prefix -> reuse valid prefix,
        |                             compute changed suffix
        |
        +-- no reusable prefix -> fresh prefill
```

The existing append-only continuation path may remain an optimized fast path.

It must not be the only valid cache-reuse path.

The design must explicitly cover:

- longest/common reusable prefix determination
- cache identity
- tokenizer/template/model identity
- tool/schema-sensitive prompt identity where applicable
- immutable committed cache boundaries
- MTP auxiliary-state consistency
- safe reuse after branching
- capacity accounting
- eviction
- retirement
- cache hit/miss observability

For MTP, target KV, DSpark committed context and any other execution state required for continuation must describe the same reusable frontier.

Do not assume that speculative rollback is automatically a general conversation-prefix rewind mechanism.

If safe arbitrary rewind is not available, use an architecture such as reusable immutable prefix blocks/checkpoints rather than introducing unsafe mutation.

---

## 4. Conversation and cache decoupling

Application conversation identity and executable cache identity are separate concepts.

A conversation may reuse:

- all of an existing cache
- part of an existing cache
- a prefix produced by another request
- no existing cache

Likewise, cached prefixes may outlive the request which created them, subject to the local cache lifetime/resource policy.

Do not force ordinary applications to adopt ds41f session identity merely to obtain prefix reuse.

---

## 5. Execution-profile transparency

`standard-off`, `mtp-singleton-v1`, and future execution implementations are internal execution profiles.

The long-term public model is:

```text
ds41f server
    |
    +-- qualified execution backend selected internally/configurably
```

rather than separate application protocols for each execution backend.

An execution profile may impose legitimate resource limits.

It must not leak implementation machinery into normal application code without necessity.

---

## 6. Streaming, cancellation and disconnects

Production behavior must be defined for:

- non-streaming completion
- SSE streaming
- client disconnect
- request cancellation
- completed generation whose transport is lost
- retry after ambiguous delivery
- server-side failure before admission
- server-side failure after execution ownership transfer

Existing M52R/MTP settlement and exact-retry work should be reused.

Do not weaken correctness merely to imitate a stateless HTTP handler.

Instead, hide the machinery behind the production serving boundary.

Partial SSE text must not become canonical model history merely because bytes reached a socket.

---

## 7. Scheduling and concurrency scope

Serving architecture must explicitly define concurrency rather than accidentally inherit it from an execution experiment.

For the initial local production release it is acceptable to support:

- one active generation at a time

if that is the intentional supported envelope.

However, distinguish:

- one active generation
- one model instance
- one live application conversation
- one reusable cache entry
- one HTTP connection

These are not equivalent constraints.

`singleton` execution must not accidentally imply that the server can understand only one conversation over its lifetime.

Multi-request cache reuse and ordinary conversation switching are required even if execution remains single-flight.

High-throughput continuous batching may remain deferred.

---

## 8. Sampling and generation parameters

The server must explicitly define the supported OpenAI-compatible generation subset.

Ordinary parameters must either:

- work correctly, or
- be rejected predictably as unsupported.

Execution-profile implementation details must not silently change sampling semantics.

Greedy-only MTP may remain a temporary bounded capability only if it is explicitly classified as a remaining production-serving limitation.

It must not be mistaken for full API completion.

---

## 9. Tool calling

Tool calling is an application-level serving capability.

A production OpenAI-compatible tool API should not require applications to know MTP consuming-horizon or certificate mechanics.

Existing M52R tool/effect correctness should remain authoritative internally.

`mtp-serving-v1` now supports ordinary generic function declarations, model-generated
calls and ID-bound tool result continuations, including JSON/SSE and paired prefix
reuse across round trips. DeepSeek recipe owns conversion, rendering, parsing and
semantic projection; execution of tools belongs to the client. See
[implementation and real acceptance](mtp-production-serving.md#ordinary-function-tools-and-opencode).

The weather-only schema remains unchanged qualification evidence for
`mtp-singleton-v1`, not a restriction of the general ordinary function API.
The historical singleton 8192-total / 768-output bound is not ordinary production
policy; ordinary serving now exposes 1,048,576 total / 393,216 output capability.

Web search, PDF, Vision and other ds41f application capabilities remain independently scoped and must not be accidentally bundled into foundational serving work.

---

## 10. Memory and resource management

Serving ownership must include explicit policies for:

- model residency
- executable KV
- reusable prefix cache
- allocator cache
- auxiliary MTP state
- Engram/SSD state where applicable
- artifact persistence
- memory ceilings
- eviction
- cleanup after cancellation/failure

Resource correctness is not established merely because one long-lived append-only session remains coherent.

---

## 11. Operational configuration

Normal use must not require reading source code.

Supported configuration must have intentional defaults.

Development-machine-specific paths must not be normal release defaults.

The production user surface must clearly document:

- checkpoint
- cache location
- host/port
- execution backend/profile if user-selectable
- context/output limits
- resource controls
- diagnostics
- supported/unsupported features

Unsupported configuration should fail clearly rather than silently change execution mode.

---

## 12. Observability

A production runtime should expose enough information to explain behavior without qualification instrumentation.

At minimum consider:

- model readiness
- active execution backend
- prompt tokens
- generated tokens
- reused/prefilled prompt tokens
- cache hit/reuse frontier
- request latency
- decode throughput where practical
- rejection reason
- resource-limit failures

Internal qualification traces and production observability are separate concerns.

---

## 13. Failure isolation

Distinguish:

- invalid client request
- unsupported capability
- cache miss
- cache incompatibility
- request cancellation
- recoverable transport ambiguity
- request-local execution failure
- poisoned execution state
- fatal model/runtime failure

A normal cache miss or edited conversation must never be treated as an architectural corruption merely because the previous implementation assumed append-only continuation.

---

# Serving parity matrix

Before declaring production-serving completion, maintain one concise matrix covering the domains above.

Each row must contain:

```text
Capability
Reference expectation
Current ds41f behavior
Classification
Production blocker?
Implementation owner/path
Closure evidence
```

The matrix is a decision tool, not an evidence-generation project.

Do not create a large compatibility catalog.

Only record distinctions that affect ds41f's actual supported local-runtime scope.

---

# Anti-rewrite rule

The serving-parity effort must not become another full runtime rewrite.

The qualified execution cores are assets.

In particular:

- preserve M50R candidate topology constraints
- preserve M51R canonical/lifecycle ownership
- preserve M52R application settlement/effect correctness
- preserve M53R reproducible dependency/admission identity
- preserve qualified DwarfStar/MLX prefill
- preserve qualified standard-OFF production implementation where useful

Change execution architecture only when a concrete serving requirement cannot be satisfied above it.

Before any destructive or large structural replacement:

1. identify the exact serving blocker,
2. show why the existing seam cannot satisfy it,
3. identify the smallest viable architectural change,
4. preserve a direct correctness/performance comparison,
5. stop if the proposed rewrite merely promises to recover established behavior later.

"Cleaner architecture" is not sufficient justification.

---

# Anti-meandering rule

Progress is measured by production-serving blockers removed.

It is not measured by:

- number of tests
- number of milestone documents
- number of probes
- qualification artifact volume
- compatibility tables
- new infrastructure
- broader benchmark coverage

New validation work must correspond to a concrete serving boundary being introduced or changed.

Do not run broad confidence campaigns after a blocker is already closed.

---

# Release interpretation

Until this document's required serving scope is closed:

**ds41f is not production-release-complete as a general LLM runtime server.**

Existing M55R evidence remains valid as bounded MTP execution/application qualification.

Do not delete or falsify historical release evidence.

Instead, current authoritative documentation must state that the broader production-serving release decision is reopened because the previous release scope omitted ordinary runtime-server requirements.

`ds41f-runtime` promotion remains deferred.

---

# New milestone series

Production-serving closure uses a new milestone series:

```text
SP1, SP2, ...
```

The new name is intentional.

It separates production-serving completion from the historical M-series execution/correctness milestones and prevents accidental continuation of the old objective function.

Do not create one SP milestone per minor issue.

Create a new milestone only when a coherent production-serving boundary needs implementation and a clear PASS/BLOCK decision.

---

# SP1 — Serving/API Authority Reconstruction (historical starting assessment)

**Historical scope of the remainder of this document:** the SP1 assessment,
B1/B2 unresolved contracts, conditional matrix and proposed next work below
record the starting state. They are preserved, not current blockers or an ongoing
ban on serving implementation. B1/B2 were subsequently implemented/closed for
`mtp-serving-v1`; the current status and production decision rule above govern.
Broader release gates remain open.

The milestone boundary is intentionally revised. SP1 is an authority/restoration
**decision**, not a serving implementation milestone. The former foundation
implementation exit condition is superseded; all broader serving domains and
release criteria above remain unchanged.

**SP1 PASS does not close production-serving functionality or release blockers. It establishes the architecture and ownership basis for subsequent implementation.**

PASS requires responsibility/path-level owners and concrete execution-state
transitions sufficient to define a bounded next implementation task. In
particular, target state, DSpark state and committed prompt frontiers must be
coherent across cache publication/restore and P5 ownership transfer. A small
adapter or separately successful subsystem runs do not establish this.
Materially unresolved transformation, restore or ownership contracts require
BLOCK, with the minimal additional evidence specified.

SP1 neither restores serving nor authorizes SP2. No new API adapter, cache
manager, production abstraction, execution-core restructuring, default-profile
change, release/promotion work or `ds41f-runtime` change belongs here.

---

# SP1 decision — BLOCK (historical assessment)

Assessment baseline: `55d1c0ae4d236313d5ab888a988ffcbc82cb969e`.
Documentation-only assessment; no production code changed, diagnostic probe
created, or fresh model execution performed. Source inspection resolves several
owners, but not the two state contracts below. Stop at this decision.

## Examined identities and evidence scope

- **oMLX 0.7.0:** exact admitted `third_party/mtp/omlx-source.tar.gz`, SHA256
  `26cc224a5fa77d8576589764a56ba8053ac31fe9f4904d8ba60305ed841f33e6`;
  upstream base `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, attributed candidate
  `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`. The archive, not upstream HEAD
  or the base commit alone, is the serving/execution reference. Attribution and
  patch identity are in `third_party/mtp/sources.json`; normal-local installed
  file/dependency admission is in `third_party/mtp/normal-local.json`.
- **DeepSeek recipe 0.1.1:** base `8cadfede7063c896b944e7bae05daa3549ae97ea`,
  attributed candidate `29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`; base archive
  SHA256 `5b71ea6837ad3eb54a07da2b7ba1dd0a4ce658b585cdd2db24f4b83d9f879c22`.
  Examined admitted M52R source archive `recipe-m52r-source.tar.gz`, SHA256
  `1f4d42fd700b876ef36dd71a95db40cf88c3fa948dedc603e0676cd16e1db66b`, including
  `m52r-recipe-consuming-eof.patch` SHA256
  `560a791ff684a60ad366c3f46b961cb004f7819cfab95b4400bd0aa0d43b3794`.
  These are official semantics with the recorded consuming-state integration
  delta, not a claim that unmodified upstream contains every qualification hook.
  Archive hashes were recomputed during SP1 and match admission records.
- **Retained direct evidence:** [M50R](milestone-50r-omlx-candidate-baseline.md),
  [M51R](milestone-51r-canonical-lifecycle-connection.md) and
  [M52R](milestone-52r-application-integration.md) connect recipe semantics,
  DwarfStar/P5 and native oMLX BatchGenerator/DSpark in the specialized singleton.
  They do **not** connect the generic oMLX Scheduler/BlockAwarePrefixCache.
  [Standard-OFF qualification](standard-off-200k-production-qualification.md)
  connects first-party execution/P5, cancellation and exact idle persistence;
  [multimodal qualification](multimodal-production-qualification.md) covers its
  stated image envelope. Neither establishes request-crossing oMLX cache reuse.
- **Reference evidence:** admitted oMLX serving/scheduler/cache source and recipe
  source examined independently. Older ordinary oMLX runs and the historical
  [P5 smoke](p5-live-cache-handoff.md) are only references for this proposed
  composition (the latter used a different dependency identity). No new upstream
  HTTP demonstration is claimed. No observed generic-serving + recipe + qualified
  ds41f prefix-restore composition exists in this assessment.

A fresh ordinary HTTP completion would not answer either blocker; no broad
feature, performance or model-loading campaign was justified.

## Current standard-OFF path: valid recipe assets, separate runtime owners

`serving/server.py:prepare_request` selects `ChatCompletionRequest`,
`ResponsesRequest` or `MessagesRequest` from `PROTOCOL_TYPES`, constructs the
native request from JSON, calls `convert(ConversionOptions)`, then
`DeepseekV41Encoding.with_tokenizer(...).render_conversation/encode`.
The tokenizer is the official V41 `static/tokenizers/v41/tokenizer.json`;
normal OFF loading uses the admitted `resource_admission.load_protocol_tokenizer`.
Recipe conversion owns message/tool transformation, thinking/reasoning effort,
response-format and stop parsing options. It is already a substantial official
API path, not a failed compatibility layer.

Recipe sources: `deepseek-recipe/src/protocol/openai/chat_completion/request/`,
`openai/responses/request/`, `anthropic/messages/request/`, `request/options.rs`,
`stream/{processor,semantic,state_machine}.rs`, and
`deepseek-recipe-encoding/src/v4/{mod,dsv41}.rs`. V41 rendering normalizes
messages, inserts tool/response-format system instructions, applies official
special tokens and the assistant thinking/required-tool prefix; encoding tokenizes
that rendered prompt. It is not oMLX's independent chat-template route.
Chat conversion validates tools/names/parameters, maps named choice to required
selection, and sets reasoning/tool/JSON parsing options. Responses additionally
handles typed input items and custom tool names; Messages transforms system and
content blocks. `ConversionOptions` defaults thinking on, Responses web-search
ignore and Messages web-search reject; MTP explicitly selects thinking off.
These defaults do not qualify web search or every protocol feature for ds41f.

`response_body` constructs the official request's chunk generator, forwards
include-usage/custom-tool-name settings, then uses `StreamProcessor` with the
**converted parsing options** and tokenizer. Token chunks, backend finish and
iterator EOF pass through the official processor; response objects accumulate
its official chunks for JSON, or ds41f frames their JSON as SSE. Recipe owns
reasoning/tool grammar, stop stashing, usage and finish projection. In
`processor.rs:canonical_finish_reason`, matched stop takes precedence; backend
Stop with tool calls becomes ToolCalls, Length stays Length, and missing backend
finish becomes EndOfStream. EOF must not be replaced by an invented successful
tool terminal. M52R consuming forks distinguish call certification from complete
response settlement. ds41f's adapters do not need a parallel grammar.

Ordinary OFF `DeepSeekRecipeRuntimeBackend.infer` takes the single-flight lock,
loads the first-party model, prefills `ids[:-1]` through DENSE_P0_P7, hands the
same live cache to `TargetGenerationSession` via P5, and consumes the held-out
terminal once. It returns `InferenceChunk.ready/token/finish`; ordinary requests
prefill fresh and report zero prefix hits. Model aliases, resolved capacity,
qualified image expansion, and execution capability checks are ds41f admission,
not protocol conversion. Sampling currently maps temperature/top_p to MLX.
Worker cancellation is shielded until mutations finish, then stop/close retires
state; iterator/SSE closure is shielded too.

Optional OFF `/v1/sessions` owns `M11RecipeToolSession`/M8 continuation, exact
encoded-history extension, streaming/recovery records and idle persist/restore.
That is separate from the ordinary stateless endpoint, but still couples reusable
execution state to a selected application session. Exact idle restoration is not
a generic partial-prefix cache. Preserve recipe preparation/projection and
qualified resource/cancellation checks as long-term thin assets; generic HTTP,
queueing and session-as-cache policy are not intrinsic recipe responsibilities.

## Current MTP path and why its specialized components exist

`mtp-singleton-v1` is strict dependency/admission and bounded capability identity,
not the intended ordinary API contract. `LocalMTPBackend` in `mtp_public.py`
subclasses `InternalMTPQualificationBackend`; despite the latter's historical
qualification-only docstring, it is reachable in the current public profile.
It rejects ordinary `infer`; `LocalBoundary` rejects ordinary Chat Completions
and admits session-scoped Chat Completions instead.

- `LocalBoundary`: trusted loopback authority/origin checks, one bounded
  preparation slot, body/read/send limits. Original requirement: bounded ingress
  and transport stalls without revoking native mutation ownership. Generic
  transport/admission policy, not DeepSeek grammar.
- `/v1/sessions`, singleton issued IDs and exact-extension admission: original
  requirement: one cache/ring lifetime with no unsupported rewind, stale reuse or
  hidden replay. Execution correctness warrants internal lifetime identity, but
  not one public conversation for the entire server lifetime.
- `X-DS41F-Request-Sequence`/request fences: original requirement: distinguish
  duplicate, conflicting, active and expired requests; freeze exact body/outcome
  before transport loss. Transport/retrieval identity, not prompt/cache identity.
- Outcome/certificate projection: original requirement: prove recipe-representable
  consumed history and supported tool completion before retry/re-entry or effect
  permission. Internal semantic/settlement safeguards are necessary; ordinary
  clients need not carry certificates or reconstruct raw canonical tokens.
- `LocalMTPClient`/`InternalLocalClient`: original requirement: own transcript,
  frozen request bytes, reconcile ambiguous delivery and reserve effects once.
  Application orchestration and optional dedicated-client protocol; bounded
  weather schema/ledger and qualification controllers are not general tool API
  semantics. Do not promote their scope into OpenAI compatibility.

Actual execution: `internal_mtp._start` appends only new prompt suffix with
`DeferredPrefillAppend`, taps same-forward reduced layer inputs 37/38/39, appends
native committed DSpark context, attaches it to `LivePrefillResult`, and P5
transfers target state to `OMLXMTPGenerationSession`. Official processor plus
`RecipeSemanticGuard` binds consuming semantics to the native horizon. Native
BatchGenerator owns proposal/verify/rollback/commit; `_settle` quiesces it,
retains target cache/rings/canonical IDs, retires predictions and the owner, then
freezes response/events/certificate/fence outcome. Persistence/restore is explicitly
unsupported for this profile. `rec.canonical`, cache, rings, previous envelope
and next sequence are all retained in one live session: correct for its qualified
contract, unnecessarily restrictive for ordinary independent requests.

## Pinned generic oMLX serving and cache contracts

Archive-relative source anchors:
`omlx/server.py`, `engine/batched.py`, `engine_core.py`, `request.py`,
`scheduler.py`, `cache/{prefix_cache,paged_cache,type_handlers,deepseek_v41_delta}.py`,
`patches/mlx_lm_mtp/{batch_generator,prompt_priming,deepseek_v4_dspark}.py`.

HTTP ingress obtains an engine/model lease; disconnect fan-out is request-scoped
(`_with_request_disconnect_abort`). BatchedEngine currently applies its own
chat template/message/tool plumbing, makes `SamplingParams`, and delegates to
EngineCore's asynchronous request/stream lifecycle. Scheduler separates waiting
and running requests, request ID and BatchGenerator UID. `Request` separately
stores full prompt IDs, remaining IDs, cached count, prompt cache, block table,
output IDs, sampling and cache-store policy. This is useful generic machinery,
not permission to replace recipe conversion or output parsing with oMLX's
independent template/tool parsers.

`_prepare_prefix_cache_for_request` fetches chain-hashed prefixes, acquires shared
block references, reconstructs cache objects, applies optional model restore,
and computes the suffix from the **actual restored** count. Invalid/missing
blocks become misses or shorter hits. Exact hits need N-1 state for the final
input: non-sliceable/stateful cases fall back to prefill rather than pretending
that generic trim can undo recurrence. Capacity/eviction belongs to PagedCache
and tiered hot/SSD machinery, not conversation deletion. Model name and optional
media keys participate in hashes; layout signatures/metadata guard restores.
Those keys do not by themselves prove strict checkpoint/tokenizer/template/
execution-profile identity. ds41f's admitted identity must namespace reuse;
recipe-generated IDs include tool/schema rendering, but numeric token equality
across tokenizer/model identities is not sufficient.

The cache handles non-KV state, not just attention tensors. V41 has specialized
storage-only deltas: `compact_state` validates seven slots, absolute offsets,
compression ratio and packed KV/index dimensions; `restore_chain` validates
contiguous absolute ranges and retains the terminal window/compressor/history
state while concatenating packed rows, then creates `DeepseekV41Cache`. This
is storage reconstruction, not proven same-live P5 transfer. ArraysCache and
other recurrent families use boundary/exact-state mechanisms rather than unsafe
arbitrary slicing. Existence of those mechanisms does not prove every qualified
V41 frontier is captured or paired with DSpark.

`_cleanup_finished` synchronizes, selects cacheable prompt/output boundaries,
extracts/merges/materializes state and dispatches async storage. Removal is
retained until storage finishes; abort/failure clears request admission/priming
and releases request block tables. Cache stats, phase timers, request counts,
SSD/capacity/pressure and admin snapshots provide generic observability. These
are reference lifecycle contracts, **not** M52R semantic commit certificates:
backend completion or delivered token count cannot alone authorize publication
of qualified MTP state.

## Highest-priority compatibility finding: target state is not DSpark state

Required reusable execution checkpoint at committed frontier **C**:

1. All 40 target layers' seven slots: offsets, packed window, compressed KV,
   index keys, compressor KV/gate tails and Engram integer lookback, plus layout/
   compression/padding metadata and compatible model/SSD resource lifetime.
   This includes non-KV state; target offset equality alone is insufficient.
2. All three DSpark rings: projected committed hidden context, absolute offsets,
   actual capacity **128**, physical modulo slot order, stage/layer identity and
   native install/take ownership. Draft KV is ephemeral, never committed context.
3. Complete encoded prefix IDs through C and strict checkpoint/tokenizer/recipe/
   layout/profile identity; prompt-priming context must reference that same C.
4. No reusable request-local proposal, verify stash, rejected suffix, queued
   unconsumed prediction, semantic horizon pending ordinal or RNG claim. These
   are retired at settlement, not serialized into a new request. Qualified MTP
   is greedy; non-greedy draw-state equivalence is not established.

Pinned oMLX **does** have a generic MTP sidecar, but it is Lightning/head priming:
`_MtpPrefixSnapshot(boundary_tokens, mtp_cache, pending_hidden)`, memory-only,
bounded LRU, keyed by exact target chain tip. Restore requires that target tip
remain live; eviction/hash-map clear drops sidecars. At target C it restores
head pairs through C-1 plus hidden(token[C-1]) awaiting the next pair. This is
coherent for that head's different representation, not a DSpark offset mismatch
and not evidence for DSpark restoration.

Crucially, `prompt_priming._prepare_prefix_context` explicitly returns False for
`_omlx_dspark_decode_enabled`. `deepseek_v4_dspark.capture_prompt` instead builds
or appends the host priming slot; noncontiguous offsets create a new context.
A capture starting at nonzero offset can acquire that absolute offset without
possessing the earlier retained context rows. `install_committed_context` checks
ring offsets against target state; it does not reconstruct lost rows or recover
an earlier physical ring from a later ring. No DSpark prefix-cache snapshot/store/
restore companion is defined in this examined chain. The generic missing-sidecar
fallback to unprimed MTP cannot stand in for **qualified** full committed priming.

For reusable qualified DSpark state, target C = ring absolute C = encoded
committed prefix C. Physical rows must match the committed tail, not merely
report C. Arbitrary earlier-history rewind cannot be inferred from speculative
rollback: a bounded ring at M has overwritten rows required at N < M. Use an
exact earlier paired checkpoint or miss; never restore target N plus rings M.
Exact-hit N-to-N-1 target fallback/trim must choose a corresponding paired DSpark
checkpoint too. The existing generic head sidecar does not establish this.

## P5 and canonical publication: established local transfer, unresolved generic adoption

Established local seam: request owns immutable prefix IDs through P=N-1;
DwarfStar's committed runner owns the same 40-layer live cache and same-forward
DSpark taps/rings through P. `LivePrefillResult` reserves/freezes the producer;
P5 validates committed transaction/frontiers/layout, detaches/revokes the producer,
consumes the result, installs native committed context and inserts only terminal
`ids[P]` with `all_tokens=ids[:P]`. Native BatchGenerator becomes sole executable
owner. Failed transfer/start burns the result; it is not retryable mutable state.
This neither replays prefix nor repacks the whole cache. The current wrapper
constructs its own BatchGenerator, **not** the generic Scheduler's active batch.

Established retirement seam: qualified quiescence reconciles execution-ahead
queue with consumed history, drains only already committed safe suffix, discards
the final future prediction, or consumes exactly one canonical token when target
is one behind. It extracts native row views, checks all target/ring offsets and
canonical IDs, removes UID and retires native prediction/proposal ownership.
Semantic failure or failed removal poisons and burns state; it cannot publish idle
reuse. Recipe turn completion/call certification is separate from socket delivery.

Intended generic ownership transition (not implemented or approved): scheduler
leases a complete compatible checkpoint, execution acquires exclusive mutable
state through P5/native install, then execution settles before any immutable
cache publication. Cache retention owns snapshots, not an active executable
alias. Cancellation must wait for the native worker, settle or burn, then release
leases. Eviction frees snapshots/blocks only; it never deletes application history
or a retained response. Acquisition cannot expose mutable shared state to two
requests, even with only one active generation.

The existing local transfer describes **move** ownership without reconstruction;
cache reuse requires an independently defined snapshot/restore or exclusive
lease operation. The generic storage path reconstructs packed target state and
has its own completion/remove schedule. It has not established where P5 adopts
into that schedule or where M52R quiescence delays publication/removal.
There is a useful existing structural seam: Scheduler `_do_external_prefill`
holds out the terminal, and `_insert_prefilled_request` / `_schedule_waiting`
insert prefilled cache plus terminal into **its own** BatchGenerator and register
request/UID ownership. Thus adoption is not structurally absent and a new
scheduler is not justified. However, those paths currently run their own prefill,
cache finalization, stop state machine and completion lifecycle, not P5 producer
revocation plus the qualified semantic guard/quiescence transaction. B2 concerns
that specific state/ownership adaptation, not the mere existence of `insert`.
Attaching
the current wrapper beside Scheduler would leave two schedulers/executable
lifetimes, not a bounded integration. No safe whole-cache repack or prompt replay
is authorized to bridge this gap.

## Minimal blocking set and smallest additional evidence

**B1 — Paired reusable checkpoint/restore contract.** Unresolved: how generic
prefix ownership captures and restores complete target **and DSpark** state at
the same committed C, including partial-prefix and exact-hit terminal holdout,
without replay or interpreting speculative rollback as conversation rewind.
Available source provides V41 target delta reconstruction and a different MTP
head sidecar; DSpark deliberately bypasses it. Retained model evidence transfers
live rings only and exercises no generic prefix fetch. Thus neither contents nor
snapshot immutability/restore ownership is established for the required pair.
Smallest additional evidence: identify a concrete existing or proposed paired
checkpoint capture/restore operation and its physical ring/target boundary
representation, copy/transfer policy, identity namespace, failure/eviction rules
and miss behavior. If source cannot establish it, one isolated diagnostic of a
real partial hit after ring wrap, plus the exact-hit holdout decision, must compare
restored target slots/ring contents/frontiers to the committed checkpoint and
show no prompt replay. A normal completion or offsets-only probe is insufficient.
This may reveal that ds41f must retain checkpoint-state ownership; upstream cache
manager ownership is **not yet a safe restoration decision**.

**B2 — Scheduler adoption and semantic publication/retirement contract.**
Unresolved: how the generic scheduler becomes the sole executable owner of P5
state and native DSpark priming, and gates extraction/store/remove on the existing
canonical/recipe settlement result rather than backend finish or partial SSE.
Source shows P5 entering a separately constructed BatchGenerator. Generic
Scheduler already inserts externally prefilled cache/terminal state, but owns
its own batch/completion/store queues without the qualified P5 revocation,
DSpark installation and semantic settlement contract. These are promising
insertion sites, not proof of the required shared ownership transition. Smallest additional evidence: a concrete
transition trace specifying pre/post owners, native cache/UID adoption, held-out
terminal, semantic guard attachment, quiesce-before-publication ordering and
failure/cancel-before-removal behavior, including async-store retention/release.
If source reasoning is insufficient, isolate one real handoff and one committed-
queue cancellation with the proposed ownership chain, proving same-live transfer,
unchanged topology and exactly one owner/retirement. Do not build a server to
obtain this evidence. A small adapter signature alone does not resolve B2.

These are architectural blockers, not requests for a broad qualification suite.
Both must resolve before destructive scheduler/cache restoration is safe. Missing
direct composition alone is not the BLOCK reason; materially missing state and
ownership transitions are. SP1 stops here rather than implementing them.

## Intended API split (conditional on B1/B2)

```text
ordinary OpenAI request / application-owned messages
  -> generic HTTP lease, body/error/disconnect lifecycle (candidate: pinned oMLX)
  -> official recipe schema conversion, encoding and parsing options
  -> generic scheduler request + identity-scoped compatible prefix lookup
  -> exclusive qualified DwarfStar/P5/native MTP execution and settlement
  -> official recipe response/chunk processing with consuming safeguards
  -> generic JSON/SSE delivery (not canonical acknowledgement)
```

Recipe owns supported request schema/messages, reasoning controls, tool
declarations/choice, model conversation encoding, response schema/finish reasons,
usage and chunks. Generic serving owns model registry/aliases, transport errors,
HTTP/SSE lifecycle and delivery; ds41f validates admitted model identity and
concrete execution limitations. Recipe converts generation parameters; execution
implements only qualified sampling, and admission must reject unsupported controls
rather than silently invent semantics. Chat conversion currently validates seed/
penalties but its `InferenceOptions` passes max_tokens/temperature/top_p only:
schema acceptance does not prove runtime support. Ordinary OFF image preparation
remains its qualified model-execution seam; generic oMLX VLM routing does not
qualify MTP images. Responses/Messages adapters remain assets, not mandatory new
serving closure scope. Recipe conversion errors retain official bodies; generic
layer supplies lifecycle/resource error mapping. No parallel DeepSeek grammar.

## Distinct identities and ordinary workload owners

HTTP request identifies one delivery attempt; application conversation is the
client's message history; encoded prompt is recipe/model/tokenizer-bound IDs;
scheduler request is an admitted work/UID lifetime; prefix cache is immutable
compatible committed state at C; active generation owns exclusive mutation;
canonical output is consumed/semantically settled output; MTP state contains
rings plus transient speculative machinery. None is interchangeable.

oMLX separates request ID, UID, block table/hash chain, prompt and output, with
no compulsory conversation ID. OFF separates ordinary HTTP requests and optional
sessions. MTP separates emitted/consumed/delivered tokens internally, but its
public session collapses conversation, reusable target/rings, canonical history
and retry lifetime. The latter is qualified containment, not general API identity.

New/unrelated conversation: recipe encodes full request, scheduler admits fresh
or compatible prefix. Exact append: cache reuses a safe complete prefix, execution
computes suffix. Last-turn edit/regeneration: cache uses only unchanged committed
prompt checkpoints; prior assistant output is not compulsory history. Earlier
edit/branch: longest compatible **paired checkpoint**, not live ring rewind.
Shortened context: earlier checkpoint or miss, never trim recurrence blindly.
Shared system prefix: reuse by encoded identity across requests, not session ID.
Applications own edits/branches; recipe owns their interpretation; cache owner
chooses safe reuse/miss; execution enforces state coherence. None of these cases
is implemented or newly qualified by SP1.

## Responsibility/path decision map

RESTORE = existing official/upstream owner should be authoritative; KEEP =
necessary qualified ds41f ownership; ADAPT = correct candidate owner needs an
integration seam; DELETE = leave the ordinary production path in later authorized
work. ADAPT entries marked blocked are **conditional candidates**, not restoration
approval; DELETE is not permission to remove current code before replacement.

| Responsibility/path | Classification and owner/boundary |
|---|---|
| HTTP transport, generic errors/leases | RESTORE pinned oMLX machinery; ADAPT disconnect to settlement (B2) |
| API models/DeepSeek conversion/encoding | KEEP current thin delegation to official recipe; no oMLX template replacement |
| Output parsing/projection, tool/reasoning grammar | KEEP recipe processor/generators; KEEP consuming guard where qualified |
| Scheduler queues/request crossing | ADAPT pinned Scheduler candidate, blocked B2; retain current lease until resolved |
| Capacity/lookup/eviction/cache manager | ADAPT pinned cache candidate, blocked B1/B2; no unconditional restoration |
| Cache-state publication/restore | KEEP ds41f committed-state gates; paired snapshot authority unresolved B1 |
| Prefill and qualified numerical execution | KEEP DwarfStar/MLX and first-party OFF execution |
| P5 transfer/terminal holdout | KEEP same-live producer revocation; ADAPT scheduler adoption, blocked B2 |
| MTP/decode physical path | KEEP qualified native oMLX topology plus ds41f frontier/lifecycle safeguards |
| Sampling | KEEP qualified MLX/native sampler; ADAPT recipe options/admission, no stochastic-MTP claim |
| Tool effect semantics | KEEP certification prerequisite; application effect reservations optional, not grammar |
| Cancellation/streaming/backpressure | RESTORE generic transport machinery; KEEP worker shielding/settlement; ADAPT B2 |
| Retry/outcome retention | KEEP immutable settled projection; ADAPT internal request identity; no automatic HTTP exactly-once |
| Observability | RESTORE generic stats/resource counters; ADAPT true committed/reused frontier reporting |
| Application-visible sessions/sequences/certificates | DELETE compulsory ordinary-path dependency; retain optional diagnostics/dedicated qualification |

Later removal/demotion candidates: `LocalBoundary`'s endpoint/session restriction
(not its ingress safety requirements), mandatory `/v1/sessions`/sequence headers,
public certificate/outcome envelopes, exact-extension-as-only-admission rule,
and dedicated-client orchestration as prerequisite for ordinary serving. Their
original stale-state/retry/effect requirements remain. Alternative generic owners
have lookup/lifecycle machinery, but B1/B2 prevent proving that restoration is
smaller and safer yet. Do not delete `internal_mtp` wholesale: its same-forward
priming, semantic guard, canonical quiescence and worker settlement are assets.
Likewise retain OFF recipe preparation/projection, not another adapter around MTP.

## M52R guarantees: preserve findings, not an enlarged public promise

- **A — Model-output settlement:** KEEP internal consuming recipe/canonical
  agreement, immutable certified call prefix, queue drain/prediction retirement,
  fail-closed incomplete tool output. Streaming deltas alone authorize neither
  reusable generated cache nor effects. Valid committed prompt checkpoints may
  be reusable independently of successful response delivery; speculative or
  transport-partial generated state may not.
- **B — Publication/retrieval:** complete tool calls/outcome are frozen before
  transport completion; exact identified retries can project retained data
  without regeneration. Generic HTTP retry without retained request identity
  need not denote the same generation. Whole-outcome reconnect is not per-event
  ACK or proof that a client observed every call. Optional idempotency/retrieval
  semantics need explicit future retention/identity policy, not compulsory MTP
  sequence/session protocol.
- **C — Server-owned effects:** if offered, server must reserve/authorize only
  certified supported settled calls, reject conflicting duplicate authority,
  reuse known results and never rerun uncertain reservations. M52R's concrete
  ledger lives in the dedicated client; it does not establish a generic server
  effect executor or durable crash-safe exactly-once ledger.
- **D — External client effects:** ordinary clients choose whether/how to execute
  tools. Server cannot guarantee exactly-once real-world external execution from
  an OpenAI response. Dedicated-client fencing/reconciliation is optional
  application semantics; weather-only schema, bounded ledger and qualification
  controller are qualification scope, not universal server guarantees.

## Next scope and unchanged release gate

Smallest proposed next **investigation**, requiring separate authorization:
resolve B1's paired checkpoint state operation and B2's scheduler adoption/
settlement transition, using existing sources/evidence first and only the named
isolated diagnostics if needed. No SP2 implementation is pre-authorized; a bounded
restoration implementation task cannot safely be finalized while these contracts
remain unresolved. No elaborate follow-up series is prescribed.

All broader required serving domains remain open to actual closure evidence:
ordinary MTP ingress, independent edited/branched requests, qualified prefix reuse
and capacity/lifecycle, transparent profiles, streaming/disconnect/retrieval,
supported sampling/tools, resource policy, configuration and production
observability. Neither this BLOCK nor a later SP1 PASS changes their release gate.

**ds41f is not production-release-complete as a general LLM runtime server.**
Actual restoration and capability closure require later explicit authorization.
Historical M-series evidence is unchanged; default MTP and `ds41f-runtime`
promotion remain deferred.
