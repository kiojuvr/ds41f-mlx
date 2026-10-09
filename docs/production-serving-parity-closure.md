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

Recent inspection exposed several examples:

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

The serving layer must eventually support ordinary function declarations and tool result messages for the supported tool scope.

A weather-only schema is qualification evidence, not a general production tool API.

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

# SP1 — Production Serving Foundation

SP1 is the first implementation milestone under this document.

Its purpose is not to produce a report.

Its purpose is to establish the missing production-serving architecture and close the foundational blockers exposed so far.

SP1 begins with a bounded comparison against mature serving runtimes only to identify missing responsibilities.

Then implementation begins immediately.

The first priority areas are:

1. stable OpenAI-compatible Chat Completions ingress independent of MTP internals,
2. request/conversation semantics that do not assume append-only history,
3. explicit reusable-prefix / KV cache management,
4. execution-profile isolation beneath the public API,
5. correct streaming/cancellation/retry integration through the existing MTP settlement machinery,
6. ordinary operator configuration and defaults.

Do not begin portable packaging, broad Web capability expansion, long-context research, additional model optimization or unrelated release infrastructure.

---

# SP1 exit condition

SP1 PASS requires evidence that ds41f has crossed from a specialized qualified execution harness into an ordinary local LLM serving architecture.

At minimum:

- a normal OpenAI-compatible client can use `/v1/chat/completions`,
- the client does not know ds41f MTP session/fence/sequence mechanics,
- MTP remains an internal execution implementation,
- append-only continuation remains fast,
- edited/regenerated/branched prompts have correct defined cache behavior,
- reusable prefixes are preserved when safe instead of forcing full exact-history continuation,
- unrelated/new prompts work without protocol corruption,
- cache hit/miss/reuse can be observed,
- streaming and disconnect do not violate canonical state,
- supported tool/sampling boundaries are stated accurately,
- ordinary startup/configuration does not depend on development-machine source knowledge,
- existing qualified MTP physical behavior is not silently replaced or materially regressed.

SP1 may remain single-flight.

Multi-user throughput and distributed serving are not required for SP1.

A PASS must represent an actually usable runtime-server path, not only a design or test harness.

---

# Release gate after SP1

SP1 does not automatically imply final release.

After SP1, reassess the remaining parity matrix.

If no material production-serving blocker remains for the declared local-runtime scope, proceed directly to release/default-profile/operator-documentation closure.

If concrete blockers remain, group them into the smallest coherent subsequent SP milestone.

Do not invent follow-up milestones merely because the numbering exists.
