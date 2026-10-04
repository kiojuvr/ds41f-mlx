# Final runtime target

## Authority and scope

This is the canonical prospective architectural target for `ds41f-mlx`, independent
of milestone scheduling or prior chat context. The final public form is a
**self-contained source runtime repository** for the official
DeepSeek-V4.1-Flash checkpoint on the documented Apple Silicon platform.

This document defines the destination, not an implementation plan or a claim that
it has already been reached. [Runtime strategy](runtime-strategy.md),
[operations](operations.md), the [release manifest](../release/ds41f-release.json)
and scoped qualification records continue to describe the current runtime.
Production/default MTP, DSpark and speculation remain OFF; this target neither
promotes MTP nor broadens any qualified capability. The separately implemented
[M41 local candidate](milestone-41-local-mtp-release-candidate.md) has its own
narrow [setup/capability contract](mtp-local-release-candidate.md).

The [reference release and promotion strategy](reference-release-and-promotion-strategy.md)
refines the repository model: `ds41f-mlx` remains the sole development authority;
a future `ds41f-runtime` is its deterministic release projection, not a second
implementation line. M42 semantic closure/Reference Release R1 precedes M43
extraction. This does not schedule full oMLX replacement or change qualification.

## Fresh-clone expectation

At the intended public state, a user or coding agent can:

1. `git clone` **one** promoted `ds41f-runtime` source repository (developed in `ds41f-mlx`);
2. read repository-owned README/setup and operations instructions;
3. install ordinary reproducible package/system dependencies;
4. obtain the official DeepSeek-V4.1-Flash checkpoint;
5. build or configure the runtime using those instructions;
6. run repository-owned provenance and qualification commands;
7. start the runtime.

Hardware, platform, memory, storage and toolchain requirements must be explicit
(the current primary target is Mac Studio M3 Ultra 512 GB with MLX/Metal and
SSD-backed Engram). Checkpoint and storage locations remain legitimate operator
configuration. No prior chat, historical qualification environment or undocumented
local knowledge is a setup prerequisite.

Normal final installation must **not** require cloning multiple runtime
implementation repositories, manually checking out magic historical commit hashes,
maintaining locally patched development checkouts, using `/tmp` source authorities,
reproducing undocumented patches, or setting environment variables to arbitrary
development source trees. Historical qualification repositories/environments are
evidence, not normal runtime dependencies.

Ordinary installable dependencies such as Python packages, MLX, FastAPI, system
libraries and build tools remain acceptable; reproducible version locks are not
prohibited. The official checkpoint is an intentionally external model asset.

## Dependency destination

### oMLX: temporary implementation substrate

Today the OFF release manifest pins oMLX and OFF configuration imports its
checkout; M41's separate candidate installs repository-exported source as a package.
M44's ds41f `TargetGenerationSession` now owns standard-off target scheduling,
sampling, consumed-token history, generation lifecycle and exact-list idle
transfer after dense P0–P7 prefill/P5 handoff. Model loading/target-forward math,
packed cache representation, kernels and SSD Engram remain temporarily oMLX-owned. Internal guarded qualification additionally uses a
patched oMLX candidate. This is a **temporary implementation substrate**, not the
intended final public architecture or the authority defining ds41f contracts.
The self-contained native reference core does not by itself close this production
dependency.

Functionality currently supplied through oMLX is expected eventually to become
owned by `ds41f-mlx` or otherwise cease to require a separate oMLX repository
checkout. This includes model loading/execution support, decode execution,
GenerationBatch-equivalent state, cache/KV ownership, continuation state,
generation lifecycle/state transitions, cancellation/quiescence, cache
transfer/extraction, and—as applicable to an explicitly supported capability—MTP
lifecycle, DSpark integration/state and proposal/verification lifecycle.
Ownership replacement does not imply enabling those experimental capabilities.
The next selected ownership frontier is target-forward/all-layer cache mutation
on the existing P7-compatible packed representation; no new capability is implied.

### deepseek-recipe: protocol authority, not a developer-checkout requirement

Unlike oMLX, the current official recipe implementation remains an important
protocol/prompt/response authority, including tokenizer, tools, thinking,
streaming and semantic preview behavior. Current setup prefers an installed
native recipe package but still requires a checkout for tokenizer/provenance;
internal qualification uses an exact candidate native build. Merely installing
a package in the OFF setup does not eliminate that checkout requirement. M41's
separate source/native provisioning delivers the candidate and tokenizer without
requiring a developer checkout; OFF migration remains separate.

The final dependency must provide all required functionality and resources via
**a normal reproducible package dependency, repository-owned functionality, or
another self-contained reproducible mechanism**. This document does not choose
among them now. Users must not manually clone a particular recipe development
revision or reproduce undocumented local patches to run ds41f. Changing delivery
or source ownership is not permission to replace official recipe semantics with
an independent guessed grammar.

## Authority and dependency direction

```text
Public/local API and operator contract
        ↓
Admission / sessions / recovery / lifecycle
        ↓
ds41f runtime interfaces and ownership
        ↓
ds41f model execution implementation
        ↓
MLX / ordinary system or package dependencies
        ↓
Official DeepSeek-V4.1-Flash checkpoint
```

This is an authority/dependency boundary, not a rigid module or class design.
Official checkpoint/reference semantics remain model authority. Temporary adapters
may exist during development, but external runtime repositories, scheduler types,
checkout paths and donor revision conventions must not leak into the final public
contract. ds41f owns its interfaces, state transitions and qualification obligations.

## Stable ds41f contracts from M33–M40

Implementation dependencies may change without reopening these established
contracts unless new evidence requires a substantive architectural revision.
Their qualified scopes remain bounded: internal guarded proofs are not claims
that the current public OFF API implements every internal feature.

- **Semantic-horizon ownership** ([M33](milestone-33-semantic-horizon.md),
  [qualification](milestone-33-protocol-qualification.md)): authoritative recipe
  preview does not advance canonical state. A predicted completing response stays
  unforwarded until canonical emission; ownership binds generation and absolute
  response ordinal, not token identity. Permitted one-token quiescence repair
  materializes only the canonical terminal, with no successor sample/proposal/verify.
- **Canonical versus representable state** ([M36](milestone-36-client-recovery-qualification.md),
  [M36R](milestone-36r-recovery-admission.md)): sole canonical native history,
  certified ordinary protocol history, client transcript and transport observations
  are distinct. Coherent caches, valid JSON or delivered SSE do not prove an
  ordinary continuation is representable.
- **Native/cache lifecycle and exact-prefix continuation**
  ([M34](milestone-34-operational-qualification.md),
  [M35](milestone-35-http-sse-qualification.md)): one executable cache authority;
  ownership transfer survives prior-owner close without reviving revoked owners.
  Shielded native phases and exact response leases govern coherent-boundary
  cancellation, settlement and retirement. Actual ordinary recipe re-encoding
  must strictly extend retained canonical IDs before mutation. Retained sessions
  allow **zero hidden model-history replay, cache reconstruction or full repack**;
  explicit fresh-session prefill is a different operation.
- **Request/outcome fencing and certified recovery** (M36R): bind session lifetime,
  monotonic sequence and exact body bytes; same-identity settlement observation
  never generates again, changed/expired identities reject. Recover only from a
  positive authoritative reconstruction certificate, then independently check
  the real continuation prefix. Writes/yields and absent observations are not ACKs.
- **Conservative negative/poison handling** (M36–M36R,
  [M37](milestone-37-local-client-integration.md),
  [M38](milestone-38-client-operational-soak.md)): unrepresentable state is not
  repaired by guessing, sampling onward, token splicing or invented tool results.
  Internal ownership failures poison; no ambiguous idle publication. Explicit
  retirement/fresh-session decisions preserve only legitimate application history;
  failed retirement blocks reuse. Unknown lifecycle/effect outcomes stop automatic
  progress rather than triggering replacement or retries with new identities.
- **Client-side effect ownership** (M37–M38): tools execute outside the server,
  only from certified completed calls. A bounded living-client ledger reserves
  before execution and reuses stored results; ambiguous effects never auto-repeat.
  This is not crash-safe or distributed exactly-once execution.
- **Finite process-lifetime identity/admission and bounded diagnostics**
  ([M39](milestone-39-lifetime-admission.md)): server-issued no-wrap identities,
  live-only execution authority and explicit exhaustion denial fence retired work
  independently of diagnostic eviction. Bounded summaries/traces are not an
  outcome archive; successful retirement releases closed payload/native authority.
  Restart does not restore that process's identity or recovery authority.
- **Explicit capability profiles and local security assumptions**
  ([M40](milestone-40-release-readiness.md)): capability/dependency identity and
  admission must be explicit, not implicit fallback or request-selected acceleration.
  Profiles carry their own supported envelope and acceptance. Local/single-operator
  operation trusts local operator peers; IDs are not authentication secrets and
  loopback alone is not access control. Remote/multi-user serving requires a new
  security design. M40's proposed profile, ingress and Host/Origin requirements
  remain prospective release obligations, not implemented/qualified guarantees.

These are **ds41f contracts**, not properties defined by oMLX, even where oMLX
currently implements part of them. Neither donor removal nor fresh packaging
silently inherits broader qualification, persistence, concurrency or recovery scope.

## Publication, packaging and historical evidence

The temporary multi-repository development configuration must not be treated as
the desired final public runtime architecture. Public/runtime completion is to be
evaluated against this target as well as correctness, practical performance and
capability qualification. Future milestones decide **when and how far** to move
toward it; there is no dependency-removal milestone number or fixed date here.
The destination is not reopened merely because an adapter remains useful.

Self-containment means source-repository closure and reproducible documented setup,
**not a general distribution system**. Wheels, installers, tarball distributions,
Homebrew formulae, portable binaries and a fully vendored third-party ecosystem
are not required. Existing bundle work remains valid scoped work, not a mandatory
final delivery form. A normal clone plus documented environment construction is
sufficient. M40's candidate delivery choices are scoped historical proposals, not
universal packaging requirements for this target.

M33–M40 evidence remains valid for the architecture and behaviors it established,
including negative results and superseded blockers. Dependency replacement should
reuse it as specification/oracle where source ownership permits, with explicit
identity/scope reconciliation and rerunning affected qualification when implementation
changes. Preserve provenance and attribution. Do not rewrite historical milestone
results, paths or dependency identities to look self-contained retroactively:
this target is prospective architecture, not retroactive history.
