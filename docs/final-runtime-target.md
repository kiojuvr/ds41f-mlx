# Final runtime target

## Authority and scope

This page defines the **prospective destination**, not present qualification or
an implementation schedule: a self-contained source runtime repository for the
official DeepSeek-V4.1-Flash checkpoint on the documented Apple Silicon platform.

[Architecture](architecture.md), [capabilities](capabilities.md) and
[operations](operations.md) describe what exists and is supported now.
[Implementation plan](implementation-plan.md) owns remaining work;
[qualification](qualification.md) governs evidence interpretation. Completed OFF
text execution, bounded image qualification, explicit ordinary MTP semantics and
a bounded singleton release do not mean this whole destination has been reached.
MTP remains OFF by default. Internet/multi-user readiness, broader release and
future projection are separate decisions.

## Fresh-clone expectation

At the intended public state, a user or coding agent can:

1. clone **one documented source runtime repository**;
2. read repository-owned setup and operations instructions;
3. install reproducible package/system dependencies;
4. obtain the official checkpoint;
5. build/configure the runtime using those instructions;
6. run repository-owned provenance and scoped qualification commands;
7. start and operate the supported runtime.

The repository may be the source-clone development repository or an explicitly
qualified future projection. A permanent separate `ds41f-runtime` repository is
not required. The existing directory of that name is unused old generation,
not deployment or accepted release history. If projection is selected, it flows
one-way from qualified `ds41f-mlx`; no independent implementation or release-only
semantic fixes. Detailed rules belong to the
[reference/promotion strategy](reference-release-and-promotion-strategy.md).

Hardware, platform, memory, storage and toolchain requirements must be explicit.
The primary platform is Mac Studio M3 Ultra 512 GB with MLX/Metal and SSD-backed
Engram. Checkpoint/cache/storage locations remain legitimate operator choices;
prior chat, a historical qualification environment and undocumented local
knowledge must not be setup prerequisites.

Normal final installation must not require multiple runtime implementation
repositories, manual magic historical checkouts, locally patched developer trees,
`/tmp` source authorities, undocumented patches or environment variables pointing
to arbitrary donor source. Ordinary packages, system libraries/build tools and
reproducible version locks remain acceptable. The official checkpoint is an
intentionally external model asset, not a missing repository component.

## Dependency destination

### Execution ownership

Standard-OFF retains first-party checkpoint loading, model/numerical execution,
packed storage, SSD-backed Engram and generation/state/resource authority.
Lower-level attributed MLX/Metal or stateless acceleration primitives may remain
reproducible dependencies; donor-count reduction is not permission to replace the
qualified core. No donor model fallback is part of the supported OFF path.

MTP retains **pinned oMLX physical architecture authority**, DwarfStar-derived
prefill topology with qualified MLX numerics, and ds41f semantic/application/
lifecycle/resource authority. Self-contained delivery or source transfer must
preserve physical topology, sole execution ownership and practical performance.
Do not reimplement an algorithm and promise to recover speed later. The
[MTP doctrine](mtp-architecture-reset.md) defines the structural conformance gate;
this target does not reopen OFF implementation or revive the discarded M50–M56
producer.

### Official recipe and semantic assets

Official recipe conversion, tokenizer, prompt/tool/reasoning grammar, streaming,
response projection and consuming semantics remain authoritative. The final
delivery must provide required functionality and assets through normal reproducible
packages, repository-owned functionality or another documented self-contained
mechanism. This page does not preselect among them.

Current installed execution admission is defined in
[runtime admission](runtime-admission.md), and exact setup/assets in
[operations](operations.md). A developer checkout or historical source archive is
not runtime authority merely because it was used in an old qualification.
Changing source ownership or package delivery never authorizes an independent
guessed DeepSeek grammar, omitted semantic asset or weakened identity check.

## Public contract and authority direction

```text
Public/local API and operator contract
  -> admission / application continuity / recovery / lifecycle
  -> ds41f state and execution interfaces
  -> qualified execution implementation
     OFF: first-party; MTP physical topology: pinned oMLX
     prefill: DwarfStar-derived topology + qualified MLX numerics
  -> MLX / ordinary reproducible dependencies
  -> official checkpoint
```

This is an authority boundary, not a predetermined class/module design.
Checkpoint/model/protocol semantics remain authoritative. Temporary adapters may
exist, but donor scheduler types, checkout paths and revision conventions must
not become compulsory final application knowledge. Explicit capability and
operational admission remain legitimate public controls.

## Contracts that delivery changes must preserve

These are present semantic constraints on future delivery work, not claims that
every profile implements every historical internal feature. Current normative
owners are linked here; exact historical proof scopes stay with their receipts.

| Contract | Current definition |
| --- | --- |
| One executable authority; consumed versus predicted history; same-live transfer and no retained-history replay/repack | [State and transfer](session-state.md), [generation](generation.md) |
| Consuming semantic horizon bound to generation/ordinal; preview cannot advance canonical state or authorize effects | [Generation/settlement](generation.md) |
| Canonical native state, recipe-representable history, transcript and transport observations are distinct | [State categories](session-state.md), [logical sessions](production-sessions.md) |
| Immutable identity-bound outcomes; conservative unknown/poison disposition; no invented tool results or automatic uncertain retries | [Logical sessions](production-sessions.md), historical application state (`c1d1f716:docs/web-application.md` in Git) |
| Model/allocator/KV/paired/Engram lifetimes, cleanup and finite identity/pin retention remain intentional | [Runtime strategy](runtime-strategy.md), [capabilities](capabilities.md) |
| Explicit profiles and admission, no silent fallback or inherited scope; local IDs are not authentication | [Capabilities](capabilities.md), [operations](operations.md) |

Retained native sessions, logical application sessions and ordinary independent
requests have different continuation/retry/cache-miss rules. Exact-prefix
retained continuation does not make every ordinary request append-only. A fresh
independent-request miss is not permission to replay a retained session when its
required pair is lost. Transport writes are not settlement or client ACKs.

Tools execute in the client under the applicable completed-call/effect policy.
Living-client reservations and stored results are not crash-safe or distributed
exactly-once guarantees. Dependency replacement cannot silently broaden recovery,
persistence, concurrency, security or multimodal scope.

## Publication and qualification

Self-containment means source closure and reproducible documented setup,
**not a general distribution system**. Wheels, installers, Homebrew formulae,
portable binaries, tarball delivery and a fully vendored ecosystem are not
mandatory. Existing bundle work remains valid scoped work, not the required final
public form. A normal clone plus documented environment construction is sufficient.

Implementation qualification, installed-payload admission, immutable R1 regression
and release approval answer different questions. Full R1 re-verification belongs
to an appropriate stable boundary, not every intermediate milestone. A later
projection needs exact source/payload/reference identity, independent setup and
its scoped receipt; historical extraction or source-clone approval does not
certify later HEAD. This document schedules no package, model or release campaign.

Preserve provenance, attribution, negative results and superseded blockers.
Reuse historical evidence as scoped specification/oracle where identities and
ownership permit, and validate the affected delta when implementation changes.
Never rewrite old results, paths or dependency identities to make them appear
self-contained retroactively.

## Earlier target edition

The pre-compilation edition (`c1d1f716:docs/archive/editorial-baseline/final-runtime-target.md.txt` in Git)
retains the original milestone account, M33–M40 source pointers and former
release-repository/checkout wording. It is source evidence, not an assertion that
the current old `ds41f-runtime` directory is release history or that historical
checkout requirements remain today's installed-execution contract. Current
normative definitions are in the owners above; the destination is prospective.
