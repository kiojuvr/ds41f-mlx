# Current implementation plan

This page owns present development priorities and decision rules, not a milestone
chronicle or optimization backlog. [Architecture](architecture.md) describes what
exists; [capabilities](capabilities.md) owns support/limits; [operations](operations.md)
owns deployment. Historical plans and receipts retain their original decisions,
not present authorization or a required sequence of new milestones.

## Upstream-first architecture and compatibility policy

**Follow official DeepSeek Harness, DeepSeek Recipe and DeepInfer architecture
wherever feasible. Implement only the integration and execution differences
necessary for Apple Silicon / MLX.** This section owns development decisions,
not a new capability, execution or operation contract; those remain with
[architecture](architecture.md), [capabilities](capabilities.md),
[Runtime strategy](runtime-strategy.md) and [operations](operations.md).

### Authority and responsibility boundaries

- **Official DeepSeek Harness** owns agent/tool orchestration, conversation
  history and session projection, attachment references and tool-image
  acquisition, image offload decisions, conversation compaction, session forks,
  resume and user-facing lifecycle semantics. Prefer its official extension
  points, event model and request projection. Do not reproduce these inside
  ds41f-mlx merely to compensate for an integration gap.
- **Official DeepSeek Recipe** owns protocol conversion, message/tool semantics,
  content ordering and multimodal placeholders, canonical encoding/decoding,
  response and tool-call accumulation. Do not add a competing Messages parser,
  serializer or semantic translation layer. Existing bounded Runtime admission
  still owns security and supported capabilities.
- **Official DeepInfer** is the reference architecture for multimodal request
  preparation, cache-aware image preprocessing, KV Cache hit/miss behavior,
  scheduler/prefill responsibilities, model execution semantics, image/token
  alignment and cache-aware resource accounting where applicable. Its code is
  not necessarily directly portable: follow architecture and numerical semantics,
  not CUDA-specific mechanisms.
- **ds41f-mlx** retains MLX/Metal numerical execution, pinned oMLX Target/DSpark
  integration, Apple Silicon memory/resource admission, Runtime lifecycle and
  installed package admission, compatibility glue between Recipe, DeepInfer
  semantics and the selected MLX backend, provenance, numerical verification and
  fail-closed consistency checks. Do not construct competing high-level agent,
  session or protocol architectures. Existing Runtime canonical-state and paired
  cache safeguards are not a replacement for Harness session ownership.

### Departures from upstream

Every proposed architectural divergence must establish:

1. The exact upstream revision and source behavior.
2. The corresponding existing local implementation.
3. Whether upstream already supplies the capability or extension point.
4. The concrete incompatibility or Apple Silicon-specific constraint.
5. Why configuration, an adapter or a small integration change is insufficient.
6. The minimum justified local deviation.
7. Required numerical, semantic and lifecycle acceptance evidence.
8. Conditions under which the deviation can later be removed.

Prefer adapting an existing implementation over a parallel framework. Avoid
duplicate ownership, compatibility shims without demonstrated need, speculative
abstractions and permanent forks of official semantics. Prior qualification does
not by itself make a local implementation an architectural requirement. Existing
safety constraints nevertheless remain enforced until a replacement is explicitly
verified; this policy authorizes no replay, recovery or other implementation.

### Compatibility versus currently qualified restrictions

**Conservative Runtime guards protect the currently qualified implementation;
they do not automatically define permanent model or upstream architectural
limitations.** In particular, rejection of historical image replay/re-encoding is
a current local safeguard, not a universal restriction of official DeepInfer.
Changing it requires an independently justified, numerically correct cache-recovery
design. Independent Messages C=0 and whole-image-safe C>0 source rules now reuse
the existing MTP execution schedule with genuine selected paired state where
present; aggregate conditional-discount and exact logical-lease safeguards remain.
The bounded reconstruction path is now deployed; general C2 qualification remains PARTIAL. See the [operational completion contract](multimodal-runtime-plan.md).

The same distinction applies to context/multimodal resource limits, KV Cache
eviction/reconstruction, stateful Messages support, file ownership/lifecycle,
compaction and historical request projection. Current no-replay rules in
[architecture](architecture.md), [Runtime strategy](runtime-strategy.md) and
[Vision integration](mtp-vision-integration.md) continue to govern execution;
they are not permanent prohibitions on investigating upstream-compatible designs.

### Completed milestone and current Multimodal Runtime objective

Conditional cached-image admission is implemented and narrowly qualified for
Flash-1017: fresh work remains bounded by **36,864 ViT patches**; historical
patch discount requires compatible complete paired coverage and pre-restore
layout validation. The independent [paired-byte bugfix](../artifacts/paired-byte-qualification/report.md)
removes the former 12 MiB projected-pair and 192 MiB retained-root qualification
ceilings, with finite native evidence, not a reopened milestone.
Independent source-byte, fresh-pixel, body, context and queue guards remain.
The post-completion [historical-image aggregate bugfix](../artifacts/historical-image-aggregate-admission/report.md)
removes cumulative covered-history pixel charging, with focused isolated native
evidence and a separately authorized [b2f7654 Production installation](../artifacts/production-multimodal-b2f7654/report.md).
This maintenance and bounded installed acceptance do not reopen phases.
The [cached-image admission report](../artifacts/cached-image-admission/report.md)
establishes twelve screenshot occurrences without overrides, not an image-count
ceiling or 600-occurrence simultaneous model execution.

The **Visual Feedback Loop milestone is complete within its demonstrated scope**,
not an ongoing demonstration campaign. [Gate F0](../artifacts/harness-files/gate-f0-report.md)
established bounded real-model official File-ID/nested-tool-result transport on
macOS. The [repaired loop](../artifacts/repaired-visual-loop/report.md) established
image-grounded correction and cached continuation, but remains **PARTIAL**: the
eighth transferred screenshot did not reach inference because conditional
paired-byte admission rejected it. [Capabilities](capabilities.md#applications)
owns the measured scope, residual visual defect and unresolved byte-guard/trace
retention evidence boundaries.

**Multimodal-Runtime is operationally complete and promoted to the existing
Flash-1017 Production installation.** The [project contract](multimodal-runtime-plan.md)
explicitly distinguishes achieved operation from the original broader C1–C9
qualification; the [completion record](../artifacts/multimodal-runtime/production-completion.md)
binds source, wheel, rollback and installed acceptance. Official Harness acquisition,
paired reuse, offload, automatic compaction, reconstruction and reacquisition pass
the finite C8 workload. No concrete implementation blocker remains within the
Thinking OFF Production envelope. C2/C7 and broader C6/C9 remain PARTIAL; no
historical failed attempt is relabeled. Same-process model reclamation, general
numerical equivalence and Windows Harness qualification do not block this closure.
Windows operation is post-closure work; no new phase/campaign is selected.

### Future official-source investigation priorities

These comparisons are retained technical context for
[closed phases](multimodal-runtime-plan.md#closed-phases-and-remaining-non-blocking-work), not a separate
project sequence or permission to start investigation, implementation, prototyping,
benchmarking or acceptance. Use these already reviewed reference snapshots, not
claims about current upstream HEAD or equality with installed packages:

| Project | Pinned revision |
| --- | --- |
| DeepSeek Harness | `687f282e8625c6b8d11ff58211a90524be2e4317` |
| DeepInfer | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| DeepSeek Recipe source review | `935817c66f5d6e64838f419692545f06787e485f` |

Pinned source links and packaged Recipe distinctions are retained in the
[Visual Feedback Loop source review](flash-1017-visual-feedback-loop-plan.md#2-historical-source-review-and-evidence-scope).
Future upstream-versus-local comparisons should distinguish completed cached-image
skipping/admission from recovery after a changed canonical projection:

- **A — Cached images versus reconstruction:** DeepInfer's
  `deepinfer/ve/multimodal_processor.py`,
  `deepinfer/model_executor/models/deepseek_vl.py` and
  `deepinfer/ve/prepare_tensor_input_vl.py` identify fully cached image spans,
  omit their binary payloads and skip active Vision work. Local
  `runtime/multimodal.py:prepare_multimodal/encode_new` and
  `serving/production_mtp.py` already supply metadata-first validation and
  fresh-only encoding after paired lookup. Compare remaining recovery semantics,
  not a missing cached-image skip. [Capabilities](capabilities.md#bounded-inline-images)
  owns the implemented conditional admission contract; the historical
  [Q1/Q2 envelope](flash-1017-visual-endurance-envelope.md) retains only its
  original geometry/admission and visual-quality scope.
- **B — Harness offload:** inspect
  `packages/compaction/compaction-image-offload/`,
  `packages/llm/llm/src/content.ts` and
  `packages/llm/llm-deepseek/src/images.ts`. Harness retains attachments and records
  `image/offload` selections, replacing selected request images with deterministic
  text placeholders. This changes the canonical prefix and may invalidate provider
  cache reuse. Audit integration with ds41f-mlx, not a competing image-history manager.
- **C — Harness compaction:** inspect `packages/compaction/compaction-basic/`.
  Harness already owns summarization-based history compaction. Determine what a
  compatible Runtime must support after official projection changes the canonical
  prompt; do not implement Runtime-side conversation compaction.
- **D — Paired Target/DSpark cache recovery:** compare official DeepInfer cache
  misses/recomputation with local paired integrity, distinguishing a complete hit,
  partial text-prefix hit, fully cached historical image span, partially covered
  historical image span, changed image or historical prompt, and complete miss.
  Any future replay/recovery must preserve Target/DSpark numerical consistency,
  canonical settlement and exactly-once tool-effect boundaries. Current paired
  integrity rules remain in force; this investigation does not authorize recovery
  implementation.

No automatic follow-on investigation is selected. General Harness compatibility
remains bounded at transport/storage and unsupported recovery boundaries, not a
missing normal offload/compaction reconstruction implementation. Official rc.2
offload does not recover arbitrary Runtime 400 errors. Independent Messages C=0
and whole-image-safe C>0 reconstruction is deployed and exercised through the
finite integrated workload. True image intersections and broader conditional-
discount reconstruction remain unqualified. No replacement history/cache owner
or follow-on campaign is introduced. The
[Q2 report](../artifacts/visual-feedback-loop/phase-4-q2-report.md) retains its
partial grounding and four-image continuity scope; it did not test seven images.

### Files and session lifetime

Files redesign and full implementation belong to a separate campaign. Prefer
volatile, session-associated attachment lifetimes over general durable storage
where official Harness semantics permit. The current W2 process-local Files store
is bounded, not the finalized lifecycle design; see [W2 scope](harness-image-files.md).
[Gate F0](../artifacts/harness-files/gate-f0-report.md) establishes limited real-model
File-ID transport compatibility, not general session ownership or storage
qualification. Session ownership, cancellation cleanup, fork behavior and expiry
synchronization are not established as implemented. Attachment-byte lifetime must
remain separate from executable paired-KV lifetime.

### Development sequencing and approval

**Official source inspection → local implementation comparison → concrete gap
identification → minimal design decision → focused verification → implementation
only when justified.** Before adding a feature, determine whether official
architecture already supplies it. Before changing a conservative guard, establish
the exact safety condition it protects. Before claiming compatibility, verify the
real installed package and target execution path.

Preserve scope-qualified evidence: historical PASS does not qualify a changed
implementation. Prioritize architectural compatibility and reduced long-term
maintenance over feature-specific workarounds. Performance regressions and
numerical correctness remain independent approval criteria. This documentation
policy changes no implementation, supported limits, classification or qualification.

## DeepInfer current disposition

DeepInfer has integrated official Flash-1017, Recipe 0.2 native Chat/Messages,
metadata-first resource-budgeted inline Vision and direct official Harness
text/read-only-tool continuation. Dedicated installed operation and acceptance are
recorded in [operations](operations.md#deepinfer-daily-operation-and-harness) and
the [development record](deepinfer-development.md). Historical Recipe/Messages
investigation deferrals are not current blockers; broader release qualification,
general Harness images/Files lifecycle, cancellation and dynamic tool updates are
not implied; later bounded Files/Vision evidence is scoped above.
The original-Flash release assessment below retains its own source/payload scope.

The bundled Browser Chat/Chat Server and application Vision/PDF workflows are
retired. The existing TUI is solely a Runtime operator console; no Web launch
controls remain. External clients own model-specific orchestration and tool
effects. No replacement Harness is bundled or qualified by this retirement;
successful inference is not comprehensive Agent/Harness compatibility.

## Production-serving closure (superseding authority)

**READY_FOR_SCOPED_RELEASE** for the current single-operator local / explicitly
trusted-private-LAN general runtime, assessed at implementation commit
`79e96fdd3d12165caf40a885ef3773a768efbcda`.
[Runtime completeness](runtime-completeness.md#general-local-runtime-release-assessment)
owns the supported scope, evidence and findings: no concrete blocker remains.
This supersedes the pending general-runtime assessment, not the unsupported
Internet/multi-user scope or historical singleton qualification boundaries.

`mtp-serving-v1` is now the production-semantics boundary. Paired target/DSpark
restoration and Scheduler/P5 settlement/publication/retirement are implemented for
the supported ordinary path. Generic recipe-owned tools, independent edits/branches,
long-context/output capability, logical sessions and versioned generation/frontend
controls are implemented. The original SP1 BLOCK and implementation hold are
historical; do not turn them back into current blockers.

The governing rule is to protect established runtime capability, not inherit
qualification-era restrictions. Start with the real model/runtime boundary, justify
current constraints and validate the changed serving boundary with real supported
workloads. The detailed [serving closure account](production-serving-parity-closure.md)
retains the original rationale, requirements and SP1 investigation; its historical
unresolved-contract and next-task sections do not reopen closed work.

## Closed foundations to preserve

| Boundary | Established asset, not a new work request |
| --- | --- |
| Standard-OFF core | First-party checkpoint/model/packed state/SSD Engram execution; generation, transaction, state-production and admitted-resource ownership |
| Prefill/continuation | DENSE_P0_P7/P7 architecture, same-live P5 terminal transfer, exact committed continuation and no hidden replay/repack |
| Ordinary MTP core | Qualified pinned oMLX physical execution topology; ds41f canonical/recipe/lifecycle integration, paired checkpoints and one Scheduler |
| API authority | Recipe conversion/encoding/parsing; ds41f security/capability/resource admission, not duplicate grammar |
| Application continuity | Canonical logical outcomes, opaque Scheduler pins, correlation fencing, completed-tool authority and conservative unknown-effect handling |
| Installed admission | Non-editable executable payload and exact semantic assets, separate host ABI compatibility; no checkout equality/local seals |
| Generation/frontends | Request-owned sampling, retained-session reasoning and explicit workflow snapshots; implemented Browser/TUI controls |

Do not rebuild these merely to make ownership look cleaner. The failed first-party
MTP execution line is not a production basis. MTP/decode physical architecture
remains pinned oMLX; prefill retains DwarfStar-derived topology with qualified MLX
numerics. Semantic/application/lifecycle ownership does not require a new execution
graph. Material unexplained candidate-performance regressions block architectural
replacement; recovering speed later is not an acceptable plan.

## Post-M48 production boundary and remaining work (bounded execution scope)

This heading is retained for existing roadmap links. It distinguishes core evidence
from whole-product completion; it is not the old milestone sequence.

The first-party OFF text core has real 200K and 512K/768K/near-1M evidence, including
continuation to 1,048,576 consumed tokens, cancellation/re-entry, idle persistence
and exact fresh-process restore. Resource repairs include model-lifetime allocator
cache/wired-residency ownership and retirement of passive P6 admission references.
Do not discard these boundaries or replace actual runs with configured maxima.

The separate bounded OFF image core uses the same packed state/generation stack,
with new-image-only encoding and exact identity/restore behavior. Its receipt alone
does not qualify MTP images; the separate [ordinary Vision→MTP closure](mtp-vision-integration.md)
now establishes that bounded ordinary path with real-image acceptance. Neither
qualifies larger multimodal context or universal visual quality. These text/image
results remain source-scoped core evidence, not a new long-HTTP/agent release claim. Limits and evidence links live in capabilities rather
than being recopied here.

## Runtime completeness and local default closure

Required DeepSeek-V4.1-Flash model/runtime core is **COMPLETE**; no current local
promotion blocker remains. `mtp-serving-v1` is the ordinary local default, with
explicit OFF fallback. The bounded GLM distribution promotion is also closed:
native is the M3 Ultra local MTP and standard-OFF GLM default, portable is explicit,
and native admission failure never changes profile. The
[bounded OFF publication](standard-off-native-publication.md) reuses the exact
qualified recipe artifact and closes its prior authority drift. No further kernel/performance campaign
is authorized by that packaging work. See [distribution admission](runtime-admission.md#glm-execution-distributions).
[Closure assessment](runtime-completeness.md) owns the
model/ds4/oMLX comparison, operational decision and final closed performance dispositions.
No new numerical execution topology or qualification campaign is required.

## Documentation hygiene / authority convergence closure

**COMPLETE; general-runtime release assessment is now closed separately.** Current
owners distinguish closed model/runtime core, ordinary MTP default and real
separate-machine trusted-LAN PASS from deferred capabilities, unsupported scope
and optional research. Known identified post-0.7.0 performance work and its
subsequent installed delivery are closed, not a pre-assessment backlog.

Historical Web gaps, inventory next stages and intermediate performance
recommendations retain provenance with supersession notes; they do not select
current tasks. No document was relocated and no immutable receipt was refreshed.
Repository-owned capacity/reasoning/cancellation/operator mocks are cleaned up;
40 focused tests pass across the four affected/session-owner modules. The two
previously stale session-owner cases already pass. Upstream full-VLM's undeclared
`openai_harmony` remains upstream/environment evidence, not repository fixture debt
or an established supported-lane production blocker.

This cleanup found **no new concrete production blocker**. It performs no new
qualification campaign, performance research, packaging/publication, full R1 or
release assessment. The subsequent assessment above evaluates current implementation
and its supported scope, not old closed work or historical TODO wording.

## Release disposition and deferred scope

1. **General-runtime assessment: closed / ready within scope.** Current API,
   installed admission/operation, ownership, failure containment, supported clients
   and observability were reviewed against implementation and retained acceptance.
   Separate-machine trusted-LAN PASS and the Aider omitted-output correction remain
   closed. No concrete production repair task was selected.
2. **Keep distinct decisions distinct.** Broader application features, hosts,
   context, security and concurrency need their own justified scope/evidence; they
   are not prerequisites for this single-operator release boundary. Implementation
   ownership feasibility does not authorize a transfer or create a release gate.
3. **Scoped publication executed separately.**
   [DS41F-SCOPED-LOCAL-20261015](../release/scoped-local-20261015/release.json)
   records the exact installed MTP/OFF payloads, fresh acceptance and publication
   through operations' existing route. Assessment and publication do not expand
   supported scope. No further implementation ownership work is selected;
   post-release maintenance needs an explicit new objective.

This is a priorities/decision boundary, not authorization to implement every
reference-server feature, run a broad benchmark ladder or create a new milestone
for each small issue. Existing scoped R1 runs remain facts about their inputs;
neither their existence nor deferring broader re-verification certifies later HEAD.

## Decision rule for production changes

1. Start from established model/runtime capability and supported clients.
2. Retain a restriction only for a concrete current correctness, state, resource,
   compatibility, security or explicit product-scope reason—not historical bounds
   or a successful fixture list.
3. Separate capability, application/operator policy and defaults. Maximum output
   is not a mandatory workload or allocation reservation.
4. Reuse qualified core evidence; validate the changed boundary. Return to core
   investigation only when concrete regression evidence warrants it.
5. Treat real supported client/workload failures as first-class evidence:
   identify owner → smallest repair → affected acceptance → return to operation.
6. Stop confidence-building when the concrete blocker is closed. Progress is
   removed production blockers, not probes, artifacts, tests or milestones.

Temporary absence from artifacts, packages or provisioning does not authorize
permanent capability removal, default/profile demotion or non-distribution;
those changes require an explicit decision with rationale, not inheritance of
the reduced artifact inventory.

Before structural replacement, name the blocker, explain why existing seams cannot
satisfy it, compare directly against preserved semantics/performance, and choose
the smallest justified change. Do not weaken official precision/state contracts,
invent tool completion, replay retained history or accept a second executable owner.

## Reference and release decisions

Official checkpoint/model/protocol semantics outrank implementation qualification
and the executable R1 oracle. A discovered reference defect requires investigation
and affected qualification, potentially a new reference version—not preservation
of a golden bug or rewriting immutable R1 material to fit a candidate.

Develop and qualify in `ds41f-mlx`. A future source projection is one-way and must
bind its exact payload, source/reference identities and independently qualified
receipt. Runtime-only semantic patches and two-way implementation lines are not
supported. A receipt qualifies its historical payload, not later development HEAD;
do not regenerate it simply to keep HEAD promotable.

The existing `ds41f-runtime` directory is unused old generation, not deployment or
accepted release history. Maintaining it is not current work. Source-clone operation
or package publication does not require a permanent separate release repository.
Detailed retained rules: [reference/promotion strategy](reference-release-and-promotion-strategy.md)
and [R1 contract](reference-release-r1.md).

## Planning guardrails

- Document structural consequences before changing architecture; preserve qualified
  topology unless a concrete boundary requires revision.
- Validate changed execution, state, protocol and resource owners at their scope.
  Synthetic seams do not substitute for affected model behavior.
- Record exact workload/runtime/asset/hardware identity for performance claims;
  do not sell source-pinned measurements as an installed deployment SLA.
- Keep native/reference, experimental/rejected selectors, current production and
  prospective capability explicitly distinct.
- Compilation, inspect, application acceptance, model qualification and release
  approval answer different questions. None is a substitute for another.
