# Current implementation plan

This page owns present development priorities and decision rules, not a milestone
chronicle or optimization backlog. [Architecture](architecture.md) describes what
exists; [capabilities](capabilities.md) owns support/limits; [operations](operations.md)
owns deployment. Historical plans and receipts retain their original decisions,
not present authorization or a required sequence of new milestones.

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
