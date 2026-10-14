# ds41f-mlx documentation

## Start here

- [Project entrance](../README.md) — purpose and defaults.
- [Architecture](architecture.md) — components and execution ownership.
- [Capabilities](capabilities.md) — profile support, limits and exclusions.
- [Operations](operations.md) — installed setup, launch and retirement.
- [Implementation decisions](implementation-plan.md) — remaining priorities.

These are current owning pages, **not an at-HEAD qualification badge**.
Ordinary `mtp-serving-v1` is the local default; `standard-off` remains an explicit
fallback, distinct from bounded singleton. [Runtime completeness and promotion](runtime-completeness.md)
records core COMPLETE, no current local blocker and separated upstream performance work.
General-runtime release boundary assessment remains separate;
separate-machine trusted-LAN acceptance is **CLOSED / PASS** (real Windows → Mac,
with operator correlation), as recorded in capabilities. The old `ds41f-runtime` directory is
unused generation, not current deployment or accepted release history.

## Specialist contracts

| Responsibility | Owning pages |
| --- | --- |
| Execution and resources | [Runtime strategy](runtime-strategy.md), [installed admission](runtime-admission.md), [prefill/P5](production-mlx-prefill-substrate.md), [scaffold roles](scaffold-classification.md) |
| Numerical fidelity | [Correctness](correctness.md), [attention](attention.md), [HC/MoE](moe-hc.md), [Engram](engram.md) |
| Canonical state and generation | [Session state](session-state.md), [generation](generation.md), [policy implementation](generation-policy-investigation.md), [OFF live delivery](stateful-live-delivery.md) |
| HTTP and sessions | [OFF API](api.md), [ordinary recipe admission](ordinary-api-boundary.md), [ordinary MTP ownership](mtp-production-serving.md), [logical sessions](production-sessions.md), [bounded singleton/setup](mtp-local-release-candidate.md) |
| Application ownership | [Web application](web-application.md), [Browser interface](chat-interface.md), [Operator console](operator-chat.md), [policy snapshots](generation-policy-frontends.md) |
| Application resources | [Actual-request fitting](dynamic-capability-budget.md), [retained PDF workflow](pdf-document-workflow.md), [trusted private LAN](private-lan.md), [operator observation](operator-live-observability.md) |
| Proof and provenance | [Qualification](qualification.md), [performance interpretation](performance.md), [provenance](provenance.md), [release review](release-qualification.md) |
| Development/delivery decisions | [Serving closure](production-serving-parity-closure.md), [MTP physical doctrine](mtp-architecture-reset.md), [reference/promotion strategy](reference-release-and-promotion-strategy.md), [R1 guide](reference-release-r1.md), [future target](final-runtime-target.md) |

Status, startup and profile limits belong to their owners; historical PASS/BLOCK,
next-task paragraphs and old setup commands must not override these contracts.
R1 material/verifier, installed admission, source projection and release approval
are different authorities and proof scopes.

## Source-scoped evidence and history

The following links are **evidence/navigation**, not normative definitions or
instructions to repeat qualification. Results apply to their recorded inputs;
source changes need explicit affected-scope reconciliation, not inherited badges.

- Distribution assessment: [GLM native promotion](native-profile-promotion-assessment.md)
  — viable with bounded artifact/admission/selected-path work; no default or publication change.

- OFF cores: [200K](standard-off-200k-production-qualification.md),
  [very long context](very-long-context-production-qualification.md),
  [bounded multimodal](multimodal-production-qualification.md).
- Application receipts: [Tool Completion](tool-capability-completion.md),
  [PDF acceptance](pdf-document-acceptance.md), [TUI acceptance](operator-chat-status.md).
- Diagnostics: [MTP verification compute](mtp-verification-compute-investigation.md),
  [fresh startup](mtp-startup-investigation.md), [target dense prefill](target-prefill-investigation.md).
  Their old next-task/sealing/reproduction text is source-scoped history, not current setup.
- Historical milestones: [M42 closure](archive/development/milestone-42-reference-release.md),
  [M43 extraction](archive/development/milestone-43-release-repository-extraction.md),
  [M54R/M55R disposition](milestone-54r-55r-operational-release.md).
  None certifies later HEAD, the old runtime directory or broader ordinary release.
- [Superseded MTP source archive](archive/mtp-reconstruction/README.md) — preserved
  reconstruction/results, not a selected producer or instruction to resume that line.
- [Literal editorial editions](archive/editorial-baseline/README.md) — byte-preserved
  pre-compilation text, including old relative links; never live operator instructions.
- [Authority inventory](authority-inventory.md) — frozen pre-compilation baseline,
  not a regenerated current classification or full paragraph audit.
- [Environment consolidation](environment-consolidation.md) — adopted-payload relocation
  and retirement evidence, not latest-checkout publication or new acceptance.

Other retained milestone/diagnostic documents are discoverable through the complete
[classification map](doc-classification.json). Historical material need not be moved
or deleted to become non-normative; do not treat its old CANONICAL designation as current.

## Documentation policy

[Policy](documentation-policy.md) defines CURRENT / HISTORICAL / EVIDENCE / RELEASE,
explicit contextual evidence links and the static checker. Compilation/index closure
is an editorial result, not a model, installed payload, default-MTP or release decision.
