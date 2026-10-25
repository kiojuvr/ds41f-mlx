# ds41f-mlx documentation

Supported Runtime: package-fixed original **Flash** or **Flash-1017**, **MTP-only**,
Recipe **0.2** Chat/Messages. The operator application is the Runtime-only TUI.
Standard-OFF, Browser Chat, Web Tool execution and PDF application workflows are
retired. Historical acceptance does not qualify either variant at later HEAD.

## Current owners

| Need | Documentation |
| --- | --- |
| Installation, provisioning, operation and TUI | [Operations](operations.md), [operator console](operator-chat.md), [observability](operator-live-observability.md), [private LAN](private-lan.md) |
| Checkpoints, immutable packages and admission | [Capabilities](capabilities.md), [Runtime admission](runtime-admission.md) |
| MTP execution and Recipe 0.2 APIs | [Architecture](architecture.md), [ordinary MTP](mtp-production-serving.md), [API boundary](ordinary-api-boundary.md), [bounded singleton](mtp-local-release-candidate.md), [physical ownership](mtp-architecture-reset.md) |
| Logical sessions and physical caches | [Logical sessions](production-sessions.md), [session state](session-state.md), [generation](generation.md), [generation controls](generation-policy-investigation.md) |
| Numerical fidelity | [Correctness](correctness.md), [attention](attention.md), [HC/MoE](moe-hc.md), [Engram](engram.md), [prefill/P5](production-mlx-prefill-substrate.md) |
| Attribution and reproducibility | [Provenance](provenance.md), [third-party sources](../third_party/mtp/README.md), [reference R1](reference-release-r1.md) |
| Verification and release | [Source test baseline](verification.md), [qualification](qualification.md), [release review](release-qualification.md), [reference/promotion strategy](reference-release-and-promotion-strategy.md) |
| Measurements | [Public benchmark](public-performance.md), [performance interpretation](performance.md) |
| Development boundaries | [Implementation decisions](implementation-plan.md), [Runtime strategy](runtime-strategy.md), [scaffold roles](scaffold-classification.md) |

The [implementation plan](implementation-plan.md#completed-milestone-and-current-multimodal-runtime-objective)
owns the current Multimodal Runtime continuity objective after the completed,
scope-qualified Visual Feedback Loop milestone. [Capabilities](capabilities.md#applications)
scopes the later Gate F0 and repaired-loop evidence; neither is comprehensive
Harness qualification or authorization for follow-on implementation.

## Retained evidence

The [scoped release record](../release/scoped-local-20261015/release.json) binds
historical installed packages and acceptance, including retired OFF assets. Those
records, numerical specifications under `archive/validation/`, source attribution
under `archive/reference/`, and the hash-verified reconstruction source archive
remain for concrete proof/reproduction consumers—not as supported setup guidance.
Other unlisted reports and JSON receipts are source-scoped evidence, not current
contract owners. Obsolete milestone prose and editorial duplicates are available
in Git at `c1d1f716:<path>` rather than a new archive.

[Policy](documentation-policy.md) and the [small current-owner map](doc-classification.json)
define this distinction. See [tools](../tools/README.md) for surviving verification
entry points. No documentation cleanup constitutes inference, release promotion
or renewed qualification.
