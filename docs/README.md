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
| Measurements | [oMLX External Benchmark](../artifacts/omlx-external-benchmark/report.md), [performance interpretation](performance.md) |
| Development boundaries | [Implementation decisions](implementation-plan.md), [Runtime strategy](runtime-strategy.md), [scaffold roles](scaffold-classification.md) |

The [Multimodal Runtime plan](multimodal-runtime-plan.md)
owns the operational completion contract and preserved broader C1–C9 limits.
**Operationally complete and promoted**: the
[completion record](../artifacts/multimodal-runtime/production-completion.md) binds
installed identity, acceptance and rollback. [Capabilities](capabilities.md#applications)
scopes finite Harness continuity evidence; Windows qualification is post-closure
operation. No further development phase or campaign is selected.

## Retained evidence

The [upstream performance audit](upstream-performance-audit.md) compares the actual
patched runtime with DwarfStar, oMLX through 0.7.4.dev1 and DeepInfer at the
2026-11-04 cutoff. It is investigation evidence and decision options, not an
approved optimization plan or dependency upgrade.

The [official Harness 0.2.2-alpha.2 Native Messages audit](../artifacts/harness-alpha2-compatibility/report.md)
preserves source/installed identities, the compatibility matrix, dynamic-tool gaps
and prior rc.2 execution evidence. It is source-scoped evidence and recommendations,
not alpha.2 E2E acceptance, startup authority or an approved implementation plan.
The later [official alpha.2 Web acceptance](../artifacts/harness-alpha2-e2e/report.md)
supplies bounded model-backed evidence for tools, Skills, `/goal`, manual
`/compact`, Files/images and text cancellation. [Capabilities](capabilities.md)
consolidates current status without rewriting the audit's historical findings.

The [official Harness 0.2.2-rc.1 campaign](../artifacts/harness-rc1-compatibility/report.md)
is **CLOSED — PASS within the verified Native Web / Native Messages compatibility scope**:
41 real-model requests and 15 exact-ID Tool pairs. Broader Plugin lifecycle
coverage remains partial; `tool_removal` remains unsupported and Session-log upload
remains disabled. This is not complete upstream feature parity.
Windows Desktop-specific behavior is outside this compatibility campaign and is not included in its verdict.
There is no pending Windows qualification gate or follow-on campaign.

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
