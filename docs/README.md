# ds41f-mlx documentation

Canonical current-state documents:

- [Runtime strategy](runtime-strategy.md) — production architecture and non-selected paths
- [Implementation plan](implementation-plan.md) — staged implementation status and current selector decisions
- [Architecture](architecture.md) — runtime components and ownership boundaries
- [Correctness](correctness.md) — backend-local fidelity policy and authority hierarchy
- [Session state](session-state.md) — persistent, runtime-owned, and call-local state contract
- [Generation](generation.md) — prefill/decode/commit/termination lifecycle
- [Operations](operations.md) — configuration, provenance inspection, launch, and unified qualification
- [API](api.md) — release-scope public HTTP contract
- [M19 local web client](m19-local-web-client.md) — browser client and client-side tool boundary
- [M20 oMLX release migration](m20-omlx-release-migration.md) — promoted exact upstream 0.7.0 MTP-OFF baseline and bounded qualification evidence
- [M24 operational soak status](milestone-24-operational-soak-status.md) — bounded real-agent/client tool-loop soak, persistence/restore, interruption recovery, and observability evidence
- [M25 MTP decision](milestone-25-mtp-decision.md) — pinned upstream lifecycle audit, real-model extraction failure, diagnostic A/B and continued-default-OFF decision
- [M30 recipe semantic preview](milestone-30-recipe-semantic-preview.md) — pinned parser audit and missing upstream preview API; protocol gate remains blocked and MTP remains OFF
- [M34 operational qualification](milestone-34-operational-qualification.md) — bounded guarded singleton soak/recovery qualified; native cancellation ownership-transfer defect fixed, production/public MTP remains disabled
- [M33 protocol qualification](milestone-33-protocol-qualification.md) — PROTOCOL_GATE_SOLVED for the isolated native singleton; M34 operational qualification authorized, production MTP remains OFF
- [M33 semantic horizon design](milestone-33-semantic-horizon.md) — strong unforwarded-terminal invariant and native initialization/chain/alignment ownership
- [M32 native recipe gate](milestone-32-native-recipe-gate.md) — full ARM64/OpenCV build and native parity pass; initialization commit edge blocks semantic MTP qualification, no M33 authorization
- [M31 recipe semantic session](milestone-31-recipe-semantic-session.md) — candidate upstream extension and source parity; real native build blocked, no live semantic MTP gate
- [Performance](performance.md) — qualified performance class and baselines
- [Provenance](provenance.md) — checkpoint/source/import/dependency identity
- [Qualification](qualification.md) — current qualified, optional, unqualified, and non-goal areas
- [Release qualification](release-qualification.md) — scoped release statement and regression matrix

Historical milestone documents and diagnostic notes remain in `docs/` as evidence. They are not rewritten to make development appear linear; the documents above are authoritative for current release claims.
