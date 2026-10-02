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
- [Performance](performance.md) — qualified performance class and baselines
- [Provenance](provenance.md) — checkpoint/source/import/dependency identity
- [Qualification](qualification.md) — current qualified, optional, unqualified, and non-goal areas
- [Release qualification](release-qualification.md) — scoped release statement and regression matrix

Historical milestone documents and diagnostic notes remain in `docs/` as evidence. They are not rewritten to make development appear linear; the documents above are authoritative for current release claims.
