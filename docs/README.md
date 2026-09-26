# ds41f-mlx documentation

This directory is the canonical current-state documentation for the runtime at HEAD.  It is organized by subsystem, not by development chronology.

## Canonical current-state docs

- [Architecture](architecture.md) — runtime structure and source layout
- [Correctness](correctness.md) — authority hierarchy and non-claims
- [Session state](session-state.md) — persistent, runtime-owned, and call-local state contract
- [Attention](attention.md) — SWA, compressed producer, reuse consumers, indexing, and production kernels
- [HC / MoE](moe-hc.md) — hyper-connections, routing, experts, and residency policy
- [Engram](engram.md) — SSD-backed store, MLX projection, and backbone integration
- [Generation](generation.md) — prompt, decode, logits, sampling, commit, stop/cancel lifecycle
- [API](api.md) — current HTTP/runtime surface and bridge status
- [Performance](performance.md) — current performance claims and unqualified areas
- [Provenance](provenance.md) — checkpoint/source/import identity and dependency policy
- [Qualification](qualification.md) — current qualification matrix

## Machine-readable classification

- [Documentation classification map](doc-classification.json) records how old prose was archived or absorbed.
- [Scaffold classification](scaffold-classification.md) records the current role of Python/native pre-import scaffold files.

## Historical archive

`docs/archive/` contains non-normative development history, validation closeouts, migration notes, and detailed historical references.  Archive documents preserve evidence context but are not required to understand current runtime behavior and must not override the canonical docs above.
