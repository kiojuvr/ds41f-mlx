# Attention runtime

The native attention subsystem consists of per-layer SWA, compressed producer layers, reused consumer layers, global/shared publications, indexer/candidate machinery, and fixed-tile Metal kernels.

## Topology

- Window attention uses a bounded sliding window KV cache.
- Compressed producer layers create compressed KV and associated index/query material.
- Index source layers publish K/index state for top-k candidate selection.
- Candidate sources produce top-k candidate sets.
- Candidate consumers and reuse regions read source-layer publications through `SharedAttention`.
- Reused layers combine local window state with source publications according to the model topology.

The exact layer map is represented in the imported native block/backbone implementation and provenance manifests.  Canonical behavior is publication-based: consumers read committed source state only.

## State and publication

Persistent attention state includes:

- SWA window KV per layer
- compressed KV from producer layers
- GlobalKV/source publications
- indexer K
- candidate/top-k lifecycle data
- compressor pending KV/score when compression ratio requires cross-call buffering

Reset clears state; fork copies committed state; continuation appends to it.  Invalid input must be atomic with respect to state mutation.

## Production implementation

Accepted production implementation includes:

- fixed-tile attention lineage
- packed work-list topology where enabled by the imported production path
- batched split-K kernels
- ragged tail kernels
- width-one path
- request-boundary compact topology
- KV quantization and compressed rope/index query machinery

Relevant sources:

- `native/src/attention/*`
- `native/src/cache/global_kv.cpp`
- `native/metal/attention/*`
- `native/include/dsv41/shared_attention.hpp`
- `native/include/dsv41/compressed_layer.hpp`
- `native/include/dsv41/reused_layer.hpp`

## Rejected historical alternatives

Rejected/deferred candidates such as wide/rectangular/packed experimental alternatives are not production-selectable in the canonical build.  They may appear in archive/provenance as historical performance work but are not part of the current architecture.

## Qualification status

Attention source and checkpoint-free tests are local.  Full MLX-enabled runtime execution requires an MLX-capable build environment and remains explicitly unvalidated in the current import environment.
