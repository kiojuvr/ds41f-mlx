# P7 overlap/scheduling architecture

Status: **base implementation target defined for qualification**. P7 does not change the production selector, does not begin P8, and does not use throughput as an acceptance criterion.

## Qualified base backend

The qualified oMLX loader is invoked with:

```text
preserve_mtp=False
engram_ssd_offload=True
moe_expert_offload_resident_fraction=None
```

Pinned loading materializes ordinary model state with `mx.eval(model.parameters())` and `model.eval()`. Therefore P7's base backend is:

```text
FULL_RESIDENT_BACKBONE_SSD_ENGRAM
```

Classification for this backend:

- ordinary layer weights: resident / already ready;
- MoE expert weights: resident / already ready;
- Engram tables: SSD-backed / command scheduling required.

DwarfStar layer mmap/read-ahead, encoder expert page residency, and streaming expert acquire/release are **NOT_APPLICABLE_FOR_FULL_RESIDENT_BACKBONE** once admission proves that model/config condition. `SSD_READ_AHEAD`, `BEGIN_LAYER`, and `END_LAYER` are still handled by policy: they return `ALREADY_READY` / no-eviction under the proven resident policy rather than being ignored.

Admission for the base backend must fail closed unless all are true:

- no installed `OffloadedExpert` modules;
- no active `_moe_offload_plan`;
- loader config has no expert offload and has SSD Engram enabled;
- ordinary model parameters have passed loader materialization;
- Engram layers remain `DiskEngramEmbedding` in SSD mode.

## Optional backend explicitly outside base P7

Pinned oMLX also has a real alternative architecture:

```text
P7 optional backend: EXPERT_OFFLOAD
status: NOT QUALIFIED / OUTSIDE BASE P7
```

It changes weight storage, resident expert lifetime, materialization behavior, memory policy, and MoE execution. It must not be used to satisfy the base P7 residency criterion.

## Engram donor constraint

The base P7 Engram scheduler borrows the model-owned donor:

```python
language_model._engram_prefetch
```

It must not create or close a second `EngramPrefetch`. Request completion drains/releases the borrowed scheduling capability; model close remains responsible for donor `.close()`.

Pinned oMLX `EngramPrefetch.submit(embed, ids)` holds one exact pending request. `DiskEngramEmbedding.__call__(indices)` reuses it only when the later host IDs exactly match (`np.array_equal(requested, host)`). Therefore this is invalid for chunked ds41f execution:

```text
submit full 16384 IDs
layer1 consumes 8192-ID chunk
```

The ds41f adapter must submit the exact hash-ID slice that the next `ENCODE_ROWS` Engram consumer will request. Submitted IDs and consumer IDs are derived through the same `_slice_engram_hashes(...)` logic; there is no second history/hash authority.

Chunk pipeline:

```text
PREFETCH_ENGRAM0 before layer0
  submit exact first layer1 chunk
layer1 chunk0 consumes exact prefetch
  submit layer1 chunk1 immediately
...
PREFETCH_ENGRAM1 before layer2
  submit exact first layer14 chunk
layers2..13 overlap table1 first chunk I/O
layer14 chunks consume/submit-next
final drain
```

## Command stream authority

P7 scheduling is driven only by real `SweepCommand`s:

```text
PREFETCH_ENGRAM0
PREFETCH_ENGRAM1
SSD_READ_AHEAD
BEGIN_LAYER
ENCODE_ROWS
END_LAYER
```

The P6 custom `_segment_commands()` path emits the required P7 commands in the same authority-correct positions: table0 before layer0, table1 before layer2, and resident-policy read-ahead only for actually executed layers. Source-only P6 segments do not schedule decoder-layer weight work.

## Coordinator

One request-scoped owner coordinates scheduling:

```text
SchedulingCoordinator
  +-- EngramPrefetchController      ACTIVE
  +-- ResidencyPolicy               FULL_RESIDENT_BACKBONE admission/proof
  +-- ReadAheadPolicy               RESIDENT_ALREADY_READY for normal weights
  +-- MaterializationPolicy         ACTIVE async boundaries
  +-- SchedulingTelemetry           evidence only
```

The coordinator owns only scheduling state: active Engram table, expected chunk, pending donor request, consumer position, and request capability. It does not own Engram history, cache slot6, h/pre authority, publication frontiers, KV/index state, or P5 handoff state.

Lifecycle:

```text
runner created -> coordinator admitted
BEGIN_INVALIDATE -> request scheduling active
commands execute -> scheduling transitions follow commands
successful seal -> drain outstanding Engram request
P5 reservation/transfer or close/failure -> revoke and drain
```

No scheduling completion may mutate cache/publication/frontiers.

## Materialization and CPU boundary

P7 materialization mirrors oMLX donor intent with `mx.async_eval` around Engram SSD dependencies:

```text
before Engram host/SSD dependency: optional mx.async_eval(h_chunk, pre_chunk)
Engram lookup/combine
 after Engram incorporation: mx.async_eval(h_after_engram, pre_chunk)
```

It does not add unconditional per-layer `mx.eval` and does not call `mx.synchronize()` on the normal overlap path. P6's existing `P6_SOURCE_COMPLETE_AND_DETACH_CONE -> mx.eval(...)` remains a correctness/lifetime boundary and is not weakened by P7.

Forbidden on the production hot path: activation h/pre CPU round trips, KV/index/cache tensor CPU round trips, weight diagnostic conversions, and digest/list/NumPy conversion of model activations/state. Allowed as the explicit storage boundary: Engram lookup/hash IDs to host lookup representation and selected SSD Engram rows back to MLX.

## Acceptance evidence

Structural tests must prove command order, exact submitted IDs equal consumed IDs, donor exact-match behavior, mismatch fallback behavior, next-chunk submission after consumption, failure drain/revoke, stale coordinator rejection, resident admission fail-closed cases, and materialization boundaries.

Real qualification must compare P7 enabled with a scheduling-disabled control for state/correctness only: frontiers, cache geometry, Engram history, justified compressed KV/index equality, and P5 zero-replay behavior. Qualification evidence must include donor-backed prefetch submissions, exact-match consumptions, fallback synchronous reads, and drains. A run where prefetched work is discarded because IDs do not match does not qualify.

P8 graph reuse, custom Metal kernels, command-buffer fusion, and performance-driven graph changes remain out of scope.
