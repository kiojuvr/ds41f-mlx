# P7 overlap/scheduling architecture reconstruction

Status: **architecture reconstructed; implementation not started**. This document pins the P7 authorities and defines the implementation package, lifetimes, materialization policy, tests, and qualification sequence. It does not change the production selector, implement P8 graph reuse, introduce custom kernels, or use performance as an acceptance criterion.

## Authorities

- DwarfStar authority: `antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac`.
  - Use for execution/scheduling topology, read-ahead timing, expert/weight lifetime, and overlap intent.
  - Do not port GGUF formats, CUDA/Metal kernels, or native tensor math blindly.
- oMLX authority: local pinned checkout `/Users/kioju/omlx-0.7.0.dev2`, revision `b390b31e0c6831225fed0f24d278eb1db7fcb68b`.
  - Use for official FP8 model semantics, Engram implementation, SSD-backed Engram behavior, MLX operations, MoE/SwitchLinear structure, and available `mx.eval`/`mx.async_eval` mechanisms.

P7 is separate from P6. P6 defers decoder work and bounds carry lifetime; P7 schedules Engram I/O, weight/expert residency, read-ahead, and explicit materialization. Do not merge these features.

## DwarfStar P7 authority findings

### Engram prefetch commands 102/103 and SSD read-ahead command 104

The ds41f planner vocabulary names commands 102/103/104 as `PREFETCH_ENGRAM0`, `PREFETCH_ENGRAM1`, and `SSD_READ_AHEAD`. In the pinned DwarfStar source the real Engram overlap is implemented in `ds41_graph_prefill_sweep` rather than as externally dispatched enum cases:

- Before the layer loop, when `total_count >= 1024` and Engram prefetch is not disabled, DwarfStar starts table-0 prefetch for the whole new token range: `ds41_engram_prefetch_start(&engram_prefetch, g, 0, total_count)`.
- At `il == 2`, after layer 1 has had an opportunity to consume table 0, it starts table-1 prefetch for the same range.
- At Engram consumer layers (`il == 1` for table 0, `il == 14` for table 1), DwarfStar either waits until the needed rows are ready (`ds41_engram_prefetch_wait(..., off + count, ...)`) in pipelined mode or joins the whole prefetch before consumption.
- The prefetch thread reads Engram rows in 2048-row chunks, publishes `ready` only after a completed chunk, supports cooperative cancellation, and is always joined at layer/range end. It owns no model state; it fills a prefetch buffer that is copied into the active batch before Engram math.
- Duplicate issue is not an arbitrary queue: a single `ds41_engram_prefetch` object is reused. Table 1 replaces table 0 after table 0 has been consumed/settled by command order. The final join drains or cancels any remaining in-flight read.

Event timeline relative to command phases:

```text
setup/hash for new token range
PREFETCH_ENGRAM0 issue before BEGIN_LAYER 0
BEGIN_LAYER 0 / ENCODE_ROWS layer0 / SWAP / PUBLISH / END_LAYER 0
BEGIN_LAYER 1
  before Engram1 consume: wait/join table0 to required rows
  copy prefetched rows to batch; execute Engram1 + layer1
END_LAYER 1
PREFETCH_ENGRAM1 issue before BEGIN_LAYER 2
layers2..13 overlap with table1 read
BEGIN_LAYER 14
  before Engram14 consume: wait/join table1 to required rows
  copy prefetched rows to batch; execute Engram14 + layer14
END_LAYER 14
final range cleanup: join/cancel any active prefetch
```

Command 104 (`SSD_READ_AHEAD`) maps to DwarfStar's layer/weight streaming read-ahead rather than model semantics. On Apple, `metal_graph_stream_prepare_start_if_needed(..., il + 1, ...)` starts next-layer preparation while the current layer computes; `metal_graph_stream_prepare_join_layer` waits before using the current layer; `metal_graph_stream_prepare_join_all` drains at exit. This is scheduling-only and must be cancellable/drained. It is not a correctness requirement when the data is already resident.

### Expert/weight residency authority

DwarfStar has several distinct mechanisms:

| Mechanism | Source behavior | Classification | Portable intent |
|---|---|---|---|
| `ds41_encoder_acquire/release` | On Apple only, for long non-wide streaming prefill, borrows the configured expert-cache budget, mlocks encoder expert pages for layers0..19, disables streaming expert cache, and restores budget/release with `madvise(DONTNEED)` | memory-capacity policy + scheduling optimization | optional encoder-weight residency controller; not part of P6 deferral |
| Wide layer streaming prepare | Joins current-layer mapping, maps current layer, starts next-layer prepare when useful | scheduling optimization | overlap next layer weight readiness with current layer compute |
| Routed expert cache prefetch | Non-Apple path prefetches current/next expert tables for large rows | scheduling optimization / implementation detail | if backend exposes expert-cache APIs, prefetch selected tables; otherwise omit |
| `ds41_prefill_seed` | After a non-encoder-only layer's final chunk, reads selected routed experts from current mapped layer and seeds GPU expert cache with recent/high-frequency experts | scheduling optimization for following decode | optional expert-cache seeding after real routing exists |
| Encoder residency early return for wide carry | Wide prefill returns without full encoder residency | important separation | P6 wide/deferred carry is not an encoder-residency mechanism |

None of these owns persistent model state. They may improve I/O overlap or memory pressure but must not affect cache/frontier correctness.

## oMLX donor/capability map

| DwarfStar action | oMLX capability | Mapping decision |
|---|---|---|
| Engram SSD-backed tables | `DiskEngramEmbedding` in `storage.py`; mmap-backed `TensorFile`; `engram_ssd_offload=True` loader path | direct donor exists |
| Engram prefetch | `EngramPrefetch.submit/drain/forward`; one pending CPU read per model; `LanguageModel.__call__` submits first Engram before loop, submits second after first, uses `mx.async_eval(h, pre)` around Engram boundaries | direct donor exists, but current ds41f command runner does not use it |
| Engram read-ahead of unseen pages | `TensorFile._prefetch_pages` reads unseen mmap pages concurrently before gather | donor exists inside oMLX storage, not command-driven |
| MLX async materialization | `mx.async_eval` used in oMLX language Engram overlap and kernels | MLX primitive and donor examples exist |
| MLX eval/synchronize | `mx.eval` at loading/calibration/kernel drains; `mx.synchronize` in loading/memory cleanup | primitives exist; P7 must choose boundaries |
| MoE/expert implementation | `MoE` uses `SwitchLinear` / `QuantizedProjection`; no DwarfStar-style routed expert cache owner in the V4.1 language module | model semantics donor exists; residency controller required in ds41f if needed |
| Expert offload/residency | `moe_offload.py` and engine-pool admission estimate expert savings; not a command-level per-layer acquire/release API for this P6 runner | not directly portable; use as capacity/admission evidence only |
| Layer weight read-ahead | MLX lazy arrays and Python objects exist; no oMLX DwarfStar layer-map/read-ahead wrapper | requires ds41f scheduler or intentionally omit |

## Current ds41f P7 gap matrix

| Item | Current state | Classification |
|---|---|---|
| `PREFETCH_ENGRAM0/1` planner command names | Present in `SweepCommandKind`; P6 custom plan does not emit them; runner falls through to `arena.apply_command` only | PLANNER_ONLY / CONNECTED_BUT_NOOP if emitted |
| `SSD_READ_AHEAD` | Present in planner vocabulary; no runner read-ahead behavior | PLANNER_ONLY / CONNECTED_BUT_NOOP if emitted |
| `BEGIN_LAYER` / `END_LAYER` scheduling | Runner records/applies commands, no scheduling coordinator action | CONNECTED_BUT_NOOP |
| Expert residency owner | No real ds41f P7 owner; no acquire/release lifecycle | MISSING |
| `MlxEvaluationPolicy` | `evaluate_after_batch=False` default; optional `mx.eval(*batch_values)` and optional synchronize after complete batch | CONNECTED_AND_ACTIVE only when explicitly enabled; default inactive |
| `mx.async_eval` in P3-P6 production runner | No command-runner use | MISSING |
| `mx.eval` production boundaries | `MlxEvaluationPolicy` optional; P6 source boundary explicitly materializes persistent source/cone for lifetime correctness | CONNECTED_AND_ACTIVE at P6 correctness boundary |
| P6 source-boundary eval | Required lifetime/correctness boundary; must remain explicit | CONNECTED_AND_ACTIVE |
| P5 handoff materialization | Validates cache/frontiers and transfers same cache; no replay/repack | CONNECTED_AND_ACTIVE |

## Proposed P7 package

One owner should coordinate all scheduling lifetimes:

```text
SchedulingCoordinator
  +-- EngramPrefetchController
  +-- ReadAheadController
  +-- WeightResidencyController
  +-- MaterializationPolicy
  +-- SchedulingTelemetry
```

The coordinator is request-owned and capability-scoped. It receives the same command stream as the block runner. It may prepare/prefetch/async-evaluate, but it must not publish model state, mutate cache frontiers, or outlive the append/session capability.

## Engram lifecycle

States:

```text
IDLE -> ISSUE(table, ids, rows) -> IN_FLIGHT -> CONSUMER_READY(rows)
     -> CONSUME(copy/use rows) -> RELEASE
     -> CANCELLED/FAILED -> DRAINED
```

Rules:

- Ordinary sweep: issue table0 before layer0; wait/consume at layer1; issue table1 after layer1/before layer2; wait/consume at layer14; drain at sweep end.
- P6 source-only sweep: layers0..19 execute, so both Engram layers are real consumers; same lifecycle applies. No decoder-layer Engram scheduling is needed because there are no Engram layers after 19 in V4.1.
- P6 completing sweep: same Engram lifecycle for the new encoder range; decoder suffix scheduling must not re-read Engram for skipped old decoder rows.
- Ordinary one-token tail: controller may choose no prefetch when below threshold; direct synchronous oMLX read remains correct.
- Failure/cancellation: cancel/drain prefetch before request capability is released. Stale completion must not mutate a new request; prefetched buffers are request-owned or embedded only while guarded by the same controller.

The live cache/history remains the only model-state authority. Prefetch is an I/O staging optimization.

## Weight/expert residency lifecycle

States:

```text
UNPREPARED -> READ_AHEAD_IN_FLIGHT(layer) -> READY(layer)
            -> ACQUIRED(layer/expert-set) -> CONSUMED -> RELEASED/EVICTED
            -> CANCELLED/DRAINED
```

Policy:

- Layer weights may be readied after `END_LAYER L-1` or once the command stream proves `BEGIN_LAYER L` is next.
- Current-layer readiness must be joined before executing that layer.
- Next-layer read-ahead may overlap current-layer compute but may not force synchronization merely for telemetry.
- Encoder and decoder differ only by command reachability: P6 source-only sweeps must not schedule layers20..39. Completing sweeps schedule only actually executed decoder suffix layers.
- Expert cache seeding may occur only after routed expert selections exist for a real layer/chunk, and it must be treated as scheduling state, not persistent model state.
- Full model residency is not assumed. Memory-capacity policy should prefer bounded, request-scoped acquisition and release.

## Materialization policy

| Boundary | Operation | Reason |
|---|---|---|
| Setup/model load | existing loader `mx.eval(model.parameters())`; no P7 change | checkpoint readiness |
| Prefetch issue/read-ahead issue | no `mx.eval`; CPU/file scheduling only | overlap, no model tensor result required |
| Command batch normal execution | optional `mx.async_eval` on selected live tensors; no unconditional eval | overlap without draining graph |
| Publication | no eval unless publication owns persistent tensor whose producer may otherwise be retained incorrectly | separate correctness from telemetry |
| P6 source-complete boundary | `mx.eval`/materialize as currently required | lifetime correctness: persistent source/cone must survive parent retirement |
| Final seal | no new eval beyond existing validation needs | expose sealed cache/frontiers |
| P5 handoff/bootstrap | bootstrap forward naturally materializes active decode state; no prompt replay | transfer authority |
| Diagnostics | explicit `mx.eval`/`mx.synchronize` allowed behind tools | evidence only, not hot path |

Correctness boundaries (P6 source materialization, final ownership transfer) must not be weakened for overlap. Overlap boundaries (`async_eval`, prefetch, read-ahead) must not introduce unnecessary synchronization.

## Failure and cancellation policy

- Engram/read-ahead/expert tasks can be abandoned only if their worker/future is cancelled or drained before closing request capability.
- No scheduling task may write public cache frontiers or publication state after owner revocation.
- Stale completions may fill only request-private staging buffers still owned by the revoked coordinator; they must be discarded on drain.
- If a required current-layer readiness wait fails, the append/session fails closed and the cache remains inadmissible.
- P5 reservation/transfer forbids further scheduling work by the prefill runner.

## Structural acceptance tests before implementation

Required evidence:

1. `PREFETCH_ENGRAM0/1` commands perform real prefetch work and are consumed at Engram layers.
2. `SSD_READ_AHEAD` / layer read-ahead performs real work or is explicitly omitted by backend policy, not counted as complete.
3. Expert residency acquire/release performs real acquire/release under a fake/recording backend.
4. P6 source-only sweep does not schedule skipped decoder layers20..39.
5. Prefetch lifetime follows command order and is drained/cancelled on failure.
6. No stale request can publish state or mutate a new request.
7. Materialization occurs only at declared boundaries.
8. No CPU tensor round-trip on hot path.
9. P5 remains same-cache zero-replay.

Telemetry counters alone are not sufficient; tests must prove real controller actions against recording/donor-backed mechanisms.

## Qualification sequence

Before any throughput benchmark:

```text
structural fake/recording tests
-> bounded real 2K
-> bounded real 8K
-> P6 16K complete
-> P6 pending boundary
-> A-24577
```

Use 49K only if scheduling lifetime requires it. P7 is complete only when the structural gate is satisfied. P8 graph reuse, custom Metal kernels, command-buffer fusion, and performance-driven tuning remain deferred.

## Implementation order after review

1. Add `SchedulingCoordinator` interfaces and recording controllers; connect to command runner without changing semantics.
2. Implement Engram prefetch controller using oMLX `EngramPrefetch`/`DiskEngramEmbedding` where available.
3. Implement materialization policy with explicit `async_eval`/`eval` boundary declarations.
4. Add read-ahead/weight-residency recording backend and backend policy hooks.
5. Add real bounded qualification tools and only then evaluate overlap effects.
