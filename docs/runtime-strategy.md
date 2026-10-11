# Execution selection and resource lifetime

This specialist note explains production selector and resource-lifetime rules.
It is not another global status, roadmap or release document. Start with
[architecture](architecture.md), [capabilities](capabilities.md) and
[operations](operations.md); future changes follow [implementation decisions](implementation-plan.md).

## Selected execution

| Boundary | Selected owner / rule |
| --- | --- |
| Protocol | Official `deepseek-recipe` conversion, encoding and response processing |
| Prefill | DENSE_P0_P7 dense FP8/MLX, DwarfStar-derived topology; P7 SSD-backed Engram with tile 2048 |
| Terminal handoff | P5 revokes the prefill producer and supplies the held-out terminal once |
| Standard-OFF | First-party model execution, `TargetGenerationSession`, `TargetForwardTransaction` and `DecodeStateProducer` |
| Ordinary MTP | Pinned oMLX physical MTP topology, Scheduler-owned batch, ds41f paired-state and canonical/recipe lifecycle integration |
| Native C++ core | Reference/qualification and selectively reusable components, not the production selector |

OFF and MTP are separately admitted installed payloads, not dynamic request switches
or fallback backends. OFF first-party math/storage does not imply rebuilding MTP's
proposal/verification/rollback graph. An architecture change must preserve actual
state and performance contracts, not promise to recover them after a rewrite.

The old one-chunk prefill, reference vertical-slice serving and P8 tile-native carry
remain diagnostic/experimental, not selectable production defaults. Their build,
fixture or historical throughput results grant no promotion. Source attribution,
R1 fixtures and old engines retained for comparison are not active fallback paths.

## OFF model-lifetime resource policy

`runtime/resource_admission.py:AdmittedResources` binds model/configuration,
admitted numerical payloads and concrete checkpoint/tokenizer/SSD resources.
Execution requires a live capability; retirement revokes it. Changed/unadmitted
resources reject entry, and an invalidated transaction cannot publish continuation.
Installed admission and host compatibility are detailed in
[runtime admission](runtime-admission.md), not inferred from directory names.

The OFF lifetime owns two process-global MLX policies **before model allocation**,
through dense prefill, generation and idle P6:

- Free allocation cache is at most **32 GiB**, never increasing a caller's smaller
  limit. This bounds reusable freed buffers, not live model/KV state.
- GPU residency uses the device-recommended wired budget for the entire model
  lifetime, not only an active decode lease. Prefill/idle work must not depend on
  a previously retired generation owner for resident weights.

Overlapping OFF allocator-policy owners reject. Retirement/load-failure cleanup
synchronizes and restores previous settings; ownership is not silently transferred
when cleanup fails. No per-token cache clear, executable-KV eviction, numerical
change, second state owner or promise of arbitrary working-set feasibility follows.

P5 also retires the **passive** P6 setup/certificate backlink after transfer or burn.
It was admission evidence while ready, not executable authority after revocation.
Dropping that cycle releases old publications without waiting for cyclic GC; it
does not alter live packed state, diagnostic aliases or one-shot transfer guards.

## Continuation and images

OFF returns the exact coherent live list to idle continuation; validated same-backend
artifacts remain dormant until admitted restore. MTP reuses complete immutable
paired checkpoints under its Scheduler/cache rules, never an OFF artifact or live
ring rewind. Detail: [state contracts](session-state.md).

The bounded OFF image path preserves this execution topology. CPU preprocessing
binds bytes and expanded spans; the worker encodes only new images, injects sparse
absolute embeddings and slices VL/Engram masks per command. Idle artifacts contain
image identities/packed state, not pixels or encoder activations. Restore and
cancellation materialize/drain protected work before publication or fence release.
Image limits and profile exclusions live in capabilities, not a second strategy.

## Evidence versus selection

Long-context resource repairs and rejected selector experiments are recorded in
source-scoped reports. They are not startup dependencies or today's benchmark.
Use [qualification interpretation](qualification.md) and
[performance interpretation](performance.md); native build success, an old PASS
or a configured maximum does not select a production path or qualify later HEAD.
