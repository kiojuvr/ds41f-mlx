# Production MLX prefill substrate

This page owns **the selected prefill facade and hot-path handoff boundary**.
[Architecture](architecture.md) owns profile execution;
[session state](session-state.md) owns coherent frontiers;
[runtime strategy](runtime-strategy.md) owns resource lifetimes. The
source edition (`c1d1f716:docs/archive/editorial-baseline/production-mlx-prefill-substrate.md.txt` in Git)
preserves the earlier practical one-chunk/oMLX-operation substrate description,
not current missing-work instructions.

## Production API

`ds41f_mlx.runtime.dwarfstar_prefill.DwarfStarMLXPrefillSession` is a compatibility
facade for **`DenseP0P7PrefillSession` / `DENSE_P0_P7`**, not the retained legacy
one-chunk implementation. It accepts the already-loaded profile model, creates a
fresh request-local cache, executes `DeferredPrefillAppend`, requires its commit
certificate and returns `DenseP0P7PrefillResult` containing the `LivePrefillResult`.
Empty prefix, invalid binding or forbidden production configuration fails closed.

The selected route is dense FP8/MLX P0–P7 with P6 committed state and P7 SSD-Engram
read-ahead overlap. OFF uses admitted first-party model operations; ordinary MTP
uses its pinned model/primitives within the Scheduler-owned topology. A historical
`omlx` class/configuration name does not select a donor OFF model. No portable
state export, full cache repack or reference-model execution belongs on this path.

## Handoff

P5 validates the commit/frontier, revokes the prefill producer and transfers the
**same live packed cache** plus the held-out terminal token exactly once to the
profile's sole generation owner. The terminal must be consumed once, not omitted
or decoded twice. Transfer is not another independent executor or cache copy.

- OFF: admitted target-generation ownership and consumed canonical settlement.
- Ordinary MTP: Scheduler's sole `BatchGenerator`/`GenerationBatch`; request-local
  prefill does not own a second Scheduler, prefix cache or speculative loop.

Historical `OMLXGenerationSession.from_prefilled_cache` is a compatibility API,
not proof that every profile runs the old donor BatchGenerator transaction.
`PrefillContinuationState` belongs to explicit portable/reference flows; production
must not export through NumPy and reconstruct a second cache for hot handoff.
Append/idle reuse must obey the owning P6/paired contract rather than prefill old
history or recover a missing pair by replay.

## Guard policy

The reference vertical-slice guard remains active. Production must not reach
`DwarfStarPrefillVerticalSliceExecutor`, `OfficialModelMath`,
`tools.run_m4_omlx_base_decode_qualification.build_prefill_state` or official-derived
validation `block()` helpers; reaching that route is
`PRODUCTION_PREFILL_REGRESSED_TO_REFERENCE_VERTICAL_SLICE`.

P7 overlap is required for the selected dense configuration. P8 tile-native carry
is rejected/forbidden for production; old one-chunk/reference serving is not a
fallback after admission/execution failure. Tool-only/native prototypes, build
success or selected numerical microbenchmarks do not alter this selector.

## Scope and provenance

This is the DwarfStar-derived dense topology implemented in real MLX with admitted
numerical/Metal primitives, not a claim that the historical native DwarfStar Metal
runtime is restored. The earlier page's general "performance mechanisms not yet
restored" statement does not describe present P6/P7/P5 ownership or prescribe a
native carry/graph rewrite. Remaining measured limits belong to
[performance](performance.md), not a speculative restoration list.

Source/checkpoint/reference authority, exact executable identity and compatible
host environment are separate [provenance](provenance.md) responsibilities.
[MTP doctrine](mtp-architecture-reset.md) protects its physical graph. Compilation
performs no real-model prefill, native/package rebuild, dispatch change or new
qualification.
