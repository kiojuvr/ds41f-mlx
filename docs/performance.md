# Performance status

This document records current performance-relevant conclusions without preserving the full benchmark diary.

## Architecture interpretation

The current native attention, MoE, Engram, state, and generation paths are measured as the current correctness/reference runtime. Their components may be reused in production where architecturally justified, but this document does not promote the current native execution topology as final production architecture.

The current production-prefill candidate for Milestone 6 is the existing **dense P0-P7 DwarfStar-derived FP8/MLX path** with `P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, `P7_ENGRAM_TILE=2048`, and ds41f `TargetGenerationSession` MTP-OFF decode (M44). The earlier GenerationBatch measurements below are historical comparison evidence; M44 records fresh matched measurements for the changed engine. P8 optimization search is complete: `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`; the tile-native implementation is retained as experimental/default OFF and is not production-selected.

## Post-`2f928cd` target dense prefill

The [real first-party investigation](target-prefill-investigation.md) separates
short-shape occupancy, the 8192-token P6 work-elimination transition, source/tail
amortization, lazy GPU debt and actual native/portable dispatch. The native baseline
rises from 226 tok/s at 64 tokens to 525 just below 8192 and about 930 at 64K.
Dense MXFP8/GEMM and MoE variant controls reject a broad leaf-kernel campaign.
Reusing normal target packed attention in P6's 127-old-row suffix geometry
measures about 1.85 s headroom, but changes decoder-window bytes and a paired
32-token decode is 2.3% slower. The candidate is **not selected**, despite passing
native state/restore and full OFF R1 checks. No production execution changes:
planner, precision, source/cache ownership, resources and donor/MTP remain
unchanged. See the investigation for dispatch loss, affected qualification and
the subsequent optimization seam; a prefill gain is not permission to accept a steady regression.

## Post-`d006e1f` R1 verification compute investigation

The [bounded real-model investigation](mtp-verification-compute-investigation.md)
re-establishes OFF/MTP R1 independently and maps four-row verification, expert-local
occupancy, packed-attention dispatch and head costs. Dense MXFP8 already uses wide
few-row kernels; sparse experts average 1.35 active rows. A private BF16 head
prototype gives 3.54× microkernel speedup but only about 0.8–1.7% modeled request
compute gain. The highest-value next boundary is fresh DSpark context materialization
and P5 startup, not a general few-row MMA kernel campaign. No production dispatch,
Web, qualification scope or release/promotion changes were made.

## Post-M48 first-party 200K production baseline

Fresh core evidence is recorded in [standard-OFF 200K production qualification](standard-off-200k-production-qualification.md)
and `artifacts/standard-off-200k/summary.json`: real first-party checkpoint/model
execution, 200K prefill followed by sustained decode/continuation, idle persistence,
fresh-process exact restore, cancellation/re-entry and bounded allocator resources.
This closes a long-session free-allocation retention defect with a model-lifetime
32 GiB cache budget, not arithmetic/kernel changes. Historical M6/M20 endpoint
measurements below are comparison evidence, not substitutes for this current run.

## Very-long-context supported text core

The [fresh very-long qualification](very-long-context-production-qualification.md)
uses actual first-party OFF prefill/decode/continuation and fresh exact restore,
not configured-length admission. 524,288 / 786,432 / 1,040,090 initial contexts
prefill in 762.91 / 1,310.01 / 1,947.39 s (687 / 600 / 534 tok/s), with sustained
decode approximately 17.30 / 16.57 / 16.07 tok/s. Actual final frontier reaches
**1,048,576**. This is the measured supported checkpoint text-core ceiling, not
physical memory exhaustion or qualification beyond the checkpoint context range.
Initial ingestion is batch/maintenance-class; 2K suffix bootstrap remains about
7.6 / 9.1 / 10.5 s. No separate suffix optimization was required.

512K exposed OS compression/jetsam with the default zero wired budget during
prefill. The admitted lifetime now owns recommended GPU residency in addition
to the unchanged 32 GiB allocator-cache budget. The same 512K fixture and higher
frontiers qualify after repair: peak MLX active 320.44 GB, sampled active plus
cache 348.60 GB, minimum sampled system headroom 96.41 GB, no corrected swap growth.
P5 also retires a passive certificate/setup cycle that retained old source buffers
until GC. Identical-token/all-slot requalification keeps idle allocation growth
near 5 MB instead of gigabytes across these turns; no live state is discarded.
See `artifacts/very-long-context/summary.json` for lifecycle, SSD and persistence
costs/limits. No new release, R1 or runtime promotion is implied.

## Historical native reference baseline

The first current post-import native performance baseline is recorded in:

- `artifacts/performance/native-short-context-baseline.json`

Harness: `native/tests/qualification/native_perf.cpp`, target `dsv41-native-perf`.

Measured identity:

- git commit tested: `b9ac930bc0112f55684a7bdbda7dac463176a9ae`
- checkpoint revision: `dba1be0a40aa45a94ad051997016db3960a90277`
- checkpoint index SHA256: `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- MLX version: `0.32.2`
- hardware: Mac Studio M3 Ultra, `Mac15,14`, 512 GiB unified memory
- Engram mode: SSD-backed production Engram metadata/store path
- selector state: current production defaults recorded in the artifact

Timer methodology: benchmark timers use `std::chrono::steady_clock`; the runtime forward paths already evaluate hidden/pre-mix tensors at state publication boundaries, and the benchmark additionally evaluates the sliced logits with `mlx::core::eval` at prefill and every decode step before stopping timers. Model construction is excluded from prefill/decode timing.

Current measured baseline, one measured run per case because full-model runs are expensive:

| Case | Time | Throughput |
| --- | ---: | ---: |
| model construction/load to ready | 137.294 s | n/a |
| prefill 128 tokens | 5.193 s | 24.647 tok/s |
| prefill 512 tokens | 12.554 s | 40.783 tok/s |
| prefill 1024 tokens | 21.963 s | 46.623 tok/s |
| prefill 2048 tokens | 40.774 s | 50.228 tok/s |
| prefill 4096 tokens | 80.881 s | 50.642 tok/s |
| prefill 8192 tokens | 160.480 s | 51.047 tok/s |
| decode 1024 prompt + 32 generated | 103.342 s | 0.310 tok/s total |
| decode 1024 prompt + 128 generated | 409.580 s | 0.313 tok/s total |

Decode first-token latency after prefill was about 3.03 s in the 32-token run and 3.05 s in the 128-token run. Subsequent decode throughput was about 0.31 tok/s.

Memory observations from the same artifact:

- after model construction: process footprint about 302.0 GB; MLX active about 300.1 GB; no swap reported
- final: process footprint about 318.2 GB; MLX peak about 303.4 GB; no swap reported

Engram-specific read/page counters are not currently exposed through a non-invasive production interface, so this baseline records synchronized wall-clock behavior and memory telemetry only.

## Current qualification interpretation

MLX-enabled native build/tests remain qualified as reference evidence. The old native ~51 tok/s prefill / ~0.31 tok/s decode path is preserved historically as a **native reference runtime**, not the current practical production candidate.

Historical Milestone 6 performance qualification was **qualified through 200K** for the dense P0-P7 FP8/MLX path. The measurement includes serving-path prefill to a committed live `DeepseekV41Cache`, P5 same-cache terminal bootstrap, first generated token, bounded decode throughput, memory, P6/P7/P5 evidence, and comparison to the recorded oMLX baseline.

Historical M6 result: `artifacts/m6/performance-qualification/result.json`. Decision: `M6_PERFORMANCE_QUALIFIED_200K` against dev2. M20 retains the unchanged prefill/context-ladder scope and freshly confirms the endpoint against exact upstream 0.7.0: `artifacts/m20/release-200k-endpoint.json` reports 242.88 s prefill, 19.00 tok/s decode, 18.73 tok/s live continuation, 319.79 GB MLX peak, zero replay/repack and coherent all-layer frontiers. No intermediate ladder was rerun.

## Historical and external baselines

Historical measurements remain useful only as scoped archive/provenance context. They are not current native production qualification by themselves and are kept separate from the baseline above.

The supplied oMLX `0.7.0.dev2` official-checkpoint baseline on the target machine records about 176-194 tok/s prefill and 29-37 tok/s decode across 32K-200K contexts with about 292.8-293.0 GB peak memory. This is a production architecture comparison baseline, not a correctness authority.

Historical DwarfStar Q4 resident and ds41f DwarfStar-derived prototype measurements are scaling-shape and architecture evidence only unless their fixture, checkpoint representation, and semantics match the official runtime scope.

## Rejected optimization measurements

Rejected/deferred candidates include wide/rectangular attention alternatives, fused mHC decode candidate, performance-only decode graphs, non-promoted late residency variants, and P8 `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`. Tile-native carry structurally eliminated source/encoder dense carry writes/slices at 16384, but the ~0.57% warm median difference was inside run-to-run noise and the final-boundary cache proxy regressed by ~1.5 GB. These are not selectable as canonical production implementations.

## Release performance sanity

The scoped release relies on the dense P0-P7 performance class already qualified through 200K, practical ds41f-owned target decode above the interactive floor, EOS termination avoiding unnecessary post-answer generation, and same-backend persistence/restore that is operationally small relative to model load/inference.

Operator-facing performance claims are regression class claims, not a new benchmark competition. Do not broaden them beyond the recorded checkpoint/runtime/hardware/config provenance.

## M20 dependency A/B

Same MTP-OFF configuration on the target: dev2 median synchronized decode
20.25 tok/s; release 19.65 tok/s initially and 20.17 tok/s on repeat. Load was
63.22 s versus 69.02/62.69 s; median continuation bootstrap ~0.169 s on both.
No reproducible meaningful regression was found. Current RSS grew ~0.28 GiB
across 12 follow-ups on each side, with coherent state growth and empty retired
generators. See [M20](m20-omlx-release-migration.md) for timing scope, per-turn
trend, native/package identity and bounded-session limits.

## Unqualified areas

- batch throughput serving;
- general MTP/DSpark/speculative performance claims beyond the bounded R1 investigation;
- non-target hardware performance;
- vision/multimodal performance;
- cross-runtime KV portability costs.

Do not run or cite benchmarks as release qualification unless the run records exact checkpoint, runtime, build, prompt/generation lengths, options, and hardware.
