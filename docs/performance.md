# Performance status

This document records current performance-relevant conclusions without preserving the full benchmark diary.

## Architecture interpretation

The current native attention, MoE, Engram, state, and generation paths are measured as the current correctness/reference runtime. Their components may be reused in production where architecturally justified, but this document does not promote the current native execution topology as final production architecture.

The current production-prefill candidate for Milestone 6 is the existing **dense P0-P7 DwarfStar-derived FP8/MLX path** with `P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, `P7_ENGRAM_TILE=2048`, and oMLX `GenerationBatch` MTP-OFF decode. P8 optimization search is complete: `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`; the tile-native implementation is retained as experimental/default OFF and is not production-selected.

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

Current production-candidate performance qualification is Milestone 6 and is now **qualified through 200K** for the dense P0-P7 FP8/MLX path. The measurement includes serving-path prefill to a committed live `DeepseekV41Cache`, P5 same-cache terminal bootstrap, first generated token, bounded decode throughput, memory, P6/P7/P5 evidence, and comparison to the recorded oMLX baseline.

M6 canonical result: `artifacts/m6/performance-qualification/result.json`. Decision: `M6_PERFORMANCE_QUALIFIED_200K`.

## Historical and external baselines

Historical measurements remain useful only as scoped archive/provenance context. They are not current native production qualification by themselves and are kept separate from the baseline above.

The supplied oMLX `0.7.0.dev2` official-checkpoint baseline on the target machine records about 176-194 tok/s prefill and 29-37 tok/s decode across 32K-200K contexts with about 292.8-293.0 GB peak memory. This is a production architecture comparison baseline, not a correctness authority.

Historical DwarfStar Q4 resident and ds41f DwarfStar-derived prototype measurements are scaling-shape and architecture evidence only unless their fixture, checkpoint representation, and semantics match the official runtime scope.

## Rejected optimization measurements

Rejected/deferred candidates include wide/rectangular attention alternatives, fused mHC decode candidate, performance-only decode graphs, non-promoted late residency variants, and P8 `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`. Tile-native carry structurally eliminated source/encoder dense carry writes/slices at 16384, but the ~0.57% warm median difference was inside run-to-run noise and the final-boundary cache proxy regressed by ~1.5 GB. These are not selectable as canonical production implementations.

## Unqualified areas

- long-session robustness beyond one long request
- native HTTP serving performance
- release performance qualification

Do not run or cite benchmarks as release qualification unless the run records exact checkpoint, runtime, build, prompt/generation lengths, options, and hardware.
