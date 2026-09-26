# Performance status

This document records current performance-relevant conclusions without preserving the full benchmark diary.

## Accepted implementation lineage

The native attention runtime keeps the accepted fixed-tile lineage with split-K/ragged/width-one support and request-boundary compact topology. This lineage is the production direction for current native attention source.

MoE keeps the current gate/routing/shared-routed merge arithmetic and grouped expert pipeline. Engram keeps SSD-backed storage to avoid impractical resident-buffer memory use.

## Current native baseline

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

MLX-enabled native build/tests are now qualified in the current environment. Bounded full-checkpoint native execution is also qualified for the smoke scope: the official checkpoint opens, full prefill executes for the bounded fixture, and one-token generation executes.

The current short-context performance state is **measured but unqualified**. The baseline is trustworthy enough to guide the next optimization task, but it does not demonstrate practical decode performance. The dominant observed bottleneck for the current production path is **DECODE**.

## Historical baselines

Historical measurements remain useful only as archive/provenance context. They are not current native production qualification by themselves and are kept separate from the baseline above.

## Rejected optimization measurements

Rejected/deferred candidates include wide/rectangular attention alternatives, fused mHC decode candidate, performance-only decode graphs, and non-promoted late residency variants. These are not selectable as canonical production implementations.

## Unqualified areas

- short-context production latency/throughput: measured but not qualified as practical
- long-context behavior
- native HTTP serving performance
- release performance qualification

Do not run or cite benchmarks as release qualification unless the run records exact checkpoint, runtime, build, prompt/generation lengths, options, and hardware.
