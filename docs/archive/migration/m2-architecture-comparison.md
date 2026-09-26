# M2 architecture comparison and first bounded measurement

M2 does not port DwarfStar and does not begin with kernel optimization.  The
first question is whether the known-good oMLX production path shows a structural
performance gap against a concrete upper bound that cannot be reasonably
explained by DwarfStar's different weight precision alone.

## Baseline identities

- oMLX runtime baseline: `$HOME/omlx-0.7.0.dev2`
- pinned upstream revision: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
- local known-good oMLX patch identity: recorded by M0/M1 artifacts
- official checkpoint: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`
- known-good settings include `mtp_enabled=true`, `deepseek_v41_engram_ssd_offload=true`, `dflash_enabled=false`, `vlm_mtp_enabled=false`, `moe_expert_offload_enabled=false`
- DwarfStar donor: `$HOME/ds4`

M0, M0.5 and M1 remain closed.

## Structural comparison summary

| Area | oMLX known-good path | DwarfStar path | M2 implication |
| --- | --- | --- | --- |
| CED / CSA2 prefill topology | MLX/Python module graph with DeepSeek-V4.1-specific packed attention, indexer, HC, Engram and MoE kernels. | Explicit C/Metal layer-major prefill sweep, carry buffers, suffix handling and state publication. | Candidate family, but only if scaling-shape evidence shows an oMLX structural gap. |
| Decode M=1 | Production server path with embedded DSpark/MTP active under known-good settings. | V4.1 docs state DSpark/pipeline are not implemented for V4.1; target-only comparisons are confounded. | Do not replace oMLX DSpark with DwarfStar target-only decode. |
| Routed MoE | Gate/top-k plus short grouped expert and sorted prefill paths; paired gate/up and fused reductions where eligible. | Expert/cache scheduling is a first-class stage with resident/streaming assumptions and explicit work buffers. | Possible later candidate; not first without MoE-specific dominance evidence. |
| Expert-major work lists | oMLX sorts large prefill expert rows and uses grouped short decode/verify paths. | DwarfStar makes routed batches and ownership/cache behavior explicit. | Compatible conceptually, but DwarfStar quantized expert kernels are not directly portable. |
| Metal fusion granularity | Targeted custom kernels around packed attention, indexer/top-k, HC, grouped expert and sorted-combine. | Broader stage fusion around attention output, HC, routed/shared FFN, carry/index publication. | Investigate structural scheduling before individual kernels. |
| Command-buffer/submission topology | Mostly MLX lazy graph/eval boundaries; some explicit async eval for Engram prefetch. | Explicit begin/end/drain boundaries and staged command submission. | Candidate only if measured overhead/scaling shape supports it. |
| Attention / CSA2 | Compressed KV, candidate blocks, packed sparse attention, packed index scores/top-k. | Raw window plus compressed KV publication, indexed mixed batch attention and candidate filtering. | Semantics align; state publication differs. |
| Indexer topology | Packed index scores/top-k and candidate path via MLX/custom kernels. | Batched projections, packed Q/K, batched top-k for large visible rows. | Candidate only if indexer dominates or scaling evidence points there. |
| Weight layout/lifetime | Official safetensors and official precision; Engram SSD-backed. | GGUF Q2/Q4/imatrix recipes; Engram disk-only. | DwarfStar layout/quantization is not directly promotable. |
| Residency / Unified Memory | MLX-managed arrays/lifetime; M0.5/M1 known-good path uses no MoE offload. | Static context buffers, aliasing, carry compaction, explicit mapping/streaming. | Lifetime ideas may transfer; weight-format assumptions must not. |
| Temporary materialization/copies | MLX arrays for cache merge/extract, sorted routed inputs, inverse/order and outputs. | Explicit scratch aliasing and carry buffers. | Candidate if bounded evidence shows materialization overhead. |
| Cache/state publication | `DeepseekV41Cache` extraction/merge per row through MLX path. | Explicit raw window, compressed KV, index cache, carry and suffix publication. | Most plausible difference behind prefill scaling shape. |
| Prefill scheduling | Production oMLX sustains known long-context prefill around 180-190 tok/s. | Q4 resident reference shows ~340 tok/s at +4K and ~640-716 tok/s at +8K/+16K. | Absolute gap is not enough because precision differs; scaling shape is the key signal. |
| Synchronization boundaries | Mostly implicit through MLX/server timings. | Explicit command-buffer and CPU/GPU overlap boundaries. | Candidate only after scaling/overhead evidence. |

## Candidate families

1. **Layer-major prefill / state-publication topology** — leading hypothesis,
   not selected for implementation until scaling-shape evidence is sufficient.
2. **Decode M=1 submission/static-buffer topology** — lower priority because
   oMLX DSpark/MTP is already known-good and DwarfStar V4.1 target-only decode
   is not equivalent.
3. **Routed MoE expert-major scheduling/materialization** — lower priority
   because oMLX already has several V4.1 MoE specializations and needs
   dominance evidence before changes.

## Promotion rule for Candidate 1

DwarfStar absolute throughput is treated only as an architectural upper bound.
There is no supported precision adjustment between DwarfStar Q4 resident and
oMLX official precision.  Candidate 1 can be promoted only from a scaling-shape
or overhead difference that cannot reasonably be explained by weight precision
alone.

Signals that can promote Candidate 1:

- oMLX throughput remains nearly flat as added prefill chunks grow while the
  DwarfStar reference shows a clear jump at larger added chunks;
- or oMLX shows fixed/near-fixed overhead in cache/state publication,
  Python/MLX materialization, command submission/synchronization, temporary
  copies, or Engram overlap;
- or oMLX fails to amortize larger prefill chunks where DwarfStar does.

## First bounded measurement

Tool:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_omlx_prefill_shape.py \
  --frontiers 4096,8192,16384 \
  --out artifacts/m2/omlx-prefill-shape/shape-4k-8k-16k.json
```

This starts the production oMLX server with prefix cache enabled, uses the
known-good model settings, sends increasing `promessi_sposi.txt` prefixes, and
uses `max_tokens=1`.  This is a bounded architectural measurement, not
long-context qualification.

Result artifact:

```text
artifacts/m2/omlx-prefill-shape/shape-4k-8k-16k.json
```

| Content frontier | Prompt tokens | Cached tokens | Uncached tokens | Prefill seconds | Uncached t/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4,126 | 0 | 4,126 | 23.42 | 176.17 |
| 8,192 | 8,222 | 4,096 | 4,126 | 21.79 | 189.35 |
| 16,384 | 16,414 | 8,192 | 8,222 | 43.16 | 190.50 |

Swap, swapins, swapouts, compression and decompression deltas were all zero for
the measured requests.

DwarfStar Q4 resident upper-bound shape on the same class of host, from
`$HOME/ds4/QA_BEFORE_RELEASES.md`:

| Frontier | Added tokens | Prefill t/s |
| ---: | ---: | ---: |
| 4,096 | 4,096 | 341.75 |
| 8,192 | 4,096 | 339.10 |
| 16,384 | 8,192 | 641.49 |
| 32,768 | 16,384 | 715.96 |

## M2 decision after first measurement

The first oMLX measurement shows a flat uncached suffix shape from +4K to +8K
(~189-190 tok/s), while the DwarfStar upper-bound reference jumps from ~340
tok/s at +4K to ~641 tok/s at +8K.  The absolute gap is not itself decisive,
but the scaling-shape difference is now real evidence for Candidate 1 as the
leading implementation target.

## Narrow attribution check

Tool:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_omlx_prefill_trace.py \
  --prompt-lengths 4096,8192 \
  --out artifacts/m2/omlx-prefill-trace/trace-4k-8k.json
```

This uses oMLX's existing admin throughput benchmark because that path sets
`benchmark_trace=True` and causes the production scheduler to log per-prefill
chunk timing.  It is still a bounded M2 attribution measurement, not
qualification.

Result artifact:

```text
artifacts/m2/omlx-prefill-trace/trace-4k-8k.json
```

Benchmark summary:

| Prompt tokens | Processing t/s | TTFT | Peak MLX active | GPU util avg | Swap delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 191.5 | 21.39 s | 292.11 GiB | 99.1% | 0 |
| 8,192 | 194.5 | 42.12 s | 292.07 GiB | 99.6% | 0 |

Scheduler trace summary:

| Prompt | Chunks | Boundary | Model/cache ms sum | Overhead ms sum | Overhead fraction |
| ---: | --- | --- | ---: | ---: | ---: |
| 4,096 | `[2048, 2047]` | 2,048-token cache blocks | 21,247.079 | 23.809 | 0.112% |
| 8,192 | `[2048, 2048, 2048, 2047]` | 2,048-token cache blocks | 41,953.072 | 42.210 | 0.101% |

Attribution:

- The flat oMLX scaling shape is not explained by API/server overhead: traced
  non-model overhead is about 0.1% of prefill time.
- It is not explained by swap or host memory pressure in these fixtures: swap,
  swapins and swapouts stayed at zero; GPU utilization was ~99%.
- It is not explained by paged prefix-cache reconstruction in the first scaling
  artifact: prefix restore/reconstruct log entries were sub-10ms, while the
  +4K/+8K suffix prefills took ~22s/~43s.
- The production scheduler is executing the prefill as repeated ~2,048-token
  external chunks because boundary snapshots are enabled at a 2,048-token cache
  block size.  Doubling the prompt from 4K to 8K doubles the number of nearly
  identical chunks instead of exposing a larger-chunk amortization regime.

This narrows Candidate 1 from a broad "layer-major prefill" idea to a concrete
execution difference: DwarfStar's upper-bound path has a large-added-chunk regime
that oMLX's current production prefill path does not enter because it remains
2,048-chunked.

## Boundary/step causality check

The most information-efficient next question was whether the 2,048-token shape
itself was causal.  Two diagnostic-only runs were made.  They do not modify oMLX
source and are not promotion evidence by themselves.

### Prefix cache disabled

Tool:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_omlx_prefill_trace.py \
  --prompt-lengths 4096,8192 \
  --no-prefix-cache \
  --out artifacts/m2/omlx-prefill-trace/trace-4k-8k-no-cache.json
```

Result:

| Prompt tokens | Processing t/s | Chunks | Boundary enabled | Overhead fraction |
| ---: | ---: | --- | --- | ---: |
| 4,096 | 191.6 | `[2048, 2047]` | false | 0.087% |
| 8,192 | 194.5 | `[2048, 2048, 2048, 2047]` | false | 0.066% |

Disabling the prefix cache disables boundary snapshots but does not change the
2,048-token prefill chunking or throughput.  Therefore the observed flat scaling
is not caused by paged-prefix-cache boundary snapshot overhead.

### Diagnostic larger scheduler step

Tool:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_omlx_prefill_trace.py \
  --prompt-lengths 4096,8192 \
  --no-prefix-cache \
  --prefill-step-patch 4096 \
  --out artifacts/m2/omlx-prefill-trace/trace-4k-8k-no-cache-step4096.json
```

This uses an isolated `sitecustomize` monkeypatch for the server process only,
changing `Scheduler._base_prefill_step_size` to 4,096.  It is a diagnostic
causality check, not an implementation.

| Prompt tokens | Processing t/s | Chunks | Peak MLX active | Swap delta |
| ---: | ---: | --- | ---: | ---: |
| 4,096 | 186.2 | `[4095]` | 295.33 GiB | 0 |
| 8,192 | 195.0 | `[4096, 4095]` | 296.06 GiB | 0 |

A further 8K-only diagnostic with `--prefill-step-patch 8192` produced a single
8,191-token chunk but regressed to 186.8 tok/s and raised peak MLX active memory
to 302.50 GiB:

```text
artifacts/m2/omlx-prefill-trace/trace-8k-no-cache-step8192.json
```

### Boundary/step decision

The simple "move the 2,048 boundary / make chunks larger" hypothesis is
rejected for now:

- no-cache still uses 2,048-token chunks and keeps the same throughput;
- 4,096-token chunks do not improve throughput;
- an 8,192-token chunk regresses throughput and increases memory materially;
- traced scheduler overhead remains negligible in every run.

Candidate 1 should therefore not be implemented as a trivial prefill-step or
cache-boundary size change.  The remaining plausible DwarfStar difference is
not the scalar chunk length alone, but the deeper layer-major/state-publication
execution topology: DwarfStar processes and publishes state in a different
layer-major/carry-buffer regime, while oMLX repeats full model chunks through
the MLX module graph.

The next implementation target, if pursued, must be a minimal state-publication
or layer-major prototype that changes execution topology, not merely a larger
scheduler chunk.  Do not start 32K qualification before such a prototype shows a
bounded short/medium win and passes bounded correctness.
