# P8 graph-reuse/performance architecture

Status: **design review only; no P8 optimization implemented or selected**.  P8 begins only after the scoped P0-P7 structural gate closure recorded in `docs/dwarfstar-v41-prefill-fp8-mlx-architecture-plan.md`.

## 1. Frozen P0-P7 authority

P8 is an optimization phase, not a correctness architecture. It must preserve:

- DwarfStar sweep/segment/command order as execution authority.
- Request-owned arena/carry lifetime and P6 deferred-decoder suffix lifetime.
- PublicationManager frontier visibility.
- P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM` scheduling with `P7_ENGRAM_TILE = 2048` and zero foreground Engram fallback on qualifying enabled paths.
- Live `DeepseekV41Cache` as the final handoff authority and P5 zero prompt replay.
- No forbidden CPU fallback for h/pre activations, KV/index/cache, weights, or hot-path tensor digest/list conversion.

Qualified P7 environment: Python 3.13.15, MLX 0.32.2, NumPy 2.3.5, oMLX 0.7.0.dev2, oMLX `b390b31e0c6831225fed0f24d278eb1db7fcb68b`; `preserve_mtp=False`, `engram_ssd_offload=True`, `moe_expert_offload_resident_fraction=None`. Expert offload remains outside scope.

## 2. Clean P8 baseline measurements

Artifacts: `artifacts/p8-baseline/*.json` and matching `*.supervisor.log`.  Runs used the existing `abef4ff` path plus qualification-only aggregate timing/memory instrumentation; P7 remains enabled except the explicit controls.

| case | total prefill wall s | effective prefill tok/s | P5 bootstrap/decode | peak MLX active/cache/peak memory | RSS / swap observation | Engram reads |
|---|---:|---:|---|---|---|---|
| P7 complete-2048 | 2.896 | 707.11 | PASS, zero replay, 2 decoded tokens `[235, 223]`, P5 wall 4.567s | active 301,276,568,418; cache 9,923,148,805; peak 304,111,422,833 bytes | `ru_maxrss` raw platform value 19,151,912,960; no supervisor memory abort | bg 2 / fg 0 |
| P7 complete-8192 | 9.373 | 873.99 | PASS, zero replay, tokens `[1, 201]`, P5 wall 5.832s | active 301,362,294,832; cache 20,889,211,900; peak 310,986,708,472 | raw `ru_maxrss` 19,151,421,440; no abort | bg 8 / fg 0 |
| P7 complete-16384 | 17.626 | 929.53 | PASS, zero replay, tokens `[1, 0]`, P5 wall 5.759s | active 301,390,950,812; cache 30,887,908,008; peak 319,629,252,136 | raw `ru_maxrss` 19,151,552,512; no abort | bg 16 / fg 0 |
| P7 A-24577 | 26.750 | 918.78 | PASS, zero replay, tokens `[90, 1]`, P5 wall 5.629s | active 301,394,575,092; cache 31,080,071,653; peak 319,629,252,136 | raw `ru_maxrss` 19,151,290,368; no abort | bg 26 / fg 0 |
| P7-off attribution control 8192 | 10.153 | 806.89 | not run; control only | active 301,362,294,836; cache 24,727,778,300; peak 310,986,708,476 | raw `ru_maxrss` 19,148,963,840; no abort | bg 0 / fg 2 |
| P7-off attribution control 16384 | 18.657 | 878.16 | not run; control only | active 301,390,950,816; cache 34,842,833,576; peak 319,629,252,140 | raw `ru_maxrss` 19,151,699,968; no abort | bg 0 / fg 4 |

Attribution controls are not production candidates. On this pass P7 overlap improves 8192 prefill by ~7.7% and 16384 by ~5.5%, with foreground SSD reads eliminated.

Cold/warm distinction: these are cold first-request measurements including checkpoint load in the supervised process wall, but the table's prefill time excludes checkpoint load and sums only segment execution. No warm repeated same-shape run has been promoted yet; P8 benchmark methodology below requires both cold and warm evaluation.

## 3. MLX laziness/materialization timeline

Timing classifications:

- Embedding/hash setup: Python/graph construction plus lazy MLX graph creation; Engram hash/index host representation is an allowed SSD-storage boundary.
- `ENCODE_ROWS`: Python/graph construction, async submission, oMLX state mutation, and possible implicit synchronization. The aggregate wall time is not per-kernel GPU compute.
- P7 Engram micro-pipeline: SSD/host I/O plus MLX graph work around Engram rows; explicit `mx.async_eval` before/after Engram incorporation; no normal-path `mx.synchronize()`.
- `P6_SOURCE_COMPLETE_AND_DETACH_CONE`: explicit materialization by `mx.eval` of persistent source state and owned-row copy/detach; synchronized/lifetime-required boundary.
- PublicationManager: Python/shared metadata mutation; no intentional tensor materialization.
- Final seal/commit: transaction state and final cache-frontier assertions.
- P5 handoff/decode: handoff to generation and bounded decode; MLX synchronizations may occur inside generation/session internals.

Do not interpret Python call duration around lazy operations as GPU execution duration unless the boundary explicitly materializes or synchronizes.

## 4. Materialization/synchronization map on P0-P7 path

| occurrence | file/function | trigger | values materialized | correctness-required? | lifetime-required? | overlap-only? | diagnostic-only? | candidate for P8 removal/grouping? |
|---|---|---|---|---|---|---|---|---|
| loader materialization | pinned oMLX loader | checkpoint load | model parameters | yes, backend admission | yes | no | no | no; outside prefill timing |
| `mx.eval(*values)` | `OfficialFP8MLXBlockRunner._p6_eval_persistent_source_state` | `P6_SOURCE_COMPLETE_AND_DETACH_CONE` | layer 0..19 window KV and source layer slots 2/3/4/5 | yes | yes, detaches source-only cone | no | no | maybe group with adjacent detach, but cannot remove blindly |
| `mx.eval(out)` | `_owned_row_copy` | P6 detach compact row copy fallback | compact h/pre/next rows | yes when fallback used | yes | no | no | maybe replace with clearer copy primitive if measured |
| `mx.async_eval(h_chunk, pre_chunk)` | `MaterializationPolicy.before_engram_dependency` | P7 Engram before SSD dependency | current h/pre microtile | no semantic change | no | yes | no | tune grouping only if donor overlap preserved |
| `mx.async_eval(h_after, pre_chunk)` | `MaterializationPolicy.after_engram_incorporated` | P7 after Engram incorporation | h after Engram and pre microtile | no semantic change | no | yes | no | candidate grouping; must not serialize donor pipeline |
| `mx.eval(*tensors)` | `MlxEvaluationPolicy.maybe_eval_batch` | optional batch policy | command outputs | no in current qualified default | no | no | not enabled | keep disabled unless explicitly qualified |
| `mx.synchronize()` | `MlxEvaluationPolicy.maybe_eval_batch` | optional policy | global stream | no in current default | no | no | not enabled | do not enable for timing |
| P5/generation internals | oMLX `OMLXGenerationSession` and MLX ops | terminal bootstrap/decode | decode logits/cache updates | yes for decode | yes | no | no | audit separately; not P8 prefill first unit |
| oMLX Attention/MoE internals | pinned oMLX modules | projection/attention/MoE | may internally use compiled/custom kernels | yes | yes | no | no | compile only pure regions, not cache mutation |

Repository inventory (`rg`) found production-path `mx.eval`, `mx.async_eval`, and `mx.synchronize` at the runner/p7 policy points above plus diagnostic/tool occurrences outside the hot path.

## 5. Shape-class inventory

| shape class | static tensor shapes? | dynamic absolute_start only? | cache mutation? | Python dict/shared mutation? | publication side effects? | compiled pure subgraph candidate? |
|---|---|---:|---:|---:|---:|---|
| encoder 8192 tile, layers 0..19 | mostly stable per layer, same row count | yes for positions | yes | yes | yes | only inner pure math; full layer requires functionalization |
| Engram 2048 microtile | stable rows and hash columns | offset changes IDs | no cache mutation in Engram math | no | no | candidate if it does not serialize SSD donor |
| encoder short/final tile | variable: 1, 2048/8192/16384 families | yes | yes | yes | yes | shape registry needed; compile only recurring sizes |
| decoder suffix layers 20..39 | finite family `Q(L)=1+(39-L)*127` | yes | yes | yes | source/index capture | promising per-layer/shape after functionalization |
| ordinary 1-token tail | stable single-token decode/full block | yes | yes | yes | yes | decode-owned; not first P8 prefill unit |
| P5 decode token | stable 1-token | yes | yes | session-owned | generation state | outside initial P8 prefill package |

## 6. Existing fused-kernel and optimized-operation inventory

Pinned oMLX already uses substantial optimized paths:

- `@mx.compile`: `norm`, `_rope`, `pack_activation`, quantization fallback, and several HC/helper functions in `language.py`.
- Quantized projections: `QuantizedProjection.project_quantized` reaches `mx.quantized_matmul` for packed FP8/FP4 weights when no expert indices are supplied.
- Activation packing/quantization: `quantize_activation` uses custom `quantize_fp8_activation` fast path for 8-bit/group32/BF16-or-FP32 non-CPU shapes; otherwise compiled fallback.
- Attention: for length > 8, 64 heads, 512 head dim, BF16 q, and available GLM symbol, uses `glm_fast.deepseek_v41_packed_attention`; otherwise uses pinned `packed_sparse_attention` Metal kernel.
- HC: `fused_hc_pre_norm`, `fused_hc_post`, and projection kernels are already custom Metal.
- MoE: routes through oMLX/GLM MoE/DSA helpers and quantized/switch-linear paths where predicates match.
- Indexer/compressor: `pack_activation(bits=4)` and `packed_index_scores`/`packed_index_topk` kernels are present.
- Engram storage: SSD mmap selected-row I/O with Metal-backed arrays on read; P7 donor prefetch now active.

Fast-path reach findings:

| component | existing optimized donor exists? | current call shape reaches it? | fallback currently selected? |
|---|---|---|---|
| HC pre/post | yes | yes for production Block path | no known ds41f fallback |
| QuantizedProjection | yes (`mx.quantized_matmul`) | yes for packed weights/projections | no, except routed expert variants have separate predicates |
| Attention long prefill | yes GLM packed attention and packed sparse kernel | likely reaches GLM for length>8 BF16 geometry; suffix/tail may use packed sparse | suffix/tail length families may miss GLM length>8 on very small rows |
| Engram 2048 | SSD donor/read rows optimized; no bespoke compute kernel proposed | reaches donor for 2048 microtiles | full 8192 request misses donor due 16 MiB limit; P7 microtiles fix it |
| MoE | existing GLM/switch/quantized routes | requires deeper timing to prove per-shape route | unknown for all shapes |
| packed activation | yes | used for KV/compressed/index packing | no known ds41f fallback |

Fast paths unexpectedly missed: the old 8192 full Engram prefetch missed donor admission because selected bytes exceeded 16 MiB. Qualified P7 microtiling fixes this. No other confirmed missed fused path is proven yet; P8 first implementation should add a fast-path verifier rather than writing kernels.

## 7. Hot-path attribution

Use materialized/synchronized boundaries and command-family aggregate wall time, not Python call time alone.

| hot region | share/evidence | frequency | shape class | existing fused path? | graph-reuse candidate? | custom-kernel candidate? | semantic risk |
|---|---|---:|---|---|---|---|---|
| encoder `ENCODE_ROWS` layers 0..19 | 8K: 6.98s of 9.37s; 16K: 13.06s of 17.63s; A: 19.23s of 26.75s | 20 per 8192 tile | encoder 8192 tile | many oMLX fused paths already | yes, inner pure regions only | not first | high if cache/publication included |
| P7 Engram micro-pipeline | 8K: 4.11s; 16K: 7.12s; A: 10.14s, overlapped with encoder totals | 2 Engram layers per tile, 2048 microtiles | Engram 2048 | SSD donor optimized | maybe for Engram math/reassembly | no until no donor/fused path | medium due SSD overlap |
| P6 source detach | 8K: 2.33s; 16K: 4.48s; A: 6.79s across two detach calls | once per source-only/final segment | persistent source boundary | MLX eval | grouping candidate | no | high; lifetime boundary |
| P5 handoff/decode | ~4.6-5.8s including two decode tokens | once | decode token | oMLX generation | outside first prefill unit | no | high, decode authority |
| decoder suffix 20..39 | <0.14s in aggregate command wall in current lazy timing | 20 calls | finite Q(L) family | existing kernels | maybe but not hot yet | no | medium |
| PublicationManager metadata | microsecond-level in command wall | per command/layer | metadata | n/a | maybe negligible | no | low |

## 8. ds41f orchestration overhead map

| overhead source | classification | evidence / action |
|---|---|---|
| SweepCommand object iteration | negligible | command-stream non-encode timings are microseconds; keep until proven otherwise |
| arena slicing/writes | unknown/measurable candidate | `_write_rows` may retain graph parents or perform true slice update depending MLX assignment semantics; audit before changing |
| publication dictionaries | negligible to unknown | command wall for publication is tiny, but graph retention via shared tensors must be checked with memory traces |
| cache slot helpers | negligible | no visible wall share; correctness critical |
| microtile concatenation | unknown/measurable | `mx.concatenate` reassembles Engram microtiles; may create lazy concat graph or copy when materialized |
| Python list/tuple construction | negligible | not visible in aggregate timing |
| per-command policy calls | negligible | read-ahead/begin/end timings are tiny |
| telemetry | negligible in production; measurable in qualification logs | P8 instrumentation is qualification-only; do not leave giant per-kernel logs |
| qualification guards | negligible/unknown | `assert_no_reference_hot_path` remains but should be sampled under P8 if optimizing Python overhead |

## 9. Specific audits

### Engram microtile concatenate

Current P7 performs 2048-row Engram calls and then `mx.concatenate(values, axis=1)` back to the original command row shape. No `mx.eval` is forced at the concatenate boundary. Therefore the concatenate is likely lazy until a later layer/materialization boundary, but it can add graph nodes and may become a copy when consumed/materialized. P8 alternative design: an exact-semantic reassembly strategy using preallocated/write-slice or a compiled assembly helper, but only if profiling proves material cost and without changing `P7_ENGRAM_TILE` or donor overlap.

### carry `_write_rows`

`_write_rows()` uses an `assign_rows` hook if present; otherwise `base[:, offset:end] = update`. It does not explicitly concatenate whole 8K buffers. Performance risks are MLX slice-update graph retention and parent lifetime, not correctness. P8 should inspect active-memory deltas and graph size around repeated writes before replacing it.

### PublicationManager overhead

Publication handling appears to be metadata over shared persistent tensors. Current aggregate timings are negligible. Remaining risk is dictionary retention of tensors keeping graphs alive longer than intended; P8 should add object-count/tensor-reference telemetry, not replace the manager first.

### decoder suffix graph reuse

Suffix query geometry is deterministic: `Q(L)=1+(39-L)*127`. Best candidate is one compiled/registered pure query math function per layer/shape after separating cache mutation and publication capture. A single parameterized graph is less likely under MLX 0.32.2 because layer modules/weights and shapes differ.

### encoder 8192 graph reuse

The shape is highly repetitive, but different layer modules/weights mean compile reuse is likely per function/module and shape, not automatically shared structurally across all 20 layers. P8 should verify MLX 0.32.2 cache behavior empirically with a no-semantics microprobe before compiling production blocks.

### P7 Engram graph reuse

Engram math has a stable 2048-token shape, but compile boundaries must not serialize donor scheduling. Candidate scope is the pure combine/math after selected rows return to MLX; SSD submit/drain and `mx.async_eval` remain outside.

## 10. MLX 0.32.2 compile compatibility audit

`mx.compile` exists in MLX 0.32.2, but pinned oMLX already uses it for pure helper functions. P8 classifications:

| candidate function | classification | reason |
|---|---|---|
| full layer block | REQUIRES_FUNCTIONALIZATION | mutates cache, shared dicts, PublicationManager state, uses modules and absolute positions |
| Attention full call | REQUIRES_FUNCTIONALIZATION | mutates cache/shared, branches on ratio/length, concatenates caches |
| MoE full call | UNKNOWN / REQUIRES_FUNCTIONALIZATION | routed expert state and module dispatch; inspect route predicates first |
| HC transforms | COMPILE_SAFE_PURE_REGION | already compiled/fused in oMLX where pure |
| Engram math | REQUIRES_FUNCTIONALIZATION | storage/I/O outside; pure combine may be safe |
| suffix query math | REQUIRES_FUNCTIONALIZATION | finite shapes, but cache/shared mutation must be externalized |
| packed activation | COMPILE_SAFE_PURE_REGION | already compiled/fast-pathed in oMLX |
| DeepseekV41Cache mutation | NOT_COMPILE_SUITABLE | Python object mutation and ownership semantics |
| PublicationManager mutation | NOT_COMPILE_SUITABLE | Python dict/shared-state side effects |
| donor scheduling | NOT_COMPILE_SUITABLE | host/SSD/threadpool side effects |

## 11. P8 optimization package design

Conceptual package:

```text
P8ExecutionOptimizer
  +-- ShapeClassRegistry
  +-- GraphReusePolicy
  +-- MaterializationOptimizer
  +-- ExistingKernelFastPathVerifier
  +-- OptionalNativeKernelRegistry
  +-- PerformanceTelemetry
```

Responsibilities:

- `ShapeClassRegistry`: records stable shape classes, absolute-start variability, layer/module identity, and side-effect boundaries.
- `GraphReusePolicy`: decides compile eligibility only for pure tensor regions; never compiles cache/publication/donor ownership blindly.
- `MaterializationOptimizer`: audits and proposes grouping/removal only for non-correctness materialization boundaries.
- `ExistingKernelFastPathVerifier`: checks that ds41f shapes, dtype, device, layout, and quantization mode reach existing oMLX/MLX fast paths before any native kernel proposal.
- `OptionalNativeKernelRegistry`: disabled by default; accepts candidates only after measured hotspot, no adequate existing path, stable semantics/shapes, expected gain, and bounded correctness surface.
- `PerformanceTelemetry`: low-overhead aggregates by command kind/phase/layer range; no xctrace, no per-kernel giant logs, no forced component `mx.eval`.

Feature flags/rollback should be package-level (for example `P8ExecutionOptimizer.enabled` plus per-component rollback switches), not unrelated isolated flags.

## 12. Candidate implementation order

1. **Telemetry/verification hardening**: keep aggregate command-family timing, add fast-path predicate reports for Attention/MoE/HC/quantized projection/packed activation, and add graph/materialization labels. No semantic change.
2. **ShapeClassRegistry + verifier**: record actual 8K/16K/A shape classes and whether oMLX fused paths are reached. This is the recommended first P8 implementation unit.
3. **Materialization grouping study**: quantify P6 detach and Engram concat/write-row graph retention; propose grouping only if correctness/lifetime boundaries remain explicit.
4. **GraphReusePolicy prototype**: compile a single pure helper region already known to be side-effect free or already compiled in oMLX; compare cold and warm costs.
5. **Encoder pure-region functionalization**: only after proving compile cache behavior and isolating cache/publication mutation.
6. **Optional native kernels**: last resort; no candidate is promoted at design review time.

## 13. Rejected/non-applicable candidates

- Enabling expert offload: rejected/out of P7/P8 scope.
- Changing production selector: rejected for this task.
- Compiling full stateful layer calls: rejected until functionalized.
- Forcing `mx.eval()` at model-component boundaries to time components: rejected; would change laziness and overlap.
- Writing custom Metal for HC/quantized projection/attention before verifying existing oMLX fast paths: rejected.
- Removing P6 source detach: rejected; correctness/lifetime boundary.
- P7-off architecture: rejected as production baseline; attribution control only.

## 14. Benchmark methodology and promotion rules

Benchmarks must separate:

- checkpoint load;
- cold first request, including initial graph compilation/cache warmup;
- warm repeated same-shape runs;
- steady-state prefill for 8192, 16384, and A-24577;
- P5 bootstrap;
- decode.

All long measurements remain under `tools/qualification_supervisor.py` with the 180s no-progress guard, pathological <10 tok/s guard, 4x normalized regression guard where a baseline exists, and memory/swap pathology stop.

Each optimization must have: baseline, candidate, correctness comparison, memory comparison, performance comparison, and rollback switch. Promotion requires unchanged P0-P7 invariants, P5 zero replay, valid cache/publication state, no new CPU fallback, no abnormal memory growth, and repeatable performance gain. P8 may reject a candidate because it is slower.

## 15. Promotion/rollback criteria

Rollback immediately if any candidate:

- changes command ordering/frontiers/coverage;
- invalidates P6 failure/drain/reuse or P5 same-cache handoff;
- introduces foreground Engram fallback on P7-enabled paths;
- introduces forbidden CPU activation/cache fallback;
- increases peak/cache memory pathologically;
- regresses normalized prefill wall time by >4x under the supervisor;
- fails cold first-request or warm same-shape methodology.

Promotion is scoped per optimization and does not promote the production selector or global runtime completion.
