# P8 graph-reuse/performance architecture

Status: **P8 optimization search complete**.  The scoped P0-P7 structural gate is closed; P8 evidence, implementation, and A/B evaluation are recorded here. Final decision: `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`.

## 1. Frozen P0-P7 authority

P8 was an optimization phase, not a correctness architecture. It preserved:

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

## 11. P8 optimization package design (historical design)

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

## 12. Candidate implementation order (superseded by final P8 decision)

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

## 14. Benchmark methodology and promotion rules (used for P8 closure)

Benchmarks must separate:

- checkpoint load;
- cold first request, including initial graph compilation/cache warmup;
- warm repeated same-shape runs;
- steady-state prefill for 8192, 16384, and A-24577;
- P5 bootstrap;
- decode.

All long measurements remain under `tools/qualification_supervisor.py` with the 180s no-progress guard, pathological <10 tok/s guard, 4x normalized regression guard where a baseline exists, and memory/swap pathology stop.

Each optimization must have: baseline, candidate, correctness comparison, memory comparison, performance comparison, and rollback switch. Promotion requires unchanged P0-P7 invariants, P5 zero replay, valid cache/publication state, no new CPU fallback, no abnormal memory growth, and repeatable performance gain. P8 may reject a candidate because it is slower.

## 15. Promotion/rollback criteria (used for P8 closure)

Rollback immediately if any candidate:

- changes command ordering/frontiers/coverage;
- invalidates P6 failure/drain/reuse or P5 same-cache handoff;
- introduces foreground Engram fallback on P7-enabled paths;
- introduces forbidden CPU activation/cache fallback;
- increases peak/cache memory pathologically;
- regresses normalized prefill wall time by >4x under the supervisor;
- fails cold first-request or warm same-shape methodology.

Promotion is scoped per optimization and does not promote the production selector or global runtime completion.

## 16. P8 evidence package implementation results (2026-10-01)

Implemented package: `ds41f_mlx/prefill_fp8_mlx/p8_optimizer.py`.

Components are opt-in only (`DS41F_P8_OPTIMIZER=1` or explicit `P8ExecutionOptimizer(enabled=True)`) and do not alter the production selector. `GraphReusePolicy` and `MaterializationOptimizer` remain planned interfaces only; they perform no production work. No custom kernels and no whole stateful layer compilation were introduced.

Evidence artifacts:

- `artifacts/p8-evidence/p8-coldwarm-8192.json`
- `artifacts/p8-evidence/p8-coldwarm-8192.supervisor.log`
- `artifacts/p8-evidence/p8-coldwarm-16384.json`
- `artifacts/p8-evidence/p8-coldwarm-16384.supervisor.log`

Structural regression: `PYTHONPATH=tests /Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 -m unittest discover -s tests -v` passed: 84 tests OK.

Same-process cold/warm results:

| family | cold prefill s | warm1 s | warm2 s | warm3 s | warm median s | cold/warm |
|---|---:|---:|---:|---:|---:|---:|
| 8192 | 8.9079 | 8.8008 | 8.6887 | 8.7345 | 8.7345 | 1.0199 |
| 16384 | 16.4453 | 16.3612 | 16.3126 | 16.4070 | 16.3612 | 1.0051 |

Interpretation: outcome B from the P8 review taxonomy. First-request to warm improvement is small and within the same order as run-to-run noise; graph construction/cache warmup is not currently proven dominant.

Observed shape classes:

- 8192-family direct request: command geometry is two 4096 encoder command chunks per encoder layer, plus 2048 Engram microtiles, decoder suffix `Q(L)`, and ordinary 1-token tail. This is recorded as `encoder_short_final` command shape rather than a single `encoder8192` command, while the request family remains 8192.
- 16384-family P6 request: source segment records actual `encoder8192` command shape classes (40 distinct layer/module signatures across two 8192 source commands), plus decoder suffix `Q(L)`, ordinary 1-token tail, and a final short/P6 boundary signature.
- Shape registry stores scalar metadata only and separates shape identity from layer/module identity, weight identity class, and side-effect class.

Fast-path observations from the cold requests:

| component | 8192 fast/fallback calls | 16384 fast/fallback calls | notes |
|---|---:|---:|---|
| HC fused paths | 389 / 0 | 375 / 0 | `fused_hc_projection`, `fused_hc_pre_norm`, `fused_hc_post` directly observed. |
| Attention | 0 / 23 | 0 / 20 | `packed_sparse_attention` directly observed. `mlx._glm.deepseek_v41_packed_attention` was `UNAVAILABLE` at the wrapped seam, so GLM fast-path absence is seam-limited evidence, not a kernel-failure claim. |
| Index | 12 / 0 | 11 / 0 | `packed_index_topk` and `packed_index_scores` directly observed; source/publication geometry determines use. |
| Quantized projection | 501 / 0 | 487 / 0 | `mx.quantized_matmul` directly observed. |
| MoE | 62 / 0 | 59 / 0 | `combine_sorted_experts` observed. Grouped/gather QMM symbols wrapped in the attempted modules were `UNAVAILABLE`; classify as `UNAVAILABLE`, not fallback. |

P7 donor evidence: foreground Engram fallback was 0 for all cold/warm repeats. Background microtile reads were 8 for each 8192-family request and 32 for each 16384-family request (two 8192 source segments, two Engram layers, four 2048 microtiles per layer per source segment).

`_write_rows` audit: real MLX used the `MLX slice assignment` branch in all observed writes (126 calls in 8192 cold, 120 in 16384 cold). No `assign_rows` hook was observed.

Engram concat/reassembly audit: 8192-family cold recorded four two-part concatenations to `[1, 4096, 4, 5120]`; 16384-family cold recorded four four-part concatenations to `[1, 8192, 4, 5120]`. Python microtile references are released after reassembly in the instrumentation scope. No artificial evaluation was inserted at concat.

P6 detach materialization attribution: 16384 cold recorded `P6_SOURCE_COMPLETE_AND_DETACH_CONE` elapsed wall time 4.4411s. Memory proxy before/after was active 312,847,817,016 -> 314,864,784,632 bytes, cache 14,647,420,570 -> 17,009,208,026 bytes, peak 315,667,354,880 -> 319,629,252,136 bytes. This remains an inclusive boundary: upstream lazy graph materialization may be charged here.

Limitations:

- Wrapping `mlx._glm.deepseek_v41_packed_attention` and `mlx._glm.deepseek_v41_grouped_expert` reported `UNAVAILABLE`; pinned oMLX may expose GLM symbols through a different seam. These are not classified as fallbacks.
- Activation quantization is recorded at a donor-level route (`quantize_activation`) plus `mx.quantized_matmul`; per-internal primitive branch remains partially caller-branch evidence.
- P5/decode timing fields are reserved in the same-process harness output but not populated by this initial prefill-focused evidence run.
- Aggregate regions are inclusive/overlapping: `ENCODE_ROWS` includes nested Engram micro-pipeline work, and P6 detach can materialize preceding lazy work. No percentages should be summed across overlapping regions.

## 17. P8 first-target decision / next-unit contract

Decision: **recommend no optimization yet**.

Reason: none of the allowed target classes currently satisfies all selection criteria simultaneously.

- Existing fast-path miss: not proven. HC, index, quantized matmul, and MoE combine routes are observed. Attention GLM/grouped symbols are unobservable at the attempted seam and must remain `UNAVAILABLE/UNKNOWN`, not a missed optimization.
- Graph reuse / compile: cold-to-warm ratios are only 1.0199 (8192) and 1.0051 (16384), so compile/cache warmup is not proven material.
- Engram concat/reassembly: concat shape and retention evidence exists, but materialized cost is not isolated from downstream lazy graph execution.
- `_write_rows`: branch evidence exists (`MLX slice assignment`), but large impact is not yet measured.
- P6 detach: large inclusive time is measured, but P6 detach is a required lifetime/correctness boundary and removal is explicitly out of scope. Attribution must be refined before proposing grouping or copy reduction.

Concrete next-unit specification (evidence-only, not an optimization):

- target: isolate materialization attribution for Engram concat and `_write_rows` graph retention feeding the existing `P6_SOURCE_COMPLETE_AND_DETACH_CONE` boundary.
- measured reason: P6 detach is 4.4411s inclusive on 16384 and memory rises by ~2.02GB active / ~2.36GB cache, while concat and slice-write graph nodes remain plausible upstream contributors.
- affected functions: `SchedulingCoordinator.apply_engram_micro_pipeline`, `_concat_sequence`, `_write_rows`, `OfficialFP8MLXBlockRunner._p6_materialize_and_detach`.
- frozen invariants: DwarfStar command order, P6 deferred-decoder geometry, `P7_ENGRAM_TILE=2048`, PublicationManager ownership, DeepseekV41Cache authority, P5 zero replay, qualified boundaries, and zero foreground Engram fallback.
- candidate change: none in the next unit; add only more precise telemetry around existing references/liveness and already-required materialization boundaries.
- correctness comparison: identical P0-P7 structural tests, P7 donor reads, frontiers, cache slot structure, and no production selector change.
- performance benchmark: same-process 8192 and 16384 cold + 3 warm, same deterministic tokens, fresh cache each request, separate shape-family process.
- memory benchmark: active/cache/peak memory before encoder tile, after Engram reassembly, after layer19, before P6 detach, after P6 detach.
- promotion threshold for any future candidate: repeatable absolute and percentage improvement outside warm run-to-run noise, no correctness regression, no meaningful memory regression, and no hidden first-request catastrophe.
- rollback mechanism: package-level P8 flag; any candidate must have a separate rollback switch and default disabled state.
- stop conditions: foreground Engram fallback > 0, P5 replay, command/frontier mismatch, new tensor-to-NumPy/list conversion, new `mx.eval`/`mx.synchronize` in measured internal path, abnormal memory growth, or failure to isolate a non-required boundary.


## 18. P8 detach-source attribution instrumentation update (2026-10-01)

This update is evidence-only. It does not change the production selector, does not add `mx.compile`, custom Metal, or production materialization boundaries, and does not alter P6 detach or P7 Engram microtiling.

Pinned MLX 0.32.2 slice-assignment authority: `ml-explore/mlx` v0.32.2 `python/src/indexing.cpp` routes simple Python slice assignment such as `base[:, start:end] = update` through `mlx_compute_slice_update_args(...)`, `slice_update(src, update, starts, stops, strides)`, and `src.overwrite_descriptor(...)`. Therefore the observed `_write_rows` real branch is classified as `MLX_SLICE_UPDATE_DESCRIPTOR`, not conventional eager mutable in-place assignment. Source inspection alone is not evidence that this is slow; it is only the primary hypothesis to measure.

Telemetry additions:

- `_p6_materialize_and_detach()` now records existing mandatory substeps separately: `persistent_source_eval`, `owned_h_copy`, `owned_next_copy`, `owned_pre_copy`, `arena_detach_rebind`, and `materialize_boundary_bookkeeping`.
- `_p6_eval_persistent_source_state()` records metadata for values passed to its mandatory `mx.eval(*values)`: layer, slot, slot class (`window_KV`, `source_compressed_KV`, `index_K`, pending slots), shape, dtype, logical producer class, and known logical bytes.
- `_owned_row_copy()` records the actual branch (`detach_rows`, `.copy()`, `mx.copy()`, `mx.array()`, or fake adapter), role, input/slice/output shape, rows, elapsed wall time, and memory before/after. A bounded real-MLX probe on MLX 0.32.2 found MLX arrays do not expose `.copy()` at this seam and the current fallback branch is `mx.array()`.
- `_write_rows()` records Python object id before/after and classifies real slice assignment as `MLX_SLICE_UPDATE_DESCRIPTOR`. Descriptor identity is not inspected through private pointers and arrays are not retained.
- P8 lineage telemetry adds scalar event IDs for `ENGRAM_CONCAT`, `BLOCK_OUTPUT`, and `SLICE_UPDATE_WRITE`; records contain only metadata and producer/consumer IDs.
- Engram microtile concat telemetry records layer, transformer command rows, number of inputs, input row counts, output rows/dtype/shape, and the next-consumer class. `microtile_python_references_released_after_reassembly=True` remains only Python-list evidence; underlying MLX graph ancestry release is explicitly `NOT_PROVEN`.
- Diagnostic-only barrier mode is available via `DS41F_P8_DIAGNOSTIC_BARRIER=carry` or `persistent`. These runs are labeled `NON_QUALIFYING_ATTRIBUTION_PROBE` and default OFF.

Structural suite: `PYTHONPATH=tests /Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 -m unittest discover -s tests -v` passed 87 tests OK. Added tests prove detach substep telemetry preserves execution, lineage telemetry retains no fake tensors, write-chain accounting is coherent, diagnostic barriers default OFF, and no extra production eval is inserted by default.

Synthetic MLX structural probes: `tools/p8_synthetic_mlx_graph_probes.py` writes `artifacts/p8/synthetic_mlx_graph_probes.json`. The reduced probe used MLX 0.32.2, shape `[1, 1024, 128]`, rows/update 64. Slice-update graph-build time rose from ~0.000015s at 1 write to ~0.000034s at 32 writes; final eval after the first warmup stayed ~0.0007-0.0010s in this small case. Concat/update probes cover 2-way and 4-way concat feeding 1/4/16 slice updates. These are synthetic structural evidence only, not model-performance equivalence.

Real 8192/16384 cold+warm attribution runs with the new substep telemetry have not been completed in this commit; therefore no optimization target is selected. Required next evidence remains: baseline 8192, baseline 16384 cold + 3 warm with P7 foreground fallback 0, detach substep median/min/max/range, then non-qualifying carry and persistent-cache prepayment probes.

Current target-selection result: **NO_TARGET_YET**. The evidence is sufficient to correct terminology and to collect the required attribution, but not sufficient to choose among `WRITE_ROWS_SLICE_UPDATE_CHAIN`, `ENGRAM_CONCAT_GRAPH`, `PERSISTENT_CACHE_GRAPH_RETENTION`, `OWNED_CONE_COPY`, or another measured source.

## 17. Refreshed P8 attribution evidence (commit ecf3b1a, MLX 0.32.2)

Artifacts: `artifacts/p8-attribution/p8-baseline-8192.json`, `p8-baseline-16384.json`, `p8-probe-carry-16384.json`, `p8-probe-persistent-16384.json`, and `summary.txt`. Structural gate: `PYTHONPATH=tests /Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 -m unittest discover -s tests -v` => 87/87 PASS.

### Qualifying baselines

Both baselines used fresh cache per request, P8 verification on, diagnostic barrier off, correct final cache frontiers, and foreground Engram fallback 0.

| tokens | cold prefill s | warm1 | warm2 | warm3 | warm median | warm range | bg Engram reads/run | diagnostic barrier |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 8.898815 | 8.646203 | 8.667183 | 8.751718 | 8.667183 | 0.105515 | 8 | off |
| 16384 | 16.312919 | 16.325394 | 16.294619 | 16.215167 | 16.294619 | 0.110226 | 32 | off |

The refreshed 16384 detach reproduces the prior ~4.44s observation: cold 4.425615s, warm median 4.426134s.

### 16384 detach substeps (qualifying baseline)

| substep | cold | warm1 | warm2 | warm3 | warm median | warm min | warm max | warm range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistent_source_eval | 4.413029 | 4.416260 | 4.451661 | 4.423165 | 4.423165 | 4.416260 | 4.451661 | 0.035401 |
| owned_h_copy | 0.007766 | 0.004339 | 0.001220 | 0.001258 | 0.001258 | 0.001220 | 0.004339 | 0.003120 |
| owned_next_copy | 0.004113 | 0.001170 | 0.001197 | 0.001169 | 0.001170 | 0.001169 | 0.001197 | 0.000028 |
| owned_pre_copy | 0.000378 | 0.000223 | 0.000201 | 0.000367 | 0.000223 | 0.000201 | 0.000367 | 0.000166 |
| arena_detach_rebind | 0.000017 | 0.000015 | 0.000012 | 0.000013 | 0.000013 | 0.000012 | 0.000015 | 0.000003 |
| materialize_boundary_bookkeeping | 0.000003 | 0.000003 | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000003 | 0.000001 |
| P6_SOURCE_COMPLETE_AND_DETACH_CONE | 4.425615 | 4.422223 | 4.454450 | 4.426134 | 4.426134 | 4.422223 | 4.454450 | 0.032227 |

### Owned row copy branch (real 16384 baseline, cold)

All three compact copies used the real `mx.array()` branch.

| role | rows | input shape | output shape | elapsed s | memory before active/cache/peak | memory after active/cache/peak |
|---|---:|---|---|---:|---|---|
| batch_cur_hc | 2541 | `[1,16384,4,5120]` | `[1,2541,4,5120]` | 0.007766 | 316004324600 / 15666703066 / 319629252136 | 316108412152 / 15666703066 / 319629252136 |
| batch_next_hc | 2414 | `[1,16384,4,5120]` | `[1,2414,4,5120]` | 0.004113 | 316108412152 / 15666703066 / 319629252136 | 316207289592 / 15666703066 / 319629252136 |
| carry.pre | 2541 | `[1,16384,4]` | `[1,2541,4]` | 0.000378 | 316207289592 / 15666703066 / 319629252136 | 316207355128 / 15666637530 / 319629252136 |

Owned copies are measured materializations but do not dominate the 4s-class detach.

### Persistent-source eval inventory (real 16384 baseline, cold)

Total: 34 values, 15,933,440 known logical bytes, wall 4.413029s. Memory before active/cache/peak: 312847817016 / 14647420570 / 315667354880. Memory after: 316163052796 / 15507974870 / 319629252136.

| slot class | value count | known logical bytes | layers |
|---|---:|---:|---|
| window_KV | 20 | 1,351,680 | 0..19 |
| source_compressed_KV | 4 | 11,796,480 | 2..20 |
| index_K | 4 | 2,785,280 | 2..20 |
| compressor_pending_input | 3 | 0 | 2..14 |
| indexer_pending_input | 3 | 0 | 2..14 |

### Slice-update chain proxy (real 16384 baseline, cold)

This is a slice-update chain proxy only. `CarryState.swap()` swaps TensorSlot references; it does not itself materialize tensors, and write counts are not proven MLX graph depth.

| physical arena role | write count | offset sequence summary | row sizes summary | layers contributing |
|---|---:|---|---|---|
| batch_cur_hc | 30 | first twenty alternate `0,8192`; then `254,508,...,2540` | first twenty `8192`; then `2287,2033,...,1` | odd 1..19 twice, then odd 21..39 |
| batch_next_hc | 30 | first twenty alternate `0,8192`; then `0,254,508,...,2286` | first twenty `8192`; then `2414,2160,...,128` | even 0..18 twice, then even 20..38 |
| carry.pre | 60 | first forty alternate `0,8192`; then `127,254,...,2540` | first forty `8192`; then `2414,2287,...,1` | layers 0..19 twice, then 20..39 |

### Engram concat lineage (real 16384 baseline, cold)

Four Engram concat events were recorded: layers 1 and 14, two events each. Each had four 2048-row microtile inputs, output rows 8192, next consumer `BLOCK_OUTPUT_THEN_WRITE_ROWS`, and no stronger `SLICE_UPDATE_WRITE` lineage than the following write-row proxy. Python references released = proven. MLX graph ancestry released = NOT_PROVEN.

### Attribution probes (non-qualifying)

Warm medians are used below. Diagnostic probes are attribution-only and are not production candidates.

| component | baseline | carry-prepay | persistent-prepay |
|---|---:|---:|---:|
| diagnostic prepayment | 0.000000 | 4.414981 | 4.423886 |
| persistent source eval | 4.423165 | 0.007855 | 0.000006 |
| h copy | 0.001258 | 0.003368 | 0.002482 |
| next copy | 0.001170 | 0.001153 | 0.001171 |
| pre copy | 0.000223 | 0.000205 | 0.000207 |
| rebind/bookkeeping | 0.000015 | 0.000016 | 0.000015 |
| inclusive detach total | 4.426134 | 4.427812 | 4.428052 |
| post-prepayment remainder | 4.426134 | 0.012831 | 0.004166 |

Correction note: the previous derived row named `prepayment + detach` double-counted diagnostic prepayment, because total detach was already inclusive of diagnostic prepayment in the carry/persistent probe implementation. Raw measurements remain valid.

Carry-prepayment run: `DS41F_P8_DIAGNOSTIC_BARRIER=carry`, 16384, cold+2 warm. It moved almost all subsequent `persistent_source_eval` to near-zero (~0.0079s). The inclusive detach total remained ~4.43s because the diagnostic prepayment is inside the P6 detach interval; the post-prepayment remainder is ~0.0128s. This indicates the carry barrier materializes the same upstream work before the named persistent eval substep; it is not an optimization and it was not an 8.8s path.

Persistent-prepayment run: `DS41F_P8_DIAGNOSTIC_BARRIER=persistent`, 16384, cold+2 warm. The prepayment (~4.42s) matches baseline `persistent_source_eval`; subsequent `persistent_source_eval` becomes near-zero (~6 microseconds). The inclusive detach total remained ~4.43s and the post-prepayment remainder is ~0.0042s. This validates that the dominant mandatory materialization cost can be moved earlier by instrumentation, not removed.

### Target decision

Decision: **NO_TARGET_YET**.

The refreshed evidence shows the P6 detach wall time is predominantly the required materialization point for upstream model computation producing persistent KV/index/cache state. The persistent prepayment cleanly moves the named eval cost earlier; owned copies, rebind/bookkeeping, slice-update write count alone, and Engram concat lineage do not account for a material portion after persistent state is already evaluated. No optimization target is selected from this evidence. The next P8 investigation should return to actual model hot regions that produce the persistent tensors, rather than arena bookkeeping or removing the mandatory `mx.eval` boundary.

## 18. Upstream materialization staircase probe (non-qualifying)

Implementation location: `tools/run_p8_same_process_cold_warm.py --p8-upstream-staircase`. The tool monkeypatches `OfficialFP8MLXBlockRunner.execute_command` only inside the diagnostic worker process. It does not add normal runtime barriers and does not change cache contents, write/repack tensors, export continuation state, add `mx.compile`, or alter the production selector.

Artifact: `artifacts/p8-attribution/p8-upstream-staircase-16384.json`. Classification: `NON_QUALIFYING_ATTRIBUTION_PROBE`. Run shape: 16384, P7 enabled, P8 verification on, cold + 3 warm, fresh process.

Total prefill context only: cold 16.644413s; warm1/2/3 16.582033 / 16.644607 / 16.635724s; warm median 16.635724s. This is not an optimization comparison because intermediate barriers intentionally perturb laziness/overlap. P7 evidence remained correct: foreground Engram fallback 0, background Engram reads 32 per run, final cache frontiers all 16384.

### Staircase cut timings

| cut | cold | warm1 | warm2 | warm3 | warm median | persistent values at cut | known logical bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUT A layers 0..4 | 2.093937 | 2.104484 | 2.143912 | 2.127007 | 2.127007 | 9 | 3,254,272 |
| CUT B layers 5..9 | 1.335398 | 1.335355 | 1.345423 | 1.335528 | 1.335528 | 18 | 6,508,544 |
| CUT C layers 10..14 | 0.563182 | 0.564567 | 0.563399 | 0.563233 | 0.563399 | 27 | 9,762,816 |
| CUT D layers 15..19 | 3.844412 | 3.842966 | 3.844719 | 3.842584 | 3.842966 | 32 | 10,100,736 |
| CUT E layer20 source publication | 0.311084 | 0.299448 | 0.299260 | 0.303500 | 0.299448 | 36 | 15,933,440 |
| final residual P6 persistent eval | 0.007778 | 0.009100 | 0.007806 | 0.008625 | 0.008625 | 34 | 15,933,440 |

Warm-median staircase attribution shares within staircase materialization total only:

| region | materialization wall | share of staircase materialization |
|---|---:|---:|
| layers 0..4 | 2.127007 | 26.01% |
| layers 5..9 | 1.335528 | 16.33% |
| layers 10..14 | 0.563399 | 6.89% |
| layers 15..19 | 3.842966 | 47.00% |
| layer20 source publication | 0.299448 | 3.66% |
| final residual P6 eval | 0.008625 | 0.11% |

Staircase warm-median sum `CUT A..E + final residual` = 8.176973s, versus baseline qualifying `persistent_source_eval` warm median = 4.423165s. This is materially larger (~1.85x), so the intermediate barriers perturb execution/fusion/overlap enough that the absolute split is suspect. The near-zero final residual validates that the bundle prepaid the relevant persistent graph, but the split should be used only as coarse qualitative evidence.

### Staircase memory snapshots, cold run

| cut | active before | cache before | peak before | active after | cache after | peak after | carry shapes |
|---|---:|---:|---:|---:|---:|---:|---|
| CUT A | 304333395528 | 14711498889 | 307085808837 | 305860219700 | 15275649949 | 309423871064 | current/next `[1,16384,4,5120]`, pre `[1,16384,4]` |
| CUT B | 308389271092 | 14779410677 | 311416804621 | 309240615808 | 15280630697 | 312804267172 | current/next `[1,16384,4,5120]`, pre `[1,16384,4]` |
| CUT C | 312814261792 | 14606657891 | 315600246016 | 312654566348 | 15458692023 | 316218217712 | current/next `[1,16384,4,5120]`, pre `[1,16384,4]` |
| CUT D | 312621012484 | 15492245949 | 316218217712 | 315998491684 | 15492246429 | 319528588616 | current/next `[1,16384,4,5120]`, pre `[1,16384,4]` |
| CUT E | 314812717356 | 17400637212 | 319528588616 | 314877073288 | 22500177768 | 319528588616 | current `[1,2414,4,5120]`, next `[1,2541,4,5120]`, pre `[1,2541,4]` |

### Investigation target decision after staircase

Decision: **NO_TARGET_YET**.

Reason: the staircase was complete enough to reduce final residual persistent eval to ~0.0086s warm median, but its materialization sum is materially larger than the original ~4.42s boundary. Therefore the barrier-induced split is not reliable enough to select `COMMON_ENCODER_BLOCK_COMPUTE`, `ENGRAM_SPECIALIZED_COMPUTE`, `SOURCE_LAYER_PERSISTENT_COMPUTE`, or `LAYER20_SOURCE_PUBLICATION`. It does not justify returning to arena bookkeeping; those paths remain excluded as primary 4-second contributors.

Next diagnostic contract if this is revisited: design a less-perturbing upstream compute attribution method before component-level profiling. If a future low-perturbation pass points to ordinary encoder math, select one non-Engram/non-source layer such as layer 17 and decompose using the pinned-oMLX Block order only: attn HC mixes, attn pre-norm, Attention, attn HC post, ffn HC mixes, ffn pre-norm, MoE, ffn HC post. Do not duplicate or rewrite the math in the diagnostic.

## 19. Phase transition: E2E representation architecture audit (2026-10-01)

Barrier-based attribution is complete. The current conclusion is frozen:

- The ~4.42 s P6 `persistent_source_eval` is primarily a materialization point for upstream lazy computation.
- It is not evidence that P6 detach bookkeeping or owned cone copies themselves cost ~4.42 s.
- Carry/persistent prepayment only moves the same upstream lazy computation to an earlier `mx.eval`.
- The five-cut materialization staircase is preserved as raw evidence, but is now classified as **coverage validation: useful** and **performance attribution: too perturbing** because it inflated materialization by ~1.85x.
- Further `mx.eval` insertion is rejected as the primary P8 attribution method because it changes laziness, fusion, and overlap.

The primary P8 question is no longer “which is slower: Attention, MoE, or HC?” It is now:

```text
Can the same correct model computation be expressed with fewer E2E representation boundaries?
```

New audit deliverable: `docs/p8-e2e-dataflow-representation-audit.md`.

Selected architecture-level candidate: **TILE_NATIVE_CARRY**. This is not implemented in this task. The candidate preserves DwarfStar command geometry, publication visibility, DeepseekV41Cache semantics, SSD Engram donor behavior, P6 ownership/materialization, and P5 no-replay handoff, while targeting repeated implementation-only boundaries:

```text
Block h/pre output
 -> dense carry write
 -> layer swap
 -> dense carry slice
 -> next Block
```

Candidate shape:

```text
TileSpan / TileCarryState
 -> unchanged oMLX Block on tile tensors
 -> next-layer tile object directly
 -> assemble once only at layer19->20 if full-source publication requires it
 -> direct final-cone extraction from retained tiles
```

For the 16384 source path, the structural before/after target is 120 dense carry writes -> 0 dense writes, 158 carry slices -> near-zero for layer transport, full-range layer assembly every layer -> at most one layer19->20 assembly, while keeping P6 persistent materialization unchanged. 8192 and A-24577 counts are recorded in the audit.

Kernel/component optimization is explicitly deferred: no Attention-specific, MoE-specific, HC-specific, custom Metal, or additional `mx.compile` work should become primary until tile-native carry is either implemented and qualified or proven not viable/no-op. Future performance experiments must use E2E 8192, 16384, and A-24577 cold/warm measurements, not new internal eval barriers.

## 24. TILE_NATIVE_CARRY implementation and A/B result (2026-10-01)

Implementation status: **implemented behind disabled A/B switch** `DS41F_P8_TILE_NATIVE_CARRY=1`. Default remains OFF and the production selector is unchanged.

Implemented pieces:

- `TileSpan`: immutable metadata plus tensor reference (`value`, `logical_offset`, `rows`, `role`, optional absolute-start metadata). Creating a span does not copy tensor content.
- `TileCarryState`: request-local current/next/pre tile maps built from actual `ENCODE_ROWS` command spans. It follows planner offsets/row counts and does not define an independent tiling policy.
- Tile-native source transport in `OfficialFP8MLXBlockRunner`: source/encoder `ENCODE_ROWS` obtains h/pre directly from tile maps, calls unchanged pinned oMLX Block, and binds `h_out`/`pre_out` as next-layer tiles instead of calling `_write_rows`.
- Logical `SWAP_HC_AFTER_LAYER` remains in the command stream and command history; the tile bridge performs the physical tile-vector swap and preserves ping-pong spare semantics.
- Layer19 full-source bridge: ordered source tiles are assembled exactly once for current `publish_full_source(h_full, pre_full, ...)`. Single-tile source can reuse the tile directly.
- P6 final cone extraction: after unchanged `_p6_eval_persistent_source_state()`, the retained h/next/pre cones are extracted from tile spans and then copied through the same owned compact-copy proof path.
- Candidate admission is request/segment-start only. If not admitted before execution, dense remains available. Once tile-native execution begins, invariant failure aborts the request rather than falling back mid-request.

Structural tests: `PYTHONPATH=tests /Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 -m unittest discover -s tests -v` passed **97 tests OK**. Added tests cover default-off switch, admission, exact lookup, two-tile transport, logical swap, missing tile fail-closed, gap/overlap rejection, ordered full-source assembly, single-tile no-op assembly, same-tile/cross-tile cone extraction, and retirement of tile parents.

### 16384 same-revision A/B

Artifacts:

- dense control: `artifacts/p8-tile-native/dense-16384-v2.json`
- tile candidate: `artifacts/p8-tile-native/tile-16384-v2.json`
- P5 dense smoke: `artifacts/p8-tile-native/dense-p5-16384.json`
- P5 tile smoke: `artifacts/p8-tile-native/tile-p5-16384.json`

Both A/B runs used fresh process/model, deterministic 16384 tokens, P7 enabled, P8 verification, cold + 3 warm. No new eval barriers, no `mx.compile`, no custom Metal, no donor oMLX changes.

| path | cold s | warm1 | warm2 | warm3 | warm median | warm range | warm tok/s | bg Engram reads/run | fg fallback | cold write_rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 16.365340 | 16.338032 | 16.349117 | 16.207646 | 16.338032 | 0.141471 | 1002.81 | 32 | 0 | 120 |
| tile-native | 16.353992 | 16.245373 | 16.190254 | 16.248494 | 16.245373 | 0.058240 | 1008.53 | 32 | 0 | 40 |

Structural operation result for cold 16384:

- dense source+suffix `_write_rows`: 120 total.
- tile-native `_write_rows`: 40 total, all decoder suffix; source/encoder dense carry writes were eliminated.
- tile telemetry at P6 detach: `dense_carry_writes=0`, `dense_layer_transport_slices=0`, `logical_swaps=20`, `tile_input_bindings=80`, `tile_output_bindings=80`, `full_source_assemblies=1`, `final_cone_tile_slices=3`, `final_cone_minimal_concats=0`, `initial_tile_views=4`.

State/P7/P5 evidence:

- all 40 cache frontiers reached 16384 in both A/B runs;
- P7 foreground fallback stayed 0 and expected background microtile reads were preserved;
- cache slot shape/dtype inventory in P5 matrix remained valid for source layers and tail layers;
- P6 C/E/D/T completed at 16384;
- source coverage remained `{2:16384, 8:16384, 14:16384, 20:16384}`;
- P5 tile smoke PASS: prompt replay `0`, full cache repack `0`, export `false`, generated tokens `[1, 0]`, decode frontiers advanced.

Memory proxy from final P6 materialization boundary after-state:

| path | active bytes cold | cache bytes cold | peak bytes cold | active bytes warm typical | cache bytes warm typical | peak bytes warm typical |
|---|---:|---:|---:|---:|---:|---:|
| dense | 314,864,784,632 | 17,004,751,578 | 319,629,252,136 | ~315,152,036,4xx | ~17.38-17.42B | 319,916,503,956 |
| tile-native | 314,864,915,708 | 18,558,717,170 | 318,765,159,990 | ~315,152,167,5xx | ~18.93-18.98B | 319,052,411,806 |

Interpretation: tile-native eliminated the intended source carry writes/slices, but warm median improvement was only ~0.092659 s (~0.57%), smaller than the same-revision dense warm run-to-run range (~0.141471 s). Candidate cache-memory proxy was also ~1.5 GB higher at the sampled final boundary, while peak proxy was lower. This is not a repeatable practical E2E gain.

### P8 completion decision

Decision: **TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN**.

Because the coherent structural candidate removed the intended operations but did not produce repeatable E2E value outside observed noise, P8 optimization search is **COMPLETE**. The qualified production candidate for Milestone 6 is the existing dense P0-P7 path. Tile-native carry remains retained experimental code, default OFF, and not production-selected. Per P8 stop rule, do not continue into Engram concat tweaks, Attention/MoE/HC profiling, `mx.compile`, or custom kernels under this task. Attention/MoE/HC further optimization is not pursued / not required for M6 unless a future explicit product/runtime requirement reopens component optimization.
