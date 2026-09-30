# Command-local oMLX suffix math closeout (after 004734e)

Historical implementation closeout. Real-smoke status and connected-path completion evidence are superseded by [P3/P4 real MLX qualification](p3-p4-real-mlx-qualification.md) from `05da536`.

## Authorities and scope

- Execution topology: `antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac`, specifically `ds41_decoder_prepare` and the surrounding command/dependency-cone execution. The full layer20 source prepare and the immediately preceding 127 local dependency rows remain distinct planner commands.
- Mathematical donor: oMLX baseline `b390b31e0c6831225fed0f24d278eb1db7fcb68b`, reviewed against `jundot/omlx@36493634c1d602abbc4293f8d12590008a7209b8` (`language.py`: Block `ced_tail`, Attention `ced_kv/ced_kv_start`, Indexer `latent_start`, stale-window removal and stored-window length).
- No CED scheduler, whole-layer-loop policy, installed-package patch, runtime selector change, P5/P6/P7 work, or benchmark was introduced.

## Math changes

`ds41f_mlx/prefill_fp8_mlx/omlx_suffix_math.py` now executes explicit command-local Block math. Unknown semantics fail with `SuffixMathError`; tests may supply explicit hooks. There is no ordinary `Block.__call__` fallback.

Local prepare is HC pre-norm -> loaded `_input_projections` KV result -> loaded KV norm -> official RoPE at command absolute positions -> official packing -> replacement of cache1 (truncated only to normal window size). It never reads/concatenates stale cache1, and never advances cache0. Queries use `min(start, window_size, stored_old_length)`. A first query after prepare therefore has 127 preceding rows plus its own row. The explicit official sparse-attention implementation is used, not the long-prefill fused kernel that assumes a full old window.

Full-source prepare does HC pre-norm and calls the loaded compressor exactly once. It appends/replaces the proper cumulative prefix in cache2 with official compressed RoPE and 4-bit/group16/e4m3 packing. Index K is factored from Indexer: loaded `wk` -> loaded `k_norm` -> compressed-position RoPE -> 4-bit pack -> prefix-trim/append cache3. No full-prompt query projection, top-k, idx, or candidates is produced. For layer20 ratio=1 the official compressor leaves pending slots4/5 unchanged; other ratios retain the loaded compressor's pending-state semantics.

Layer20 queries use producer-private prepared KV/index K, without calling the compressor, wk, or writing cache2/3. The path is official HC mixes/pre-norm, query and local KV projection/norm, query/local RoPE, query index projection/weights and top-k/candidates, sparse attention, inverse RoPE/output projection, HC post, FFN HC mixes/pre-norm, loaded official MoE, HC post, h/pre return. Layers21..39 use the same explicit Block path and actual old-window length. Index-source refresh layers use the factored query-only index path, including candidate-restricted ranking. The runner captures idx/candidates for each exact command span.

## Publication and frontiers

- Ordinary compressed consumers require KV + idx (+ candidates after layer20).
- Index refresh consumers require KV + index K (+ candidates), not previous idx.
- KV/index producers use their own source path.
- `producer_shared_for_span` overlays only the executing layer's pending cumulative state. Consumer shared dictionaries never expose layer20's pending source. Only `PUBLISH_FRONTIER(layer=20)` changes global visibility.
- A new row-span producer supersedes the previous producer. Partial consumption slices exact requested rows instead of passing an oversized tensor to a smaller query cone.
- Full-source/local prepare leave cache0 unchanged. Commit asserts every one of the 40 logical frontiers equals `base_frontier + plan.count`.

## Validation

`python3 -m unittest discover -s tests -q`: **34 tests passed**. New regressions execute operation-recording HC/projection/RoPE/packing/compressor/index/sparse/MoE adapters, not merely scheduling flags. They cover fallback rejection, stale-window removal and exact row identities, first-query geometry, separated source/query production, immutable cache2/3 through multiple queries, candidate/idx spans, refresh topology, private publication, and three complete live wide sweeps. Each sweep checks all 40 frontiers and one full-source compressor call.

`python3 -m compileall -q ds41f_mlx/prefill_fp8_mlx tools/run_prefill_fp8_mlx_smoke.py`: passed. Recording tests establish structure, **not real numerical equivalence**.

## Bounded real smoke

Tool: `tools/run_prefill_fp8_mlx_smoke.py`. It loads the intended DeepSeek-V4.1 checkpoint with MTP disabled and drives only planner -> executor setup -> plan commands -> official block runner. It materializes command results explicitly and reports structure, without timing or throughput. Report fields include all 40 frontiers, layer20 cache2/cache3 and pending geometry, all local windows, full-source prepare count, continuation export count, and cache repack count.

```bash
python3 tools/run_prefill_fp8_mlx_smoke.py --tokens 2048 --mtp-off --no-benchmark
python3 tools/run_prefill_fp8_mlx_smoke.py --tokens 8192 --mtp-off --exercise-suffix --no-benchmark
```

Both commands were attempted here and returned `NOT RUN: No module named 'mlx'` (exit 2).

- Real short smoke: **NOT RUN**.
- Real wide smoke: **NOT RUN**.
- Numerical parity/model-memory execution: unverified until real MLX/model runs are available.

## ArchitectureCompletion

No completion telemetry was promoted from these recording tests. All current default fields remain false:

- sweep_planner_owns_order
- whole_prefix_layer_loop_absent
- dwarfstar_carry_lifetime
- deferred_decoder_suffix_lifetime
- frontiers_drive_execution
- scheduling_hooks_effective
- materialization_boundaries_explicit
- no_cpu_hot_path_roundtrip
- no_intermediate_cache_repack
- one_live_cache_handoff_no_replay

P5 handoff and later phases remain out of scope.
