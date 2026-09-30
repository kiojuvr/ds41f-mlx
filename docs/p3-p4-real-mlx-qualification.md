# P3/P4 real MLX qualification from 05da536

Status: **P3/P4 bounded real qualification passed; separate P5 handoff task may begin. STOP here.** This is not production selection, a full architecture gate, performance evidence, or broad model numerical qualification.

## Actual runtime authority

Recorded **before model execution** in `artifacts/p3-p4-real-mlx-qualification/runtime-authority.json`:

- Target: arm64 Mac Studio, macOS 26.5.2, 512 GiB physical memory.
- Python: `/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3`, 3.13.15. System `/usr/bin/python3` is 3.9.6 and is not the qualification interpreter.
- oMLX distribution and exposed package version: **0.7.0.dev2**.
- MLX: **0.32.2**, Metal available.
- Imported language module: `/Users/kioju/omlx-0.7.0.dev2/omlx/patches/deepseek_v41/language.py`.
- SHA256: `2c64bef36c9fc4a6007cd34e7b120142ced072a8d491cf1f83f2c5b1a956763f`.
- Git revision: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`.
- Checkout is **not clean**: pre-existing changes in `encoding.py` and `processing.py`, plus an untracked literal-image-token test. These files were not modified in this task. Their digests and the tracked diff are captured in the evidence directory.
- All reviewed module operations are exposed. Actual signatures are recorded; the numerical qualification also binds the adapter's positional/keyword contracts before loading. There was **no API drift blocker**. Loaded layer modules were exercised successfully, not merely inspected.

The runtime is recorded, not required to equal the baseline. Mathematical review authorities remain `b390b31...` and the `36493634...` CED donor; no installed package was patched for suffix execution.

Checkpoint: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`. Config/index/tokenizer digests are in `environment-and-checkpoint.json`. Inputs are the smoke's fixed `[1]` token ID repeated 2048/8192 times. This is a bounded fixture, not broad prompt coverage.

## Short smoke: initial failure and minimal fix

The first requested 2048 smoke **failed before planner/setup/block execution**. Classification: **environment/dependency — memory residency configuration**.

The diagnostic rerun traced Metal insufficient-memory failure to official `loading.load -> DiskEngramEmbedding.make_resident -> _resident_buffer -> mx.synchronize`. The tool inherited `engram_ssd_offload=False`. Checkpoint tensor headers account for 202,758,032,400 bytes of Engram tables plus 307,527,990,600 bytes of non-Engram storage. No suffix math was implicated.

Smallest fix: pass `engram_ssd_offload=True` to the existing official loader, matching the already-reviewed target runtime setting. No weights, operators, scheduler, residency optimization, fallback, or serving selector was changed. Committed separately as **a2b06a5** (`fix: use reviewed SSD Engram mode for real prefill smoke`).

Evidence: `short-initial.log`, `short-load-diagnostic.log`, `load-memory-diagnosis.json`.

## Real structural smoke results

Run with the target interpreter selected explicitly:

```bash
export PATH=/Users/kioju/.venvs/omlx-0.7.0.dev2/bin:$PATH
python3 tools/run_prefill_fp8_mlx_smoke.py --tokens 2048 --mtp-off --no-benchmark
python3 tools/run_prefill_fp8_mlx_smoke.py --tokens 8192 --mtp-off --exercise-suffix --no-benchmark
```

The wide smoke was run **only after** the corrected short smoke passed.

| Invariant | 2048 short | 8192 suffix |
|---|---|---|
| Real checkpoint load, planner topology, embedding/Engram setup, command execution | PASS | PASS |
| All 40 final cache frontiers | 2048 | 8192 |
| Layer20 cache2 (packed compressed KV, uint8) | `[1,2048,288]` | `[1,8192,288]` |
| Layer20 cache3 (packed index K, uint8) | `[1,2048,68]` | `[1,8192,68]` |
| Layer20 ratio-1 slots4/5 | both `[1,0,512]` bfloat16 | both `[1,0,512]` bfloat16 |
| All 40 local-window caches | `[1,128,528]` uint8 | `[1,128,528]` uint8 |
| Adapter full-source prepare count | 0 (ordinary source path) | 1 |
| PrefillContinuationState exports | 0 | 0 |
| Full-cache repacks | 0 | 0 |
| Final-prefix-logits requirement | absent/suppressed | absent/suppressed |

Complete original smoke output and exit status are retained in `short-ssd-engram.log/.exit` and `wide-initial.log/.exit`. These are real passes, not inferred from recording tests. The short run executed the ordinary non-suffix layer20 source path successfully. Persistent slots were non-null at commit and accepted by real `mx.eval`; the qualification rerun additionally records every layer/slot's geometry.

## Bounded numerical and operation evidence

Only after both smokes passed:

```bash
python3 tools/qualify_prefill_fp8_mlx.py \
  --out artifacts/p3-p4-real-mlx-qualification/bounded-numerical.json
```

This evidence-only tool loads the same model/settings and drives the same planner/setup/command/runner pipeline. Process-local recording wrappers call the original loaded operations and are restored on exit; they do not change the installed package or operator math.

Results: **PASS**, 289 planner commands, all layers 0..39 executed.

1. **Same-input source comparison:** immediately after full-source prepare, compare adapter cache2/cache3 against the loaded ordinary oMLX `Attention.__call__` source path with identical full encoder-final HC pre-norm input and absolute start 0. The reference attention output is not evaluated or used as a decoder oracle. Both compressed KV and index K have **zero mismatched packed bytes**, identical dtypes/geometries. This checks the source split independently against the combined ordinary path.
2. **Actual source operations:** command execution calls the loaded layer20 compressor exactly once (`start=0, rows=8192`) and loaded indexer `wk` exactly once (`rows=8192`). The reference comparison's one compressor/wk invocation is recorded **separately** and excluded from the command count. Every layer20 query chunk preserves cache2/cache3 byte-for-byte against the independent reference source.
3. **Real stale-window injection:** before each decoder local prepare, inject a deliberately non-contiguous 128-row packed cache1 filled with byte 255. Prepared cache1 exactly equals an independently executed current dependency-row HC pre-norm/KV projection/norm/absolute RoPE/pack path: **zero mismatched bytes** for all layers20..39, each `[1,127,528]` uint8. All prepare commands leave slot0 unchanged.
4. **First-query geometry:** each decoder layer's first query records old length **127**. Layer20 starts at 5778; layer39 at 8191. Final local caches are all `[1,128,528]` uint8.
5. **Fallback/source/publication safeguards:** an instrumentation guard rejects any ordinary decoder `Block.__call__`; observed calls: **zero**. Full-source prepare creates no row spans and the layer20 source remains producer-private before the publication frontier. Queries do not regenerate/append global sources.
6. **Final structure:** all 40 frontiers equal 8192, ratio-1 pending slots empty, zero continuation exports/repacks, final-prefix logits suppressed and not computed. Full per-layer slot geometries are retained in the result.

Policy: `docs/correctness.md`. Exact bytes are required here **only because the source/window comparisons use identical loaded operations on the same input**. No tolerance was fitted, and no cross-backend canonical reduction ordering was introduced. No final hidden/logit equality is imposed against ordinary full decoder replay: the exact DwarfStar dependency cone intentionally differs from full replay, so whole-decoder equality is not established by construction. Encoder-side whole-path comparison, final-token behavior, broad determinism, varied prompts, and real multi-sweep continuation are **not claimed** by this bounded numerical check.

**First proven semantic divergence: none** in the checked boundaries. No numerical fix, speculative optimization, or further model investigation was performed after these passes.

## ArchitectureCompletion reassessment

Definitions come from the architecture plan's **production-prefill** structural gate. Production still selects the old path; do not promote global serving completion flags from this diagnostic connection. Record the following new-path evidence separately (also in `qualification-summary.json`):

| Field | New connected path evidence | Complete production gate |
|---|---|---|
| sweep_planner_owns_order | **qualified within new path**: real command iteration drives execution | not production-selected |
| whole_prefix_layer_loop_absent | **qualified within new path**: no independent transformer loop | not production-selected |
| dwarfstar_carry_lifetime | partial: real request arena/ping-pong/views execute; full lifetime/retirement gate not qualified | false |
| deferred_decoder_suffix_lifetime | suffix work cone executes, but full-size carry still retained; deferred/resume lifetime is P6 | false |
| frontiers_drive_execution | **qualified within new path**: publication visibility, prepares and all 40 logical frontiers | not production-selected |
| scheduling_hooks_effective | P7 hooks remain incomplete/no-op | false |
| materialization_boundaries_explicit | **qualified within this diagnostic path**: explicit command materialization and gated comparison checks | production/P7 grouping not qualified |
| no_cpu_hot_path_roundtrip | not qualified; cache-frontier scalar reads and official Engram disk/host handling remain to scope | false |
| no_intermediate_cache_repack | **qualified within new path**: real runs report zero | not production-selected |
| one_live_cache_handoff_no_replay | not exercised/implemented for this package; P5 | false |

The complete P0-P7 structural gate remains **false**, specifically including P5 handoff, P6 deferred lifetime, P7 scheduling, and production selection. This is not a mechanical all-false qualification report: connected-path facts above are positively qualified while their production scope remains explicit.

## Completion / stop

- Real 2048 smoke: **PASS** after the separately committed loading fix.
- Real 8192 suffix smoke: **PASS**.
- Bounded same-input numerical/state checks: **PASS**, no proven mismatch.
- Regression tests in the target Python environment: **34 passed**; both tools compile; whitespace checks pass.
- **P3/P4 qualification is sufficient to begin a separate P5 handoff task. P5 was not begun.**
- No runtime selector change, benchmark, throughput result, speculative optimization, P6 implementation, or P7 implementation.
