# M50 stopped-boundary evidence

Decision: **NO / BLOCKED; speculative transaction not implemented or qualified.**
Base `5e784d9e83a85c497cd192f87184e9b01a1343eb`. Runtime source/guards unchanged.
Architecture and state inventory: `docs/milestone-50-canonical-speculative-target-transaction.md`.

## Fresh results

- `state-boundary.json`: real official checkpoint, first-party model and active
  M47 admission. Exit 0, expected BLOCKED decision, not a qualification PASS.
- Frontier 4095 → 4103, eight canonical one-token OFF commits. All 40 frontiers,
  all 280 slot shapes/dtypes/raw-byte hashes and admission metadata observed.
  Exact packed retained-window bytes and compressed/index old-prefix bytes match
  each preceding OFF state. All 40 layers evict an oldest window row per step
  (window 128). Source layers 2/8/14 consume the pre-pooling KV/gate remainder at
  4096, 4098, 4100, 4102. No undo publication is available in the current producer.
- Several-position shapes reject before mutation; all hashes unchanged. One-token
  preflight remains admissible. No speculative mutation was attempted.
- History equals prompt plus consumed OFF tokens; exact-list idle cancellation /
  return, no pending sample or target-pending marker. SSD prefetch drained;
  retired M47 capability rejects new execution.
- Prompt replay **0**, full-cache repack **0**, P5 handoff **1**, OFF control only.
- `owner-regressions.log`: **97 passed**. Existing layer/producer/barrier faults,
  aliases, cancellation, SSD future drain and admission retirement covered with
  existing test fixtures. These are NOT real-checkpoint speculative fault tests.
- No accepted-prefix OFF/native oracle comparison: no speculative transaction
  exists. No all-accept, partial-accept, reject-to-one, speculative cancellation,
  rollback cycles or journal allocation/retirement PASS is claimed.
- `source-audit.json`: binds probe, unchanged canonical seams, result/log bytes
  and installed native donor sources inspected as algorithmic reference only.

Commands (existing standard-OFF environment; no environment/package changes):

```sh
.venv/bin/python -m tools.probe_m50_state_boundary \
  --output artifacts/m50/state-boundary.json
.venv/bin/python -m pytest -q \
  tests/test_m44_target_generation.py tests/test_m45_target_forward.py \
  tests/test_m46_state_production.py tests/test_m46_completion.py \
  tests/test_m46_engram_reads.py tests/test_m47_resource_admission.py \
  tests/test_m48_model_execution.py
.venv/bin/python -m py_compile tools/probe_m50_state_boundary.py
```

No full R1 rerun, native candidate execution, performance qualification, release
extraction or runtime promotion. Supplied uncommitted M49 files and unrelated
worktree changes were preserved, not included in this M50 commit.
