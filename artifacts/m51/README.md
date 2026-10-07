# M51 evidence — PASS, state primitive only

Architecture: `docs/milestone-51-accepted-prefix-journal.md`.
Baseline: M50 `bdd08e715d884e453f5b846c3ab8856253d2a931`.
No generation acceptance/scheduler integration or runtime promotion.

## Final evidence

- `checkpoint.json` / `.log` / `.exit`: **PASS / YES_PRIMITIVE_ONLY**, exit 0;
  fresh official checkpoint with active first-party M47/M48 resource admission.
  Runtime 553.9 s including cold complete payload admission.
- B=8 tentative positions under one pending exact-list lease. Repeated accepted
  counts **8 / 4 / 1 / 0 / 6**, frontier **4095 → 4114**. All 280 slots and row
  admission metadata match independent canonical OFF shape/dtype/raw bytes.
- Fresh independent-prefix trials cover **every k=1..7**. Actual compression group
  completion and chronological window eviction occur in every tentative span.
- Cancellation before and after completion restores exact initial OFF state.
- **15 real-checkpoint faults**: before/after materialization, halfway through
  state materialization, prepare, publish slots 0..6, lengths, left_padding,
  publication boundary, and parent frontier hook. All 40 aliases burn and further
  target admission rejects; journal references retire in every failure case.
- Successful retirement has no journal payload/output/cache/model/stream handles.
  Subsequent OFF owner returns the **same list** at idle. SSD futures drain;
  model close revokes admitted resource capability.
- **28 fresh setups** each record replay **0**, full-cache repack **0**, P5 handoff
  **1**. Oracle prefills are independent test setup, not rollback replay.
- Matched frozen M50/current OFF, 32 positions: all state bytes identical;
  median latency **50.441 → 50.691 ms** (**+0.50%**, small sample only).
- `owner-regressions.log`: **136 passed** (97 retained + 39 journal/guard tests),
  including max B=32 / accepted 0,1,19,31,32 on real MLX reduced state; context-
  independent payload at F=5/1001/10001; exact-list replacement burn; owner close;
  metadata corruption/admission overrun; context-managed cancel/burn.
- `prefix-storage.json` / `.log`: physical allocation PASS. Bounded byte-copy
  payloads do not retain large parents. Plain prefix slices can retain a tentative
  parent; `array`/`contiguous` do not universally detach same-bucket slices.
  Explicit gather creates separate canonical packed-byte outputs at actual row
  widths **528/288/68**; all aliases retired returns active memory to zero.

## Costs and limits (not hidden by the counters)

Journal payload before settlement: **267,480–279,768 bytes** at B=8, independent
of context. Initial tails plus new pre-pooling projections/gates and evictions
are detached bounded storage. `payload_bytes` counts bytes copied, not allocator
capacity; bounded canonical remainder copies at preparation can add to that count.

Settlement copies already packed selected window/prefix bytes, **2,703,360 bytes**
on all accept and **6,358,056–6,364,820 bytes** in the repeated partial/reject trials.
Shortened compressed/index source prefixes require **context-sized byte copies**;
M51 does NOT claim O(B) settlement work or zero memcpy. These outputs are canonical
physical state, not journal snapshots. No pooling, quantization or full-cache
repack runs at settlement. B=32 is reduced-state tested; fresh checkpoint runs use
B=8. No native rollback execution oracle was run; native donor semantics are
reference only, independent OFF is the state oracle.

Burned cache aliases are inert, not a recovery checkpoint. Their holders must
release those invalid objects; retirement clears the child resources and drains
work, it does not promise to erase arbitrary externally retained Python arrays or
exception tracebacks. OFF has no journal allocation/copy path.

## Historical attempts

- `checkpoint-initial.*`: preliminary numerical/prefix/cancel/seven-fault run;
  FAILED only at its final observer (incorrect coordinator `_prefetched` attribute).
  Not used as PASS evidence. The corrected observer checks each embedding.
- `checkpoint-pre-regression.*`: corrected state/fault PASS, preceding the final
  matched OFF performance/control run. Final completion uses `checkpoint.*`.
- `source-audit.json`: final source and evidence SHA-256 bindings, plus baseline.

## Reproduce (existing qualified environment; no package changes)

```sh
.venv/bin/python -m tools.probe_m51_accepted_prefix \
  --output artifacts/m51/checkpoint.json
.venv/bin/python -m tools.probe_m51_prefix_storage \
  --output artifacts/m51/prefix-storage.json
.venv/bin/python -m pytest -q \
  tests/test_m44_target_generation.py tests/test_m45_target_forward.py \
  tests/test_m46_state_production.py tests/test_m46_completion.py \
  tests/test_m46_engram_reads.py tests/test_m47_resource_admission.py \
  tests/test_m48_model_execution.py tests/test_m51_accepted_prefix.py
```

The next boundary is M50's generation-owned history/lookahead/RNG/terminal/response
integration with the child publication receipt. M51 installs no acceptance loop.
