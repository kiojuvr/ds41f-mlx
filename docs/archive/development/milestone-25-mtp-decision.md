# M25 — pinned upstream MTP lifecycle decision

## Decision: REJECT/DEFER MTP; production default remains OFF

MTP is **not qualified as optional** and no serving/configuration switch is
exposed. The upstream decode loop was exercised against the official checkpoint,
but the existing extraction API does not provide ds41f's arbitrary committed
idle boundary. A throughput improvement cannot compensate for this prerequisite
failure. This is a lifecycle rejection of the pinned integration, **not** a claim
that speculative decoding is mathematically incorrect or intrinsically slow.

Production remains the M24 control: DENSE_P0_P7, resident backbone/SSD Engram,
P5 terminal holdout, upstream GenerationBatch, MTP OFF, deepseek-recipe. No
prefill, P7, P5, recipe, Rust, packaging or kernel implementation was changed.

## Exact upstream audit

Authority: oMLX `v0.7.0`, revision
`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, clean tracked source;
mlx-lm `0.31.4.dev132+g94cdcae13`, MLX `0.32.2`, Python `3.13.15`.
Executable/native/checkpoint identities are recorded in `artifacts/m25/quick-final.json`;
probe-specific source hashes, configurations and packages are in `on.json` and
`off.json`. Historical M20/M24 evidence is not relabeled as fresh MTP evidence.

The actual V4.1 adapter differs from generic DeepSeek-V4 comments in the shared
patch package:

- `omlx/patches/deepseek_v41/loading.py`: official-source loading with
  `preserve_mtp=True` retains and materializes draft weights. Backbone and SSD
  Engram loading remain upstream-owned.
- `language.py`: `LanguageModel` inherits `DSparkMixin`, constructs stages via
  `dspark.make_stages`, and exposes the native MTP hooks. The checkpoint itself
  declares block size 5 and target-layer capture IDs. There is no independent
  legacy V4.1 MTP head selection in this pinned adapter. Reusing those hooks is
  a dependency observation, not a separate ds41f DSpark project or custom drafter.
- `mtp.py`: `configure_mtp` sets both native MTP and DSpark-named flags, chains
  drafts, caps depth to checkpoint block size, uses independent target verification,
  and supplies an acceptance-only controller (`min(max_depth, accepted+1)`).
  This controller never exits based on wall-clock time; generic shared-loop
  parking behavior must not be attributed to V4.1 without evidence.
- Native prompt capture in `__call__` folds selected target-layer hidden states
  into a bounded draft-context cache via `deepseek_v4_dspark.capture_prompt`.
  Absolute target offsets enforce contiguity; omitted spans reset capture.
  `take_primed` transfers that host cursor into per-UID loop state. The generic
  priming module provides UID scopes and cleanup; it delegates native take/drop
  to the V4.1 adapter. The MTP cache is a **draft context**, not a second target
  authority, but its queue and pending anchors matter to extraction.
- Importing `omlx.scheduler` installs ordinary scheduler patches, **not MTP
  dispatch**. The diagnostic explicitly calls upstream `cache_rollback.apply()`
  and `batch_generator.apply()`. Native V4.1 supplies model hooks already; no
  generic model adapter was copied or speculative loop reimplemented.
- `batch_generator._post_init_mtp` forwards the sampled main token once,
  seeds two confirmed response tokens, and initializes draft history. Activation
  is lazy on the first decode call, not during terminal bootstrap.
- `_run_verify_cycle_chain` verifies `[next_main, drafts...]`. Greedy accepts a
  matching causal prefix; stochastic acceptance uses target/draft probabilities
  and residual sampling. This milestone tests greedy only. Remaining-token and
  stop matchers clamp acceptance; protocol correctness is still a separate gate.
- Native `mtp_partial_rollback` commits already-computed causal window KV,
  compression tails, compressed/index KV, offsets and Engram history from a
  verify snapshot. It clears the draft stash. This is verification rollback,
  **not** an arbitrary emitted-token-boundary rewind API.
- `_emit_response` changes emitted history separately from the target frontier;
  `GenerationBatch.extract_cache` calls layer row extraction, and
  `BatchGenerator.extract_cache` returns that row plus emitted history. Neither
  API reconciles the MTP queue. Natural finish removes MTP state after returning
  the cache. Removal/filter and close release UID priming ownership; dropping a
  UID is not a commit operation.
- `_reconcile_mtp_to_standard` builds a new cache and re-prefills the entire
  emitted history in chunks. It is called on certain fallback/reshape paths.
  This is expressly forbidden by ds41f. The probe blocks that upstream function
  and records attempted calls; it never uses replay to manufacture a pass.
- `_materialize_mtp_boundary_emit` is an upstream **block-boundary** helper,
  not a general exact-idle-extraction API. It seeds another queued token and
  retains the ordinary pipeline skew after that token. Merely enabling a block
  alignment flag is not a proof of arbitrary cancellation correctness.

## Chosen boundary architecture

The narrow experimental boundary is existing DENSE_P0_P7 → one held-out terminal
→ upstream BatchGenerator directly. It is isolated in
`tools/run_m25_mtp_boundary_probe.py`, not exposed through HTTP or a parallel
production session class. P7 remains MTP-OFF; draft priming sees terminal capture
only. Frozen prefix row views fork diagnostic scenarios without reconstructing
tensors. They are not a qualified continuation or persistence mechanism.

This deliberately does not weaken idle contracts to accept a lagging cache,
truncate token history, count queued future tokens as already delivered, replay a
prefix, or clear offsets without repairing compressor/Engram state. One lagging
terminal could potentially be forwarded once by a future adapter, but **that
alone does not solve ahead-of-emission cancellation inside an accepted queue**.
A safe future adapter needs an upstream-supported committed frontier transition,
with explicit queue/anchor ownership, snapshot lifetime and no replay fallback.
No impossibility claim is made about a future such API.

The only production code change is a fail-closed M9 save/restore guard: preserved
MTP weights or active native MTP cannot use the current OFF-only artifact. This
check runs before MLX import, filesystem access or cache mutation. M9's seven
slots per target layer do not represent pending queues, draft caches or cursors.
The artifact schema is unchanged; ordinary OFF save/restore is rechecked in a
fresh M24 workload. There is no MTP persistence/process-restart success claim.

## Evidence scope and prerequisite stop

Canonical new artifacts are under `artifacts/m25/`:

- `on.json`, `off.json`: sequential real-model A/B, 2048 and 200000-token prefixes,
  initial terminal bootstrap, lengths 1/2/64, interruption after 1/7 responses,
  all-40 extraction offsets, queue/depth/work/acceptance telemetry and resources.
- `decision.json`: machine-readable decision, comparisons, gate coverage and
  explicit NOT_RUN/BLOCKED outcomes.
- `off-operational-soak.json`: fresh OFF agent/tool workload, interruption recovery,
  persist → process restart → restore → continued tool turns.
- `structural.log`, `cargo.log`, `ctest.log`, `quick-final.json`: regressions.
- `inactive-attempt.json`: preserved failed *experiment setup*, not MTP evidence.
  It retained weights and set a flag but omitted dispatch installation. Its
  apparent coherent extraction/ordinary throughput must not qualify MTP.

Actual MTP activation, accepted drafts and rejected-prefix rollback occur in the
corrected run. Idle extraction fails on length2, ordinary64 and cancel7. During
ordinary decode the cache also advances beyond emitted history while accepted
response tokens remain queued. This prevents publishing an M8 idle authority
and therefore blocks repeated stateful turns, tool-result re-entry, MTP restore
and the MTP M24-style soak. Those gates are **not run**, not inherited from OFF,
and not simulated with fresh-prefill fallback. It would be misleading to report
an agent-session performance A/B for an unrepresentable continuation state.

| 64-token diagnostic decode | OFF | MTP | MTP/OFF |
| --- | ---: | ---: | ---: |
| 2048-token prefix | 19.93 tok/s | 24.69 tok/s | 1.239× |
| 200000-token prefix | 19.08 tok/s | 19.98 tok/s | 1.047× |

Warm-case first emitted-token steps are 53.05 → 65.15 ms (short) and
54.18 → 68.52 ms (200K); these exclude terminal bootstrap (~49–54 ms) and
prefill. The first cold MTP length1 call costs 212 ms. MTP retains about 8.05 GB
additional active weight memory: ~309.15 versus ~301.10 GB after load. Cumulative
200K peaks are 328.02 versus 319.97 GB. Prefill is unchanged (~244 versus 243 s).
RSS/resource samples are in the raw artifacts; no leak or swap-growth proof is
claimed.

Last live-state counter snapshots in ordinary64 show 36 accepted drafts / 26
cycles with 15 rejection cycles at short context; 27 / 34 with 29 rejection
cycles at 200K. Upstream counter acceptance falls from 70.6% to 48.2%; that
denominator counts checked prefix positions, not all physical draft proposals.
Snapshots may omit terminal-call updates. There are 50 observed successful native
partial-rollback calls across scenarios, with actual all-40 offset reductions;
no failed rollback or attempted forbidden history reconciliation occurs. Adaptive
depth and emission queues are preserved per step in the raw artifacts.

Exact comparisons and acceptance/rollback telemetry are in `decision.json`.
Short/200K results are diagnostic throughput, not practical end-to-end benefit.
Early/mid/late windows within each 64-token decode are not early/mid/late *agent
session* measurements. No claim of retained long-session advantage is made.

## Regression policy and remaining work

Fresh OFF M24-style soak: 10 turns / 10 tool cycles, frontier 1702, all idle
frontiers coherent, zero replay/repack, interruption recovered from the exact
committed response, idle persistence at 1024, clean process restart/restore and
five further turns. Early/mid/late median decode is 20.05 / 20.00 / 19.95 tok/s;
first-token steps stay ~50.4–50.5 ms. Median append cost is 31.5 / 40.4 / 24.0 ms
(the initial prefill is a distinct cost, not a normal append). RSS is 12.43–13.17
GB. Session close succeeds and final backend mutex is unlocked. The 660 s harness
wall time includes two fixed lazy-start readiness polling windows; it is not all
inference time. This is a fresh bounded OFF result, not an MTP result.

Production structural tests pass (137 tests, 32 subtests); Rust passes 7 tests
and native CTest passes 5. Canonical quick and real Rust→HTTP release acceptance
are rerun separately; see their artifacts for exact identities and scopes. Initial quick
qualification found an existing M24 documentation-classification omission; M24
and M25 are now classified without changing historical evidence. An unrestricted
repository-wide pytest collection also discovers five pre-existing Engram runner
tests referring to absent `native/tools/benchmark/run_engram_cache.py`; these
unrelated reference-tool failures are retained in `regression.log` with baseline
blob/absent-runner proof in `preexisting-reference-test-failures.json`, not fixed or
silently excluded from that result. The scoped production `tests/` suite is
reported separately.

Canonical quick and real Rust→HTTP release acceptance both pass. Evidence is
qualified against the recorded working-tree runtime source digest before the
milestone commit, not falsely labeled as an earlier committed runtime. Docs and
diagnostic artifacts do not authorize MTP serving.

Reproduce diagnostics sequentially in the pinned release environment (write a
new evidence directory):

```bash
for mode in ON OFF; do
  PYTHONPATH="$HOME/omlx-0.7.0.release:$PWD" \
    "$HOME/.venvs/omlx-0.7.0.release/bin/python" -B \
    tools/run_m25_mtp_boundary_probe.py --mtp "$mode" \
    --contexts 2048,200000 --out "artifacts/m25-repro/$mode.json"
done
```

MTP remains unsupported in release metadata and configuration. OFF is both the
production default and the only qualified serving decode mode. Future work, if
pursued, must resolve the upstream commit/extraction boundary before adding a
serving option, then freshly qualify recipe EOS/tool behavior, repeated turns,
persistence and real early/mid/late agent-session performance. Full prompt
priming and stochastic configurations are unqualified. No DSpark/DFlash project,
new kernels, cache repacking or unrelated architecture work was begun.
