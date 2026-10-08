# M50R — oMLX candidate baseline freeze assessment

**PASS — bounded candidate baseline freeze. M51R is still not authorized.**
The three baseline closure requirements below now have fresh evidence. This is
not broad capability, cold-start, severe-pressure or fault/burn qualification.
M51R's baseline prerequisite is satisfied; commencing integration still requires
explicit authorization. No runtime, producer, profile, dependency, packaging or
release change is made. [M49R](mtp-architecture-reset.md) remains governing authority; archived
M50–M56 implementations were not executed or restored.

Evidence and reproduction: [`artifacts/m50r/README.md`](../artifacts/m50r/README.md).
`tools/report_m50r_candidate.py` verifies receipt/source/checkpoint consistency,
computes dispersion and validates supplementary source-pinned receipts before
emitting `decision: PASS`, `m51r_prerequisite_satisfied: true`,
`m51r_authorized: false`. Missing numerical/physical/environment receipts still
BLOCK; offset consistency or fast rates alone cannot approve state.

## Actual candidate identity

- Source tree: current master **`ddc9caf`**, unchanged runtime restart bytes.
  Imported ds41f is this source clone; oMLX and recipe are installed inside
  `$HOME/.venvs/ds41f-mtp-investigation`, not a donor checkout.
- Inspected executable identity SHA256:
  **`9329cb3a6ea4248c9f9127f6615caa89176947dc38b9cddca1fed3dc1bb3316b`**.
  Receipts contain the complete runtime/package-content inventory, actual import
  origins, native and transitive Homebrew dylib hashes, lock and build record.
- oMLX delivered archive SHA256 `26cc224a5fa77d8576589764a56ba8053ac31fe9f4904d8ba60305ed841f33e6`;
  upstream base `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, attributed candidate
  `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`. The archive/installed content,
  **not this base revision alone**, identifies execution.
- Recipe archive SHA256 `5b71ea6837ad3eb54a07da2b7ba1dd0a4ce658b585cdd2db24f4b83d9f879c22`;
  native SHA256 `e1ac924fef25f2f21acf3b4b6be771bb6f1365ef3c93320e10aad2a6b087e027`.
  This is the existing candidate native, not the archived consuming-EOF wheel.
- Python 3.13.15, MLX 0.32.2, mlx-lm 0.31.4.dev132+g94cdcae13,
  stock source-JIT oMLX without optional compiled kernels; SSD-backed Engram.
  Model config declares `dtype=fp8`; this is **not** all-FP8 arithmetic:
  quantized activations/packed KV, BF16 projections and FP32 attention/sinks and
  hyper-connections retain their source-defined mixed precision.
- Official DeepSeek-V4.1-Flash asset revision
  `dba1be0a40aa45a94ad051997016db3960a90277`, 48 shards,
  **510,296,708,312 bytes**. The identity capture freshly streams SHA256 over all
  weight shards and checks against LFS metadata (not just shard sizes or config).
  Metadata/tokenizers are independently pinned by candidate inspection.
- Mac Studio Mac15,14, Apple M3 Ultra, 549,755,813,888 memory bytes;
  macOS 26.5.2 / 25F84. `identity.json` records GPU/device limits, interpreter
  binary hash, OS build and thermal report. No thermal/performance warning was
  reported by `pmset`; that is not continuous GPU/thermal telemetry.

## Physical path and state facts

These refer to the installed files pinned in the receipts, not upstream's name
or the reconstruction's classes. The original five-workload trace counters are **API regions**, never inferred
Metal dispatch counts. A separate native-boundary/Metal assay below measures
actual imports/materializations, resources and device intervals.

1. **Prefill:** public `LocalMTPBackend` → `InternalMTPQualificationBackend._start`
   → `DeferredPrefillAppend.execute_all` / DENSE_P0_P7, with existing P5 handoff.
   Taps are input-side reduced target hidden states at layers **37,38,39**, from
   the same forward; projected committed context is appended into three native
   DSpark rings. Prefix is consumed once; the terminal prompt token bootstraps
   `BatchGenerator`. No oMLX prefill layer-loop substitution.
2. **Proposal:** `DSparkMixin.dspark_forward` appends committed target hidden
   context, then `proposal_forward` computes a **parallel** anchor/noise block
   through three DSpark stages. Markov bias/sample dependency is left-to-right;
   this is not three separate target transactions. DSpark attention uses the
   committed physical ring plus ephemeral draft KV, FP32 SDPA/sinks, with **no
   causal draft mask**. Draft keys never enter committed rings.
3. **Actual depth:** `_start` requests depth 5, but
   `OMLXMTPGenerationSession.__post_init__` configures from `n_mtp_layers=3`.
   The actual session maximum is **3 drafts**, not 5. The acceptance-based
   controller chooses `min(max_depth, accepted+1)`, not wall-time-based depth.
   Proposal enqueue precedes controller observation: the numeric mismatch trace
   still uses width 3 for the immediately following proposal, then produces
   width 1. Do not replace this observed asynchronous ordering with an assumed
   immediate-depth update. This explains the historical width-4 geometry;
   “repairing” the discrepancy in M50R would change the baseline.
4. **Verification:** `_run_verify_cycle_chain` forwards `[next_main,d1..dk]`
   once with `n_confirmed=1`; all 40 target layers preserve the block width.
   Each layer's input projections and local/compressed sparse attention use
   causal indices, then a packed attention region. Widths <=4 take the stock
   packed sparse source-JIT branch, not the optional length>8 compiled route.
   Greedy prefix acceptance and correction/bonus IDs resolve through the
   existing single host-vector materialization. Next proposals are dispatched
   via `mx.async_eval`, so enqueue phase times are not isolated device costs.
5. **Rollback/commit:** target cache is one executable authority: **40 × 7
   slots** (offset, packed window, compressed KV, index K, compressor KV/gate
   tails, integer Engram lookback), plus cache metadata/padding/lengths.
   Verification retains input/snapshots/window/compressor publications in the
   native stash. `mtp_partial_rollback` selects the already-computed causal
   prefix `accepted+1`, truncates pooled/index publications, reconstructs tails
   from captured projections and computes integer Engram prefix history. It
   does not re-execute target layers for each accepted token. Native `mx.eval`
   here is part of the candidate, not audit instrumentation.
6. **Queue/frontier:** verification enqueues accepted drafts and one correction
   or bonus. Physical target/DSpark context may be **ahead of canonical emitted
   history**; the last queued prediction is unconsumed. At idle the existing
   quiescence drains the already-committed queued suffix, discards its final
   future prediction, or materializes exactly one already-canonical token if
   target is one behind with an empty queue. No new proposal/verification,
   prompt replay or full cache repack is allowed during settlement.
7. **Rings:** native `DSparkContextCache.max_size` is **128**, with absolute
   offset and physical `absolute_position % max_size` order. Appending may
   temporarily concatenate chronological views and remap them to physical
   slots; attention reads physical order without a decode-time chronology
   rotation. The serialized quiescence `window_size=5` comes from block-size
   metadata in ds41f, **not actual ring capacity**. Preserve actual layout,
   not that misleading scalar. Ordinary trials end at frontier **283**, so
   ring wrap is exercised. Offset equality alone is not byte correctness; the
   independent primitive/live oracle below now establishes ring contents.
8. **Lifecycle:** sole singleton worker, response lease and operation lock;
   start/next/settle are shielded native regions, cancellation is admitted at
   coherent boundaries. Semantic preview binds candidate IDs/UID/ordinal before
   canonical recipe observation. The terminal path retains the exact native
   state/cache, settles before terminal publication, clears prediction ownership
   and retires the generator. Socket yield is never canonical-history ACK.
   DELETE drains/clears session aliases and native prime context, synchronizes
   and clears cache; model residency remains process-scoped.
9. **Storage/transfer/sync source map:** `storage.py` uses selected-row mmap
   gathers that copy CPU rows under locks, then `decode_array` creates MLX arrays
   and converts/dequantizes them to BF16. `EngramPrefetch` permits one pending
   model read on one `v41-engram` worker (16 MiB admission bound); submit drains
   the previous future and each forward drains again. A separate page-I/O pool
   is bounded at 48 workers and uses `pread` for previously unseen pages when
   row-count/size conditions hold. Those asynchronous reads are outside this
   profiler. Main-worker ordinary verify regions observe 20 embedding calls,
   20 prefetch submissions, 30 drains and 40 `decode_array` regions across ten
   cycles: these are regions, **not actual DMA byte counts**. Cache offset
   `.item()`, CPU Engram token/history conversion, acceptance `.tolist()`, native
   rollback `mx.eval`, per-next generation-stream `synchronize`, and idle/retire
   eval/sync are separate source sites; the acceptance-vector sync is not proof
   of one total sync per cycle. The M3 speculative command-buffer helper requests
   `(200 ops,512 MiB)` only if its optional extension is available; this pinned
   stock-JIT installation forbids all oMLX `.so` files, so that setter is a no-op,
   **not evidence that the requested caps govern Metal execution**. Actual MLX
   internal dispatch/copy/cap details are not supplied by this source map. The
   separate assay below resolves native boundaries and actual Metal activity,
   while explicitly bounding opaque compute-kernel internals.

The ordinary rate trials have 41 canonical generated IDs, including suppressed
backend stop ID 1, 10 cycles, 29 accepted drafts and one rejection. All 40 target
and three DSpark offsets equal frontier 283 after settlement; queue/prediction
owners are retired, and replay/repack counters are zero. These prove **relations
and lifecycle receipts, not independent correctness of the 280 state slots**.
Independent content validation is supplied separately below.
The ordinary trial's `rejects=1` is a backend-stop clamp: GPU acceptance was 3,
final commit acceptance was 2. It must not be relabeled a model proposal mismatch.

### Observed complete causal paths (diagnostic process only)

The machine receipt preserves ordered events and queue/history metadata; its
width histogram includes captured width-1 initialization but **not**
prefill/bootstrap or separate settlement materialization. Verify-only layer and
attention counts are grouped separately, not total model work.

| Trace workload | Captured initialization / verify widths | Verify projection / packed-attention regions | Settled frontier |
|---|---|---:|---:|
| Ordinary repeated OK | 1 × width 1; 10 × width 4 | 400 / 400 | 283 |
| Required Paris weather tool | 1 × width 1; 8 × width 4 | 320 / 320 | 546 |
| First-response disconnect | 1 × width 1; no verify cycle | 0 / 0 | 243 |
| Exact varied numeric sequence | 1 × width 1; 16 × width 4 | 640 / 640 | 359 |
| Disconnect with committed queue | 1 × width 1; 2 × width 4 | 80 / 80 | 251 |

- **Actual model rejection:** numeric prompt has 299 IDs, generated count 60.
  After fourteen full accepts, cycle 15 has `k=3`, `m_gpu=m=0`. Native rollback
  reports `before=356`, `count=1`, `end=357`; all three rings advance only to
  357, and queue contains correction **16**, not rejected drafts. Next cycle
  still verifies width 4, with `m_gpu=1`, stop-clamped `m=0`; rollback ends at
  358 and queue contains backend stop **1**. Canonical emission reaches 359;
  settlement materializes that one consumed stop once and appends taps/rings,
  with no new proposal/verify/replay/repack. This is a real mismatch path,
  distinct from the ordinary fixture's stop clamp.
- **Protected tool terminal:** prompt 512, generated 34. Last width-4 cycle
  commits physical target/rings at 545 and queues `[1718,128825,10699,32]`.
  Native preview binds `OwnedTerminal(ordinal=33, region=chain, UID=0)` with
  exact IDs and `DSML_TOOL_CALL_BLOCK_END` identity `(140,159)`; the terminal
  branch schedules no successor proposal. Canonical observation consumes token
  32 at its owned ordinal, clears pending prediction, and settlement advances
  target/rings once to 546 before terminal publication. Result is the official
  recipe's completed `lookup_weather({"city":"Paris"})` / tool_calls, not an
  executed tool effect. Parent/subclass `_settle` events are nested regions of
  **one** settlement, not two target executions.
- **Cancellation:** real SSE socket closes after first data frame with a
  diagnostic 30 ms delivery delay (never applied to rate trials).
  `CancelledError` is recorded; shielded cleanup settles canonical first token
  **11932** at 243, discards pending future **20370**, and performs **zero**
  target forwards, proposals or verify cycles during quiescence. DELETE succeeds
  and removes the lifetime. A second socket test closes after eight data frames:
  two verify cycles have committed physical target/rings at **251**, while
  emitted canonical history is **250** and queue holds one accepted draft plus
  an unconsumed bonus (both 20370). Shielded quiescence drains exactly the one
  committed draft into history 251 and discards the bonus; no forward, proposal,
  verify, replay or repack is issued. Nine canonical generated IDs thus exist
  despite eight pre-cleanup emissions; delivery is not ACK. DELETE retires the
  aliases. Cancellation inside an in-flight verify mutation and uncertain-
  mutation fault/burn matrices are **not freshly covered** here.

In active requests, the sole owner's `history` is current canonical emission;
`rec.canonical` is the retained idle prefix until settlement publishes it.
Socket yields and successful client reads do not update transport ACK (`0`
here). At idle the owner is retired and retained history/40 target offsets/three
ring absolute offsets agree. Numeric `depth_drafted` stats count **considered**
prefixes (the loop breaks on rejection): sum 44, while sixteen observed width-3
draft cycles propose 48 IDs. Do not use that sum as actual proposal work.
Backend stop paths may still have a dispatched future proposal before stop
emission; semantic-tool terminal and settlement branches have distinct lifetime
behavior, not a generic “no future work” rule.

The original worker profiler cannot reliably observe MLX nanobind calls and
misses Engram's other thread and Metal. Its zeros remain **not** zero transfers
or synchronization. It is retained unchanged; the separate closure assay uses
all-thread original-call-once wrappers, CPython native-call monitoring and real
Metal records instead of retroactively interpreting those zeros.

## Separate, bounded performance observations

Uninstrumented here means **no added profiler/wrapper/synchronization**;
existing candidate telemetry and per-next native synchronization stay intact.
The separate trace process must not supply rates. The public profile/server uses
`LocalH11Protocol` with canonical launcher options, one preloaded singleton worker,
loopback and sequential fresh sessions. Lifespan is off and the model is explicitly
loaded before trials: **no model-cold HTTP/TTFT launch claim**.

Exact request bytes/hash and canonical IDs are in each receipt: 242 prompt IDs,
greedy temperature 0, reasoning none, output budget 64, SSE; repeated “OK” ordinary
workload. One excluded warm-up, then **six fresh sessions** in one loaded process.
All measured sessions have identical output/frontier/acceptance work. This is not
an OFF comparison, stochastic/tool throughput test or broad context qualification.

| Metric | Mean ± sample SD | Range |
|---|---:|---:|
| Sum-of-`_next` decode rate, canonical tok/s | 40.378 ± 0.037 | 40.331–40.421 |
| Settled post-handoff rate, canonical tok/s | 38.219 ± 0.046 | 38.146–38.267 |
| HTTP request latency, s | 2.435 ± 0.008 | 2.430–2.451 |
| HTTP canonical tok/s | 16.838 ± 0.055 | 16.727–16.871 |
| First nonempty content SSE / TTFT, s | 1.424 ± 0.007 | 1.420–1.438 |
| Prefill + handoff/bootstrap, s | 1.360 ± 0.007 | 1.356–1.374 |
| Settlement/report/owner cleanup, s | 0.04893 ± 0.00029 | 0.04866–0.04944 |
| Built-in backbone telemetry, s | 0.84259 ± 0.00077 | 0.84156–0.84354 |
| Built-in proposal enqueue telemetry, s | 0.03196 ± 0.00022 | 0.03159–0.03223 |
| Built-in cache ops telemetry, s | 0.002757 ± 0.000075 | 0.002672–0.002858 |

Settled post-handoff rate uses `canonical_count / (elapsed_s - load_s -
prefill_handoff_s)`, including worker/delivery/recipe/settlement overhead. The
sum-of-next rate excludes those gaps. Backbone telemetry includes enqueue, host
materialization and outstanding lazy work; proposal telemetry is host enqueue,
not proposal GPU completion. Phase values overlap and **must not be added**.
Neither rate is numerically interchangeable with historical M56's profiled
end-to-end decode epoch or JSON HTTP request; the difference is not a speedup
claim. First SSE frame may be protocol metadata; TTFT uses actual content.

Process load: 90.72 s; warm-up HTTP 4.95 s / next-rate 39.62 tok/s, excluded.
Filesystem/model/JIT/Engram caches were not forced cold; prior excluded attempts
and full model loads affect system residency. Trials are model-resident and
session-fresh, not process-fresh. Post-trial active MLX: 309.17–309.21 GB,
cache ~2.484 GB, peak 311.56 GB; process RSS ~7.246 GB is **not** unified Metal
allocation. Available system memory 119.50–120.75 GB, load average ~2.0, no swap
allocation reported. Raw `vm_stat`/swap counters are retained; platform page-in
counters must not be relabeled GPU transfers or actual swap traffic. Session
retirement does not unload 309 GB of model weights.

**Screening policy before integration:** >5% decode or verification regression
outside measured uncertainty blocks advancement and requires causal investigation.
This is not a 5% structural-change allowance. Current CVs (~0.09% next-rate,
0.12% settled rate, 0.33% HTTP, 0.49% TTFT) characterize only this within-process
sample. They do **not** establish a cross-process/day uncertainty allowance;
that envelope is not established by this original run. The bounded reload
extension below establishes a limited empirical envelope, not universal noise. Future comparisons must rerun the
matched candidate under their actual residency/pressure conditions when this
uncertainty is material, rather than granting a blanket allowance. No slowdown
is accepted into an optimization backlog.

## Closure evidence: independent prefix contents

`prefix-oracle.json` is **16/16 primitive cases plus 2/2 actual-public-cycle
observations PASS**, exact byte equality, not a tolerance or offset/digest-only
approval. Before-frontiers **127,128,255,256** cross compressor boundaries and
physical ring wrap; accepted drafts **0,1,2,3** mean consumed target rows **1..4**
(the anchor is always consumed). All 40 layers' seven slots plus padding/lengths
are compared: packed windows, pooled compressed KV/index K, compressor KV/gate
tails, integer Engram lookback, including empty/None fields. All 16 Indexer
index/candidate publication fields, complete consumed-row vocabulary logits,
same-forward taps and each of the three physical committed rings match.

The subject uses native width-4 verification/rollback. The reference uses a
**separate same-width numerical forward with a changed unconsumed suffix**, then
CPU absolute-position/group-completion/window/tail selection, CPU compressed-ID
history update and absolute-position modulo ring placement. It does not call
candidate rollback or use its stash/snapshot selectors. Numerical arithmetic and
immutable model/token-map parameters are the existing qualified oMLX/MLX, **not
an independent reimplementation of model arithmetic**. The independent property
is causal suffix invariance and accepted-prefix state/layout selection. Padding
and lengths are None in this singleton; nonempty padded batches are not covered.

The existing public ordinary request's cycle 1 (full accept) and cycle 10
(stop-clamped partial accept) retain immutable before/after/tap/publication
references through external observation. No oracle eval/read occurs while that
owner is live. After DELETE/retirement, their state/taps/full logits/publications
and ring bytes are checked against the independent reference; append inputs also
match the same-forward taps. There is one ordinary candidate scheduler and **zero
live oracle schedulers**, no shadow executable authority, archived imports or
restored producer/journal. Historical `scope`/`live_scheduler:false` boilerplate
in the executed receipt refers to the offline oracle; its explicit `live_cases`
record the normal candidate observation and retirement. Executed tool snapshots
are retained unchanged; current tool labels clarify that distinction.

## Closure evidence: actual bridges and Metal, separately from rates

`movement.json`, `movement-summary.json`, `shared-bridge.json` and `metal/` cover
one separately warmed identical public request. Original-call-once wrappers run
on both `ds41f-recipe-infer_0` and `v41-engram_0`; CPython monitoring additionally
observes nanobind constructor/`item`/`tolist` CALL/RETURN without replacing the
array type. No additional eval/sync is inserted. Output/frontier match the frozen
fixture; owner/prediction retire before inspection. These are **diagnostics, not
performance samples**.

- 28 Engram worker row reads, 56 owned CPU selected-row copies, **3,598,848
  bytes**; 28 submits and 43 drains. Warm mmap gathers, not cold SSD throughput;
  no observed `pread` is not proof of zero cold page I/O.
- 56 decode/import sites; **98 actual NumPy-source native constructors carry
  4,035,408 source payload bytes**. The 4,909 constructors also include 4,783
  Python-list and 28 scalar sources. This is source payload, not GPU DMA volume.
  Separately tested uint8/uint32/int64/float32 GPU-buffer imports snapshot and
  copy their CPU source (distinct pointers; later source mutation cannot affect
  device values); two NumPy exports share a non-owning buffer view. All observed
  pipeline Metal allocations use **Shared** storage. Exports may wait for lazy
  readiness and dtype conversion may create an owning CPU copy: 56 observed
  exports, 13 owning results, 111,664 returned payload bytes.
- Native materialization sites: **1,467 `item`**, **15 `tolist`**, including ten
  eight-element acceptance vectors and five scalar vectors. These are sites,
  **not a count of actual fences**: already-ready shared data need not wait.
  Explicit API regions: 69 async evals, 7 evals, 46 synchronizations. Rollback
  has its own eval; settlement and retirement have distinct eval/sync sites.
  Eleven parallel-proposal enqueues include the future proposal later discarded
  at ordinary stop. Asynchronous lifetime is not proposal enqueue latency.
- xctrace Metal System Trace was attached **after warm-up**, request released
  while recording, and target PID/device/time/command IDs verified. Exported raw
  tables resolve actual **5,832 command-buffer submissions/completions**, 3,484
  Compute encoder intervals, 3,475 active Compute device intervals; no target
  Blit encoder is observed. 776 allocations and 776 deallocations all report
  Shared; allocated size spans **309.162–311.861 GB**. Background WindowServer
  work is excluded. Whole-request active-device interval union **2.330 s** is
  instrumented device occupancy, **not** isolated proposal/verify latency or an
  additive sum of kernel costs. Driver batching/counts are diagnostic observations,
  not exact deterministic dispatch tripwires.

**Residual observability judgment:** Shader Timeline is disabled. Individual
compute-copy kernels, per-kernel barriers, memory-page migration and isolated
per-phase GPU costs are not measured. Thus this is **not a complete internal
kernel copy/sync inventory**. It does not obstruct conformance of the **unchanged
delegated qualified primitive graph**: freeze MLX/native binary identities,
oMLX physical function bodies, dtype/device/shape/causal-row bindings, cache/ring
ownership and proposal/rollback/retirement lifetime; review any adapter for extra
operators/bridges; rerun these boundary/Metal diagnostics, the byte oracle and
matched uninstrumented rate gate. Logical concatenations (1,024 here) are not
physical-copy counts. No Blit is not no copy: compute copies remain opaque.

It **does obstruct approving a changed physical graph, library/kernel, precision,
staging route or unexplained transfer/sync delta**. Such a change requires fresh
native/kernel-level diagnostics and **BLOCK**, not extrapolation from this freeze
or a 5% allowance. oMLX owns that physical implementation; M51R must not rebuild
it. The source-preservation condition, real boundary/resource evidence and strict
changed-graph BLOCK are what make this bounded freeze sufficient—not an assertion
that opaque costs are zero. In-flight mutation fault/burn qualification and
isolated-device phase optimization remain unclaimed, not release authorization.

## Closure evidence: bounded process/residency variation

Original six resident samples are unchanged. Four **process-fresh reloads** use
ABAB order, one excluded warm-up and three fresh sessions each, with identical
identity/request/output/frontier/acceptance/width work. Pressure cases hold
**64 GiB touched nonzero entropic anonymous memory**, CPU idle, released on parent
EOF. This is bounded co-residency, not severe OS-pressure qualification.

| Reload condition | Load seconds | Next decode mean (tok/s) | Available GB |
|---|---:|---:|---:|
| reload-a | 90.777 | 40.475 | 218.65 |
| +64 GiB pressure-a | 100.666 | 40.387 | 153.81 |
| reload-b | 90.956 | 40.396 | 222.87 |
| +64 GiB pressure-b | 96.556 | 40.316 | 153.06 |

Load varies materially (~6–11% slower under co-residency); this cannot inherit
resident decode's small noise budget. Measured-session HTTP/content-TTFT means
remain ~2.435–2.439 s / 1.426–1.427 s. Small system-wide swap allocations appear
(43.65 MB, later 85.39 MB); they persist across reload and are not GPU-transfer
counters. Raw VM/swap/MLX residency snapshots are preserved. Neither filesystem,
JIT nor Engram caches were forced cold. No cold-launch HTTP claim follows.

Across **18 measured sessions / five model-loaded processes on one host/day**,
full empirical decode and existing backbone-wall ranges are each **<1%**. This
is an observed range, not a population confidence interval or a cross-day noise
waiver. Model-cold HTTP, stochastic/tool/terminal/cancel rates, longer contexts,
severe pressure and other hardware do not inherit it.

**Future matched comparison contract:** use the pinned bytes/hardware/OS/binaries,
actual prefill route, singleton generation stream, depth/precision/ring geometry,
LocalH11/socket settings, exact request bytes/context, temperature/reasoning/budget,
and canonical token count (including suppressed stop). Record actual acceptance,
widths, correction/clamp and proposal work; disclose required-semantic differences.
Use separate uninstrumented processes, excluded warm-up and >=3 fresh sessions
per matched candidate/integration condition; retain phase wall metrics, HTTP/TTFT,
startup, active/cache/peak Metal memory, available memory, swap/page counters and
co-residency. Bracket with current candidate controls and increase repeats when
control drift exceeds 1%, intervals overlap a decision threshold, residency/JIT/
pressure differs materially, or CPU/GPU/thermal contention exists. Do not subtract
this historical range as an automatic allowance. **>5% decode or verification
regression beyond resolved uncertainty BLOCKS** and requires causal investigation;
no optimization backlog. Unavoidable semantic overhead needs isolated necessity,
phase cost and unchanged physical path reviewed under M49R, never a blanket waiver.

## Frozen invariants and authorization

oMLX owns parallel proposal/causal block verification/prefix-select rollback and
native async lifetimes. DwarfStar topology plus qualified MLX numerics owns the
prefill route. ds41f owns semantics/application/lifecycle/resources. Preserve
actual depth 3, width-4 target blocks, mixed precision, same-forward taps 37–39,
three committed-only physical rings of capacity 128, one 40-layer executable
cache authority, bounded settlement and protected UID/ordinal ownership. A
single-token reconstruction, journal/child producer, hidden-tap surrogate,
cache repack/replay or deferred speed recovery is not conformant.

**M50R baseline prerequisite: closed. M51R commencement: not authorized by this
work.** No profile/dependency/runtime/default/release promotion occurs. Additional
scope or changed physical implementation must be separately qualified, and M51R
requires explicit authorization before any implementation.
