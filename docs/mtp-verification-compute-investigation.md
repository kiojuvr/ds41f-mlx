# R1 MTP verification compute investigation

Runtime base: **`d006e1f`**. Apple M3 Ultra, 512 GB, macOS 26.5.2,
MLX 0.32.2, official DeepSeek-V4.1-Flash checkpoint. No production numerical,
acceptance, cache, fallback, Web, release or promotion changes were made.

## Decision

**Do not start a general llama.cpp-style few-row MMA campaign.**

- The current public MTP path actually verifies **four token rows**, not the
  six-row, depth-five historical private benchmark. Representative dense MXFP8
  projections already use MLX's **four-row wide qmv** implementation.
- Routed experts do use a gather-matvec path. But their measured effective
  workload is predominantly **one row per active expert**, not four. There is
  no demonstrated several-fold sparse-expert dispatch opportunity here.
- There is a real, small optimization opportunity in the **BF16 vocabulary
  head**. A qualification-only weight-reuse prototype is approximately **3.5×**
  faster for four rows, with identical FP32 outputs on the captured real operand
  set. Its conservative modeled payoff is only **4–6% decode / 0.8–1.7% request
  compute time** on the three directly counted R1 workflows below.
- The highest-value next boundary is **fresh-context DSpark materialization and
  P5 MTP startup after dense prefix execution**. This consumes about **3.5–3.7 s**
  per fresh weather-tool turn, versus approximately **0.5 s** for the target
  prefix itself and **0.85–1.66 s** for all decoding. Packed attention already
  has an MMA path; a measured candidate-list dispatch opportunity is smaller
  and numerically non-identical. Investigate the startup boundary
  before a production verification-kernel campaign. Keep the head prototype as
  a separately proven, narrow follow-up candidate.

Evidence: [`artifacts/mtp-verification/`](../artifacts/mtp-verification/), especially
`summary.json`, `headroom.json`, the full R1 receipts, compressed observer records,
`dispatch-*.json`, `attention-dispatch-*.json`, and `head-prototype.json`. Operand/weight files and full GPU
resource dumps are deliberately **not committed**; the commands below regenerate
those from real execution.

## Correctness and environments

The unchanged R1 identity is
`45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.

| Gate | Result |
|---|---|
| Current standard-OFF, full R1 | **CONFORMANT / PASS**, 24 gates |
| Current mtp-singleton-v1, full R1 | **CONFORMANT / PASS**, 24 gates |
| Unchanged real MTP HTTP fixture with observers | **PASS**, 18 cases; admission, retained continuation, tools, byte loss, cancellation, negative partial tools and retirement |
| Occupancy and prototype numerical tests | **8 passed** |

The successful lanes use **different admitted environments**:

- OFF: the existing repository `.venv`, Python 3.13.14, with the first-party
  M48 model and the exact M47-admitted native artifacts.
- MTP: normally provisioned Python 3.13.15 environment
  `$HOME/.venvs/ds41f-mtp-investigation`, pinned source exports and stock-JIT
  profile, with its normal executable identity seal.

A fresh `--profile both` attempt in the MTP environment failed closed at
`deepseek_recipe._native` identity admission for OFF. A newly built MTP recipe
binary is not the existing OFF-admitted binary. **No single-environment both-lane
PASS is claimed, and no pin or admission check was relaxed.** Both successfully
admitted lanes were subsequently qualified independently. This is an environment
identity incompatibility, not a demonstrated model/correctness regression.

The fresh environment's initial seam run also lacked the already-declared
`web-binary` optional dependencies: the unchanged legacy-client refusal fixture
constructs the frozen tool registry. Installing exactly `pypdf==6.10.0` and
`pypdfium2==5.3.0`, followed by the normal seal, resolved that prerequisite.
No Web implementation or R1 fixture was edited. Excluded attempts are recorded
separately; they are not passing qualification/performance evidence.

## Current baseline: use the actual runtime, not the historical stock control

The immutable OFF fixture uses first-party `TargetGenerationSession`; the MTP
fixture launches the real admitted `mtp-singleton-v1` operator and lifecycle.
Both keep SSD Engram, canonical parsing, semantic horizon, acceptance and protected
settlement. Public metrics retain zero prompt replay / full cache repack, aligned
idle frontiers, and zero settlement proposals / verification cycles where required.

| Uninstrumented R1 workload | Output | Prefix/handoff s | Decode s | Decode tok/s | Cleanup s |
|---|---:|---:|---:|---:|---:|
| OFF weather call | 28 | 0.536 | 1.425 | 19.64 | not separately sampled |
| MTP weather call | 34 | 4.393 | 0.851 | 39.93 | 0.050 |
| MTP two weather calls | 64 | 4.386 | 1.663 | 38.49 | 0.051 |
| MTP first stored-result continuation | 10 | 0.338 | 0.266 | 37.64 | 0.049 |
| MTP second stored-result continuation | 43 | 0.432 | 1.410 | 30.50 | 0.049 |
| MTP short greeting | 3 | 13.832 cold | 0.164 | 18.31 | 0.050 |

These are workflow measurements, **not a token-identical OFF/MTP speedup claim**.
The R1-qualified backend-local trajectories produce different tool-call token
lengths. OFF's 19.64 tok/s comes from a read-only sample of its public M8 session
record; the stateless warm Responses/Messages turns take 1.083/1.066 s including
prefill and 16 output tokens. OFF's first stateless turn takes 315.534 s, dominated
by cold admission/load; MTP's first load takes 92.121 s. Do not equate these
asymmetric cold identity/load paths with decode or verification time.

## Measurement method and its limits

`tools/profile_mtp_verification.py` substitutes only the server launch command
of the unchanged real R1 MTP fixture with an opt-in observer. Normal runtime
admission still runs. It never substitutes a model, prompt, sampler, acceptance
rule, cache or protocol fixture.

- At most **32 actual verify cycles** are observed. Shapes and routing decisions
  are recorded without evaluating extra arrays in the measured cycle.
- Expert-index host transfers, isolated operator benchmarks and operand export
  occur **after the server has gracefully drained**. The harness handles Uvicorn's
  replayed SIGTERM so it can export afterward; it does not interrupt protected
  execution or shorten settlement.
- Operators are replayed with the **actual evaluated real-model operands**.
  Capture runs in a **separate process**, with at most 4 GB of operand data per
  capture. Shader/pipeline names come from Metal capture metadata, not guessed
  from the Python API name.
- A separate `--sync-phases` run evaluates attention/MoE outputs at component
  boundaries. This is a **serialized diagnostic**, not production throughput.
  Its timings include incoming HC/norm work and host submission/synchronization;
  they are neither pure GPU durations nor an additive cost model for the normal
  batched graph.

Calibration is essential: final low-overhead observation changes weather-call
and two-call decode time by about **5% and 2%**, respectively, relative to the
uninstrumented receipt. Only the uninstrumented receipt is the performance
baseline. Component synchronization increases these times by roughly **46% and
44%** and is used only for coarse consumer attribution. A capture-enabled serving
attempt was much more intrusive and its deferred allocator dump exceeded the
fixture's shutdown deadline. It is **excluded**; the final tool rejects capture
layers on serving and uses isolated captures instead.

Microbenchmarks include their own submission/synchronization costs. **Do not sum
isolated operator latencies** into a claimed production verification duration.
The observer's `target_verify` enqueue timer is also not GPU time: asynchronous
proposal work can be paid at the next acceptance transfer. Existing `backbone_ms`
has that same producer-debt caveat. The synchronized diagnostic separates draft
logits, complete draft/Markov sampling, target forward and rollback.

## Logical phase → shapes/routing → dispatch → cost

### Target verification

The lifecycle configures depth from `n_mtp_layers=3`. Every one of the 32 observed
backbone calls has input **`[1,4]`** (`next_main` plus three drafts). Global request
batch size is one. `n_confirmed=1` and the existing rollback/semantic-horizon
contracts remain intact. Public `considered_drafts` counts are horizon-limited;
they are **not physical verification row counts**. For example a short greeting
can consider only one draft while still executing the four-row target forward.

| Work | Representative real shape | Observed dispatch / cost |
|---|---|---|
| Dense target projections | `[1,4,8192] → 5120` MXFP8; Engram WKV `[1,4,6144] → 25600` | MXFP8 `quantized_matmul`; captured WKV uses `mxfp8_qmv_wide_bfloat16_t_gs_32_b_8_nv_4_kl_16_batch_0`. Isolated medians approximately 0.40–0.45 ms / 0.70–0.72 ms |
| Routed gate/up | activations `[1,4,1,1,5120]`, IDs `[1,4,6]`, packed weights `[384,2304,640]` | unsorted MXFP4 `gather_qmm`; actual `mxfp4_gather_qmv_fast_bfloat16_t_gs_32_b_4`; approximately 0.47–0.48 ms per projection |
| Routed down | `[1,4,6,1,2304]`, packed weights `[384,5120,288]` | same stock gather API; approximately 0.51–0.52 ms isolated |
| Shared experts | `[1,4,5120] → 2304 → 5120`, MXFP8 | ordinary dense quantized projections, not sparse expert batches |
| BF16 vocabulary head | `[1,4,5120] × [129280,5120]`, FP32 logits `[1,4,129280]` | `custom_kernel_v41_bf16_head_fp32_output__129280_5120_16_float_bfloat16_t_float`; **7.58–7.61 ms** isolated |
| Packed CSA2 attention | `[1,4,64,512]` queries; packed window `[1,132,528]`; pooled `[1,364,288]` or `[1,413,288]`; `wi=[1,4,128]`, `ci=[1,4,368]` or `[1,4,416]` | 496 slots: online **fused + merge**, approximately 0.90 ms; 544 slots: online **MMA scores + MMA values + merge**, approximately 0.57 ms isolated |

The stock-JIT MTP profile does **not** have the optional native expert block/pair
or grouped-expert extension available. Do not import OFF's native dispatch result
into this profile. The source's sorted/block route for longer token sequences is
not exercised by the admitted four-row verification path.

### Packed attention: a different matrix/routing quantity

The attention observer records all **1,280** target attention calls. **1,260**
use the BF16 online fused family; **20** cross the candidate-slot threshold into
its existing MMA family. Capture metadata confirms both paths with the same four
query rows and 64 heads. The dispatch quantity is `wi_slots + ci_slots`, with a
512-slot crossover, **not global verification rows**. Heads and candidate keys
provide the matrix work.

The selected real 496-slot example has 489–492 eligible keys per query; the
544-slot example has 538–541. The small padding/causal invalid tail is recorded
in `attention-occupancy.json`, rather than counted as useful key work.

A causal control calls the two existing paths on **identical** real operands:

| Real candidate slots | Fused ms | MMA ms | MMA/fused speed ratio | Difference against current path |
|---:|---:|---:|---:|---|
| 496, currently fused | 0.900 | 0.546 | 1.65× | max BF16-output absolute difference 0.015625 |
| 544, currently MMA | 0.937 | 0.573 | 1.64× | forced fused differs by at most 0.00048828125 |

This proves a local dispatch-tuning opportunity below 512, not a universal
crossover or an immediately safe change. Both are existing rounded-BF16 attention
implementations, but their results are **not bit-identical**. Any tuning needs
backend-local numerical/trajectory qualification and R1, not an assumption that
faster arithmetic is interchangeable. No dispatch threshold was changed. These
isolated savings are much smaller than the measured seconds-long startup region.

### MoE: the effective workload is sparse and expert-local

Across **1,280 target layer calls**, four rows × six routes produces 30,720
assignments, but only 22,729 active expert invocations:

| Tokens assigned to an active expert | Expert invocations |
|---:|---:|
| 1 | 17,010 |
| 2 | 3,930 |
| 3 | 1,306 |
| 4 | 483 |

Mean local occupancy is **1.352 rows**, maximum four. Approximately **74.8% of
active expert invocations are singletons**; singleton experts account for **55.4%
of assignments**. Mean active experts per layer call is 17.76 of 384 (4.63%).
The actual gathered multiplication presents a **one-row matrix per assignment**,
not a dense four-row matrix per expert. This is exactly why global row count alone
is an inadequate dispatch hypothesis.

A sorted/flattened replay of the real routes preserves output exactly but does
not demonstrate a useful speedup: approximately **0.495 ms sorted vs 0.481 ms
original** in the final head observer. Its extra sort/materialization setup is
reported separately, not hidden in a kernel-only comparison. Sorting alone does
not manufacture a multi-row expert GEMM.

Even perfect once-per-active-expert weight reuse would change routed weight-read
multiplicity by at most **1.352×**, i.e. remove **26.0%** of assignment-based reads
on this sample. This is an optimistic traffic bound, **not a measured speedup**:
current caches may already reuse those reads, and routing, shared experts,
activation rounding and dispatch do not disappear.

### Draft, ordinary decode and settlement are separate

The three DSpark stages predict a **three-position block**, independent of the
four-row target verification window. Their routed MoE has **128 experts, top-3**:
90 measured stage calls, 810 assignments / 515 active expert invocations, mean
local occupancy **1.573**, maximum three. They must not be included in the target
MoE totals above.

Committed target taps concatenate to width **15360**; one to four committed rows
are projected to width 5120 before appending the draft context. The measured
first-row projection is approximately 0.36–0.39 ms isolated. The synchronized
three-stage draft logits take about **11.5 ms**, complete draft generation and
Markov sampling about **12.3 ms** median. Source inspection additionally shows
that the draft logits use the same BF16 vocabulary projection with three rows;
the conservative headroom model below counts **target heads only**.

Ordinary target decode/terminal bootstrap is a one-row target operation, not a
four-row verification operation. Protected idle settlement takes approximately
50 ms in the baseline and can make one required alignment target forward. It
makes **no new proposals or verification cycles**. Neither this cost nor prefix
execution is assigned to verification kernels.

### Dominant verification consumers

Over the serialized diagnostic's 32 calls:

- Target attention-entry → attention-output (including incoming HC/norm):
  **2.003 s**.
- Target MoE-entry → MoE-output (including incoming HC/norm/shared/routing):
  **1.772 s**. The additional **0.148 s draft MoE** is excluded from that total.
- Complete target forwards: **4.214 s**; the remainder includes head, other
  HC/Engram/cache work, and diagnostic overhead.
- Complete draft generation: **0.370 s**; rollback: **0.008 s** over six calls.

Attention/projection and MoE regions are the large verification consumers; the
head is a real but smaller leaf. These diagnostic totals show where work is,
**not exact production percentages**. Normal observed cycles total approximately
3.01 s and their existing backbone counters include asynchronous draft debt.

## Shape/dispatch cliffs and the small prototype

Dense row controls on real weights are already sublinear: a representative
`8192 → 5120` MXFP8 projection takes approximately **0.34–0.38 ms at one row**,
**0.40–0.45 ms at four**, and **0.60–0.73 ms at sixteen**. Separate real WKV
captures show ordinary fast qmv at one row, wide variants for 2–5 rows, then qmm
for the larger controls. No several-fold cost discontinuity was demonstrated in
this tested dense range. These are shape controls, not changes to R1 depth.

The BF16 head is different: its current shader assigns each query its own
threadgroups and repeats the weight-reading dot loop. Paired real-operand
benchmarks expose the linear row cost directly:

| Rows | Current ms | Prototype ms | Speedup |
|---:|---:|---:|---:|
| 1 | 2.127 | 2.133 | 1.00× |
| 2 | 3.941 | 2.130 | 1.85× |
| 3 | 5.759 | 2.129 | 2.70× |
| 4 | 7.578 | 2.140 | **3.54×** |
| 5 | 9.395 | 2.154 | 4.36× |

`tools/bench_mtp_head.py` retains BF16 storage and the existing FP32 float4 dot /
SIMD reduction, loading each weight vector once for all rows. It is **not MMA**,
not a runtime selector and not a production kernel. Rows 1–4 use prefixes of the
real captured input; row 5 repeats its first row as a shape control, not an actual
R1 depth. Each size uses 20 alternating
measurement pairs after warmup. Every measured FP32 output and argmax is identical
on the captured real verification rows, plus five nonzero small numerical tests.
This finite proof does not claim universal numerical qualification or a full R1
run with the prototype installed. The existing head's >5-row fallback is preserved
in production and is outside the admitted target verification shape.

## End-to-end headroom and next task

`headroom.json` multiplies the measured **5.44 ms** four-row head saving by actual
observed target-head counts (8 / 16 / 2). It holds all other costs constant.
Request compute is prefix/handoff + decode + cleanup; cold load and transport/
client reconciliation are excluded and would only dilute the modeled gain:

| R1 workflow | Modeled decode gain | Modeled request-compute gain |
|---|---:|---:|
| Fresh one weather call | 5.4% | **0.8%** |
| Fresh two weather calls | 5.5% | **1.4%** |
| First retained result continuation | 4.3% | **1.7%** |

These are defensibly modeled gains, **not measured end-to-end prototype speedups**.
Draft-head reuse could add benefit, but is not silently counted. No acceptance,
verification row or model semantics was removed to obtain the microkernel result.
Even eliminating **all decoding** would cap fresh single-call improvement at about
**1.19×** under the recorded warm prefix/handoff/cleanup costs.

The warm fresh single-call prefix/handoff breakdown in the final head observer is:

| Existing boundary | Seconds |
|---|---:|
| `DeferredPrefillAppend.execute_all` target prefix, 292 positions | 0.532 |
| After prefix → DSpark context keys evaluated and ready | **1.498** |
| P5 `handoff_to_generation`, inclusive | **2.214** |
| Of that handoff: held-out terminal bootstrap | 0.924 |
| Of that handoff: remaining admission/construction/frontier work | **1.290** |

The first retained result continuation instead spends about 0.097 s in the target
append, 0.187 s reaching context-ready, and 0.052 s in handoff. Fresh two-call
figures are similarly 0.492 / 1.529 / 2.142 s. Context certificates themselves
are tens of microseconds, so the slow region is not merely JSON certificate
construction. These coarse boundaries do not yet prove whether the excess is
lazy tap-graph materialization, cache/startup computation, host orchestration,
or another dependency; do not rename it a measured replay or assume it can be
skipped. The existing semantic replay/repack counters remain zero.

**Next task:** trace the fresh-context `hidden` tap concatenation,
`dspark_append_context` → `mx.eval(ring.keys)` boundary, and P5 admission/start
(including the ordinary held-out terminal forward). Establish which work is
recomputed/materialized or serialized and optimize its ownership/dispatch while
preserving exact tap provenance, one-shot cache transfer, frontiers and fallbacks.
The target dense prefix itself is already near OFF's measured append cost.

For prioritization only: reducing the measured 3.71 s post-prefix region by 25%
or 50% would model roughly **1.21× / 1.54×** fresh single-call request compute
speedup. Attainability is **not established**. This is a much larger investigation
boundary than the proven target-head leaf or the optimistic sparse-weight-reuse
bound. Do not spend a broad production-kernel campaign on the latter first.

## Reproduction

Use the unchanged [R1 contract](reference-release-r1.md) and normal MTP setup/seal.
The qualified MTP setup for this run is retained in `setup.log`; the extra PDF
pins are declared already in `pyproject.toml`. Use the existing admitted OFF
interpreter separately rather than rewriting its native identity pin.

```sh
export DS41F_CHECKPOINT=/path/to/official/DeepSeek-V4.1-Flash
OFF=.venv/bin/python
MTP="$HOME/.venvs/ds41f-mtp-investigation/bin/python"
OUT=artifacts/mtp-verification

$OFF -m ds41f_mlx.reference --profile standard-off --real-model \
  --output "$OUT/r1-off.json"
$MTP -m ds41f_mlx.reference --profile mtp-singleton-v1 --real-model \
  --output "$OUT/r1-mtp.json"

# Run sequentially: one full model resident at a time. No capture layer here.
$MTP -m tools.profile_mtp_verification --cycles 32 --microbench --export-operands \
  --output "$OUT/attention-observer.json" \
  --r1-output "$OUT/r1-attention-observer-mtp.json"
$MTP -m tools.profile_mtp_verification --cycles 32 --sync-phases \
  --output "$OUT/serialized.json" --r1-output "$OUT/r1-serialized-mtp.json"

# Select the exported description by its operation/source shape, not its index.
MTL_CAPTURE_ENABLED=1 $MTP -m tools.capture_mtp_operands \
  --input "$OUT/<real-operand-description>.json" --output "$OUT/dispatch-example.json"
# Add --row-sweep only for a dense quantized projection.
$MTP -m tools.bench_mtp_head --input "$OUT/<bf16-head-description>.json" \
  --output "$OUT/head-prototype.json"
$MTP -m tools.bench_mtp_attention_dispatch --input "$OUT/<packed-attention-description>.json" \
  --output "$OUT/attention-dispatch-example.json"
$MTP -m pytest -q tests/test_mtp_compute_observer.py tests/test_mtp_head_experiment.py
$OFF -m tools.summarize_mtp_compute "$OUT"
```

Optional OFF sampling: run `tools.collect_r1_off_sessions` against the OFF fixture's
printed localhost URL while that fixture is running. It only reads the two known
R1 sessions once per second, never enables diagnostics or modifies requests.
Large local operand files and complete `.gputrace` resources may be removed after
extracting dispatch evidence; regenerating them requires the real checkpoint.

The supplied `~/llama.cpp` clone contains PR #29869 as `a3a1c4747`; its actual
few-row change was inspected. #30047 was not present in the local history and was
used as the user-supplied motivating hypothesis, not as a source or implementation
authority for ds41f. Neither PR's speedup is imported into this report.
