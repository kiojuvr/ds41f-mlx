# Fresh MTP startup: target compute debt and late Metal wiring

Base: **`12182b3`**, unchanged R1 identity
`45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
Apple M3 Ultra / 512 GB, macOS 26.5.2, MLX 0.32.2, official
DeepSeek-V4.1-Flash checkpoint. The admitted MTP and OFF environments remain
separate, exactly as in the [previous investigation](mtp-verification-compute-investigation.md).
No few-row, head, attention, checkpoint, Web or release changes.

## Result

**Acquire the already-required native generation wired-memory limit before dense
prefix allocation, not after prefix/tap evaluation.** The request-scoped lease
transfers its original restore obligation to the ordinary native BatchGenerator.
Nothing is kept wired between requests beyond the previous native contract.
Acquisition is inside measured `prefill_handoff_s`, after model load and before
prefix execution. This is not background work or latency accounting relocation.

Uninstrumented full R1 MTP, compared with `12182b3`'s uninstrumented receipt:

| Workflow | Before startup s | After startup s | Before decode s | After decode s |
|---|---:|---:|---:|---:|
| Fresh weather call (34 output tokens) | 4.393 | **1.654** | 0.851 | 0.852 |
| Fresh two weather calls (64 output tokens) | 4.386 | **1.615** | 1.663 | 1.662 |
| First retained stored-result turn | 0.338 | 0.337 | 0.266 | 0.264 |
| Second retained stored-result turn | 0.432 | 0.432 | 1.410 | 1.405 |

Fresh single-call startup falls **62% / 2.74 s**. Prefix + decode + cleanup falls
from 5.294 to 2.556 s (**2.07×**, not including load/transport). Two-call request
compute improves approximately **1.83×**. The new same-task baseline observer
measures 3.95–4.01 s rather than the earlier 4.39 s: cross-run host variability
matters. Same-process controls demonstrate approximately **1.7–2.6 s** removal
without assuming the old/new receipt difference is all caused by the patch.
Continuation and steady MTP decode are unchanged within run variation.

Evidence is in [`artifacts/mtp-startup/`](../artifacts/mtp-startup/), especially
`controls-production.json`, `summary.json`, `phases-detail.json.gz`,
`phases-joint-split.json.gz`, final phase observations, and full R1 receipts.
`controls.json` is the initial diagnostic: it bypassed the *second*, nested native
wired-limit call while retaining the first call/restore. `controls-production.json`
retests the production lease with the ordinary native constructor call retained.

## Where the old 3.5–3.7 seconds went

The phrase “completion of target prefix” previously meant return from
`DeferredPrefillAppend.execute_all`, **not completion of all target GPU work**.
Ordinary short P6 segments publish lazy arrays. Engram overlap submits some
work asynchronously; the last target taps and many cache leaves have not yet
completed when Python returns.

Bounded phase observer, warm fresh weather input (293 IDs, 292 prefix positions):

| Actual boundary before fix | Seconds | Interpretation |
|---|---:|---|
| `execute_all` | 0.532 | Host graph construction plus earlier target/Engram work; not complete GPU prefill |
| `mx.eval(ring.keys)` after `dspark_append_context` | 1.497 | Predominantly the remaining **target** prefix graph, not a DSpark transformer pass |
| Native generator constructor `mx.set_wired_limit` | **1.298** | Late Metal wired-memory transition; not cache validation/admission |
| Held-out terminal bootstrap | **0.621** | Ordinary target forward plus a removable first-forward allocation/transition penalty |
| Other orchestration/frontier/publication | approximately 0.002 | Metadata, native parser setup and one-shot transfer |

The comparable fresh two-call trace is 0.492 / 1.486 / 1.278 / 0.749 s.
Additional fresh turns repeat these penalties. A retained result turn is
0.097 / 0.187 / sub-ms constructor / 0.052 s. The old coarse handoff “admission”
region was therefore **not** seconds of P5 validation or Python certificate work.

The split-context diagnostic evaluates target hidden taps before projecting them.
With the target cache leaves also included in that evaluation, it takes
1.50–1.60 s on these fresh shapes; the subsequent ring evaluation takes only
milliseconds. Almost the entire purported context-materialization interval
moves to `target_hidden_eval`. Thus the seconds are incoming target computation
debt; calling the whole interval “DSpark model compute” was incorrect.

The remaining ~1.3 s is directly timed inside `mx.set_wired_limit`, called by
`mlx_lm.generate.BatchGenerator.__init__` through
`maybe_set_recommended_wired_limit`. The native constructor normally acquires
this limit only *after* dense prefix allocations/evaluation. Moving acquisition
before those allocations removes both this late transition and the ~0.6–0.9 s
excess in the next ordinary one-row target forward. The required terminal forward
then costs approximately **52 ms**, as it does on continuation. Native proposal
activation/sampling still happens in the first `next_token` call, counted in
`decode_s`; no proposal is moved into prefill or protected settlement.

Read-only inspection of the installed version's upstream MLX **v0.32.2** source
(commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, file hashes retained in
`mlx-source-evidence.json`) explains the API boundary. `MetalAllocator::set_wired_limit`
holds the allocator mutex and calls `ResidencySets::resize`. Raising the default
zero budget walks **all tracked allocations**, adds those that fit and commits
the touched Metal residency sets—including allocator-cached prefix buffers, not
just authoritative cache leaves. Acquiring before prefix allocation establishes
the residency budget while subsequent allocations are inserted incrementally.
The ordinary constructor's second acquisition has unchanged capacity, so `resize`
returns immediately. No residency-set tuning or MLX modification was made.

This establishes a **resource-placement cause**, not a new kernel optimization.
Phase timers do not expose Metal's internal VM/wiring operations; we do not claim
an exact page-fault count, GPU duration or CPU/GPU split inside that native call.
Nor do we label all terminal excess as duplicated transformer arithmetic. The
same computations, cache geometry and tokens become fast under the earlier
resource acquisition. No asynchronous work is moved beyond proposal readiness.
Under unchanged math/dataflow the post-enqueue region is now about **1.1 s**:
roughly 1.07 s of target completion, milliseconds of context projections, and
52 ms of terminal work. Approximately two thirds of the old post-enqueue delay
was removable resource overhead. This is measured required work on the current
implementation, not a theoretical lower bound on optimized target prefill.

## P5 primed is not yet the first proposal

The historical startup metric ends at `OMLXMTPGenerationSession.start`: target
and native primed rings include the held-out terminal prompt token, and its
logits provide the first main token. Native MTP activation is deliberately lazy,
not performed by `GenerationBatch.__init__`.

The bounded first-`_next` observation follows the path **through the actual first
proposal**, rather than assuming P5 return is draft activation:

| Warm first activation | Fresh weather | Retained weather result |
|---|---:|---:|
| First main-token target forward, evaluation and semantic preview; ready to enter `_dspark_next_drafts` | 52.8 ms | 52.3 ms |
| Draft/context append + three-stage/Markov graph enqueue | 3.3 ms | 3.1 ms |
| Remaining GPU synchronization to complete proposal | 7.2 ms | 7.3 ms |
| Whole first `next_token` | **63.4 ms** | **62.7 ms** |

`_post_init_mtp` validates the committed prompt context, previews the main token,
performs **one new-token target forward** to obtain its hidden taps and next-main
anchor, transfers the already-primed rings, and appends only that new hidden row
while producing the first draft block. It does not replay or re-project the
prompt. This genuinely new-token work cannot be harvested from earlier prompt
hidden states: its input is determined by terminal-prompt logits. Semantic
terminal previews may skip proposal generation altogether, as before.

Thus actual first-proposal readiness adds roughly **53 ms** to the P5 endpoint;
proposal completion adds about **63 ms**. Fresh and re-entry pay the same small
activation cost. It is already charged to request `decode_s`, never omitted
from end-to-end latency. It is not the several-second fresh-context asymmetry.
Evidence: `phases-first-proposal.json.gz` / `fixture-first-proposal.json`.
This split-context observer calibrates at 1.657 s startup / 0.850 s decode for the
weather call, versus 1.654 / 0.852 uninstrumented (well below 1% variation).
Only the uninstrumented receipt supplies the throughput claim.

## Fixed, length-dependent and cold effects

Direct controls load one admitted real model, then run two passes over prefix
lengths **64, 128, 256, 292, 512, 1024**. Inputs truncate/repeat the real weather
prompt IDs retained by the observer; no synthetic hidden operands or toy model.
Each case executes the actual dense prefill/tap/context math, native terminal
bootstrap, eight MTP outputs, and protected quiescence. Every mode closes/restores
the native resource limit and clears the allocator cache before the next case.
Baseline, joint target+ring evaluation, and early wiring alternate in that order.

Production-lease control, second (warm) pass; ordinary native constructor retained:

| Prefix positions | Late wiring startup s | Early wiring startup s | Early terminal s |
|---:|---:|---:|---:|
| 64 | 0.359 | 0.358 | 0.051 |
| 128 | 0.684 | 0.685 | 0.052 |
| 256 | 3.133 | 1.392 | 0.052 |
| 292 | 4.021 | 1.613 | 0.052 |
| 512 | 5.148 | 2.736 | 0.053 |
| 1024 | 7.568 | 5.327 | 0.053 |

- 64/128 warm prefixes cost approximately 0.36/0.68 s, with no seconds-long late
  wiring penalty. Earlier wiring does not accelerate these already-small paths.
- The large-path penalty is reproduced at 256 positions and remains roughly
  **2 seconds** across 256–1024, rather than scaling with prefix length. It repeats
  on warm fresh contexts; it is **not persistent kernel/JIT first use**.
- Actual target work remains length-dependent. After the fix, startup is roughly
  1.4 s at 256, 1.6 s at 292, 2.7 s at 512 and 5.3 s at 1024. All six controls
  use one ordinary complete P6 range. At 1024 more work is paid before
  `execute_all` returns, rather than at the later evaluation; the individual
  enqueue/eval timers must not be treated as independent model-compute slopes.
- First model/shape use is separate: initial 64-position baseline startup is
  about 5 s before dropping to 0.36 s. Initial HTTP greeting still has seconds of
  cold compilation/startup after the fix. Checkpoint admission/load is separately
  ~90 s in this environment. Neither is a warm fresh-context construction cost.
- A joint evaluation of all target-cache leaves and ring keys **does not remove
  the repeated fresh penalty**. Keeping all those leaves in one evaluation was
  tested, not assumed. Earlier wiring is the effective intervention.

Timing is wall-clock; mixed enqueue and evaluation phases are not pure kernel
costs. The observer does not evaluate additional arrays except in the explicitly
serialized `--split-context` / `--joint-state` controls. Only uninstrumented R1
and direct controls are used for end-to-end gains.

## Is already-produced target information being recomputed?

The current architecture already harvests the needed representation:

1. `_start` temporarily wraps target layers **37, 38, 39**. Each tap takes
   `mean(h, axis=-2)` at the *input* of the layer, preserving the singleton row.
   These layers have no Engram insertion. This matches the native `_forward`
   capture site (after any Engram, before the block), not a final output or an
   attention-cache approximation.
2. All per-layer prefix chunks are concatenated in temporal order, then taps in
   checkpoint `dspark_target_layer_ids` order: width **15360**.
3. Native `dspark_append_context` runs the checkpoint's first-stage
   `main_proj` / `main_norm`, then each stage's distinct `wkv`, KV norm, absolute
   RoPE and FP8 activation transformation. The bounded physical ring retains the
   checkpoint's absolute-position modulo-slot order. No draft keys enter it.
4. The prefix rings are published only after P6 commits and all ring keys are
   evaluated at frontier `len(ids)-1`. P5 transfers the authoritative target
   cache once; native terminal capture advances target and rings by one.

There is **one target prefix execution**, not a later prompt replay. The long
post-prefix evaluation resolves the **same lazy target graph** retained by these
native taps. Existing packed target KV is not an equivalent direct substitute:
it is a lossy, differently weighted/normed representation, not these width-5120
hidden inputs or the DSpark stage projections. There is no checkpoint-defined
exact conversion from those target cache fields to DSpark keys. Producing new DSpark keys from those
already-harvested taps is required checkpoint computation, but small.

No new context owner, incremental DSpark publication, tap harvesting scheme or
numerical projection is necessary. Eagerly evaluating the same graph earlier
would primarily move required target work between timestamps. The tested joint
materialization does not help. Do **not** remove native projections or derive
DSpark keys from target packed KV on the strength of these timings.

For the actual change the state equivalence is stronger and narrower: only the
MLX wired-resource lifetime changes. The production path retains all existing
tap sites, arithmetic, shapes, concatenation, projections, evaluation boundaries,
physical rings, cache/frontier and acceptance logic. Direct controls compare
all canonical output IDs, fully settled target-cache state digests and physical
DSpark ring digests on identical inputs; these are equal across baseline/joint/
early acquisition on all six tested lengths, including physical ring wrap.
These controls do not claim real-model deferred long-segment qualification;
public MTP's existing 8192-token context ceiling is unchanged. This is finite real-model evidence, not a universal tensor theorem.

## Ownership, correctness and qualification

`runtime/mtp_resources.py:MTPWiredLimitLease` owns **only a resource limit**.
It acquires the exact same recommended limit that the native constructor already
uses. The constructor still calls its ordinary acquisition function. After
successful construction the lease verifies that the nested old limit equals its
acquired value, and changes the native generator's restore obligation to the
original pre-request limit. The old constructor/close implementation remains in
charge of active decode and final restoration. No native dependency was edited.

Before transfer, any exception or protected cancellation synchronizes the
existing generation stream and restores the original limit. After transfer,
ordinary native `close` restores it, including failed P5 start. The lease is
acquired under the same singleton worker/shield, inside measured startup. It does
not publish speculative state or outlive native request execution. OFF never
enters this code path. No model-resident lifetime wiring, admission bypass, cache
repack, cache-content mutation or authoritative-state handoff was introduced.

Qualification:

- **Full R1 OFF and MTP: CONFORMANT / PASS, 24 gates each**, unchanged reference identity,
  independently in their normally admitted interpreters.
  Real HTTP model fixture: 18 cases / 27 admission observations, including
  disconnect, cancellation, partial-tool refusal, retirement, and re-entry.
- **7 resource tests pass**, covering normal native transfer/restore, one-shot misuse,
  pre-transfer failure/cancellation-like `BaseException`, zero prior limit,
  mismatch refusal and no-recommended-limit behavior.
- Real model controls compare exact eight-output token trajectories and settled
  target/ring state digests. Deterministic R1 weather/result trajectories retain
  generated, considered and accepted draft counts. Fresh weather acceptance
  remains **24/24**; two-call acceptance remains **46/47**.
- R1 response choices match apart from per-request issued tool-call UUIDs. Timing
  dependent UTF-8 socket-cut observation can change the `generated` counter by
  one while producing the same recovered response; this is not a token-identical
  interrupted-transport claim. R1's actual recovery/settlement predicates pass.
- Zero prompt replay / full-cache repack, aligned idle frontiers, zero settlement
  proposals/verify cycles and protected settlement remain enforced by R1.

OFF qualification and final phase/control receipts are retained alongside MTP.
The final full MTP receipt also covers defensive mismatch cleanup: a newly
constructed empty generator is closed before an untransferred lease restores
the original limit, so its later finalizer cannot overwrite that restoration.
No historical long-context, few-row MMA or unrelated Web campaign was rerun.

## Next boundary

The dominant fresh-start boundary is now **ordinary target dense-prefix work**:
approximately 0.49 s in prefix enqueue/earlier execution plus 1.07 s of deferred
target completion on the real 292-position workflow. DSpark construction itself
is milliseconds; metadata transfer and the ~52 ms terminal forward are small.
The two-call workflow's ~1.66 s steady MTP decode is now comparable to startup.
Any further startup campaign should investigate that target-prefill dataflow/
materialization boundary with true completed-GPU accounting, not rename its lazy
debt DSpark compute or start a general verification MMA campaign.

## Reproduction

Use the same normally provisioned MTP interpreter and admitted OFF interpreter as
`12182b3`. The wheel and checkpoint below are local qualified paths, not release
or packaging work. Normal source sealing is required after executable changes.

```sh
MTP="$HOME/.venvs/ds41f-mtp-investigation/bin/python"
OFF=.venv/bin/python
export DS41F_CHECKPOINT=/path/to/official/DeepSeek-V4.1-Flash
OUT=artifacts/mtp-startup
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig \
  $MTP -m ds41f_mlx.mtp_identity seal --wheel /path/to/qualified-recipe.whl

# Sequentially: only one full model resident.
$MTP -m ds41f_mlx.reference --profile mtp-singleton-v1 --real-model \
  --output "$OUT/r1-mtp-final.json"
$OFF -m ds41f_mlx.reference --profile standard-off --real-model \
  --output "$OUT/r1-off-after.json"
$MTP -m tools.profile_mtp_startup --split-context \
  --output "$OUT/phases-after.json" --r1-output "$OUT/fixture-after.json"
$MTP -m tools.bench_mtp_startup --input "$OUT/phases-after.json" \
  --output "$OUT/controls-production.json"
$MTP -m pytest -q tests/test_mtp_resources.py
$MTP -m tools.summarize_mtp_startup "$OUT"
```

The direct control's `baseline` deliberately uses the unchanged native late
acquisition path without a prepared lease; `early-wire` uses the production
helper and ordinary native constructor. It is a tool-only comparison, not a
runtime/environment/request selector. Optional observer `--joint-state` tests
unified evaluation; `--split-context` separates target hidden production from
projection and intentionally serializes those phases.

Excluded diagnostic attempts: an initial observer launch used the wrong CLI
flag; a second attempt lost its export to Uvicorn's replayed SIGTERM; an early
wiring observer imported the `mlx_lm.generate` package function instead of its
module; an early control digest failed to flatten nested cache state; and a
launch concurrent with the source edit failed closed at executable identity.
None supplies performance or qualification evidence. Final tools handle native
shutdown replay, module imports and nested state correctly; identity checks were
never relaxed. The caller's pre-existing `artifacts/web-application/chrome.log`
change is not part of this work.
