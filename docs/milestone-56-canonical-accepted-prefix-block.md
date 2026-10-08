# M56 — canonical accepted-prefix causal block execution

Base: master `5bad422cf9b72f0096e0ce63f3132f9b97fd967a` (M55).
**Canonical producer/state/application boundary PASS; bounded practical local
candidate. Full R1: 25/25 PASS**, both profiles and real model, run exactly once
on final runtime source (`artifacts/m56/full-r1.json`), not inherited from M55.
No default/profile change, new kernels or `ds41f-runtime` promotion.

## Ownership and executable structure

M52 `TargetGenerationSession.speculative_cycle` still owns proposals' admission,
causal canonical sampling/RNG, acceptance, history/frontier/lookahead and reports.
It now submits up to eight speculative inputs per **single numerical forward**
through M51 `AcceptedPrefixJournal.advance_block`, rather than calling `advance`
for each input. A 32-input journal consists of at most four numerical blocks;
this bound does not expand M53/M54 proposal/application admission.

`TargetForwardTransaction.forward` embeds/hashes the causal sequence once and
executes 40 `DecodeStateProducer.block` regions, each consuming the full sequence.
The sole existing cache list and its exact 40 packed objects remain pending.
There is no executable row view, extraction/merge, donor model call, candidate
scheduler/state, shadow cache or target replay. A width-one block uses the retained
row primitive; wider blocks do NOT loop it. OFF `validate`/`execute` remain
one-input transactions and never allocate a speculative journal.

The producer records bounded undo **for every input prefix**:

- Window: chronological rows lost at each position, including rows introduced
  earlier inside the same block when the block exceeds available window space.
- Compressor: each pre-pooling KV/gate projection; initial remainder plus accepted
  projections restores the exact accepted remainder. Completed packed groups are
  selected, never pooled/quantized/executed again.
- Engram: tokenizer-derived per-prefix lookbacks, not another generation history.
- DSpark: the same `mean(h, axis=2)` post-layer 37/38/39 taps, produced by this
  exact forward, detached per row, exposed only for consumed inputs after settlement.
- Logits: all rows materialized once, sampled only in retained causal order.

Undo/taps detach together at block completion, not after each target row/layer.
M51 completion still evaluates **all slots and metadata**. Settlement prepares and
evaluates all accepted arrays before publication; every alias stays pending until
all-layer assignments and the owner's frontier hook succeed. Any uncertain
mutation, completion, preparation, publication or owner failure burns all aliases
and drains/retire permissions. Cancellation before sampling drains the protected
block and settles prefix zero; after sampling begins it remains deferred. No OFF
fallback retries an uncertain cycle. Semantic horizon, EOF/tool/effect/retry and
re-entry owners are unchanged.

### Two numerical integration defects actually found and repaired

A naive multi-row numerical call is not canonical row equivalence. Initial real
B=8 execution changed 53 packed/remainder slots. Diagnostic layer-0 projection
and HC boundaries exposed shape-dependent GEMM/GEMV reduction differences.
The repair uses existing MLX **batched singleton GEMV**, not a Python per-row
projection loop or a new kernel. Quantized target weights remain shared read
views; small dense HC/router/output/compressor matrices use bounded RHS `take`
scratch to prevent MLX collapsing broadcast weights into a different GEMM.
Compiled HC retains OFF's fused normalization/mix graph. The existing head kernel
already has a query axis; the owned short-block scope admits up to eight queries
on that unchanged implementation. Scope is worker-local, reset even on failure;
OFF/prefill/proposal arithmetic is unchanged. Exact module hashes were updated in
M47 admission, not removed/bypassed. No Metal/MMA/FP8/MoE/attention kernel changed.

A further real greedy F=255/B=6 oracle failed at layer-2 attention. Sparse top-k
sorting puts invalid `-1` entries **before** valid indices. Using the block-end
list width shifted earlier causal keys across the qualified 64-key BF16 online
softmax tiles, despite correct future masking. The producer now retains each
row's canonical source/candidate list capacity and groups **only attention** at
width transitions, trimming the extra leading padding. Embedding, projections,
HC, MoE and head still consume the whole block. This uses existing qualified
packed attention calls, not modified numerical tolerances or a new kernel.

These failures and excluded driver/environment attempts remain in the artifacts.
They are not PASS evidence, and their fast timings are not substituted for final
matched results. There is no hidden row-forward repair/re-execution.

## Final checkpoint and operational qualification

Environment: the locally admitted `/tmp/ds41f-m55-py315` Python **3.13.15**, MLX
0.32.2, unchanged M54 consuming-EOF recipe wheel, explicit official checkpoint.
Normal local `seal`/`inspect` pass. No external dependency origin, checkpoint,
expected fixture, native identity or semantic gate is relaxed.

- Final F=4095/B=8 independent sequential OFF oracle: **all 280 slots plus
  metadata, retained full logits and same-forward taps exact**. Repeated accepted
  spans 8/4/1/0/6, every first-span prefix 0–8, real compression crossings/window
  eviction, before/after-materialization cancellation and 15 materialize/prepare/
  publish/owner faults pass. Exact list/object identity and burn are preserved.
- F=255 greedy-input oracle: B=8 and **every prefix of widths 2–7** (33 additional
  trials), including rejected suffixes, candidate-capacity/compression crossings,
  history and taps. All-state/tap/logit hashes are exact. Both cancellations,
  all 15 faults, OFF continuation and resource/SSD drain pass. Width-one remains
  the independently qualified row primitive. Oracle lists are destroyed before
  tentative lists; offline qualification execution is not serving replay.
- Real DSpark proposal matrix passes greedy and explicit-key stochastic OFF
  state/history/lookahead parity, first/partial rejection, low acceptance,
  cancellation, failures before verification/after publication, target burn,
  repeated legitimate warm-up/continuation and child/parent retirement. Its main
  64-input loop executes **12 forwards for 67 physical rows** (one bootstrap plus
  eleven six-row blocks), versus M53's 67 one-row forwards. Main core rate is
  22.00 tok/s; this is NOT the matched HTTP performance figure.
- Final affected regressions: **498 PASS**, including every maximum-span prefix,
  actual packed primitive width transitions, all block widths, sampler/RNG,
  canonical history/frontier, alias exclusion, atomic publication and faults.
- M54 actual socket envelope: **17 turns, 26 speculative cycles, 243 consumed
  inputs**. All 17 generated-ID sequences/frontiers equal M55 row execution,
  including the held consuming-EOF cancellation race and disconnect trials.
  **142 physical forwards / 260 physical rows / 5,680 layer regions**, rather
  than M55's 260 forwards / 10,400 regions. Forbidden execution **0**, actual
  ledger effects **5**, duplicates **0**. JSON/SSE, every argument delta/tool finish,
  exact-byte retry, stale/body mismatch refusal, incomplete-effect denial,
  effect-response loss, tool-result re-entry and repeated generation pass.
  Parent/child retire with receipts/producers empty; incomplete tools burn rather
  than publish an executable continuation. No stochastic application expansion.
- Final-source full R1: **25/25 PASS**, `--profile both --real-model`, exactly one
  run after the stable producer/state/socket/matched boundary. OFF protocols,
  tools, re-entry and persistence regression pass. R1's public candidate lane is
  not substituted for the first-party oracles/socket evidence above; OFF restore
  does not grant a first-party MTP persistence capability.

## Matched performance and physical topology

Same exact M55 request, **41 generated inputs**, same IDs/message/stop, **F=283**.
One excluded warm-up plus three sequential fresh sessions per lane, no concurrent
real models. Load is outside rates. Extended topology is a separate diagnostic
trial, not included in those three rates. These are executable region/frame
counts, **not GPU dispatch counts or isolated kernel timers**.

| Verification topology | M55 canonical row | M56 canonical block | existing candidate |
|---|---:|---:|---:|
| Logical cycles | 7 | 7 | 10 |
| Offered / accepted | 34 / 32 | 34 / 32 | 30 / 29 |
| Speculative target rows | 41 | 41 | 40 |
| Physical target forwards | **41** | **7** | **10** |
| Observed forward widths | 1 | **six × 6, one × 5** | ten × 4 |
| Backbone layer/projection regions | **1,640** | **280** | **400** |
| Packed attention calls | 1,640 | **748** | 400 |

Thus this is actual target/backbone amortization, not a row loop hidden behind a
block API. Attention's canonical list-width subregions explain why its counter
does not reduce all the way to 280. Whole-request first-party forward calls are
**10** for **44 physical rows**: bootstrap + two protected rows + seven blocks.
OFF has 42 one-row forwards (bootstrap + 41 inputs). Rejected speculative inputs
are planned physical rows, not canonical history or hidden re-execution.

| Lane | M55 decode / HTTP tok/s | M56 decode / HTTP tok/s |
|---|---:|---:|
| OFF | 15.61 / 10.91 | **15.51 / 10.93** |
| first-party MTP | 11.68 / 8.74 | **17.85 / 11.88** |
| existing candidate | 31.32 / 14.80 | **31.45 / 14.82** |

First-party is now **15.1% faster than OFF** in integrated decode and **8.7% faster**
through HTTP. Versus M55 it improves decode approximately **52.8%**. Mean verify
falls **2.868 → 1.661 s** (42.1%); candidate is **1.006 s**, so parity is not claimed.
Inclusive phase timings overlap and must not be summed. Counts and all matched
correctness assertions are reproduced by `tools/report_m56_block_execution.py`.

**Replay / full-cache repack / hidden target re-execution = 0 / 0 / 0.** M51's
accepted packed-byte selection copies still exist, including potentially
context-sized source-prefix copies on partial settlement. Bounded dense RHS
scratch is numerical parameter-read scratch, not an executable cache or repack.
No zero-memcpy, zero-sync or one-GPU-dispatch-per-block claim is made.

## Normal local profile assessment and remaining boundary

**YES: first-party MTP is now a coherent, practically beneficial ordinary local
operational candidate within the qualified M54 envelope.** The missing canonical
block producer is no longer a blocker. A separate normal-local-profile approval
checkpoint is now justified; it need not first achieve candidate tok/s parity.

This does **not** approve an unrestricted profile/default or release promotion.
The scope remains explicit text Chat, greedy, reasoning none, one session,
<=64 output and <=512 prompt+output reservation. The consuming-EOF native patch
still needs a normal dependency approval/admission decision; unsupported Vision,
persistence/restore, protocols, context and stochastic application modes do not
inherit OFF capabilities. `ds41f-runtime` is untouched.

The smallest remaining physical performance differences are concrete: exact
row-reduction batched GEMV (rather than candidate's block GEMM), **748 versus 400
attention subregions**, and unchanged proposal/receipt/ring boundaries. They are
not another missing accepted-prefix ownership primitive, replay/repacking, or
permission to reopen kernel research. Broader semantic/capability/native approval,
not the qualified block/state/application path, bounds profile promotion.
