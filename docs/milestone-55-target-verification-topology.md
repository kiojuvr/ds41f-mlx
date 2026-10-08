# M55 — canonical target verification topology reconciliation

Historical M55 boundary. [M56](milestone-56-canonical-accepted-prefix-block.md)
implements and qualifies the canonical block producer identified here; M55's
measurements and NOT CLOSED decision below remain historical evidence.

**NO / M55 not closed. Bounded materialization repair: qualified and committed.**
The dominant physical blocker is identified, not removed: the canonical M51
producer/undo interface accepts one input row, whereas the candidate executes
four causal rows in one backbone region. This is not an inherent cost of canonical
ownership. A canonical block producer with bounded per-prefix undo/history/tap
receipts is still required. No new kernel research or promotion is justified.

## Frozen correctness baseline

Base: M54 `193689adf153aeb92463d73bcc80bffd9cea5e4b`. Before runtime performance
changes, final M54 source passed unchanged **full R1, both profiles, real model,
25/25 gates**, in Python **3.13.15**, MLX 0.32.2. Final modified source also passed
that full command **once, 25/25 gates**. The receipts contain exact source hashes;
`tools/report_m55_topology.py` verifies current runtime source against final R1.
R1's public OFF/candidate tests do not alone qualify first-party development MTP;
the independent M51 oracle and M54 socket matrix below cover that implementation.

Environment: `/tmp/ds41f-m55-py315`, exact Homebrew 3.13.15 interpreter, installed
M54 dependencies/native recipe, normal identity `seal`/`inspect`, explicit
`DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.
No admission or fixture was weakened. Two preliminary environment attempts are
excluded: external-package-origin rejection, then missing explicit checkpoint
setting. The admitted baseline PASS preceded all runtime changes. The temporary
environment was provisioned locally, not packaged or promoted.

## Actual executable topology

Same request SHA, official checkpoint, greedy output of **41 inputs**, same
message/stop, **F=283**. Extended profiling observes existing Python frames; it
never executes extra model work or inserts synchronization. The three-sample
rate measurements exclude extended topology counting. Counters below are from
one non-warm-up matched sample, not hypothetical source paths.

| Verify-region property | first-party before | first-party after | candidate |
|---|---:|---:|---:|
| Logical speculative cycles | 7 | 7 | 10 |
| Offered / accepted proposals | 34 / 32 | 34 / 32 | 30 / 29 |
| Target verification rows | 41 | 41 | 40 |
| Physical target forward invocations | **41** | **41** | **10** |
| Rows per target forward | 1 | 1 | 4 |
| Target layer projection regions | **1,640** | **1,640** | **400** |
| Target packed attention calls | 1,640 | 1,640 | 400 |
| Target HC pre-norm calls | 3,280 | 3,280 | 800 |
| Journal/tap NumPy host-array copies in verify | **2,050** | **0** | no canonical journal |
| Journal/tap detachment evaluation topology | layer-local | row-wide | block graph / vector acceptance |
| M51 prepare / settle calls | 7 / 7 | 7 / 7 | candidate's existing undo/commit |
| Temporary packed-cache object constructors in verify | 0 | 0 | 800 |
| Cache extract / merge calls | 0 / 0 | 0 / 0 | 400 / 400 |
| Row-admission `make_mask` calls | 1,640 | 1,640 | 10 |

First-party uses `TargetGenerationSession.speculative_cycle` → per-input
`AcceptedPrefixJournal.advance` → `TargetForwardTransaction.forward` → 40 owned
`DecodeStateProducer.block` calls. Target projection, attention, MoE and head work
is constructed anew per row. It does not re-run the same row during inspection or
publication. Across the complete request, 44 forward calls equal one bootstrap
plus 43 planned inputs (including rejected inputs and protected steps).

Candidate `_run_verify_cycle_chain` constructs `[anchor,d1,d2,d3]`, invokes
`_call_backbone_captured` once, and executes each target layer on the four-row
sequence. Its existing vector greedy acceptance has one vector host receipt per
cycle. Its captured hidden tensors remain in the block graph; next proposal work
can be queued before host settlement. Candidate chain profiling also contains
next-proposal calls: those must not be mislabeled target work. Target projection
and packed-attention counters above isolate the backbone. These are executable
layer-region counts, **not GPU dispatch counts or isolated kernel timers**.

### Physical boundaries, separately from authorities

- **Evaluation / Python–MLX crossings:** before, each evicted window row,
  compressor projection, hidden tap and history used eval → NumPy readback →
  MLX reconstruction → eval. 2,050 executed copy calls imply 4,100 explicit
  detachment eval API calls, plus row-logit/tap-concat calls. Eval API calls are
  not necessarily distinct GPU synchronizations. After, one explicit evaluation
  call per verify row covers logits and allocated undo/tap outputs. Existing SSD
  Engram asynchronous prefetch boundaries remain; this is not a claim of a
  globally single-sync forward. Candidate's block receipt is not a claim that its
  entire implementation has no other native/SSD reads.
- **Temporary allocations / lifetime:** old lost rows/projections are private
  references only until the current row completes. `mx.take` on uint8 storage
  allocates independent, bounded outputs; no numerical conversion is performed.
  They are evaluated before another row, never retained as context-sized parent
  graphs. 1,968 bounded row-copy outputs replace 2,050 host copies: three per-layer
  taps become one combined same-forward tap per row. Publication receipts/rings
  retain their existing subsequent copy/evaluation boundaries.
- **Cache/state:** the same 40 objects and seven packed slots are mutated under
  the pending lease. Completion still materializes all target slots, validates
  offsets/admission metadata, and only then permits settlement. Accepted-prefix
  completion all-slot evaluation loops occur once per logical cycle (seven
  cycles, 40 layer eval calls each), not another target forward. Candidate also
  constructs/extracts/merges temporary row-cache objects inside its block model
  path (800/400/400 measured calls); constructors/views are not necessarily
  tensor copies. It is faster despite that management, not because it has no
  state checks. Canonical accepted-prefix
  preparation, independent packed allocation, all-slot validation, atomic
  publication, failure burn and alias exclusion remain intact. No executable
  shadow cache is created. Setup/settlement are per cycle, not the dominant cost.
- **Hidden capture:** same three target layers, same `mean(h, axis=2)` values;
  no hidden re-execution. Capture becomes lazy within the row. OFF/protected
  execute detaches its combined tap after its existing all-state barrier.
- **Resource lease:** canonical M47 parent admission and M53 derived child remain
  model-lifetime resources, not per-row acquisitions. Recorded wired limit is
  498,216,206,336 bytes; canonical allocator cache cap is 34,359,738,368 bytes.
  Candidate uses its existing startup wired-limit lease transferred to its
  generator and its inherited allocator policy, not a canonical M47 parent.
  No allocator/wired policy was tuned. Both restore/retire their existing leases;
  receipt and producer sets are empty after close. Different allocator policies
  are not claimed numerically identical, nor assigned a fabricated phase cost.
- **Sampling / publication:** canonical M52 still invokes its sampler only for
  retained causal rows, preserves RNG draw order and publishes history/lookahead
  after M51 settlement. Candidate uses its existing vector acceptance path.
  Semantic preview and exact-byte response publication remain separate logical
  authorities and are tiny observed phases, not excuses for the verify gap.

## What the experiment removed — and what it did not

First experiment merely delayed and consolidated host copies. Diagnostic verify
fell **3.754 → 3.376 s**. Removing those host crossings with existing allocation-
producing `take` reduced it further to **3.340 s**. The reduction is **11.0%**, not
most of the candidate gap. Inclusive target-forward frame time fell much more
because waits moved into row completion; that must not be relabeled GPU savings.

The main remaining cost is **41 one-row regions versus ten four-row regions**:
repeated Python/native graph construction, separate per-row projections/head/
backbone launches and their weight access, and a row-completion wait before the
next row. No duplicate target rows, replay or full-cache repacking explains the
factor-of-three gap. M51 settlement is only approximately 0.046 s/request.

A calibrated aggregate model is `T_first ≈ 41 × C_row`,
`T_candidate ≈ 10 × C_block4`. Extended observations give **81.46 ms/row** versus
**109.74 ms/four-row block**, predicting a **3.04×** verify ratio. Independent
three-sample ordinary measurements give **2.84×**; extended instrumentation
penalizes the much more fragmented path. This approximately 7% ratio discrepancy
is disclosed, not fitted away. The model explains the large gap through executed
region counts and amortization, not a sum of invented microkernel timings.

### Smallest remaining implementation blocker

`TargetForwardTransaction.validate`, `AcceptedPrefixJournal.advance`,
`DecodeStateProducer.block`, and `compressor_write` all require a single row.
The undo hooks record one evicted chronological row and one compressor projection
per logical input; history/taps are retained per prefix. Merely passing four rows
or bypassing these guards does **not** provide correct rollback for every accepted
prefix, compression crossing and rejected suffix.

Candidate's qualified block arithmetic is not an already-qualified **canonical
accepted-prefix block producer**. Calling its model/scheduler/verification-state
owner would abandon M45/M51 authority, which was not done. The next minimum
physical integration is an owned block producer adapter over existing numerical
primitives, with bounded per-prefix window/projection/history/tap receipts and
all-slot independent OFF qualification. Logical target/generation/recipe
separation does not require row serialization. No evidence here calls for MMA,
FP8, attention, MoE or head kernel research. This missing adapter/qualification,
not “canonical safety”, is why M55 remains **NO / not closed**.

## Numerical / state / application qualification

- Independent real-checkpoint M51 oracle, both ordinary OFF and same-forward
  DSpark-tap-enabled execution: **all 280 slots plus metadata exact**
  for accepted spans 8/4/1/0/6, every first-cycle prefix 0–8, chronological eviction,
  compression crossings, before/after-materialization cancellation and injected
  materialize/prepare/publish faults. Exact list/object identity and burn remain.
  The tap-enabled journal includes an additional 245,760 bytes for eight
  captured rows, proving that the child capture path is exercised; its frozen
  M50 OFF state comparison also passes (latency ratio **1.0098**). This is
  inherited bounded state regression, not a new long-context capability.
- New on-device-copy tests preserve uint8/int64/float32/bfloat16 bytes, empty
  shapes, and prove both retained 128-element and empty receipts release their
  million-element parent allocation. No layer-local readback occurs; a copy
  failure burns all aliases. **260 affected regressions pass**, including canonical RNG,
  history/frontier, lookahead, authorization, reservation and tool boundaries.
- Final M54 operational socket matrix: **17 turns**, **26 speculative cycles**,
  **243 consumed inputs**, **260 forwards = 17 bootstraps + 243 planned inputs**;
  forbidden donor/diagnostic target execution **0**. Tool JSON/SSE, complete and
  incomplete disconnect, byte-identical retry, stale/body mismatch refusal,
  effect-response loss, tool-result re-entry and repeated generation pass.
  **5 real ledger effects; duplicate effects 0.** Executable idle states retain
  all 40 offsets at canonical history/frontier; incomplete cancellation retires
  rather than publishing an executable prefix.
- The EOF race harness was strengthened: rearm its shared worker gate after
  earlier cases, and observe explicit cancel-route entry rather than confusing
  a disconnect-set flag with route admission. Two failed harness attempts are
  retained/excluded. Runtime stale-cancel rejection was not changed. The final
  actually-held EOF worker trial recovers every argument delta/tool_calls finish,
  preserves semantic termination, exact retry and one-effect authorization.
- **Replay / full-cache repack / hidden target re-execution = 0 / 0 / 0** in the
  first-party application measurements. Independent test-oracle execution is
  deliberately separate, not serving replay. Final parent/child inactive,
  receipts/producers **0/0**. No new stochastic application envelope is claimed.
- Final-source full R1: **25/25 PASS**. Its inherited OFF persistence regression
  does not grant first-party MTP persistence/restore capability.

## Matched performance

One excluded warm-up and **three sequential fresh sessions per lane**, no
concurrent model runs. Identical request bytes, generated IDs, message/stop and
frontier across all lanes. Model load is outside both rates. Widths remain each
existing implementation's width, not silently equalized.

| Lane | M54 decode / HTTP tok/s (historical) | M55 decode / HTTP tok/s | Offered / accepted |
|---|---:|---:|---:|
| OFF | 15.88 / 11.11 | **15.61 / 10.91** | — |
| first-party development MTP | 10.62 / 8.17 | **11.68 / 8.74** | 102 / 96 (94.12%) |
| existing candidate | 31.79 / 14.94 | **31.32 / 14.80** | 90 / 87 (96.67%) |

The M54 historical rates used its previous environment. Same-Python-3.13.15
before/intermediate/after topology trials separately establish the bounded
improvement; no historical result is relabeled a new matched three-sample run.

Mean inclusive host-wall seconds per 41-input request:

| Phase | M54 first-party | M55 OFF | M55 first-party | M55 candidate |
|---|---:|---:|---:|---:|
| Integrated decode | 3.862 | 2.627 | **3.510** | 1.309 |
| Full HTTP | 5.016 | 3.760 | **4.691** | 2.771 |
| Prefill/handoff receipt | 0.831 | 0.389 | 0.845 | 1.459 |
| Proposal | 0.176 | — | **0.180** | 0.0428 |
| Target verify | **3.229** | — | **2.868** | **1.011** |
| Protected target + canonical sampling | 0.150 | 2.622 | 0.153 | inside candidate step |
| M51 setup / settlement | 0.0566 / 0.0474 | — | **0.0473 / 0.0456** | not M51 |
| Semantic authorization / preview | 0.000440 | — | 0.000439 | 0.000498 |
| Recipe reporting | 0.00143 | inside normal loop | 0.00147 | 0.000573 |
| M8 accounting | 0.00103 | inside normal loop | 0.00108 | inside settled report |
| Final application publication + response freezing | 0.0000185 | 0.000020 | 0.000019 | inside settled report |
| Candidate settle/report / cache operations | — | — | — | 0.0687 / 0.00579 |

Phases overlap and must **not** be summed. Target-forward wall includes bootstrap
and overlaps verify; application publication is not all HTTP routing/transport.
A separate final-source residual phase receipt observes M52 cycle, tap receipt
publication and derived-ring update boundaries without relabeling the matched
three-sample rates. Its one non-warm-up sample gives:

- M52 cycle inclusive: **3.0141 s** (7 calls).
- Within that cycle, verify **2.8558 s**, setup **0.04794 s**, settlement
  **0.04562 s**; remaining sampling/history/lookahead/receipt work **0.06479 s**.
- Same-forward receipt publication: **0.06638 s** (10 calls, including bootstrap
  and protected steps); overlaps the cycle/step figures, not an extra summand.
- Derived ring publication: **0.14009 s** (10 calls, materialized projected keys).
- Protected steps **0.15151 s**, proposal **0.17999 s**.

The non-nested cycle + protected-step + proposal + ring frames predict
**3.4857 s**, versus observed integrated decode **3.4975 s**: an **11.8 ms**
remainder for reporting/control/clock boundaries. This accounts for the residual
critical path without assigning it to an unspecified safety overhead. Relative
to OFF, row verification has no block amortization, then these proposal/ring/
protected-step boundaries add work that OFF does not perform.

First-party remains **25.2% slower than OFF**, and verify remains **2.84×** the
candidate. Without four-row amortization, it pays near-OFF row execution plus
proposal, undo copies, canonical retained-row sampling, combined receipt/ring
publication and protected warm-up/semantic steps. Proposal itself is 0.180 s
versus 0.0428 s and first-party has separate ring-publication evaluations rather
than candidate's queued predraft region. Semantic/report/application-finalization
costs are not dominant. This is a concrete list of remaining execution boundaries,
not a generalized “safety cost”.

## Promotion boundary

No default change, normal-runtime or `ds41f-runtime` promotion, capability
expansion, Vision, persistence, long-context qualification, concurrency/batching,
candidate removal, release/packaging or UI work. The first-party development
Chat/greedy/single-session bounded envelope remains M54's. Before promotion:
canonical block-producer/accepted-prefix fidelity qualification and practical
matched performance recovery, then separate sampling/application/capability and
native dependency approval checkpoints. Full R1 is now PASS, not a promotion.

Evidence: `artifacts/m55/summary.json`, both full R1 receipts/gate logs,
`real-prefix-state.json`, `real-prefix-with-taps.json`, `real-operational.json`, matched lane receipts, topology
trials, affected regressions and the residual phase receipt. Reproduce comparison
with `python -m tools.report_m55_topology` in the admitted environment.
