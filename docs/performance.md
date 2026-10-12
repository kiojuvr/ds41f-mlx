# Interpreting performance evidence

This specialist guide explains measurement boundaries and recorded investigations.
It is not a benchmark diary, a second capability matrix or an installed-deployment
SLA. [Capabilities](capabilities.md) owns support/limits; [architecture](architecture.md)
owns selected topology; [qualification](qualification.md) explains proof scope.

## Compare the same work

Performance is part of useful runtime completion, but must preserve official
precision/state/protocol contracts. Record source/package/dependency and checkpoint/
tokenizer identity, hardware, profile, prompt/actual output lengths, sampler,
cache state and timer boundaries before comparing results.

Keep model load, initial prefill, retained suffix bootstrap, P5, native decode,
recipe preparation, queueing, transport, canonical settlement, lookup/reconciliation
and retirement separate. Distinguish cold/shape compilation from warm operation,
synthetic seams from real model work and injected delivery delays from normal
latency. A shorter output or early EOS is not a decode-throughput improvement.

MLX work is lazy; unsynchronized host timing can defer GPU debt into another phase.
Use each receipt's stated evaluation/synchronization boundaries. Do not introduce
extra synchronization and compare it as if methodology were unchanged.

## TTFT and reused prompts

A paired prefix hit still requires full-conversation recipe preparation, compatible
state acquisition, genuine suffix-prefill work, prompt checkpoint capture, P5
terminal bootstrap and first canonical decode. It is not zero-work generation.
HTTP-handler-entry TTFT, Scheduler-enqueue latency, first canonical output, first
client-visible chunk and final JSON completion are different timers.

Existing ordinary traces expose nested preparation/recipe and decode/decode-call
phases. Do not sum a parent with its children or assign the remaining total to
attention by subtraction. Real queue/load/resource costs remain in their stated
boundaries. `cached_tokens` counts actual restored paired state, not just a hash
lookup or four times configured cache capacity.

The [finite full-hit diagnostic](mtp-production-serving.md#full-hit-ttft-diagnostic)
found context-proportional encoding and an ordinary model/handoff floor, not a
confirmed dominant long-state copy or attention defect. Its non-monotonic medium
restore was a first-hit transient with unassigned cause. No restore optimization
was selected; the motivating long delay was not reproduced universally. Do not
turn those observations into an SLA or an automatic optimization campaign.

## Recorded investigations and execution evidence

| Report | Recorded scope / interpretation |
| --- | --- |
| [First-party OFF 200K core](standard-off-200k-production-qualification.md) | Real long prefill/decode/continuation/restore and allocator-lifetime repair; not a new long-HTTP tool-loop claim |
| [Very-long OFF core](very-long-context-production-qualification.md) | Real 512K/768K/near-1M maintenance ingestion, continuation/restore/resource behavior; initial ingestion is not interactive latency or physical-RAM exhaustion proof |
| [Target prefill investigation](target-prefill-investigation.md) | Shape occupancy, P6 work elimination, dispatch and source/tail amortization; the packed-attention candidate was not selected despite a prefill gain, because retained decode regressed |
| [MTP verification compute investigation](mtp-verification-compute-investigation.md) | Actual row/expert occupancy and verification/head costs; a private microkernel speedup is not the modeled end-to-end gain or a promoted dispatch |
| [Ordinary MTP serving](mtp-production-serving.md) | Actual cache/settlement/tool/output integration and finite TTFT evidence; no 393K-output endurance or arbitrary-prompt throughput guarantee |

Each report owns its exact measured numbers, provenance, failure/exclusion history
and candidate decision. Old native reference, oMLX migration, DwarfStar prototype
and rejected P8 measurements remain historical comparison evidence in the original
records/pre-compilation edition. The old reference engine is not the selected
production engine; a different checkpoint representation is not a model-fidelity
comparison. No measurement is refreshed by this editorial compilation.

## Memory and lifetime interpretation

Live MLX allocation, allocator free cache, process RSS, footprint, OS compression,
wired residency and system swap are different quantities. Model-dominated stable
bytes do not disprove monotonic CPU diagnostic retention; a small finite plateau
does not prove unlimited lifetime. Report actual frontier/state/container ownership
and sampling coverage alongside resource measurements.

OFF model-lifetime free-cache/wired policies and passive P6-reference retirement
are [resource contracts](runtime-strategy.md), not token-loop optimization or KV
eviction. Paired snapshot budgets must not assume cost-free array sharing.
Configured output capacity is a scalar termination/admission ceiling, not automatic
output-sized memory reservation. Preserve these distinctions when reading peaks.

## Optimization and claims

A leaf-kernel win, modeled compute fraction or one short fixture is not end-to-end
improvement. Require a concrete supported bottleneck, preserved state/precision,
matched workload and affected correctness/performance evidence. A prefill gain does
not authorize a steady decode regression or a rebuilt physical graph promising
future recovery. Rejected candidates stay non-selected until explicitly justified.

There is no universal MTP speedup, all-prompt quality guarantee, batch-throughput,
non-target-host or generalized multimodal performance claim. Use source-scoped
regression classes and workload-specific observations; stop investigating once the
concrete question is answered. Release acceptance and performance measurement are
distinct scopes, not interchangeable PASS labels.

## Post-completion compressed growth (#4365)

Runtime completeness remains **COMPLETE**, ordinary production remains MTP. This
is an isolated **post-completion performance improvement**, not renewed runtime
qualification. The 0.7.0 base, Scheduler, paired authority, pinned model topology,
DwarfStar/P6/P5, DSpark semantics, Vision, recipe and session contracts are unchanged.
No #4354/#4360/#4363 or upstream release upgrade is included.

The [build-time overlay](../third_party/mtp/compressed-growth.json) adopts upstream
`09be4f420c7409298656bcfa13699dee3a0ed938`: only compressed-KV slot 2, index-key
slot 3, singleton extraction and native partial-rollback truncation. One adaptation
preserves an already correctly sized index-key handle on ratio=2 no-publication
tokens; otherwise slicing breaks the growth chain every other token. The patch
adds 122 net dependency LOC; provisioning/admission construction adds 22 net LOC
(about **+144 production LOC**, excluding tests, driver, metadata and docs).

Two capacity buffers alternate. Short appends update only the missing tail in the
idle buffer; capacity growth, long prefill appends, external replacement and fresh
restore can still copy the prefix. Held MLX references force copy-on-write, never
an unsafe destructive write. Singleton extraction shares this advisory registry,
not a second execution owner. Multi-row extraction does not share it. Native
partial rollback retains the chain and invalidates speculative rows logically;
subsequent writes replace them. Paired capture/restore still copies only logical
arrays/metadata, not capacity/registry, with unchanged format and frontiers. There
is no replay, numeric pack/unpack or full-cache repack.

### Finite evidence and trade-off

[New evidence](../artifacts/compressed-growth/provenance.json) is separate from all
historical immutable receipts. Affected tests compare old concatenation against
new growth with **array-exact logits and all seven slots**, speculative rejection
and continuation. A 128K-row packed fixture covers singleton extraction,
truncation/overwrite, held snapshots and external replacement. Native MLX paired
capture/acquisition remains stable after repeated live/restored growth, including
image identity. Existing Scheduler cancellation/publication/reuse, wrapped rings,
P5/P6/P7 and upstream official-output/rollback/Vision tests are reused; these are
finite seam checks, not a new full-model numerical qualification.

The [128K-context packed-slot probe](../artifacts/compressed-growth/slots-128k.json)
uses the four production KV sources, ratios `(2,2,2,1)` and packed widths 288/80,
64 synchronized updates after four warmups, in old/new/new/old order. Median
slot-update time is approximately **1.30–1.32 ms → 0.24–0.26 ms**. This excludes
model computation, index scoring and DSpark, and is not production tokens/sec.
Live packed allocation at the end is **120.7 MB → 257.2 MB**: about 2.13x for these
slots, due to two buffers plus capacity headroom. The benefit is eliminating
context-sized repeated tail-publication copies/allocations, not reducing resident
state. Snapshot/restore costs are not optimized. Capacity and registry are not
checkpoint fields; the existing snapshot materialization/copy budget still applies.
Do not infer zero-copy under every alias or a near-1M HTTP speedup from this probe.

[Ordinary production before/after](../artifacts/compressed-growth/comparison.json)
was measured serially in isolated copies of the installed MTP environment
(Python 3.13.15, MLX 0.32.2). The candidate wheel passed unchanged installed
admission with the explicitly recorded overlay; the operator's installed default
was not modified. A genuine 32,762-token prompt generated 64 tokens, then two
paired repeat requests per version restored 32,760 tokens and executed the real
two-token suffix/P5/native decode. All three canonical token lists, projected
responses (except request ID/time), acceptance/depth counts and target/DSpark
frontiers **matched exactly**. Every pair settled/published at 32,826 with zero
prompt replay/full-cache repack.

Warm repeats' aggregate `decode_s` averaged **2.5514 s → 2.5394 s**, approximately
**25.08 → 25.20 completion tokens per decode-second (+0.47%)**. This trace timer
excludes prefill/restore/P5/settlement and is not wall-clock request throughput.
With two repeats the difference is within noise: no significant end-to-end
speedup is claimed. Seed/load/TTFT are not attributed to this optimization.
The finite 4K slot control was also non-regressing (roughly 0.33 → 0.28–0.29 ms);
short-session capacity overhead increases and is explicitly not a memory saving.
There is no observed semantic/performance regression in these finite seams;
Vision/cancellation coverage reuses existing affected regressions, not a new
long-image/cancellation campaign. Structural copy improvement with this small,
state-preserving delta justifies adoption despite the small production timing
change. #4365 is closed independently; subsequent performance candidates may be
evaluated separately, not bundled into this change.

## Post-completion bit-exact kernels (#4360)

**CLOSED: partial adoption**, independently of #4365. Baseline is current HEAD
`f3ab28c07cd677ca790f2ea5f7607745ccc38a4c`, with compressed growth enabled.
Upstream authority: PR #4360, commit
`f0d4cb53271456a474948edcdb38e0d2da347d79`. This is not an oMLX upgrade,
completion/default requalification, or installed-operator payload update.

### Reachability and selection

| Upstream change | Production MTP decode | DwarfStar/P6 applicability | Decision / overlap |
|---|---|---|---|
| MXFP8 GEMV | Yes: eligible dense target and DSpark projections, one row and 4/5-row verify blocks; gathered experts and other geometries stay on their existing kernels | Yes for eligible short rows/cones and completing suffix/P5; not bulk GEMM | Adopt with pinned byte decoder; projection activation rounding, expert/custom paths and dispatch topology unchanged |
| FP8 activation/KV packing | Yes: window KV FP8 packing, not FP4 compressed/index packing | Yes, via the same language/math pack entry point | **Reject**: not byte-identical to the pinned compiled pack graph; retain all current packing and activation quantization |
| Lane Sinkhorn | Yes: FP32 4x4 matrices with <=64 rows, including target/DSpark mHC | Yes for short tiles/cones/suffixes; >64 rows retain the serial kernel | Adopt: same serial row/column sum order; no mHC projection/pre/post replacement |
| Native sparse attention direct-Q and pre-unpacked KV | No: pinned BF16 dispatch returns `rounded_packed_attention` before the native attention extension | Not the qualified BF16 DwarfStar/P6 path either | Omit; would not improve production, and must not replace the qualified rounded path. No BF16 scratch/unpack kernel is added |
| Radix index top-k | No: decode/verify <=8 query rows never dispatch it | Can reach long P6 index refreshes (>8 rows, >=4096 visible keys, no candidate blocks), not just generic upstream prefill | Omit: upstream adds grouped FP32 score materialization (up to 64 MiB) plus `async_eval`/host `eval` scheduling. Not a kernel-only replacement within the retained topology; no long-context campaign or P6 scheduling redesign undertaken |
| Remaining #4360 delta | No independent runtime change | `_SCALE` refactor supports rejected packing; C++/Metal variants support omitted attention; rest are tests | Omit |

The [overlay](../third_party/mtp/bit-exact-kernels.json) changes only three oMLX
source files (`gemv.py`, `hyper_connection.py`, `quantization.py`), roughly **220
production LOC**, plus small build/provisioning overlay plumbing. It applies
AFTER #4365, strictly checks before/after hashes, and inherits the unchanged
0.7.0 dependency/native payload. Neither ds41f `model_execution`/OFF numerical
source nor MTP state ownership, Scheduler, P6, samplers, Vision or rollback is
changed. Source export, installed authority and qualification remain distinct.

### Pinned-source numerical review (not upstream's assertion)

The production environment is Python 3.13.15 / MLX 0.32.2 on Apple M3 Ultra
(`applegpu_g15d`). The installed `fp_quantized.h` was compared byte-for-byte with
MLX v0.32.2; [source identities](../artifacts/bit-exact-kernels/mlx-source-identities.json)
also bind pinned `fp8.h` and the corresponding host `quantized.cpp` reviewed for
dispatch. This host selects `fp_qmv_fast` for the admitted aligned singleton
shapes and `fp_qmv_wide` for 4/5 rows (the vector-limit minimum exceeds 5).

GEMV preserves each lane's eight-value qdot, scale multiplication, 256-value
K loop and SIMD sum for singleton rows. Wide GEMV preserves 16 K lanes,
eight float4 dots per quant group, group stepping and shuffle-down 8/4/2/1
reduction. Only output-row mapping and activation-load reuse change. E8M0
conversion is identical, including code zero's FP32 subnormal bits.
**An upstream discrepancy was found:** its FP8 decoder maps reserved `0x7f/0xff`
to NaN, while pinned `fp8.h` maps them to +/-480. The isolated backport uses the
pinned half-bit shift, multiply by 256, then sign expression instead. Tests cover
all 256 weight byte codes, not just values produced by `mx.quantize`.
Unadapted upstream's failure is preserved, not treated as qualified evidence.

Sinkhorn retains the original iteration sequence (initial column normalization,
then row/column), operand order `value + total`, epsilon addition and division.
SIMD shuffles transport entries without changing arithmetic or sum order;
reassociation/contraction remain disabled. The old `ROWS` template/guard and
serial fallback are retained, rather than importing unrelated upstream template
changes.

**Packing's rejection is concrete:** for an all-zero 32-value FP8 group, all
three production-supported dtypes yield scale byte **2 in the current compiled
graph vs 1 in upstream's fused kernel**, although value bytes are all zero.
The upstream test's new undecorated `_pack_activation` reference does not prove
identity to the current `@mx.compile` pack entry point. See the
[byte witness](../artifacts/bit-exact-kernels/packing-rejection.json) and failed
boundary tests. No attempt was made to fix/normalize this existing numerical
representation, because that would violate this task's semantics constraint.

### Finite equivalence and performance

[Receipts](../artifacts/bit-exact-kernels/provenance.json) include 39 affected
operator/connected/delivery tests, 98 lifecycle/sampling/Vision/serving seam
tests, 66 P5/P6/P7/suffix/attention tests (+26 subtests), and 80 reused upstream
rollback/official/vision tests. Connected logits and every captured cache slot
match exactly across extract, native verify, partial/full rejection rollback and
continuation, with #4365 still enabled. One unrelated cancellation mock test
fails identically on baseline and candidate: its old `SimpleNamespace` fixture
lacks `conversation`. It was recorded, not fixed or hidden as a passing test.

The [ordinary production control](../artifacts/bit-exact-kernels/comparison.json)
runs serially in isolated admitted environments: a real 4,094-token prompt,
64 output tokens, seed plus two exact prefix-hit greedy requests and one seeded
**ordinary default sampling** request per version. Hits restore 4,092 tokens
and execute the real two-token suffix/P5/MTP path. Canonical tokens, projected
output/usage, MTP acceptance/depth counts, target/DSpark frontiers and **all
retained paired target-slot/ring bytes** are identical for every corresponding
row. Replay/repack are zero throughout. A final candidate-only eight-token
request checks GEMV against pinned `mx.quantized_matmul` on real production
operands; its additional reference dispatches are excluded from timing. It
observes singleton prefill/P5 and singleton/four-row target/DSpark decode
projections. Python Sinkhorn tracing counters are not execution counts once
`hc_mixes` is compiled; Sinkhorn identity is established by source order and
independent exact tests, not fabricated counters.

Warm greedy repeats' mean `decode_s`: **2.48206 s -> 2.45094 s**;
**25.785 -> 26.112 completion tokens/decode-second (+1.27%)**. This is a bounded
phase-timer result, not wall-clock throughput or a statistically significant
benchmark campaign. Ordinary sampling's single row is an equivalence control,
not a separate throughput claim. Seed decoding (2.50223 -> 2.66666 s) includes
cold compilation/first-call effects and is not the warm attribution estimate.

Applicable qualified completing-prefill control (warm second repeat,
`prefill_handoff_s`, including suffix/P5): **111.153 -> 111.421 ms**, effectively
unchanged (+0.24%). No bulk-prefill speedup is claimed: most large-row kernels
are untouched. Cold seed prefill receipts (22.4155 -> 22.5063 s) are retained
for diagnostics, not an attributed kernel-speed measurement; the candidate's
first P5 handoff includes new first-call kernel compilation.

Measured peak MLX memory is **320,323,747,776 bytes in both versions**. No new
persistent cache, weight owner, KV expansion or score scratch is introduced;
only additional small JIT kernel/pipeline variants exist. #4365's existing
capacity storage remains unchanged. No new semantic or warm performance
regression was observed; baseline fixture failure and cold compilation effects
are explicitly separated. This small same-state delta structurally reduces
short-kernel work and is accepted despite modest end-to-end impact.

A **combined #4365 + selected #4360 fresh installed-runtime update is
recommended**, through normal package construction/admission, not an in-place
patch or broad upstream upgrade. It was not performed here. #4360 is closed;
subsequent candidates can be evaluated independently. No #4354/#4363 work was
started.

## Post-completion template reuse (#4354)

**CLOSED: selective adoption** on HEAD
`9fa18144a317c662c618b3bd227ec17980de0fe0`, independently of the completed
#4360/#4365 work. Authority: upstream PR #4354, commit
`63917f5f6a9bd429621d8e615ba958269a2fcee6`. Runtime remains **COMPLETE**, ordinary
MTP remains default, oMLX remains **0.7.0**. No #4363, arithmetic, dispatch,
state-owner, generation, Vision, native payload or architecture change.

### Reachability, overlap and exact delta

The [third overlay](../third_party/mtp/template-reuse.json) applies AFTER the
existing #4365/#4360 overlays, checking every input/output hash. It changes only
`activation.py`, `hyper_connection.py`, `kernels.py`, `routing.py`: 27 added /
17 removed dependency lines; about **50 touched production LOC (+11 net)**
including two provisioning/package-construction files. No base archive, historical
manifest or OFF numerical source is changed.

| Existing production kernel | Removed template input | Evidence / retained boundary |
| --- | --- | --- |
| Index tile selection | `K`, `NT` | Current packed-index path, including decode's growing source/candidate widths and P6 refreshes; integers now in existing metadata array |
| Sorted top-k merge | `K` | Current bounded merges; metadata gains one integer; `GROUPED`/`FANIN` unchanged |
| FP8 activation rounding | `N` | Actual target/DSpark projection activations; NOT #4360's rejected FP8 packing; element count from input shape |
| Unpaired SwiGLU rounding | `N` | Actual routed/shared expert path; count from gate shape; weighting, limit and rounding unchanged |
| Sinkhorn | `ROWS` | #4360 lane source already reads runtime shape but caller still specialized on UNUSED `ROWS`; remove it and adapt retained serial fallback guard; <=64 lane dispatch unchanged |
| Hyperconnection post | `ROWS` | Actual prefill/decode mHC; runtime input-shape size, same FMA/order/casts |
| Sorted expert combine | `ROWS` | Actual BF16 six-expert path; runtime shared shape, same expert summation order |
| Paired-concat SwiGLU | **Not adopted** | Not observed on admitted checkpoint across production requests; original generated source and `N` specialization retained, independently exact-tested; negative-control churn remains |

#4360 GEMV is untouched. Its rejected packing, omitted attention/radix work and
pinned decoder adaptation remain unchanged. There is no template-driven array
allocation, new reduction algorithm, padding, score grouping or host scheduling.
All changed operations before the original arithmetic are integer count/guard
calculations only. The same bounded tile, dtype, weighting, block/ratio, iteration,
fan-in and lane/serial geometry still specializes where necessary.

### Compiler identity and cold/transition evidence

[Receipts and method](../artifacts/template-reuse/provenance.json),
[comparison](../artifacts/template-reuse/comparison.json) and
[leaf driver](../tools/bench_template_reuse.py) distinguish three scopes.
Pinned MLX 0.32.2 source was inspected: generated template name, input dtype and
scalar/constant (<8 elements)/device address spaces, output dtype and source hash
feed `CustomKernel` -> `Device::get_library` / `get_kernel` cache keys. Counts are
**new specialization identities, not intercepted compiler invocations or GPU
execution counts**. First synchronous evaluation pays any missing library/pipeline
work; Python tracing reuse alone is not used as proof of GPU reuse.

Cold leaf processes use fresh UUID Metal function names, hence distinct generated
source/compiler identities, without deleting the operator's compiler cache.
Before/after/after/before initial probes were followed by two final-candidate
probes after the unobserved paired branch was omitted. Each shape's first evaluation
is synchronized, followed by 20 warm evaluations; input creation is outside timers.
This isolates affected first-use cost, not full-model cold-start latency.

| Representative leaf | Cold first shape, before -> after | Same-process next shape, before -> after |
| --- | --- | --- |
| FP8 activation, rows 1 -> 4 | 66.45 -> 67.82 ms | 31.90 -> 0.26 ms |
| SwiGLU, rows 1 -> 4 | 32.20 -> 34.40 ms | 32.03 -> 0.24 ms |
| Lane Sinkhorn, rows 1 -> 4 | 33.92 -> 34.43 ms | 33.68 -> 0.24 ms |
| Serial Sinkhorn, rows 65 -> 129 | 38.72 -> 40.65 ms | 38.63 -> 0.28 ms |
| Index selection, visible keys 257 -> 258 | 40.81 -> 41.63 ms | 40.39 -> 0.26 ms |
| Multi-tile index, keys 2049 -> 3073 | 72.67 -> 75.31 ms | 39.93 -> 0.57 ms |

The first genuinely new pipeline is **not eliminated or faster**; repeated shape
compilation is eliminated within each retained geometry. Leaf specialization
counts fall **43 -> 17**, including the six unadopted paired negative-control
variants in both versions (**37 -> 11 selected variants**). HC post/expert combine
still compile when their small input crosses MLX's constant/device boundary
(rows 1 -> 4 here); subsequent row changes reuse. Tile-size, fan-in and dispatch
geometry boundaries also intentionally remain. This is not a promise of one
pipeline for every possible shape.

### Actual production, warm/resource and semantic acceptance

[Production driver](../artifacts/template-reuse/production-driver.py) executes
serially in isolated admitted installed environments on Python 3.13.15 / MLX
0.32.2 / M3 Ultra. One process per version serves real **505 -> 637 -> 1142-token**
prompts, each with a seed, two paired greedy hits and a seeded ordinary default
sampling hit (64 output tokens each). Hits restore prompt-minus-two tokens and
execute genuine two-token suffix/P5/native MTP decode. New affected custom
identities per seed are **92/48/26 -> 14/1/3**; none on greedy repeats. Actual
source-count transitions reach top-k and activation/SwiGLU/Sinkhorn/post/combine;
paired-concat does not appear. Remaining variants reflect bounded geometry or
input address-space changes, not a missing removal of the selected counts.

Same-process second/third prompt TTFT is **3.650 -> 3.428 s** and
**6.001 -> 5.857 s** (wall **6.133 -> 5.869 s**, **8.450 -> 8.221 s**).
These are finite illustrative request timings, not a compilation-only saving:
request1 loads the real model, and production disk compiler cache is uncontrolled.
Request1 TTFT **88.444 -> 91.142 s** is recorded but **not attributed** to this
patch. The unique-name leaf first-use/transition control, not this model-load
noise, demonstrates the removed stalls.

Six warm greedy hits' mean decode time: **2.40695 -> 2.41188 s (+0.20%)**;
**26.590 -> 26.535 completion tokens/decode-second**. Effectively unchanged;
no material warm regression or steady-state speedup is claimed. Leaf warm timings
are likewise non-material and host-submission dominated. No benchmark campaign
or long-context extrapolation is justified.

All 12 corresponding production rows have **identical canonical tokens, projected
responses/usage/protocol, acceptance/depth counters, target/DSpark frontiers and
retained paired target-slot/DSpark-ring bytes**. Replay/repack remain **zero**.
80 operator/connected/delivery tests include byte-exact comparisons against the
CURRENT #4360/#4365 baseline for all affected operators, strides, dtypes, tie
ordering, block/explicit top-k, lane/serial Sinkhorn, verify/rejection/rollback
and continuation. 118 serving/prefix/cancellation/Vision/P5/P6/P7 seam tests
(+26 subtests) and 130 reused upstream official-output/DSpark/rollback/Vision
small-model tests pass. Two unrelated failures are reproduced on baseline:
an old cancellation mock lacks `conversation`; an upstream full-VLM-engine
subprocess requires undeclared `openai_harmony`. Neither is repaired or counted
as a pass; finite successful Vision coverage includes aligner/processor/loading,
image identity, paired restore and cancellation seams, not a fresh full-image
endurance campaign.

Peak MLX allocation is **312,626,664,100 -> 312,626,648,876 bytes**, effectively
identical; leaf peak **37,388,608 bytes** in both versions. No persistent arrays,
KV capacity, model memory or score scratch are added. Top-k metadata gains only
8 bytes per tile invocation / 4 bytes per merge plus normal shape uniforms;
pipeline variant count decreases (CPU pipeline memory itself was not measured).

**Recommend one normal installed-payload rebuild/update now that source adoption
is CLOSED.** The operator's installed runtime was not modified. Candidate-only
package construction/admission was used for acceptance; no ad-hoc reseal or
in-place operator patch. This remains a post-completion performance improvement.
