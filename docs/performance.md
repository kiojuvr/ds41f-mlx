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
