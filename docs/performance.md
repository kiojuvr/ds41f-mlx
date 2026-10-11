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
