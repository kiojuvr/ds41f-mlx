# Generation runtime

## Qualified release path

The production text release uses dense P0-P7 prefill plus P5 zero-replay handoff into ds41f `TargetGenerationSession`, its M45 `TargetForwardTransaction` and M46 `DecodeStateProducer` with MTP, DSpark, and speculative decode OFF.

```text
recipe-rendered prompt tokens
  ↓
hold out terminal prompt token once
  ↓
DENSE_P0_P7 prefill commits tokens[:-1]
  ↓
P7 SSD-backed Engram / live DeepseekV41Cache[40]
  ↓
P5 handoff supplies tokens[-1] exactly once
  ↓
ds41f single-token forward / 40-layer mutation transaction
  ↓
commit generated tokens/KV
  ↓
DeepSeek EOS, length, cancel, or continuation
```

## Commit and continuation invariants

Prefill initializes persistent cache/token state. Incremental decode consumes and extends that state. M44 commits all-token history only after M45 completes all 40 cache mutations, materializes every mutated slot, synchronizes the owned stream and checks every frontier. Continuation reuses the exact list and layer objects. Invalid input fails preflight; any transaction failure burns every layer rather than publishing or rolling back partially mutated state.

At idle committed boundaries there is one executable cache authority, prompt replay is zero, full-cache repack/reconstruction is zero, and all 40 cache offsets equal the committed frontier.

## Upstream MTP boundary (M25)

Historical M25: the pinned V4.1 MTP loop owns a draft cache, pending anchors and an emission
queue; its target frontier can lead or lag emitted history. Plain upstream row
extraction is not a ds41f idle commit. [M25](milestone-25-mtp-decision.md) records
real-model verification/rollback and the failed extraction prerequisite. MTP
was unsupported at M25. The later explicitly bounded M41 profile is separately qualified; M45 does not change it. Full-history
upstream reconciliation is forbidden; no OFF evidence transfers to MTP.

## Sampling and determinism

The release deterministic policy is backend-local: for a fixed checkpoint/runtime/backend/build/config/input/session state, deterministic greedy behavior is required relative to backend-produced logits. PyTorch or cross-backend RNG/bitstream parity is not claimed.

## Termination

DeepSeek V4.1 EOS token id `1` is configured as a TargetGenerationSession stop token. The EOS token is consumed into cache/all-token history exactly once, hidden from protocol text by the recipe layer, and reported as finish reason `stop`. Length termination reports `length`; cancellation must clean up ownership without creating a second authority.

Arbitrary detokenized request stop strings are not a stateful release feature. Stateful session endpoints reject non-empty/non-null `stop` before mutation; stateless endpoints retain official recipe behavior for their qualified scope.

## Reference native generation

The native `TextGeneration` lifecycle remains retained as reference/qualification evidence and as reusable code where architecturally justified. It is not the selected production decode topology for the release path.
