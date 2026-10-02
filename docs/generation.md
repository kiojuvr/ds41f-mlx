# Generation runtime

## Qualified release path

The production text release uses dense P0-P7 prefill plus P5 zero-replay handoff into oMLX `GenerationBatch` with MTP, DSpark, and speculative decode OFF.

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
oMLX GenerationBatch decode
  ↓
commit generated tokens/KV
  ↓
DeepSeek EOS, length, cancel, or continuation
```

## Commit and continuation invariants

Prefill initializes persistent cache/token state. Incremental decode consumes and extends that state. Token commit updates all-token history and GenerationBatch-owned cache state. Continuation reuses the committed state; invalid inputs must fail before partial commit or cache mutation.

At idle committed boundaries there is one executable cache authority, prompt replay is zero, full-cache repack/reconstruction is zero, and all 40 cache offsets equal the committed frontier.

## Sampling and determinism

The release deterministic policy is backend-local: for a fixed checkpoint/runtime/backend/build/config/input/session state, deterministic greedy behavior is required relative to backend-produced logits. PyTorch or cross-backend RNG/bitstream parity is not claimed.

## Termination

DeepSeek V4.1 EOS token id `1` is configured as a GenerationBatch stop token. The EOS token is consumed into cache/all-token history exactly once, hidden from protocol text by the recipe layer, and reported as finish reason `stop`. Length termination reports `length`; cancellation must clean up ownership without creating a second authority.

Arbitrary detokenized request stop strings are not a stateful release feature. Stateful session endpoints reject non-empty/non-null `stop` before mutation; stateless endpoints retain official recipe behavior for their qualified scope.

## Reference native generation

The native `TextGeneration` lifecycle remains retained as reference/qualification evidence and as reusable code where architecturally justified. It is not the selected production decode topology for the release path.
