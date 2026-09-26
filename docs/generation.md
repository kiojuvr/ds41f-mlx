# Generation runtime

Generation is implemented by the native text runtime and `TextGeneration` lifecycle code.

## Lifecycle

```text
prompt input
  ↓
TextFront / prefill
  ↓
TextEncoder / TextDecoder
  ↓
TextBackbone call
  ↓
logits and main_hidden
  ↓
sampling
  ↓
token commit
  ↓
stop/cancel or continuation
```

Prefill initializes persistent session state.  Incremental decode consumes and extends that state.  Token commit updates token/ngram history and any generation-owned lifecycle state.  Stop/cancel terminates the loop without inventing a second model-session implementation.

## Sampling

The imported generation loop supplies lifecycle and integration behavior.  Current ds41f validators define sampling arithmetic seams such as supplied-noise and temperature-zero behavior.

Native stochastic RNG is a runtime provider seam.  The repository does not claim PyTorch RNG bitstream parity.  If the RNG provider is adapted, the generation loop should remain intact and only the provider seam should change.

## main_hidden

`main_hidden` is a call-local generation handoff associated with the current backbone/logits call.  It is not persistent session state beyond any explicit token-commit bookkeeping performed by the generation state.

## Continuation

Continuation reuses the existing `TextBackboneState` and generation state.  Reset clears them; fork copies committed persistent state.  Invalid inputs must not partially commit tokens or mutate session state.

## Qualification status

Checkpoint-free generation-loop tests pass.  Full MLX-backed generation over the official checkpoint is not yet validated in the current import environment.
