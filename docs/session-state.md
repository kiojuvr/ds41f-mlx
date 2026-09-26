# Session state contract

The production session owner is `TextBackboneState` together with `TextEncoder`, `TextDecoder`, and `TextGeneration` state machinery.  Validation inventory helpers are not independent production session implementations.

## State classification

| State | Classification | Owner / lifetime |
| --- | --- | --- |
| Token/ngram history | persistent model/session state | text backbone/generation session; reset clears, fork copies |
| Per-layer window KV | persistent model/session state | SWA layer state; survives decode continuation |
| Source-layer compressed KV | persistent model/session state | compressed producer layers; published for reuse consumers |
| Ratio > 1 compressor pending KV/score | persistent model/session state | compressor; may span calls until publication boundary |
| Indexer K | persistent model/session state | index source layers; reused by candidate/top-k flow |
| SharedAttention publications | persistent model/session state | producer/consumer publication registry; republished when source advances |
| Candidate/top-k buffers | runtime-owned state with persistent lifecycle hooks | candidate source/consumer machinery; invalidated on reset/fork as required |
| HC pre/post `pre_mix` | call-local handoff | valid only during the subblock call that produced it |
| `main_hidden` | call-local handoff | generation step output used for commit/logit-associated bookkeeping |
| Runtime RNG | runtime-owned state | generation/sampler session; deterministic supplied-noise tests qualify arithmetic, not PyTorch bitstream parity |
| Checkpoint weights, Engram rows, expert backing | static model data / storage-backed data | read-only catalog/store/backing; not session state |

## Reset, fork, and continuation

- **reset** clears persistent session state and returns the runtime to a prompt-free state.
- **fork** duplicates persistent session state so the child can continue without mutating the parent.
- **continuation** appends new prompt/decode work to existing state while preserving already-published KV/index/candidate data.
- invalid input must not partially mutate session state.

## Prefill to first decode

Prefill populates token/ngram history, window KV, source-layer compressed KV, indexer K, and shared publications according to the layer topology.  The first decode step consumes those states rather than recomputing a fresh prompt-only session.  Engram insertion points consume current token/ngram context through the backbone and preserve SSD-backed lookup semantics.

## Ratio-2 and cross-call behavior

Compressed producer layers with compression ratio greater than one can hold pending KV/score data across calls until enough tokens exist to publish a compressed entry.  Consumers must observe only committed publications.  Producer state publication ordering is part of the session contract; consumers do not read speculative partial state.

## Publication lifecycle

1. Producer layer updates its local persistent state.
2. At a valid boundary it publishes compressed/global/index data.
3. Consumer layers read the publication for candidate selection and reused attention.
4. Reset clears publications; fork copies the currently committed publications.

## RNG lifecycle

Generation owns RNG provider state through the sampler seam.  Deterministic tests may supply explicit noise to validate arithmetic.  Native stochastic output is not claimed to match PyTorch bitstreams unless a future qualification explicitly states that.
