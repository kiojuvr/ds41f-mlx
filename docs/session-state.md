# Session state contract

M25 preserves the OFF-only idle artifact: save/restore reject models with
preserved or active MTP before I/O. The pinned upstream MTP emission queue and
target cache do not provide the required arbitrary committed frontier through
ordinary extraction; see [M25](milestone-25-mtp-decision.md). No production MTP
session or restore qualification is implied by the contracts below.

The separate internal guarded singleton contract is operationally qualified in
[M34](milestone-34-operational-qualification.md), using M33's target/DSpark/
canonical/delivery frontiers and bounded quiescence. Active cancellation transfers
native singleton cache row views before owner release, preserving subsequent P6
admission and old-owner revocation. MTP save/restore still fails closed before I/O;
this is not qualification of persisted or concurrent MTP sessions.
[M35](milestone-35-http-sse-qualification.md) qualifies bounded internal HTTP/SSE
recovery on that authority: active canonical history remains owned by generation,
recorded session history commits at quiescence, and socket writes never acknowledge
model-token ordinals. Client-observed recipe events remain a canonical prefix;
GET recovery followed by ordinary exact-prefix conversion consumes retained caches.
Protected failures require termination rather than ambiguous idle recovery.
[M36](milestone-36-client-recovery-qualification.md) establishes that native idle
coherence alone does not certify protocol reconstructability: interrupted canonical
DSML can produce an unfinished tool call whose ordinary encoding is non-prefix.
The internal backend now poisons such settlement and reports busy until retirement
releases the response lease. Its diagnostic partial tool response is not execution
permission. Full client recovery/admission qualification remains blocked.

The current native session owner is `TextBackboneState` together with `TextEncoder`, `TextDecoder`, and `TextGeneration` state machinery. It is the reference state contract for future production architecture unless a documented architecture decision replaces or wraps it. Validation inventory helpers are not independent production session implementations.

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
| DSpark/MTP priming context | speculative decode session state, not base target state | oMLX MTP-enabled prefill can create `_omlx_mtp_prime_ctx` containing `DSparkContextCache`, target-layer hidden history, and `expected_target_offset`; absent from current `PrefillContinuationState` and not required while MTP is OFF |

## Reset, fork, and continuation

- **reset** clears persistent session state and returns the runtime to a prompt-free state.
- **fork** duplicates persistent session state so the child can continue without mutating the parent.
- **continuation** appends new prompt/decode work to existing state while preserving already-published KV/index/candidate data.
- invalid input must not partially mutate session state.

## Prefill to first decode

Prefill populates token/ngram history, window KV, source-layer compressed KV, indexer K, candidate/top-k publications, final logits, and shared publications according to the layer topology.  The first decode step consumes those states rather than recomputing a fresh prompt-only session.  Engram insertion points consume current token/ngram context through the backbone and preserve SSD-backed lookup semantics.

The DwarfStar-derived prefill path commits an architecture-neutral live state represented by `ds41f_mlx.prefill_session.PrefillContinuationState`, with artifact/evidence views represented by `PrefillSessionHandoff`.  The live state contains actual runtime arrays/handles for token history, Engram hash history, per-layer window KV, source compressed KV, source index K, candidate state, top-k generations, shared publications, ownership, and empty/non-empty compressor-pending state.  The artifact handoff records digests, provenance, inventory, and ownership without pretending those digests are executable state.  HC residual/pre-mix digests remain final-output qualification evidence unless a future official continuation contract requires them across token boundaries.  oMLX target decode is selected for production, and `OMLXDecodeSession` / `OMLXDecodeStateAdapter` form the decode admission/session seam. After successful admission, `OMLXDecodeSession.cache` (`DeepseekV41Cache[40]`) is the active decode state authority; `PrefillContinuationState` is immutable input/evidence, not a second synchronized live authority. `OMLXDecodeSession` keeps CPU token history in append-only chunks for rollback/fork/reset bookkeeping; full concatenation is an explicit observation/export operation, not per-token hot-path work. The M21 Rust boundary is client/process control over local HTTP only and never becomes a session-state or cache authority.

M4 native-kernel-backed controls showed that ordinary oMLX `BatchGenerator` MTP-OFF decode can be much faster than direct one-token `_forward`, even from the same model core. P5 additionally proves the existing mlx-lm admission seam can consume an already-populated cache without prompt replay:

```python
BatchGenerator.insert(prompts=[[15]], caches=[admitted_cache], all_tokens=[[0, 3]])
```

At that seam the cache authority before decode is state after `[0,3]`; token `15` is the first GenerationBatch backbone input and is appended to token history by the GenerationBatch bootstrap; subsequent responses represent the distribution after `15`.  The current artifact records zero prompt-processing forwards from `[0,3]`.  This fast path is not yet the production authority because the DwarfStar-prefill state is not correctness-qualified against ordinary oMLX.

That fast scheduler path does not make DSpark priming part of base target correctness, but future practical speculative decode must either export or recreate the oMLX prefill-created priming contract: target-layer hidden history for `dspark_target_layer_ids`, per-MTP-layer `DSparkContextCache`, and `expected_target_offset` alignment consumed by `take_primed()`.

## Ratio-2 and cross-call behavior

Compressed producer layers with compression ratio greater than one can hold pending KV/score data across calls until enough tokens exist to publish a compressed entry.  Consumers must observe only committed publications.  Producer state publication ordering is part of the session contract; consumers do not read speculative partial state.

## Publication lifecycle

1. Producer layer updates its local persistent state.
2. At a valid boundary it publishes compressed/global/index data.
3. Consumer layers read the publication for candidate selection and reused attention.
4. Reset clears publications; fork copies the currently committed publications.

## RNG lifecycle

Generation owns RNG provider state through the sampler seam.  Deterministic tests may supply explicit noise to validate arithmetic.  Native stochastic output is not claimed to match PyTorch bitstreams unless a future qualification explicitly states that.
