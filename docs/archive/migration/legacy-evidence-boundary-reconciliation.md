# Legacy evidence reconciliation: deepseek-v41-flash-mlx vs ds41f Boundary authority

Status: audit artifact only.  No model/runtime code was changed, no model math was added, no numerical Boundary was run, and Boundary13f was not implemented.

## Repositories / SHAs

- Current repo: `kiojuvr/ds41f-mlx` at `559201eb8fefb839b3c8a734ecd59eb4e144a150`.
- Legacy repo: `kiojuvr/deepseek-v41-flash-mlx` current HEAD `1b7d0a2c7d33602437dffd44e26a33f39f189661`.
- Legacy evidence commits observed in artifacts/docs include `a961c22edaeabcb4ef9bc9e23acf5b26aa5bffd7`, `1b266485f9a11cd80d23210cc40389bd1b4fe24a`, `40ddbb6`, and fixed/reject lineage commits named in `docs/attention-reject-reaudit-40ddbb6.md`.

Machine-readable table: `artifacts/legacy-evidence-boundary-reconciliation.json`.

## Authority classes used

- **A** official model-data authority: checkpoint/config/tokenizer/raw-bit fixtures.
- **B** official semantic authority: ds41f-reviewed official source spans/contracts.
- **C** legacy implementation-equivalence evidence: local reference/optimized, chunk/tokenwise, continuation/state exactness.
- **D** engineering/lifecycle evidence: transaction, reset/fork, invalid atomicity, publication, capacity/resource behavior.
- **E** compatibility/performance only: oMLX/DwarfStar agreement, throughput, memory, donor architecture.
- **F** tainted/superseded for correctness: oMLX/DwarfStar output/tensor used as official oracle, broken provenance, or known mismatch if elevated to correctness.

Important principle applied: legacy evidence is classified claim-by-claim, not rejected because it is legacy.

## ds41f Boundary correspondence summary

Current ds41f authority already covers primitive official-source/data anchors, window/rotary, compressed KV, Indexer, candidate, sparse attention, output projection, HC, MoE, connected prefill, Engram, main_hidden/logits/sampling/return contracts, plus:

- Boundary13: decode/incremental-state official source audit only.
- Boundary13a: prefill-end persistent-state snapshot.
- Boundary13b: first incremental NgramHashState update.
- Boundary13c: first incremental position/rotary/layer0 window KV through sparse prelude.
- Boundary13d: first incremental Block0 completion and Engram@1 handoff.
- Boundary13e: first incremental Block1 completion, layer2 entry, ratio2 partial slot0 write and top-k reconstruction; stops before layer2 sparse attention.

## Legacy evidence audit by area

### SWA / decode window state

Evidence: `docs/swa-reference.md`, `artifacts/swa/*`, tests.

Classification:

- Window ring, position progression, 128→129 wrap, chunk/tokenwise equality, continuation, fork/reset, invalid position/state atomicity: **C/D reusable**.
- Q/KV projection and Attention integration with real weights: **C**, not official CUDA oracle.
- Raw checkpoint/source identity portions: **A partial**.

Reuse decision: reusable without rerun for engineering/state-transition claims and as composition support for ds41f Boundary13c.  It must not be promoted to official sparse-attn CUDA bit authority.

### Compressed attention / Indexer

Evidence: `docs/compressed-attention.md`, `artifacts/global-kv/*`.

Key interpretation:

- `3-token -> 4-token continuation` demonstrates ratio2 cross-call completion: previous-call slot0 pending KV/score is read at the next decode position, current slot1 is added, the group completes, and a latent can be produced.
- `5-token -> 6-token publication` demonstrates completed-group publication timing: new compressed latent produces packed main KV/index K rows, pending state clears/advances, and consumers can observe the published row only after completion.

Classification:

- Pending state, group completion, publication timing, packed cache lifecycle, one-token Indexer query/top-k offset, producer/consumer position checks: **C/D reusable**.
- Numeric agreement with official CUDA/source by itself: not claimed.
- Performance/layout choices: **E** when cited for optimization.

Composition: ds41f Boundary13e proves first decode pos2 partial slot0/no-publication under official source.  Legacy proves the cross-call ratio2 completion/publication state machine.  Together this is enough for lifecycle correctness; a new numerical boundary is justified only if ds41f needs exact token15 position3 digest.

### Hyper-Connections / MoE / Block

Evidence: `docs/mhc.md`, `docs/moe-block.md`, CPU/reference artifacts, `artifacts/block/reviewed-result.json`.

Classification:

- Local chunk/tokenwise Block, reset, position rejection, selected real experts: **C/D**.
- Official semantic formulas where independently source reviewed in ds41f: **B only in ds41f**, not legacy alone.
- Known cross-backend differences: mHC FP32 coefficient mismatch and tie/reduction caveats remain **not correctness authority**.

Reuse: keep as local equivalence/regression context.  Do not use legacy mHC/CPU mismatch or oMLX agreement to upgrade official correctness.

### Full backbone / decoder

Evidence: `docs/text-decoder-validation.md`, `artifacts/text-backbone/reviewed-result*.json`, reviewed prefill-129 artifacts.

Recorded comparisons are native schedule comparisons: chunk vs tokenwise, reset/fork/continuation, public state, hidden/pre_mix/logits, route tie records.  They are not official minimal-inference logits oracle comparisons.

Classification: **C/D**, with oMLX-related route diagnostics **E/F if elevated to correctness**.

Reuse: strong connected-state evidence.  With ds41f official primitive/source anchors it can avoid reauditing whole connected state machinery, but it does not replace ds41f token-specific decode numeric authority after Boundary13e.

### Production attention / connected optimized path

Evidence: `docs/attention-reject-reaudit-40ddbb6.md`, production gate artifacts.

Claims traced:

- `40ddbb6` fixed-tile lineage has local reference/optimized bit exactness over layers0-39, final persistent state, 2x128 hidden/pre_mix/logits, state/publication/hash, continuation, invalid-token atomicity.
- Older rejected candidates remain valid rejects when failures are arithmetic/reduction topology and unrelated to `40ddbb6` request-boundary fix.
- 2K/paired runs are performance/compatibility observations, not official correctness or 32K/256K qualification.

Classification: **C/D** for optimized-vs-reference parity, **E** for performance, not B.

### Engram

Evidence: `docs/engram-forward.md`, `artifacts/engram/*`, connected backbone evidence.

Classification:

- Ngram history/cache lifecycle: reusable **C/D**, and ds41f Boundary13b supplies current B-backed numeric update.
- Engram projection/gate/residual local evidence: **C**, but old artifact has explicit CPU oracle mismatch and oMLX agreement; cannot be official correctness by itself.
- Connected state evidence: **C/D** when not used as oracle.

Layer14 first-decode connected numeric value remains not proven by legacy alone.

### Generation / sampling

Evidence: `docs/sampling-reference.md`, `artifacts/text-generate/*`, generation lifecycle artifacts.

Classification:

- Greedy, supplied exponential formula, lifecycle, callback/cancel/fresh request, next_position: **C/D reusable**.
- Legacy splitmix64 native RNG: **not official RNG**; compatibility/local reproducibility only.
- API parity/performance: **E/D** depending on claim.

## Boundary13e+ planned work assessment

### Boundary13f: layer2 sparse_attn + Block2 completion

Do **not** proceed as a broad sequential reimplementation task.  There is already strong legacy C/D evidence for layer2 compressed attention lifecycle and ds41f B evidence for official source/primitive semantics.  What is still missing is narrower:

- exact ds41f token15 position3 compressed latent/index/main publication digest, if required; and/or
- exact ds41f first-decode layer2 sparse/output/Block2 digest, if composition is not accepted for that claim.

If no token-specific digest is required, Boundary13f should be cancelled.

### Boundary13g: layers3-7 shared-state consumers

Legacy consumer evidence plus ds41f source spans are enough for publication/position/lifetime claims.  A broad numerical Boundary13g should be cancelled unless exact token15 connected consumer digests are required.

### Future layer8/layer14/layer20/layer24+ and logits/sampling

- Layer8 producer and layer14 producer+Engram: source/provenance review first; numeric rerun only if exact decode digest is needed.
- Layer20 candidate source: source/provenance review first.  Legacy full decoder proves local candidate/state machinery, but first-decode current-call candidate regeneration exact digest remains unproven.
- Full decode logits/main_hidden/return: genuinely unproven after Boundary13e if exact first-decode return authority is required.
- Sampling: greedy/formula lifecycle can be reused; official stochastic RNG remains unproven.

## Proof composition examples

1. ds41f official sparse_attn/Compressor/Indexer semantics + Boundary13e pos2 partial state + legacy 3→4 and 5→6 continuation/publication ⇒ ratio2 lifecycle is proven by composition; no broad rerun required.
2. ds41f official SharedAttention source + legacy layer3-7 consumer acceptance/rejection/immutability checks ⇒ consumer lifecycle is proven by composition; no Boundary13g broad lifecycle rerun required.
3. ds41f primitive official semantic proofs + legacy full-backbone chunk/tokenwise/state exactness ⇒ connected native lifecycle is reusable; official token-specific decode logits still require separate authority if demanded.

## Final classification

### PROVEN — no further work

1. Primitive official model-data/source anchors already closed in ds41f.
2. Boundary13 source/lifetime audit.
3. Boundary13b Ngram first incremental update.
4. Boundary13c layer0 window/rotary prelude.
5. Boundary13d Block0 + Engram@1.
6. Boundary13e Block1 -> layer2 entry, pos2 partial slot0/no-publication.

### PROVEN_BY_COMPOSITION — no rerun required

1. SWA ring wrap/fork/reset/invalid-input lifecycle.
2. Ratio2 compressor cross-call completion/publication state machine.
3. SharedAttention publication and layers3-7 consumer position checks.
4. Connected native lifecycle: chunk/tokenwise, continuation, fork/reset, invalid-input atomicity.
5. Generation lifecycle and request-local native RNG reproducibility, with RNG non-authority retained.

### NEEDS_SOURCE/PROVENANCE_REVIEW_ONLY — no numerical rerun yet

1. Code-lineage mapping if legacy optimized attention performance/parity is cited in ds41f.
2. Layer20 decode candidate regeneration source/dataflow.
3. Engram@14 decode source/gate arithmetic provenance.
4. Transformer return/main_hidden packaging for first decode.

### GENUINELY_UNPROVEN — numerical work justified only for these holes

1. Token15-specific position3 ratio2 compressed latent/key/cache digest, if exact ds41f numerical digest is required.
2. First-decode layer2 sparse_attn output and Block2 digest, if not accepted by composition.
3. First-decode layer20 candidate regeneration/current-call top-k/candidate publication exact digest.
4. First-decode full logits/main_hidden/return after Boundary13e.
5. Official stochastic RNG sequence; old native splitmix64 is not official.

## Recommended action

Recommended single next task: source/provenance-only review of layer2 position3 and layer20 decode candidate regeneration to decide whether a minimal numerical boundary is actually required.  Do not implement Boundary13f yet.

## Proof-obligation necessity review

Status: source/artifact/docs/code inspection only.  No Boundary13f implementation, new model math, numerical runner, checkpoint-heavy execution, benchmark, or long-context run was performed.

Previous `GENUINELY_UNPROVEN` count: **5**.

This review separates two facts that were intentionally conflated in the previous audit:

1. a concrete numerical digest is not currently recorded; and
2. ds41f correctness/release claims actually require that concrete digest.

A missing digest is not by itself a proof obligation.

### 1. token15-specific pos3 ratio2 compressed latent/key/cache digest

Final classification: **PROVEN_BY_COMPOSITION**.

Proof composition:

```text
official Compressor transition semantics
+ Boundary13e actual pos2 partial state
+ legacy cross-call ratio2 state-machine evidence
+ official compressed-KV / Indexer publication semantics
```

This closes the semantic/state claim:

```text
pos3 reads prior partial slot0
pos3 contributes slot1
group completes
new latent appears
compress_kv publication occurs
index key publication occurs
```

Requirement analysis:

- `requirement_source`: official model semantics and session-state contract require the transition/publication behavior.
- `observable_claim`: pos3 completes the ratio2 group and publishes completed compressed/index rows at the source-defined time.
- `failure_if_missing`: without the composition, cross-call completion could be ambiguous; without the token15 digest specifically, no release claim fails unless that digest is explicitly chosen as a regression fixture.
- `numerical_digest_missing`: yes.
- `numerical_digest_required`: no.

The token15-specific latent/key/cache digest would add only a concrete regression value.

### 2. first-decode layer2 sparse_attn + Block2 exact digest

Final classification: **OPTIONAL_REGRESSION_FIXTURE**.

Mapped ds41f evidence:

```text
window KV
compressed KV
Indexer/top-k
window+compressed assembly
sparse_attn
inverse rotary
Attention output projection
HC
MoE
Block semantics
Boundary13e actual layer2 input/Q/KV/window/compressed state/current topk
```

Mapped legacy evidence:

```text
layer2 CompressedLayerReference
window + global attention
chunk/tokenwise output/state exact
continuation
Block/text connected path
full-backbone tokenwise execution
```

No unconnected operation boundary was found.  A new exact Block2 digest would establish only:

```text
for this exact token15 fixture, here is the final digest
```

Requirement analysis:

- `requirement_source`: regression convenience only for the exact token15 Block2 digest.
- `observable_claim`: layer2 sparse attention and Block2 follow already-reviewed official operation contracts from the actual Boundary13e state.
- `failure_if_missing`: no correctness/release claim fails solely because this digest is absent.
- `numerical_digest_missing`: yes.
- `numerical_digest_required`: no.

Broad Boundary13f must not be revived for this.

### 3. layer20 decode candidate regeneration / current-call topk/candidate publication digest

Final classification: **SOURCE_PROVENANCE_ONLY**.

Mapped ds41f evidence:

```text
official candidate-block semantics
official candidate-consumer semantics
official Indexer semantics
official compressed KV semantics
Boundary13 decode state lifetime audit
```

Mapped legacy evidence:

```text
legacy TextDecoderReference
legacy full-backbone tokenwise path
layer20 candidate_source behavior
ratio1 compressed KV publication
index key publication
candidate mask generation
topk publication
layers21+ candidate consumption
```

The required property is regeneration ordering, not a digest.  Official source already says prefill candidate/topk values are call-local and not reused; layer20 regenerates candidates/topk in each forward call from current compressed/index state.  Legacy tokenwise decoder/full-backbone evidence supports the implementation path executing the layer20 source and consumers every token with exact state continuation.

Requirement analysis:

- `requirement_source`: official model semantics and session-state contract for current-call regeneration; source/provenance mapping for legacy-to-ds41f correspondence.
- `observable_claim`: decode call regenerates layer20 compressed KV, index K, candidate mask/topk, and layers21+ consume current-call values rather than prefill tensors.
- `failure_if_missing`: candidate reuse across calls would remain a source/dataflow concern; missing numeric digest alone does not fail correctness.
- `numerical_digest_missing`: yes.
- `numerical_digest_required`: no.

Remaining work is source/provenance mapping of layer20 execution order and release-claim scoping.  If a future gap remains, shrink it to candidate-mask publication identity only, not “layer20 decode”.

### 4. full first-decode logits / main_hidden / Transformer return after Boundary13e

Final classification: **SOURCE_PROVENANCE_ONLY**.

Mapped ds41f evidence:

```text
Boundary12b4 Engram-connected deterministic logits
Boundary12c main_hidden capture/mean/concat
Boundary12d integrated deterministic Transformer.forward return packaging
Boundary12e target-MLX stochastic integrated return
Boundary13 decode source audit: h/pre_mix/main_hiddens are call-local; S=1 main_hidden shape [B,1,15360]
Boundary13b-e first-decode connected prefix through layer2 entry
```

Mapped legacy evidence:

```text
tokenwise full 40-layer execution
logits
state continuation
generation
```

A source/artifact search did not find legacy `main_hidden` / `main_hiddens` evidence.  Therefore legacy should not be cited as main_hidden authority.  The main_hidden claim must come from ds41f source and Boundary12c/12d-style authority.

Property remaining after composition:

- Not “full decode logits”.  Legacy tokenwise generation plus ds41f logits/head contracts already support logits/generation continuity.
- Not Blocks3-39 as a broad rerun.
- At most: decode-specific final return/main_hidden exposure scoping, if ds41f release/API claims expose core `Transformer.forward` return for decode.

Requirement analysis:

- `requirement_source`: official model semantics if ds41f claims `Transformer.forward` return; current M0 runtime API does not expose `main_hidden`.
- `observable_claim`: first-decode return tuple ordering and call-local main_hidden capture/concat follow official source.
- `failure_if_missing`: if ds41f releases a core `Transformer.forward` API with decode `main_hidden`, missing decode-specific return review could underspecify that observable.  Current chat/generation API correctness does not fail solely from absent main_hidden digest.
- `numerical_digest_missing`: yes.
- `numerical_digest_required`: no, unless release scope explicitly requires it.

No full first-decode numerical Boundary is justified.  Decide release/API exposure first; if required, make a minimal decode return assembly review/fixture only.

### 5. official stochastic RNG sequence

Final classification: **NON_GOAL**.

Boundary13 already classifies target MLX RNG as target-runtime-owned session state, not official-model persistent state.  ds41f has separated:

```text
official sampling arithmetic with supplied noise
temperature-zero sampling
target MLX stochastic sampling
MLX RNG session-key transition
integrated stochastic return
```

Requirement analysis:

- `requirement_source`: no current ds41f correctness/release requirement; optional compatibility only if a future release explicitly promises PyTorch RNG bitstream parity.
- `observable_claim`: target runtime produces reproducible stochastic sampling under the documented MLX key/noise policy and source-defined arithmetic.
- `failure_if_missing`: no ds41f correctness claim fails.
- `numerical_digest_missing`: yes for official torch RNG sequence.
- `numerical_digest_required`: no.

Do not create an official PyTorch RNG comparison boundary.

## Necessity review result

### REQUIRED_NUMERICAL_PROOF

None.

### PROVEN_BY_COMPOSITION

1. token15-specific pos3 ratio2 completion semantics.  The digest itself is optional.

### SOURCE_PROVENANCE_ONLY

1. layer20 decode candidate regeneration/current-call topk/candidate publication.
2. full first-decode logits/main_hidden/Transformer return requirement scoping.

### OPTIONAL_REGRESSION_FIXTURE

1. first-decode layer2 sparse_attn + Block2 exact digest.

### NON_GOAL

1. official stochastic RNG sequence / PyTorch bitstream parity.

## Cancelled numerical boundaries

- broad Boundary13f
- broad Boundary13g
- layer20 decode-all boundary
- full first-decode 40-layer logits/return boundary
- official PyTorch RNG bitstream boundary

## Updated recommendation

There are currently **0 REQUIRED_NUMERICAL_PROOF** items.  Do not create a new numerical Boundary now.

Next task should be non-numerical: source/provenance-only requirement scoping for layer20 candidate regeneration and decode return/main_hidden exposure.  If that confirms no release-observable gap, switch to an implementation plan for reusing/porting the legacy proven implementation evidence into the ds41f runtime.

## Phase 1 local import resolution

The reviewed legacy evidence files needed by this reconciliation have been copied into this repository and pinned in `artifacts/provenance/legacy-import.json`.

Local evidence roots:

```text
artifacts/legacy-import/evidence/swa/layer-check.json
artifacts/legacy-import/evidence/global-kv/attention-check.json
artifacts/legacy-import/evidence/global-kv/consumer-check.json
artifacts/legacy-import/evidence/global-kv/consumer-range-check.json
artifacts/legacy-import/evidence/block/reviewed-result.json
artifacts/legacy-import/evidence/text-backbone/prefill-129-route-ties.json
artifacts/legacy-import/evidence/attention/fixed-tile-backbone-20260919-203130-test.log
artifacts/legacy-import/evidence/generation-lifecycle/reviewed-temperature-result.json
```

These files are byte-copied evidence artifacts.  Any old absolute paths inside them are historical provenance only and are not live dependencies.  The old repository is no longer required to audit the imported evidence claims in this reconciliation; runtime implementation and tests are still future Phase 2 work.
