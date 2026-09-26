# Boundary 5d: candidate consumer masking + top-k publication

Status: PASS. This boundary stops before Attention KV concatenation and `sparse_attn`.

## Reviewed local pinned official source/config

Checkpoint/config: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.

Official `inference/model.py` spans recorded in `artifacts/candidate-consumer-official-reference-fixture.json`:

| Span | Lines | SHA-256 | Reviewed semantic |
|---|---:|---|---|
| `Indexer` | 488-580 | `cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75` | full Indexer scope |
| `Indexer.__init__` | 496-525 | `a043fb3fb3a0dfd386b732f18483b7e321af313e58173e3cf591a074ccbd36a5` | `self.uses_candidates = 0 <= args.candidate_source_layer < layer_id` |
| `Indexer.forward` | 527-580 | `32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d` | consumer branch masks with `~shared_attn.candidates`, then top-k/sort/`-1`/offset int32 |

Config determines `candidate_source_layer_id = 20` and `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`; therefore the first index-source consumer layer after the candidate source is layer 24.

## Contract

1. Start from consumer `index_score` after ordinary causal compressed-position masking.
2. If `uses_candidates`, apply `index_score.masked_fill(~shared_attn.candidates, -inf)`.
3. Candidate `False` positions become exactly `-inf`.
4. Candidate `True` reachable scores are preserved.
5. Existing causal `-inf` remains `-inf` after composition.
6. Run top-k over the masked scores.
7. Sort selected indices into position order.
8. Return `torch.where(idxs < compress_lens, idxs + offset, -1).int()`.
9. Publish the same int32 tensor as shared `topk_idxs`.

## Artifacts

- Fixture: `artifacts/candidate-consumer-official-reference-fixture.json`
- Validation: `artifacts/native-candidate-consumer-official-reference-validation.json`

The semantic fixture uses Boundary 5c's non-trivial candidate mask and validates masking changes top-k results, exact masked scores, causal composition, position-order sort, unreachable `-1`, offset, and int32 exactness.

The actual wiring case uses real checkpoint tensors for layer-24 consumer scoring and layer-20 index-key provenance. The short production candidate mask retains all reachable positions, so it is a wiring/publication validation only; no production pruning effect is claimed.

## Non-claims

No window/compressed KV concatenation, `sparse_attn`, Attention output projection, decode path, Block/HC, performance optimization, logits, full layer, or full model correctness is validated here.
