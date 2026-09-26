# Candidate block selection semantics contract

Boundary 5c validates candidate block selection and candidate-source publication only.

It does not validate layer >20 consumer masking, compressed-KV sparse attention integration, decode, Block/HC, or performance.

## Local config authority

From the pinned local official `config.json`:

```text
candidate_source_layer = 20
candidate_topk_blocks = 2048
candidate_block_size = 8
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
compress_ratio at layer 20 = 1
```

Layer 20 is therefore selected from config, not hard-coded by assumption.

## Reviewed official source

Official source file:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

File SHA-256:

```text
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
```

Reviewed spans:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `select_candidate_blocks` | 583-610 | `99d70a4f2befc8302203aa247495b12aae2d04d0b2125999554d9f222e219d3d` |
| `Indexer` | 488-580 | `cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75` |
| `Indexer.forward` | 527-580 | `32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d` |

Reviewed branch:

```python
if self.is_candidate_source:
    shared_attn.candidates = select_candidate_blocks(
        index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
    )
```

## Operation contract

For `select_candidate_blocks(logits, compress_lens, topk_blocks, block_size)`:

1. `logits` already contains `-inf` for unreachable compressed positions.
2. The final partial block is padded with `-inf` out to `block_size`.
3. Block score is the max score within each block.
4. The newest reachable block, `(compress_lens - 1) // block_size`, is pinned by setting its block score to `+inf` before top-k.
5. Top-k selects blocks by block score.
6. Top-k choices whose block score is still `-inf` are dropped.
7. Kept blocks are expanded with `repeat_interleave(block_size)` and truncated to the original width.
8. Candidate-source publication writes the resulting bool mask to `shared_attn.candidates` exactly.

## Artifacts

Fixture:

```text
artifacts/candidate-block-official-reference-fixture.json
```

Native validation:

```text
artifacts/native-candidate-block-official-reference-validation.json
```

The fixture contains two cases:

### A. Non-trivial function-level semantic fixture

Bounded parameters:

```text
block_size = 8
topk_blocks = 2
width = 25
```

This case exercises:

- block max scoring;
- final partial block padding with `-inf`;
- newest reachable partial block pinning;
- top-k block selection;
- dropping `-inf` blocks;
- expansion to a bool position mask;
- causal/unreachable positions.

### B. Actual layer-20 wiring fixture

Uses local production config:

```text
candidate_source_layer = 20
topk_blocks = 2048
block_size = 8
```

With the short bounded width, production `topk_blocks` retains all reachable blocks. This is explicitly a wiring/publication validation, not a production pruning claim.

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_candidate_block_fixture.py \
  --out artifacts/candidate-block-official-reference-fixture.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_candidate_block_against_official_reference.py \
  --out artifacts/native-candidate-block-official-reference-validation.json
```

Observed result:

- non-trivial semantic candidate mask: bool exact;
- partial-block pinning: pass;
- `-inf`/unreachable semantics: exact;
- layer-20 candidate mask: bool exact;
- `shared_attn.candidates` publication: bool exact.

## Non-claims

This contract does not validate:

- layer >20 candidate consumer masking;
- compressed-KV sparse attention integration;
- decode path;
- Block / HC;
- logits, full layer, full model correctness;
- production pruning behavior in the short layer-20 wiring case;
- performance.
