# ParallelEmbedding official semantics contract

This was the first manually scoped official-reference semantics contract. It now also points to the bounded official-reference-derived fixture and native validation artifact for the existing native BF16 gather primitive. The contract itself remains scoped to embedding semantics only and does not imply broader native model math correctness.

## Authority

Candidate official reference source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

File SHA-256:

```text
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
```

Reviewed source spans from `artifacts/official-reference-review.json`:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `ParallelEmbedding` | 152-178 | `1b8428b4e639bed65674a7e6e3a0387181dc31d110b5c45542d111f54e0b81f4` |
| `ParallelEmbedding.forward` | 168-178 | `fc448dd2ac5f8a483c1ed42264e8ac1e0ba12e3bac3a073c12aae28db406f4bb` |

## Source behavior summary

The official reference defines `ParallelEmbedding` as vocabulary-sharded embedding:

1. Constructor:
   - requires `vocab_size % world_size == 0`;
   - rank-local rows are `[rank * part_vocab_size, (rank + 1) * part_vocab_size)`;
   - each rank owns `weight[part_vocab_size, dim]`.
2. Forward:
   - if `world_size > 1`:
     - compute a mask for token ids outside the rank-local vocabulary range;
     - subtract `vocab_start_idx` from token ids;
     - set masked token ids to `0` before lookup;
   - compute `F.embedding(x, self.weight)`;
   - if `world_size > 1`:
     - zero masked output rows;
     - `dist.all_reduce(y)` to sum the one nonzero rank contribution;
   - return `y`.

## Operation contract

For single-rank fixtures (`world_size == 1`, `rank == 0`):

```text
output = weight[input_ids]
```

where:

- `input_ids` is an integer tensor of arbitrary prompt shape;
- `weight` is the official checkpoint `embed.weight` tensor or an explicitly bounded slice containing all referenced rows;
- output shape is `input_ids.shape + [dim]`;
- output dtype and bits are exactly the selected embedding weight dtype/bits.

For multi-rank fixtures (`world_size > 1`):

```text
part_vocab_size = vocab_size // world_size
vocab_start_idx = rank * part_vocab_size
vocab_end_idx = vocab_start_idx + part_vocab_size
mask = (ids < vocab_start_idx) | (ids >= vocab_end_idx)
local_ids = ids - vocab_start_idx
local_ids[mask] = 0
local_y = weight_rank[local_ids]
local_y[mask] = 0
output = sum(local_y over ranks)
```

The current native embedding fixture validates the single-rank raw BF16 gather subset of this contract. It is clean model-data evidence, not a full distributed embedding semantics test.

## Fixture requirements

A future `official_reference_derived` or `official_checkpoint_raw_bits` embedding fixture must include:

- `classification`: `official_checkpoint_raw_bits` for single-rank raw gather, or `official_reference_derived` for sharded semantics;
- `not_omlx_derived: true`;
- source tensor name, shard, dtype, shape, and digest;
- source span identity for `ParallelEmbedding.forward`;
- `world_size`, `rank`, `part_vocab_size`, and token ids;
- explicit statement whether all-reduce is modeled;
- expected-value provider:
  - direct raw-bit gather for single rank;
  - independently implemented sharded gather/all-reduce for multi-rank;
- non-claims.

## Non-claims

This contract does not validate:

- tokenizer behavior;
- HC repeat/pre-mask behavior;
- attention, Engram, MoE, DSpark/MTP;
- full prefill logits;
- oMLX cache/state correctness;
- distributed runtime performance.

## Generated official-reference-derived fixture

Fixture:

```text
artifacts/parallel-embedding-official-reference-fixture.json
```

Regenerate with a Python environment that has NumPy installed:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_parallel_embedding_fixture.py \
  --out artifacts/parallel-embedding-official-reference-fixture.json
```

This fixture models the reviewed `ParallelEmbedding.forward` rank masking and all-reduce contract for `world_size=8` over official `embed.weight` BF16 bits. It does not import or execute oMLX, PyTorch, MLX, or native Metal code. The independent sharded result is cross-checked against direct raw gather because exactly one rank contributes each token row.

Observed bounded fixture:

```text
classification: official_reference_derived
world_size: 8
part_vocab_size: 16160
tokens: [0, 3, 16159, 16160, 32319, 32320, 129279]
dim: 128
bit_exact: true
source_bounded_weight_bf16_bits_sha256: c1b1bbe26fe819901745c649144ff165036acb00951f492906466b7526a81622
expected_output_bf16_bits_sha256: 9284d0cbd07f860309b3f19b322fafa29f5b8b44a79daa9dc1c9ef3ccd0fe913
```

## Native validation against official-reference fixture

Native validation artifact:

```text
artifacts/native-embedding-official-reference-validation.json
```

Regenerate with:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_embedding_against_official_reference.py \
  --out artifacts/native-embedding-official-reference-validation.json
```

This runs the existing native BF16 embedding gather primitive over the same bounded `embed.weight` rows/dimensions and token ids used by `parallel-embedding-official-reference-fixture.json`, then compares the native output digest to the official-reference-derived expected output. It does not execute oMLX and does not modify native Metal code.

Observed result:

```text
bit_exact: true
mismatch_count: 0
native_output_bf16_bits_sha256: 9284d0cbd07f860309b3f19b322fafa29f5b8b44a79daa9dc1c9ef3ccd0fe913
reference_expected_output_bf16_bits_sha256: 9284d0cbd07f860309b3f19b322fafa29f5b8b44a79daa9dc1c9ef3ccd0fe913
```

## Relationship to existing native artifact

Existing clean native artifact:

```text
artifacts/m2/dwarfstar-prefill/native-official-embedding.json
```

Classification: `CLEAN_OFFICIAL_CHECKPOINT_RAW_BITS`.

It covers `world_size == 1` raw BF16 token gather for bounded rows/dimensions. It remains the first clean native data-plane primitive. The new `native-embedding-official-reference-validation.json` validates the same native primitive against the first official-reference-derived embedding semantics fixture, within the bounded token/dimension scope only.
