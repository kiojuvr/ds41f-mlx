# Official reference review metadata

This records function-level source identity for the candidate official DeepSeek-V4.1-Flash reference implementation. It is a review artifact only: no model math is executed and no numerical correctness is claimed.

## Generated artifact

```text
artifacts/official-reference-review.json
```

Regenerate with:

```sh
python3 tools/record_official_reference_review.py \
  --out artifacts/official-reference-review.json
```

## Scope

Candidate source snapshot:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
```

Reviewed files:

| File | SHA-256 |
| --- | --- |
| `inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |

Initial source-span targets include:

- `ParallelEmbedding` / `ParallelEmbedding.forward`
- `linear` / `Linear.forward`
- `RMSNorm.forward`
- `precompute_freqs_cis` / `apply_rotary_emb`
- `Compressor`, `Indexer`, `Attention`, `Gate`, `Expert`, `MoE`, `Block`, `ParallelHead`, `Transformer`
- quantized/sparse kernels in `inference/kernel.py`

## Review state

Current status: `source_identity_recorded_manual_semantics_review_required`.

This means the source spans are pinned and ready for manual semantics review, but they are not yet executable oracles. Before generating an `official_reference_derived` fixture from any target, record:

1. exact source span and hash;
2. operation contract;
3. dtype and rounding behavior;
4. input tensor source/digest;
5. expected-value provider;
6. non-claims.

## Completed bounded review targets

The following low-risk targets now have bounded official-reference-derived or independent-arithmetic fixtures and native validation artifacts:

1. `ParallelEmbedding.forward` for tensor-parallel masking/all-reduce behavior;
2. non-quantized `linear` / `Linear.forward` path;
3. `RMSNorm.forward` F32 variance/rsqrt/output-cast behavior;
4. rotary frequency/application helpers, including bounded no-YaRN and YaRN fixtures.

## Next safe review target

The current recommended next bounded target is `ParallelHead.forward`, whose source identity is now recorded in `artifacts/official-reference-review.json`:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `ParallelHead` | 997-1017 | `6247ecdd05c6a9abae3c53eb58bb1154a88e56caa1d149ec71554c405b18afa0` |
| `ParallelHead.forward` | 1008-1017 | `2537fe7c5c90707fb39100f54ef9559efed9979513f5f0f0939e063742ceaa52` |

`ParallelHead.forward` should be reviewed as a bounded output-head/read-logits seam before entering Attention/MoE/FP8/FP4/Hyper-Connections/DSpark/MTP/cache semantics. Do not start attention, MoE, FP8/FP4, Hyper-Connections, DSpark/MTP, or cache semantics from oMLX comparisons.
