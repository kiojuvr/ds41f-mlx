# Official fixture metadata schema

Future official semantic/data fixtures must carry explicit authority metadata. This document defines the minimum schema expected before any fixture can be used as official correctness evidence.

## Required top-level fields

```json
{
  "schema": "ds41f.<stage>.vN",
  "classification": "official_checkpoint_raw_bits | official_reference_derived | independent_arithmetic_contract | architecture_only | omlx_compatibility_reference | historical_unknown",
  "not_omlx_derived": true,
  "checkpoint": "...",
  "authority": {},
  "operation_contract": {},
  "inputs": {},
  "expected": {},
  "comparison": {},
  "non_claims": [],
  "ok": true
}
```

## Field rules

- `classification` determines whether the fixture may gate official correctness.
- `not_omlx_derived` must be `true` for any official model-data or model-semantics fixture.
- `authority` must identify one of:
  - official checkpoint raw tensor/config/tokenizer source;
  - official reference source file/function/span/hash;
  - independent arithmetic contract over official checkpoint tensors;
  - DwarfStar architecture authority for architecture-only fixtures.
- `operation_contract` must define the exact operation, dtype, accumulation, rounding/casting, shape, and scope.
- `inputs` must include tensor names, shards, shapes, dtypes, token ids or generated input data, and digests.
- `expected` must state exactly how expected values were generated and must not be oMLX-derived for official fixtures.
- `comparison` must distinguish bit-exact, tolerance-bounded, shape-only, and architecture-only checks.
- `non_claims` must explicitly state what the fixture does not validate.

## Classification gating

| Classification | Official model-data gate | Official semantic gate | Notes |
| --- | ---: | ---: | --- |
| `official_checkpoint_raw_bits` | yes | no | Raw bits/slices only. |
| `official_reference_derived` | yes, if using official data | yes within scope | Requires reviewed reference span and operation contract. |
| `independent_arithmetic_contract` | yes, if using official data | primitive only | Does not imply full model semantics. |
| `architecture_only` | no | no | DwarfStar topology/lifetime/submission only. |
| `omlx_compatibility_reference` | no | no | Historical/diagnostic only. |
| `historical_unknown` | no | no | Must not gate correctness. |

## Rejection conditions

A fixture must not be promoted to official correctness if:

- expected logits/cache/state/intermediate tensors come from oMLX;
- provenance of expected values is unknown;
- source file/function/hash is missing for reference-derived fixtures;
- dtype/rounding behavior is unspecified;
- it combines architecture and semantic claims without separate classifications;
- it lacks non-claims.

## Current clean examples

- `artifacts/m2/dwarfstar-prefill/native-official-embedding.json`
  - `official_checkpoint_raw_bits`
  - stage: single-rank BF16 embedding gather subset
- `artifacts/m2/dwarfstar-prefill/native-official-projection.json`
  - `independent_arithmetic_contract`
  - stage: BF16 input/weight to F32 dense linear primitive

## Current contract metadata examples

- `artifacts/parallel-embedding-semantics-contract.json`
  - `OFFICIAL_REFERENCE_SEMANTICS_CONTRACT_METADATA_NOT_NUMERICAL_VALIDATION`
  - not a numerical fixture; prepares official-reference-derived embedding fixtures

## Current official-reference-derived examples

- `artifacts/parallel-embedding-official-reference-fixture.json`
  - `official_reference_derived`
  - independently models reviewed `ParallelEmbedding.forward` rank masking/all-reduce semantics over official `embed.weight` BF16 bits
  - does not execute oMLX, PyTorch distributed runtime, MLX, or native Metal code
- `artifacts/bf16-linear-official-reference-fixture.json`
  - `official_reference_derived_independent_arithmetic_contract`
  - covers reviewed non-quantized `linear()` / `F.linear` branch with explicit F32 accumulation contract over official BF16 tensors
- `artifacts/rmsnorm-official-reference-fixture.json`
  - `official_reference_derived_independent_arithmetic_contract`
  - covers reviewed `RMSNorm.forward` with explicit F32 variance/rsqrt and BF16 round-to-nearest-even output contract

## Current native validations against official-reference-derived fixtures

- `artifacts/native-embedding-official-reference-validation.json`
  - `official_reference_derived_native_validation`
  - compares existing native BF16 embedding gather output to `parallel-embedding-official-reference-fixture.json`
  - bit-exact within bounded token/dim scope
- `artifacts/native-linear-official-reference-validation.json`
  - `official_reference_derived_native_validation`
  - compares existing native BF16 linear output to `bf16-linear-official-reference-fixture.json`
  - bit-exact within bounded primitive scope
- `artifacts/native-rmsnorm-official-reference-validation.json`
  - `official_reference_derived_native_validation`
  - compares native BF16 RMSNorm output to `rmsnorm-official-reference-fixture.json`
  - tolerance-bounded within `max_bf16_ulp_lte = 1`
