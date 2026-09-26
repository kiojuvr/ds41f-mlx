# RMSNorm official-reference-derived fixture

This contract covers the reviewed official `RMSNorm.forward` source span, records a bounded expected-value fixture, and validates the native BF16 RMSNorm primitive against that fixture within the predeclared tolerance.

## Authority

Official reference source:

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
| `RMSNorm` | 281-293 | `ac829397ad0c5f99412def7adb54ba0334397baa5c2ab795d4531f072fc47ecc` |
| `RMSNorm.forward` | 288-293 | `adb7c70ed245e8830f6692b8c026fd7f2d17518cd98b68bf973e75524a921c85` |

Official source behavior:

```python
dtype = x.dtype
x = x.float()
var = x.square().mean(-1, keepdim=True)
x = x * torch.rsqrt(var + self.eps)
return (self.weight * x).to(dtype)
```

## Bounded fixture contract

Fixture classification:

```text
official_reference_derived_independent_arithmetic_contract
```

The fixture uses official checkpoint tensors:

- input: `embed.weight` gathered rows for tokens `[0, 3]`, BF16 raw bits, shape `[2, 5120]`;
- weight: `layers.0.attn_norm.weight`, BF16 raw bits, shape `[5120]`.

Declared arithmetic contract:

```text
x_f32 = input_bf16_as_f32
w_f32 = weight_bf16_as_f32
var = mean(x_f32 ** 2, axis=-1, keepdims=True), F32
y_f32 = x_f32 * rsqrt(var + 1e-6) * w_f32
expected = y_f32 cast to BF16 round-to-nearest-even raw bits
```

Fixture:

```text
artifacts/rmsnorm-official-reference-fixture.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_rmsnorm_fixture.py \
  --out artifacts/rmsnorm-official-reference-fixture.json
```

Observed digests:

```text
intermediate_output_f32_sha256: 9d8889b78d8d5cb592abe207333c6ff0536ce6dc120234fef9b2eb1c464345e0
expected_output_bf16_bits_sha256: 055ee38bab468da2470854d34f70b857b873abed595fbe0d57f8880e74abd231
```

## Native validation

Validation artifact:

```text
artifacts/native-rmsnorm-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rmsnorm_against_official_reference.py \
  --out artifacts/native-rmsnorm-official-reference-validation.json
```

Observed result:

```text
bit_exact: false
mismatch_count: 3
max_bf16_ulp_error: 1
within_predeclared_tolerance: true
native_output_bf16_bits_sha256: d75615e3a0f185fe4d38f69a90a8419ab2dec37d8853bd2635ed20663ab618c3
reference_expected_output_bf16_bits_sha256: 055ee38bab468da2470854d34f70b857b873abed595fbe0d57f8880e74abd231
```

The tolerance contract is predeclared as `max_bf16_ulp_lte = 1`. The mismatch is limited to three BF16 elements at one BF16 ULP, attributed to backend `sqrt`/rounding differences within the declared primitive contract. This is a bounded primitive validation, not a full-layer or full-model claim.

## Non-claims

This contract does not validate:

- backend-specific PyTorch `rsqrt` behavior beyond the declared NumPy F32 contract;
- HC repeat/pre-mask behavior;
- attention, Engram, MoE, DSpark/MTP;
- logits, cache/state, full layer, or full prefill correctness;
- oMLX behavior;
- full-model native correctness beyond the bounded RMSNorm primitive contract.
