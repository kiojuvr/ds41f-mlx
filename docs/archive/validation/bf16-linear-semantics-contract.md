# BF16 linear official-reference-scoped contract

This contract covers the reviewed non-quantized BF16 branch of the official `linear()` helper and validates the existing native BF16 linear primitive against an independently generated fixture.

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
| `linear` | 181-207 | `c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada` |
| `Linear.forward` | 242-243 | `2b53749a0c9c5bd1c3ec1553ec4f107378f3050a8f585c3c773dd0b2a728404e` |

Reviewed source behavior:

- quantized FP4/FP8 weights dispatch to quantized kernels;
- otherwise `linear()` calls `torch.nn.functional.linear(x, weight)`;
- `Linear.forward` calls `linear(x, self.weight, self.bias)`.

## Bounded fixture contract

Fixture classification:

```text
official_reference_derived_independent_arithmetic_contract
```

The fixture uses official checkpoint tensors:

- input: `embed.weight` gathered rows for tokens `[0, 3]`, BF16 raw bits;
- weight: `layers.0.ffn.gate.weight`, shape `[384, 5120]`, BF16 raw bits.

Declared arithmetic contract:

```text
output = input_bf16_as_f32 @ weight_bf16_as_f32.T
output dtype = F32
accumulation = F32 over k=0..5119
```

This is an explicit primitive contract for the non-quantized linear path. It does not claim PyTorch backend-specific BF16 accumulation behavior beyond the declared F32 fixture contract.

Fixture:

```text
artifacts/bf16-linear-official-reference-fixture.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_bf16_linear_fixture.py \
  --out artifacts/bf16-linear-official-reference-fixture.json
```

Observed expected digest:

```text
expected_output_f32_sha256: 56227eb9cafee086b7fc6f403d895e0ecf418f7d45d12eb2e7822f1c0829bb70
```

## Native validation

Validation artifact:

```text
artifacts/native-linear-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_linear_against_official_reference.py \
  --out artifacts/native-linear-official-reference-validation.json
```

Observed result:

```text
bit_exact: true
mismatch_count: 0
native_output_f32_sha256: 56227eb9cafee086b7fc6f403d895e0ecf418f7d45d12eb2e7822f1c0829bb70
reference_expected_output_f32_sha256: 56227eb9cafee086b7fc6f403d895e0ecf418f7d45d12eb2e7822f1c0829bb70
```

## Non-claims

This contract does not validate:

- quantized FP8/FP4 linear kernels;
- activation quantization;
- RMSNorm or activation functions;
- MoE routing or expert execution;
- attention, Engram, Hyper-Connections, DSpark/MTP;
- logits, cache/state, full layer, or full prefill correctness;
- oMLX behavior.
