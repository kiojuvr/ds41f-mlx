# FP8 linear/GEMM semantics contract

This is Attention Stage 1a only: a bounded FP8 projection primitive contract and native validation. It does **not** integrate `wq_a -> q_norm -> wq_b -> rotary`.

## Authority

Official checkpoint/reference root:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
```

Reviewed sources:

| File | SHA-256 |
| --- | --- |
| `inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |

Reviewed spans:

| Target | Lines | Source SHA-256 | Role |
| --- | ---: | --- | --- |
| `linear` | 181-207 | `c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada` | dispatches FP8 weights to `act_quant` then `fp8_gemm` |
| `act_quant` | 98-124 in `kernel.py` | `563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb` | activation quantization wrapper |
| `fp8_gemm` | 277-307 in `kernel.py` | `cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657` | scaled FP8 GEMM wrapper |

The reviewed official `linear()` FP8 branch is:

```python
x, s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype)
return fp8_gemm(x, s, weight, weight.scale, scale_dtype, block_size=fp8_block_size)
```

with model globals:

```text
fp8_block_size = 32
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu
```

## Official checkpoint tensor provenance

Stage 1a uses the real layer-0 attention `wq_a` FP8 projection tensor.

| Tensor | Shard | Checkpoint dtype | Checkpoint shape | Role |
| --- | --- | --- | --- | --- |
| `layers.0.attn.wq_a.weight` | `model-00003-of-00048.safetensors` | `F8_E4M3` | `[1280, 5120]` | FP8 projection weight |
| `layers.0.attn.wq_a.scale` | `model-00003-of-00048.safetensors` | `F8_E8M0` | `[40, 160]` | one scale per 32x32 weight block |

The scale layout is:

```text
weight_scale[out_block, k_block]
out_block = floor(output_row / 32)
k_block = floor(input_col / 32)
```

## Operation contract

For this bounded primitive:

1. Input is BF16 activation data. The fixture uses official `embed.weight` rows only as bounded input values.
2. Activation quantization is per row and per 32-element K block:
   - `amax = max(max(abs(x_block)), 1e-4)`
   - because `scale_fmt` is set, activation scale is rounded to a power of two:
     `scale = 2^ceil(log2(amax / 448))`
   - quantized activation is E4M3FN round-to-nearest-even of `clamp(x / scale, -448, 448)`.
3. Checkpoint weights are byte-encoded `F8_E4M3` / torch `float8_e4m3fn` finite values.
4. Activation and weight scales are byte-encoded `F8_E8M0` / torch `float8_e8m0fnu`; byte `e` decodes as `2^(e-127)`.
5. Dequantization/GEMM order for each output element is:

```text
sum_k (decode_e4m3(act_q[k]) * act_scale[k_block])
    * (decode_e4m3(weight_q[k]) * weight_scale[out_block, k_block])
```

6. Accumulation is FP32 in increasing K order for the fixture.
7. Output dtype is BF16 round-to-nearest-even, matching the intended BF16 inference default dtype for the official `fp8_gemm` output path.
8. Predeclared native validation tolerance:

```text
max_bf16_ulp_lte = 0
```

## Bounded fixture

Fixture:

```text
artifacts/fp8-linear-official-reference-fixture.json
```

Scope:

- projection: `layers.0.attn.wq_a`
- input tokens: `[0, 3]`
- input slice: first 32 dimensions of `embed.weight`
- weight slice: output rows `[0, 32)`, input columns `[0, 32)`
- scale slice: `[0:1, 0:1]`
- output shape: `[2, 32]`

Expected provider is independent arithmetic reconstruction over official checkpoint bits. It does not execute oMLX, PyTorch, MLX, or native code as the expected-value provider.

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_fp8_linear_fixture.py \
  --out artifacts/fp8-linear-official-reference-fixture.json
```

Observed digest:

```text
expected_output_bf16_sha256: c719ffa7ee439f7d126820cdcba9aeef2a30caa264f44129ab7dbcd6ae74769f
```

## Native validation

Validation artifact:

```text
artifacts/native-fp8-linear-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_fp8_linear_against_official_reference.py \
  --out artifacts/native-fp8-linear-official-reference-validation.json
```

Observed result:

```text
bit_exact: true
mismatch_count: 0
max_bf16_ulp_error: 0
native Metal executed: true
```

## Non-claims

This contract does not validate:

- Attention Q-prelude integration;
- RMSNorm or rotary connection;
- sparse attention;
- KV/cache behavior;
- Compressor / Indexer;
- FP4;
- MoE / Hyper-Connections;
- DSpark/MTP;
- logits, full layer, or full model correctness;
- performance.
