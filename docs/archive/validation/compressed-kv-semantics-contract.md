# Compressed-KV Boundary 5a semantics contract

This contract validates only Compressor + compressed-KV publication:

```text
x -> Compressor.forward -> compressed latent before RoPE
  -> compressed-position rotary -> FP4 act quant -> compress_kv_cache publication
STOP before Indexer
```

## Selected layer/config

The first real compressing layer is layer 2:

```text
compress_ratios[2] = 2
kv_source_layer_ids includes 2
B = 1
S = 2
start_pos = 0
world_size = 1
```

This is prefill-only with sequence length exactly one complete compression group. Decode partial-group state is not covered.

## Reviewed official source

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `Compressor` | 429-485 | `dcd32a8debcf46c4d19d0347a3bc982e7aa70bba9746845d0b1555a7a73c8d67` |
| `Compressor.forward` | 458-485 | `cd864ce74af1194d0a30178035efa2af92b8f7d27666724b3c63cd1ca6e5a092` |
| `Attention._compress_kv` | 739-763 | `fa0b8a602b8d6e200131396219685902c525c48bfb1943740446c242352ea2d9` |
| `fp4_act_quant` | 184-204 in `kernel.py` | `1066960c1da76f484a12ae2b456daa4734eabe618f77fd781f3c98d80b05db99` |

## Actual checkpoint tensors

Layer-2 compressor tensors are in `model-00005-of-00048.safetensors`:

| Tensor | Dtype | Shape |
| --- | --- | ---: |
| `layers.2.attn.compressor.wkv.weight` | BF16 | `[512, 5120]` |
| `layers.2.attn.compressor.wgate.weight` | BF16 | `[512, 5120]` |
| `layers.2.attn.compressor.norm.weight` | BF16 | `[512]` |

## Operation contract

For `compress_ratio=2`:

1. `x` is bounded official `embed.weight` rows for tokens `[0, 3]`.
2. `Compressor.forward` promotes `x` to FP32 for the ratio>1 path.
3. `wkv` and `wgate` are dense linear projections over BF16 checkpoint weights represented as FP32 values.
4. Scores are reshaped to `[B, groups, ratio, head_dim]`; softmax is over the `ratio` axis independently for every latent dimension.
5. The latent before norm is `sum_ratio(kv * softmax(score))`, FP32.
6. The pooled latent is cast to the original input dtype (BF16) and passed through `Compressor.norm` RMSNorm, producing the compressed latent before RoPE.
7. `_compress_kv` applies rotary at compressed positions. For this fixture the single compressed group represents source position 0, so the rotary angle is zero but the stage is still checked.
8. `fp4_act_quant(latent, block_size=16, inplace=True, scale_dtype=F8_E4M3)`:
   - block scale is E4M3 round-to-nearest of `max(abs(block), 6*2^-9) / 6`;
   - values are quantized to FP4 E2M1 over `{0, .5, 1, 1.5, 2, 3, 4, 6}` with sign;
   - bytes pack two FP4 nibbles;
   - dequantized BF16 values are written back in place.
9. `compress_kv_cache[:1, 0:1] = latent` exactly.
10. Stop before Indexer / candidate selection.

Predeclared tolerance:

```text
f32_max_abs_lte = 1e-4
bf16_max_ulp_lte = 1
fp4_bytes_exact = true
cache_exact = true
```

## Artifacts

Fixture:

```text
artifacts/compressed-kv-official-reference-fixture.json
```

Native validation:

```text
artifacts/native-compressed-kv-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_compressed_kv_fixture.py \
  --out artifacts/compressed-kv-official-reference-fixture.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_compressed_kv_against_official_reference.py \
  --out artifacts/native-compressed-kv-official-reference-validation.json
```

Observed native result: all checked stages are bit-exact within the predeclared tolerance, including FP4 bytes/scales and cache publication.

## Non-claims

This does not validate:

- Indexer;
- `select_candidate_blocks` or candidate/top-k publication;
- decode partial-group state;
- sparse-attn integration with compressed KV;
- Block / HC;
- logits, full layer, full model correctness;
- performance.
