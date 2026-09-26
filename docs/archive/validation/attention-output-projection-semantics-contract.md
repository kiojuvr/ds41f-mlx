# Attention output projection semantics contract

Boundary 4 validates only:

```text
sparse_attn output
-> inverse rotary on output tail
-> grouped wo_a projection
-> wo_b projection
STOP before Block / HC residual integration
```

## Authority

Official source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

File SHA-256:

```text
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
```

Reviewed `Attention` span: lines 613-789, source SHA-256 `86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110`.

Reviewed output-projection order:

```python
apply_rotary_emb(o[..., -rd:], freqs_cis, True)
o = o.view(bsz, seqlen, self.n_local_groups, -1)
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
x = self.wo_b(o.flatten(2))
```

## Actual checkpoint tensors

Confirmed from the official checkpoint, not assumed from design notes:

| Tensor | Shard | Dtype | Shape |
| --- | --- | --- | ---: |
| `layers.0.attn.wo_a.weight` | `model-00003-of-00048.safetensors` | `F8_E4M3` | `[8192, 4096]` |
| `layers.0.attn.wo_a.scale` | `model-00003-of-00048.safetensors` | `F8_E8M0` | `[256, 128]` |
| `layers.0.attn.wo_b.weight` | `model-00003-of-00048.safetensors` | `F8_E4M3` | `[5120, 8192]` |
| `layers.0.attn.wo_b.scale` | `model-00003-of-00048.safetensors` | `F8_E8M0` | `[160, 256]` |

## Fixture scope

Fixture:

```text
artifacts/attention-output-projection-official-reference-fixture.json
```

Scope:

```text
layer = 0
B = 1
S = 2
world_size = 1
n_local_groups = 8
o_lora_rank = 1024
input = Boundary 3 sparse_attn output
```

Operation contract:

1. Input is the validated Boundary 3 sparse-attention BF16 output.
2. Apply inverse rotary to `o[..., -64:]` for positions 0..1 and copy back to BF16.
3. Reshape `o.view(B, S, 8, 4096)`.
4. Actual checkpoint `wo_a` is FP8. This fixture dequantizes `wo_a.weight` with its `[256,128]` E8M0 32x32 scale grid to BF16, then applies grouped F32-accumulating einsum:

```text
wo_a_bf16 = dequantize_fp8_weight_to_bf16(wo_a.weight, wo_a.scale)
wo_a_bf16.view(8, 1024, 4096)
out[b,s,g,r] = sum_d input[b,s,g,d] * wo_a_bf16[g,r,d]
```

5. Cast grouped `wo_a` output to BF16 RNE.
6. Flatten to `[B,S,8192]`.
7. Apply `wo_b` with the validated FP8 linear/GEMM contract, producing `[B,S,5120]` BF16.
8. Stop before Block / HC residual integration.

Tensor-parallel all-reduce semantics are not executed; world_size=1 only is validated.

Predeclared tolerance:

```text
intermediate_max_bf16_ulp_lte = 1
final_max_bf16_ulp_lte = 1
```

Current native validation is bit-exact for all recorded stages.

## Recorded intermediates

The fixture records arrays and digests for:

- sparse-attn input;
- inverse-rotary output;
- grouped reshape/layout;
- grouped `wo_a` output;
- flattened `wo_a` output;
- final `wo_b` output.

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_attention_output_projection_fixture.py \
  --out artifacts/attention-output-projection-official-reference-fixture.json
```

Final output digest:

```text
final_wo_b_output_bf16_uint16_sha256: 49afde916b9fb6eb507f9e56159312f5e813ff9da99897ba6b72720e884c9820
```

## Native validation

Validation artifact:

```text
artifacts/native-attention-output-projection-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_attention_output_projection_against_official_reference.py \
  --out artifacts/native-attention-output-projection-official-reference-validation.json
```

Observed result:

| Stage | Result | Max BF16 ULP |
| --- | --- | ---: |
| inverse rotary output | bit-exact | 0 |
| grouped reshape | bit-exact | 0 |
| grouped `wo_a` output | bit-exact | 0 |
| flattened `wo_a` output | bit-exact | 0 |
| final `wo_b` output | bit-exact | 0 |

Native Metal paths executed for all grouped `wo_a` BF16 linear group projections and the final `wo_b` FP8 projection.

## Non-claims

This contract does not validate:

- Block / HC residual integration;
- MoE;
- Compressor / Indexer;
- tensor-parallel all-reduce;
- logits, full layer, full prefill, or full model correctness;
- performance.
