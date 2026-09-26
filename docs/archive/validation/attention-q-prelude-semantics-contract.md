# Attention Q-prelude through rotary semantics contract

This document records Attention Stage 1b for layer 0 only:

```text
x -> wq_a FP8 linear -> q_norm RMSNorm -> wq_b FP8 linear -> reshape -> rotary
STOP before sparse_attn
```

It does not validate K/V, cache, sparse attention, output projection, HC, MoE, logits, a full layer, or full model correctness.

## Authority

Official checkpoint/reference root:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
```

Reviewed source identities:

| File | SHA-256 |
| --- | --- |
| `inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |

Reviewed spans used by this contract:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `Attention` / `Attention.forward` order | 613-789 | `86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110` |
| `linear` FP8 branch | 181-207 | `c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada` |
| `RMSNorm.forward` | 288-293 | `adb7c70ed245e8830f6692b8c026fd7f2d17518cd98b68bf973e75524a921c85` |
| `apply_rotary_emb` | 392-406 | `1c47da553d29f41c798a2eda82b72476473913d1201e32283de0e4977aa30b1a` |
| `kernel.py:act_quant` | 98-124 | `563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb` |
| `kernel.py:fp8_gemm` | 277-307 | `cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657` |

Reviewed `Attention.forward` Q-prelude order:

```python
freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
rd = self.rope_head_dim

qr = self.q_norm(self.wq_a(x))
q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
```

The fixture stops here, before `_window_kv`, compressed K/V, `sparse_attn`, inverse rotary, and output projection.

## Checkpoint tensors

The fixture uses full-shape real checkpoint tensors for the layer-0 Q path:

| Tensor | Shard | Dtype | Shape |
| --- | --- | --- | ---: |
| `embed.weight` | `model-00002-of-00048.safetensors` | BF16 | `[129280, 5120]` |
| `layers.0.attn.wq_a.weight` | `model-00003-of-00048.safetensors` | F8_E4M3 | `[1280, 5120]` |
| `layers.0.attn.wq_a.scale` | `model-00003-of-00048.safetensors` | F8_E8M0 | `[40, 160]` |
| `layers.0.attn.q_norm.weight` | `model-00003-of-00048.safetensors` | BF16 | `[1280]` |
| `layers.0.attn.wq_b.weight` | `model-00003-of-00048.safetensors` | F8_E4M3 | `[32768, 1280]` |
| `layers.0.attn.wq_b.scale` | `model-00003-of-00048.safetensors` | F8_E8M0 | `[1024, 40]` |

This gate exercises multiple K blocks, multiple output blocks, and the real FP8 weight-scale index rule:

```text
weight_scale[floor(output_row / 32), floor(input_col / 32)]
```

## Operation contract

Bounded fixture scope:

```text
B = 1
S = 2
layer = 0
tokens = [0, 3]
start_pos = 0
n_local_heads = 64
head_dim = 512
rope_head_dim = 64
```

Operations:

1. `x` is `embed.weight[tokens]` BF16, shape `[2, 5120]` / logical `[1, 2, 5120]`.
2. `wq_a` uses the Stage 1a FP8 linear contract with block size 32 and BF16 output, shape `[2, 1280]`.
3. `q_norm` uses the validated RMSNorm contract over dim 1280 and returns BF16, shape `[2, 1280]`.
4. `wq_b` uses the Stage 1a FP8 linear contract with block size 32 and BF16 output, shape `[2, 32768]`.
5. `unflatten(-1, (64, 512))` is row-major last-dimension reinterpretation to `[1, 2, 64, 512]`.
6. For layer 0, `compress_ratio=0`, so rotary uses no YaRN and `base=10000`. `apply_rotary_emb` is applied in place only to `q[..., -64:]` for positions 0 and 1. The f32 rotary result is copied back to BF16.
7. Explicit stop before `_window_kv` / `sparse_attn`.

Predeclared tolerance:

```text
intermediate_max_bf16_ulp_lte = 1
final_rotary_max_bf16_ulp_lte = 1
```

The current native run is bit-exact for all intermediates and final Q, which is stricter than the tolerance.

## Fixture

Fixture artifact:

```text
artifacts/attention-q-prelude-official-reference-fixture.json
```

It records expected arrays and digests for:

- input;
- `wq_a` output;
- `q_norm` output;
- `wq_b` output;
- reshaped Q;
- rotary-applied Q.

Expected-value provider: independent arithmetic composition of the existing official-reference-derived FP8 linear, RMSNorm, and rotary contracts over official checkpoint bits. It is not oMLX-derived.

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_attention_q_prelude_fixture.py \
  --out artifacts/attention-q-prelude-official-reference-fixture.json
```

Observed final digest:

```text
rotary_applied_q_bf16_uint16_sha256: a864128199748b3c06b412f87f2e7f98397c4c483263ead1ce22d5e1af3de652
```

## Native validation

Validation artifact:

```text
artifacts/native-attention-q-prelude-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_attention_q_prelude_against_official_reference.py \
  --out artifacts/native-attention-q-prelude-official-reference-validation.json
```

Observed result:

| Stage | Result | Max BF16 ULP |
| --- | --- | ---: |
| input | bit-exact | 0 |
| `wq_a` output | bit-exact | 0 |
| `q_norm` output | bit-exact | 0 |
| `wq_b` output | bit-exact | 0 |
| reshaped Q | bit-exact | 0 |
| rotary-applied Q | bit-exact | 0 |

Native Metal-backed paths executed for both FP8 linear calls, RMSNorm, and rotary.

## Non-claims

This contract does not validate:

- `_window_kv`, K/V, or cache behavior;
- compressed K/V, Compressor, Indexer, or candidate selection;
- `sparse_attn`;
- inverse rotary on attention output;
- `wo_a`, `wo_b`, output projection, residual, Block, or Hyper-Connections;
- MoE, FP4, DSpark/MTP;
- logits, full layer, full prefill, or full model correctness;
- performance.
