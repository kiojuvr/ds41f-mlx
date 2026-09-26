# Window-KV prelude semantics contract

Boundary 2 validates only the layer-0 sliding-window K/V prelude:

```text
x -> wkv FP8 linear -> kv_norm RMSNorm -> rotary on KV tail
  -> act_quant(window KV) -> window cache publication -> get_window_topk_idxs
STOP before sparse_attn
```

## Authority

Official reference root:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
```

Reviewed source spans:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `Attention._window_kv` | 700-720 | `62f9eda94cd22ee13aa115822ffede34f22aec317671411380a69eceab66a2bc` |
| `get_window_topk_idxs` | 410-426 | `20d752a018e1a7b72190468471849bcd869319b50130b5902639973237b2d0b6` |
| `linear` FP8 branch | 181-207 | `c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada` |
| `RMSNorm.forward` | 288-293 | `adb7c70ed245e8830f6692b8c026fd7f2d17518cd98b68bf973e75524a921c85` |
| `apply_rotary_emb` | 392-406 | `1c47da553d29f41c798a2eda82b72476473913d1201e32283de0e4977aa30b1a` |
| `kernel.py:act_quant` | 98-124 | `563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb` |
| `kernel.py:fp8_gemm` | 277-307 | `cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657` |

## Checkpoint tensors

Full-shape real layer-0 tensors used:

| Tensor | Shard | Dtype | Shape |
| --- | --- | --- | ---: |
| `layers.0.attn.wkv.weight` | `model-00003-of-00048.safetensors` | F8_E4M3 | `[512, 5120]` |
| `layers.0.attn.wkv.scale` | `model-00003-of-00048.safetensors` | F8_E8M0 | `[16, 160]` |
| `layers.0.attn.kv_norm.weight` | `model-00003-of-00048.safetensors` | BF16 | `[512]` |

## Fixture scope and contract

Fixture:

```text
artifacts/window-kv-prelude-official-reference-fixture.json
```

Scope:

```text
layer = 0
B = 1
S = 2
tokens = [0, 3]
start_pos = 0
window_size = 128
compress_ratio = 0
S <= window_size
```

Operation contract:

1. `x = embed.weight[tokens]`, BF16 `[2, 5120]`.
2. `wkv(x)` uses the validated FP8 linear contract with full `[512,5120]` weight and `[16,160]` scale tensors.
3. `kv_norm` uses the validated RMSNorm contract over dim 512.
4. `apply_rotary_emb(kv[..., -64:], freqs_cis)` uses no-YaRN rotary for layer 0 positions 0..1.
5. `act_quant(kv, fp8_block_size=32, scale_fmt='ue8m0', scale_dtype=float8_e8m0fnu, inplace=True)` records:
   - quantized E4M3 bytes `[1, S, 512]`;
   - E8M0 scales `[1, S, 16]`;
   - BF16 dequantized in-place window KV `[1, S, 512]`.
6. Prefill cache publication for `start_pos=0` and `S<=window_size` writes `window_kv_cache[:1, :S] = kv` exactly.
7. `get_window_topk_idxs(128, 1, S, 0)` is int32 exact. For `S=2`, expected rows are `[[0, -1], [0, 1]]`.
8. Explicit stop before `sparse_attn`.

Predeclared tolerance:

```text
intermediate_max_bf16_ulp_lte = 1
act_quant_max_bf16_ulp_lte = 0
cache_exact = true
topk_exact = true
```

## Validation

Native validation artifact:

```text
artifacts/native-window-kv-prelude-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_window_kv_prelude_fixture.py \
  --out artifacts/window-kv-prelude-official-reference-fixture.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_window_kv_prelude_against_official_reference.py \
  --out artifacts/native-window-kv-prelude-official-reference-validation.json
```

Observed native result:

| Stage | Result | Max error |
| --- | --- | ---: |
| `wkv` output | bit-exact | 0 |
| `kv_norm` output | bit-exact | 0 |
| rotary-applied KV | bit-exact | 0 |
| quantized window-KV bytes | bit-exact | 0 |
| quantized window-KV scales | bit-exact | 0 |
| dequantized window-KV | bit-exact | 0 |
| cache publication | bit-exact | 0 |
| `get_window_topk_idxs` | int32 exact | 0 |

Metal-backed native paths executed for FP8 linear, RMSNorm, rotary, and act-quant.

## Non-claims

This does not validate:

- `sparse_attn`;
- compressed KV, Compressor, Indexer, or candidate selection;
- decode ring-wrap behavior;
- output projection or inverse output rotary;
- HC, MoE, FP4, DSpark/MTP;
- logits, full layer, full prefill, or full model correctness;
- performance.
