# sparse_attn semantics contract

Boundary 3 validates only:

```text
q, window_kv, attn_sink, topk_idxs, softmax_scale -> sparse_attn
STOP before inverse rotary / wo_a / wo_b
```

## Authority

Official source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/kernel.py
```

File SHA-256:

```text
1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455
```

Reviewed spans:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `sparse_attn_kernel` | 311-389 | `5438750533acb517260b1da40a9a033e5068eaef4a0c57b2c769f2de4e686266` |
| `sparse_attn` | 392-403 | `42208bc5467f5a29efd18020b62162fa3177614293f3669d3d0b8800d6e5d704` |

## Fixture scope

Fixture:

```text
artifacts/sparse-attn-official-reference-fixture.json
```

Scope:

```text
layer = 0
B = 1
S = 2
start_pos = 0
compressed_kv = false
world_size = 1
heads = 64
head_dim = 512
topk = 2
softmax_scale = 512^-0.5
```

Inputs reuse validated boundary artifacts:

- Q: Boundary 1 rotary-applied Q from `artifacts/attention-q-prelude-official-reference-fixture.json`;
- window KV and exact `topk_idxs`: Boundary 2 from `artifacts/window-kv-prelude-official-reference-fixture.json`;
- `attn_sink`: official checkpoint tensor `layers.0.attn.attn_sink`, F32 `[64]`.

## Operation contract

For each `(batch, query_position, head)`:

1. Selected KV indices come from `topk_idxs` exactly as int32.
2. `-1` indices are invalid: they contribute no score and no value numerator. Invalid scores are treated as `-inf` for softmax.
3. Valid raw scores are:

```text
raw_score[t] = dot(q[b,m,h,:], kv[b,idx[t],:])
scaled_score[t] = raw_score[t] * softmax_scale
```

4. Row max and denominator are computed in FP32 over valid selected scores.
5. Value accumulation uses `exp(score - row_max)` rounded to BF16 before multiplying BF16 KV values, matching the official kernel's `acc_s_cast` before the value GEMM.
6. `attn_sink[h]` is not a value token. It contributes only this denominator term:

```text
exp(attn_sink[h] - row_max)
```

and has no numerator/value vector contribution.
7. All-invalid row behavior: `row_max` remains the finite lower bound `-1e30`; numerator remains zero; the sink denominator term makes the output zero.
8. Output is BF16 round-to-nearest-even.

Predeclared native tolerance:

```text
max_bf16_ulp_lte = 2
```

The current bounded validation is bit-exact.

## Fixture contents

The fixture records bounded values and digests for:

- selected KV indices;
- raw scores;
- scaled scores;
- row max;
- softmax denominator including `attn_sink`;
- denominator without `attn_sink`;
- `attn_sink` denominator term;
- BF16 attention output.

It also records that the fixture contains `-1` mask entries and that adding the sink denominator changes the bounded output, independently checking that `attn_sink` is a denominator-only sink term rather than a value token.

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_sparse_attn_fixture.py \
  --out artifacts/sparse-attn-official-reference-fixture.json
```

Observed output digest:

```text
attention_output_bf16_uint16_sha256: 40adc9d7e9df0744fcb341c0ae8c667b3fa0821bc358c594ebd90636b43a432f
```

## Native validation

Validation artifact:

```text
artifacts/native-sparse-attn-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_sparse_attn_against_official_reference.py \
  --out artifacts/native-sparse-attn-official-reference-validation.json
```

Observed result:

```text
attention output: bit-exact
max_bf16_ulp_error: 0
native Metal executed: true
```

## Non-claims

This contract does not validate:

- compressed KV, Compressor, Indexer, or candidate selection;
- inverse rotary after sparse attention;
- `wo_a`, `wo_b`, output projection, residual, Block, or Hyper-Connections;
- MoE, FP4, DSpark/MTP;
- logits, full layer, full prefill, or full model correctness;
- performance.
