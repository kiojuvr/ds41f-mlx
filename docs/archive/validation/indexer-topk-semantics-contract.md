# Indexer + top-k publication semantics contract

Boundary 5b validates layer-2 Indexer key/query/scoring and `shared_attn.topk_idxs` publication only.

```text
layer-2 compressor latent
-> Indexer key construction/publication
-> index query projection
-> index scoring
-> causal compressed-position masking
-> top-k
-> sorted position indices
-> shared topk_idxs publication
STOP before candidate block filtering
```

## Scope

```text
layer = 2
B = 1
S = 2
tokens = [0, 3]
start_pos = 0
compress_ratio = 2
world_size = 1
candidate filtering disabled for this layer
```

Boundary 5a `artifacts/compressed-kv-official-reference-fixture.json` is used as input authority for the compressor latent.

## Reviewed official source

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `Indexer` | 488-580 | `cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75` |
| `Indexer.forward` | 527-580 | `32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d` |
| `Attention._compress_topk_idxs` | 722-737 | `da4f82f646260020cdefc673e5f09ceda8280cdef2f14c424832b362bd40f640` |

## Actual checkpoint tensor provenance

Layer-2 Indexer tensors are in `model-00005-of-00048.safetensors`:

| Tensor | Dtype | Shape |
| --- | --- | ---: |
| `layers.2.attn.indexer.wk.weight` | BF16 | `[128, 512]` |
| `layers.2.attn.indexer.k_norm.weight` | BF16 | `[128]` |
| `layers.2.attn.indexer.wq_b.weight` | F8_E4M3 | `[4096, 1280]` |
| `layers.2.attn.indexer.wq_b.scale` | F8_E8M0 | `[128, 40]` |
| `layers.2.attn.indexer.weights_proj.weight` | BF16 | `[32, 5120]` |

The fixture also uses layer-2 `wq_a` / `q_norm` to construct `qr` from bounded `x`, matching the `Indexer.forward(x, qr, latent, ...)` input contract.

## Operation contract

1. Index K owner path:
   - `latent` from Boundary 5a is passed through `wk`, `k_norm`, compressed-position RoPE, then `fp4_act_quant(..., block_size=32, inplace=True)`.
   - Result is published to `k_cache` / `shared_attn.index_k` exactly.
2. Query path:
   - `qr` is built from layer-2 `wq_a` + `q_norm` over the bounded input.
   - `wq_b(qr)` is reshaped to `[B,S,32,128]`, then RoPE and FP4 in-place quant/dequant are applied.
3. `weights_proj(x)` is BF16 dense projection and scaled by `index_head_dim^-0.5 * n_heads^-0.5`.
4. Per-head score:

```text
index_score_head = einsum("bshd,btd->bsht", q, index_k)
index_score = sum_heads(relu(index_score_head) * weights[..., head, None])
```

5. Causal compressed-position mask for prefill:

```text
compress_lens = floor(arange(1, S + 1) / compress_ratio)
positions >= compress_lens are unreachable and set to -inf
```

6. Top-k is selected by score, then sorted by compressed position order. Unreachable positions become `-1`; reachable positions are shifted by `offset=2` for the window-KV prefix.
7. `shared_attn.topk_idxs` publication is exact int32.
8. Stop before `select_candidate_blocks` and candidate filtering.

Predeclared tolerance:

```text
bf16_max_ulp_lte = 1
f32_max_abs_lte = 1e-4
int32_exact = true
fp4_exact = true
```

## Artifacts

Fixture:

```text
artifacts/indexer-topk-official-reference-fixture.json
```

Native validation:

```text
artifacts/native-indexer-topk-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_indexer_topk_fixture.py \
  --out artifacts/indexer-topk-official-reference-fixture.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_indexer_topk_against_official_reference.py \
  --out artifacts/native-indexer-topk-official-reference-validation.json
```

Observed result: all checked intermediates pass; `topk_idxs` and shared publication are int32 bit-exact. The bounded output includes `-1` for the first query's unreachable compressed position and shifted index `2` for the second query.

## Non-claims

This does not validate:

- `select_candidate_blocks`;
- candidate source layer 20;
- candidate consumer masking;
- sparse-attn integration with compressed KV;
- decode path;
- Block / HC;
- logits, full layer, full model correctness;
- performance.
