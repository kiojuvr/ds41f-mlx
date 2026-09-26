# Boundary 5e: window KV + compressed KV/index assembly -> sparse_attn

Status: PASS. Scope stops immediately after `sparse_attn` output.

## Reviewed official source/config

Checkpoint/config: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.

Local pinned config confirms:

- candidate source layer: `20`
- first candidate-consumer index-source layer: `24`
- layer 24 `compress_ratio = 1`
- compressed KV source used for this bounded fixture: layer `20`
- prefill, `B=1`, `start_pos=0`, `S=2`, `world_size=1`

Recorded source spans/hashes are in `artifacts/compressed-sparse-attn-official-reference-fixture.json`:

| Span | Lines | SHA-256 | Reviewed item |
|---|---:|---|---|
| `Attention._window_kv` | 700-720 | `62f9eda94cd22ee13aa115822ffede34f22aec317671411380a69eceab66a2bc` | window KV and window topk generation |
| `Attention._compress_topk_idxs` | 722-737 | `da4f82f646260020cdefc673e5f09ceda8280cdef2f14c424832b362bd40f640` | compressed topk generation/publication |
| `Attention._compress_kv` | 739-763 | `fa0b8a602b8d6e200131396219685902c525c48bfb1943740446c242352ea2d9` | compressed prefill offset and cache read/write order |
| `Attention.forward` | 765-789 | `ab629ceb31ddb9359393b13490d5ac1dd58abc3260f7edd0ad8ea13d8a0d9e22` | concat order and `sparse_attn` call |
| `Compressor.forward` | 458-485 | `cd864ce74af1194d0a30178035efa2af92b8f7d27666724b3c63cd1ca6e5a092` | ratio=1 branch |

## Contract

For the layer-24 bounded prefill case:

```text
x
-> layer-24 local window KV
-> layer-20 ratio=1 compressed KV source
-> layer-24 candidate-consumer compressed topk_idxs
-> window topk_idxs

kv = [window_kv, compressed_kv]
topk = [window_topk_idxs, compressed_topk_idxs + window_kv_length]
-> sparse_attn(q, kv, attn_sink, topk, head_dim^-0.5)
STOP
```

Ratio=1 compressor semantics are explicit: groups are singletons, there is no gate tensor, no softmax pooling beyond identity, and every token produces one compressed latent/KV row.

Independent gates validated:

- window indices are `< window/prefill KV length`;
- valid compressed indices are `>= offset`;
- compressed index `i` maps to `concatenated_kv[offset + i]`;
- `-1` is preserved through concatenated topk;
- concatenated KV shape/digest and topk int32 tensors are exact;
- native Metal sparse-attn output is within predeclared tolerance, bit-exact in this run.

## Artifacts

- Fixture: `artifacts/compressed-sparse-attn-official-reference-fixture.json`
- Native validation: `artifacts/native-compressed-sparse-attn-validation.json`

## Non-claims

This boundary does not validate inverse rotary, `wo_a`, `wo_b`, full Attention output, decode cache/ring behavior, Block/HC, logits, full model correctness, performance, or fusion.
