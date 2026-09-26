# Boundary 8 source-region map

The local pinned authority (`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py` and `config.json`) was reviewed before each Boundary 8 extension.

## Config facts

- `compress_ratios[26..39] = 1`
- `kv_source_layer_ids = [2, 8, 14, 20]`
- `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`
- `candidate_source_layer_id = 20`

There is no KV source after layer20 in this checkpoint. Therefore `compress_kv@20`, `index_k@20`, and `candidates@20` persist through the later index-source regions. Later index sources publish only `topk_idxs` generations.

## Late source map

| source layer | source type | consumes | publishes / overwrites | consumer span |
|---:|---|---|---|---|
| 20 | KV + index + candidate | actual HC-derived attention input | `compress_kv@20`, `index_k@20`, `candidates@20`, `topk@20` | KV/index/candidates persist; topk overwritten by 24 |
| 24 | index | `compress_kv@20`, `index_k@20`, `candidates@20` | `topk@24` | layers25,26,27 |
| 28 | index | `compress_kv@20`, `index_k@20`, `candidates@20` | `topk@28` | layers29,30,31 |
| 32 | index | `compress_kv@20`, `index_k@20`, `candidates@20` | `topk@32` | future layers33,34,35 |
| 36 | index | `compress_kv@20`, `index_k@20`, `candidates@20` | `topk@36` | layers37,38,39 |

## Boundary 8a roles

| layer | compress_ratio | KV source | index source | candidate source | candidate consumer | compress_kv | index_k | candidates | topk_idxs | attention | FFN |
|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 26 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@24 | window+compressed | MoE |
| 27 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@24 | window+compressed | MoE |
| 28 | 1 | no | yes | no | yes | consume generation@20 | consume generation@20 | consume candidates@20 | overwrite topk generation@28 | window+compressed | MoE |

Boundary 8a endpoint is Block28 and stops before Block29.

## Boundary 8b roles

| layer | compress_ratio | KV source | index source | candidate source | candidate consumer | compress_kv | index_k | candidates | topk_idxs | attention | FFN |
|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 29 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@28 | window+compressed | MoE |
| 30 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@28 | window+compressed | MoE |
| 31 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@28 | window+compressed | MoE |
| 32 | 1 | no | yes | no | yes | consume generation@20 | consume generation@20 | consume candidates@20 | overwrite topk generation@32 | window+compressed | MoE |

Boundary 8b endpoint is Block32 and stops before Block33.

## Boundary 8c roles

| layer | compress_ratio | KV source | index source | candidate source | candidate consumer | compress_kv | index_k | candidates | topk_idxs | attention | FFN |
|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 33 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@32 | window+compressed | MoE |
| 34 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@32 | window+compressed | MoE |
| 35 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@32 | window+compressed | MoE |
| 36 | 1 | no | yes | no | yes | consume generation@20 | consume generation@20 | consume candidates@20 | overwrite topk generation@36 | window+compressed | MoE |
| 37 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@36 | window+compressed | MoE |
| 38 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@36 | window+compressed | MoE |
| 39 | 1 | no | no | no | no | consume generation@20 | none | preserved | consume generation@36 | window+compressed | MoE |

Boundary 8c endpoint is Block36 and stops before Block37.

## Boundary 8d roles and stop

| layer | compress_ratio | KV source | index source | candidate source | candidate consumer | compress_kv | index_k | candidates | topk_idxs | attention | FFN |
|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 37 | 1 | no | no | no | no | consume generation@20 | none | preserved, not branch-consumed | consume generation@36 | window+compressed | MoE |
| 38 | 1 | no | no | no | no | consume generation@20 | none | preserved, not branch-consumed | consume generation@36 | window+compressed | MoE |
| 39 | 1 | no | no | no | no | consume generation@20 | none | preserved, not branch-consumed | consume generation@36 | window+compressed | MoE |

`num_hidden_layers == 40`, so Block39 is the final Block. Boundary 8d stops after `Block39.forward` returns `x39_out, ffn_pre39` and before the first Transformer post-loop operation, `h = layer.hc_pre(h, pre_mix)`. It does not run post-loop HC collapse, final RMSNorm, ParallelHead, or logits.
