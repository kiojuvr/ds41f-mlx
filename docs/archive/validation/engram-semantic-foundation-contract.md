# Boundary 12b0: Engram source / config / checkpoint / tokenizer contract

Status: PASS. This is an official-source-derived semantic/data contract audit only; it performs no Engram numerical execution.

Authority root: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/`

Semantic authority is limited to local pinned `inference/model.py`, `inference/engram.py`, `inference/config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json`, and relevant safetensors shard headers. oMLX, historical runtimes, web sources, SSD/offload behavior are not semantic authority.

## Hard-gated source identities

- `inference/model.py`: `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`
  - `ParallelEngramEmbedding`, lines 288-323: `23f3b61ac17a7fe38947171365138fb294d61ee73a41973aff322a45a655b23d`
  - `Engram.forward`, lines 325-373: `ff04df4f5312b4d5170b844364ff3e0e823a51f94b7c527656d4e5b08beecaf9`
- `inference/engram.py`: `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`
  - `EngramLayout`, lines 87-126: `6ba33720ed214987cf1f7d689f99353ee310f04ef14e648d243db3d5d3f3ea85`
  - `NgramHashState`, lines 129-184: `698c03c890bc8e1b55e93cd540a678f68845d7541c9da1d3d8e9a41f1cdde64d`

## EngramLayout contract

Constructor inputs are `args.engram_layer_ids`, `engram_max_ngram_size`, `engram_n_heads`, `engram_vocab_size`, `engram_num_embeddings`, and `engram_head_dim`. If `engram_layer_ids` is empty, layout is `None`.

Released config:

- `layer_ids`: `[1, 14]`
- `max_ngram_size`: `4`
- `n_heads`: `8`
- `head_dim`: `256`
- `num_embeddings`: `[384006168, 384016682]`
- `n_hash_cols`: `(4 - 1) * 8 = 24`

`layer_hash_index` is source-defined by `layout.layer_ids.index(layer_id)`, therefore:

- layer 1 -> `layer_hash_index = 0`
- layer 14 -> `layer_hash_index = 1`

Prime bucket ranges are generated in layer order, then ngram order, then head order, with primes never reused. Per-layer offsets are cumulative sums of flattened prime bucket sizes. The full primes/offsets are recorded in `artifacts/engram-semantic-foundation-contract.json`.

## NgramHashState contract

Constructor inputs: `args`, `layout`, `tokenizer`.

Persistent buffers/state:

- `primes`: `[num_layers, max_ngram_size-1, n_heads]`
- `offsets`: `[num_layers, n_hash_cols]`
- `multipliers`: `[num_layers, max_ngram_size]`, int64
- `token_map`: `[len(tokenizer)]`
- `cache`: `[max_batch_size, max_seq_len]`, int64

Forward source order:

1. `compressed = self.token_map[input_ids]`
2. if `token_mask` supplied, masked-out tokens become `DEAD = -1`
3. write compressed ids into `cache[:batch, start_pos:start_pos+seqlen]`
4. construct absolute `positions`
5. gather current/history shifts from cache
6. maintain cumulative `blocked` state for sequence start or DEAD crossings
7. blocked sources become compressed pad id from `engram_pad_id`
8. stack tokens as `[B,L,max_ngram_size]`
9. multiply by per-layer multipliers
10. rolling XOR creates hashes for 2-gram through max-ngram
11. modulo per prime bucket and add flattened offsets

Return expression:

```python
return torch.cat(hashes, dim=-1) + self.offsets
```

Return contract:

- shape: `[B,L,num_engram_layers,(max_ngram_size-1)*n_heads]`
- released future fixture shape for `B=1,S=2`: `[1,2,2,24]`
- dtype: int64
- axis order: batch, sequence, Engram layer in `layout.layer_ids` order, hash columns ordered by ngram order then head.

## Transformer interface

`Transformer.forward` calls:

```python
engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask)
```

Then each Engram layer receives:

```python
engram_hashes[:, :, layer.engram.layer_hash_index, :]
```

This selects all batch/sequence rows and all hash columns for the current Engram layer's table. `engram_mask` is `None` for text-only, or bool `[B,S]` where `True` participates in ngrams and `False` is image/dead.

## Tokenizer dependency classification

`NgramHashState` does not hash raw token IDs directly. It builds a compressed token-id map once by iterating every tokenizer id and using:

- `tokenizer.backend_tokenizer`
- `backend.decode([token_id], skip_special_tokens=False)`
- `backend.id_to_token(token_id)` for partial UTF-8 replacement-character cases
- an Engram-specific normalizer: NFKC, NFD, strip accents, lowercase, whitespace collapse, single-space preservation, strip, restore space.

Forward hashing then uses the static `token_map`. Runtime hash computation needs the table, not full tokenizer execution, after initialization.

Tokenizer identities:

- `tokenizer.json`: `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`
- `tokenizer_config.json`: `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`

Machine-readable tokenizer subset digests are recorded in the JSON artifact.

## Engram config fields

All Engram-related config consumers are recorded in the JSON artifact, including:

- `engram_layer_ids`
- `engram_num_embeddings`
- `engram_max_ngram_size`
- `engram_vocab_size`
- `engram_n_heads`
- `engram_head_dim`
- `engram_pad_id`
- `engram_compressed_vocab_size`
- `dim`
- `hc_mult`
- `norm_eps`
- `dtype`
- cache sizing defaults for `max_batch_size` and `max_seq_len` when absent from released config.

## Checkpoint tensor inventory

Complete inventory is in `artifacts/engram-semantic-foundation-contract.json`. Engram tensors are present only for layers 1 and 14:

- `layers.{1,14}.engram.embed.weight`
- `layers.{1,14}.engram.embed.scale`
- `layers.{1,14}.engram.q_weight`
- `layers.{1,14}.engram.k_weight`
- `layers.{1,14}.engram.wkv.weight`
- `layers.{1,14}.engram.wkv.scale`

Layer 1 tensors are in `model-00047-of-00048.safetensors`; layer 14 tensors are in `model-00048-of-00048.safetensors`. Dtype, shape, logical bytes, and byte ranges are recorded from safetensors headers. No giant tensor payload hashes were computed for this audit.

## ParallelEngramEmbedding contract

Input is a hash/index tensor, for Engram normally `[B,S,n_hash_cols]`. The table is partitioned by `world_size` using `ceil(num_embeddings/world_size)`. Each rank masks out nonlocal ids, gathers local `weight` and `scale`, dequantizes fp8 rows using per-32 scales, casts to bfloat16, zeros masked entries, and all-reduces if `world_size > 1`.

For current scope `world_size=1`, all rows are local and no all-reduce occurs. Output shape is input shape plus `engram_head_dim`: `[B,S,n_hash_cols,256]`, dtype bfloat16.

## Engram.forward contract

Source-order operations:

1. `embed(hash_ids)`
2. flatten hash embeddings
3. `wkv` projection
4. split into per-HC `key` and shared `value`
5. float key, query stream, and q/k weights
6. normalize stream and key over dim
7. dot/gate computation with signed square root before sigmoid
8. optional mask: `False` token positions force gate to zero
9. residual update: `h + gate * value`, broadcasting value across HC copies
10. output cast back to input dtype

Input/output shape is `[B,L,hc_mult,dim]` -> `[B,L,hc_mult,dim]`.

## Layer1 vs layer14

Engram@1 and Engram@14 use the same class/arithmetic, but differ in:

- `layer_hash_index`: 0 vs 1
- checkpoint tensors and shards
- table row counts
- prime bucket ranges and offsets
- connected-state position: layer14 consumes stream already modified upstream, including Engram@1.

This justifies separating later validation into 12b2 and 12b3.

## Future fixture shape contract

For future fixture `tokens=[[0,3]]`, `B=1`, `S=2`, `start_pos=0`, prefill, `world_size=1`:

- `input_ids`: `[1,2]`
- `engram_mask`: `None` or bool `[1,2]` all True
- `engram_hashes`: `[1,2,2,24]` int64
- layer1 selected hashes: `[1,2,24]` int64
- layer14 selected hashes: `[1,2,24]` int64
- pre/post Engram h: `[1,2,4,5120]`
- `ParallelEngramEmbedding` output: `[1,2,24,256]` bfloat16

## 12b1 promotion gate

PASS: `NgramHashState` source contract, tokenizer representation, hash constants/config, output axis order, `start_pos=0` prefill semantics, mask semantics, layer mapping, and expected shapes are fixed. No checkpoint embedding arithmetic is needed for 12b1.

Next boundary: Boundary 12b1, `NgramHashState` bounded numerical validation for `[[0,3]]`, `start_pos=0`, prefill.

## Non-claims

- no NgramHashState numeric authority
- no Engram numerical authority
- no ParallelEngramEmbedding numerical authority
- no main_hidden numeric authority
- no Transformer.forward return correctness
- no decode/incremental hash-state correctness
- no distributed Engram correctness
- no SSD/offload semantic qualification
- no full-model correctness
- no performance/production qualification
