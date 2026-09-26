# Boundary13: decode / incremental-state source audit

Status: **PASS** (source-only semantic/state-lifetime audit; no numeric decode execution).

Authority is only pinned official source at `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`. Historical runtime/oMLX/DwarfStar are not semantic oracles. `not_omlx_derived == true`.

## Source identities

- `inference/model.py`: `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` (1309 lines, 61549 bytes)
- `inference/engram.py`: `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` (184 lines, 8138 bytes)
- `inference/config.json`: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809` (66 lines, 1982 bytes)
- `inference/generate.py`: `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` (218 lines, 8722 bytes)

Canonical span method: `raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1`.

Important spans:

| file | span | sha256 | ownership |
|---|---:|---|---|
| `generate.py` | 23-75 | `d1994eadba6f2269f79ba57c5a85df08c7903eb4f6052a7fbf8d3980b5f9db85` | official generation function |
| `generate.py` | 55-70 | `109c68796bf2265de649cd8a7a10953d920dc79a0985002d1346a4870a402425` | prefill/decode loop |
| `generate.py` | 57-62 | `bd03704a617f3f572e58676e4a6c16e7f7fa540cb2bea235389255a761ba5b92` | `Transformer.forward` call site |
| `generate.py` | 67-68 | `78db02f983aa0c4b2a4b0158bfa6ca5d46627b3c08e4e1896b860502e42bc993` | sampled-token write and `prev_pos` advance |
| `model.py` | 409-427 | `6f29e174995e5f5d3f194b335d17d68a8f65fdca5ed35c2f4ec9576f2099e03f` | window index construction |
| `model.py` | 429-486 | `f8f76a2dad907c2468a82fb0dc07550cbc696fbc17ede4c24ad9a5cf58c3410c` | `Compressor` state/forward |
| `model.py` | 488-581 | `e59d197a495bb2af96d44190ad4b8664d24f7460f36db1ea24b29807b050ac07` | `Indexer` state/forward |
| `model.py` | 613-788 | `6922d64f9accc5af89f8e5f23c21870beaac5d2a8793ab9ca09dfb2599021b82` | attention caches and update path |
| `model.py` | 1166-1179 | `4e20039aaeb433f9d3ecbd569bc3da0bdee45e09f5f5eac2cb541f6ab86b4748` | `SharedAttentionRuntime` |
| `model.py` | 1183-1273 | `3503e2aca986abba129ded558eb3db80d5045900f0c2cbe78cec768827f5c02c` | `Transformer` init/forward |
| `engram.py` | 118-184 | `4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe` | `NgramHashState` |

`kernel.py` sparse attention was reviewed as a called local implementation: it consumes `topk_idxs`, treats `-1` as masked, and owns no temporal model state.

## Official generation call sequence

`generate.py:36-54` builds a right-padded `tokens` tensor and prompt mask. `prev_pos = 0` at line 55. The loop is `for cur_pos in range(min(prompt_lens), total_len)` at line 58.

For a single prompt `[[0,3]]` with enough `max_new_tokens`:

1. First `Transformer.forward` call (`generate.py:60-65`):
   - `input_ids = tokens[:, 0:2] == [[0,3]]`
   - `start_pos = 0`
   - return member `[0]` is `output_ids` sampled by `Transformer.forward` (`model.py:1268-1272`, `sample` at `1290-1298`).
   - with deterministic `temperature=0`, prior authority/candidate fixture uses sampled token `15`; Boundary13 does not recompute it.
2. Generation loop writes/advances (`generate.py:66-68`):
   - possible prompt override by `torch.where(prompt_mask[:, cur_pos], ...)`; for this single prompt position 2 is not prompt, so sampled token is written to `tokens[:,2]`.
   - `prev_pos = cur_pos`, so `prev_pos` becomes `2`.
3. Second `Transformer.forward` call on next iteration (`cur_pos = 3`):
   - `input_ids = tokens[:, 2:3] == [[15]]` for the deterministic candidate.
   - `start_pos = 2`.
   - `S = 1`.
4. Thereafter `prev_pos` is set to each completed `cur_pos`; start position is the absolute token position of the previous loop boundary, not an inferred blind `+= S` rule. In the official loop after prefill, each call has `S=cur_pos-prev_pos=1`.

## `start_pos` consumers

- Generation slicing and advancement: `generate.py:55-68`.
- `NgramHashState.forward(input_ids,start_pos,...)`: writes `cache[:, start_pos:start_pos+S]`; builds absolute `positions` (`engram.py:151-160`).
- Rotary slices: `Attention.forward` uses `freqs_cis[start_pos:start_pos+S]` (`model.py:767`); `Indexer.forward` uses query slice `start_pos:end_pos` and compressed group positions (`model.py:540-551`); compressed KV uses group position `start_pos+1-ratio` on decode (`model.py:754-760`).
- Window KV ring placement/read index construction: `model.py:700-721`, `409-427`.
- Compressed KV/index position/capacity: `compress_len=(start_pos+S)//ratio`, writes at `start_pos//ratio` (`model.py:744`, `761`; index key analogous at `547`).
- Compressor partial group state: decode slot `start_pos % ratio`, completion `(start_pos+1)%ratio==0` (`model.py:477-484`).
- DSpark MTP attention has separate decode semantics (`model.py:1033-1067`) but is outside the primary `Transformer.forward` return; main forward does not call `forward_spec`.

For `S=1` decode there is **no new triangular intra-call causal mask** constructed. Past positions are exposed by persistent caches plus `topk_idxs` index lists; invalid positions are `-1` and `sparse_attn` masks them.

## Persistent-state ownership matrix

| state | owner | init / shape / dtype | prefill writes | first decode reads | first decode writes | reset/lifetime |
|---|---|---|---|---|---|---|
| `NgramHashState.cache` | `Transformer.engram_hash` | `torch.empty(max_batch_size,max_seq_len,int64)`, nonpersistent buffer (`engram.py:142-144`) | `[:,0:S]` compressed ids/DEAD | history via gather over prior positions | `[:,start_pos:start_pos+S]` | no explicit reset; start at 0 overwrites used prefix; model/session lifetime |
| `Attention.window_kv_cache` | every `Attention` module | zeros `[max_batch_size,window_size,head_dim]`, nonpersistent buffer (`model.py:663-668`) | ring seeded with current chunk | decode attention reads whole ring | slot `start_pos % window_size` overwritten | no explicit reset; start 0 reseeds/overwrites visible prefix; per-layer model lifetime |
| `Compressor.kv_state`/`score_state` | `Compressor` for ratio>1 source layers 2,8,14 | zeros and `-inf`, `[max_batch_size,ratio,head_dim]`, float32 (`model.py:451-456`) | trailing incomplete group stored; for `S=2,ratio=2` none | used when completing group; first decode at pos2 only writes partial | slot `start_pos%ratio` | no explicit reset beyond start0 path overwriting remainder slots used by current sequence; per-source-layer model lifetime |
| `Attention.compress_kv_cache` | KV source attention layers 2,8,14,20 | zeros `[max_batch_size,max_seq_len//ratio,head_dim]` (`model.py:670-680`) | compressed latents written by source layer | consumers read through `shared_attn.compress_kv` after source publishes | source writes completed group(s) at `start_pos//ratio` | no explicit reset; start0 writes used prefix; per-source-layer model lifetime |
| `Indexer.k_cache` | index-key owners: layers in `kv_source_layers` when their indexer owns key (2,8,14,20) | zeros `[max_batch_size,max_seq_len//ratio,index_head_dim]` (`model.py:520-526`) | index keys for compressed latents | indexers read through `shared_attn.index_k` | source writes completed group(s) at `start_pos//ratio` | no explicit reset; start0 writes used prefix; per-source-layer model lifetime |
| `shared_attn.compress_kv/index_k/topk_idxs/candidates` | module-global `SharedAttentionRuntime` | attributes initialized to `None` (`model.py:1166-1179`) | assigned while layers run | old values are not semantically consumed before producer overwrites in call order | reassigned each source/producer | process object lifetime, but semantic call-local handoff slots except pointers to persistent caches |
| `topk_idxs` | `Indexer.forward` / `Attention._compress_topk_idxs` | tensor generated in current forward | generated in prefill and assigned to shared slot | not reused as prefill value; regenerated by source before consumers | regenerated/assigned | call-local value identity |
| `candidates` | candidate source layer 20 `Indexer.forward` | bool mask from `select_candidate_blocks` | generated at layer20 in prefill | not reused as prefill value; layer20 regenerates before layers>20 consume | regenerated/assigned | call-local value identity |
| `pre_mix`, `h`, `main_hiddens` | `Transformer.forward` locals | `h` from embedding; `pre_mix` one-hot; `main_hiddens=[]` (`model.py:1252-1260`) | produced/returned as applicable | not carried | recreated | per-forward-call |
| target MLX RNG `next_session_key` | target runtime, not official model | Boundary12e digest `175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4` | advanced by target sampling, not official forward | input session key for future stochastic sampling | runtime advances outside model | target-runtime session state only |

## NgramHashState persistence and candidate structural history

`NgramHashState` owns `cache`; it is mutated in place and another `Transformer.forward` call on the same model sees prior contents. There is no reset API. The cache allocation is `torch.empty(max_batch_size,max_seq_len,dtype=int64)`, initial values undefined until written. Each call maps `input_ids` through `token_map`, applies `DEAD=-1` where `token_mask` is false, and writes `cache[:,start_pos:start_pos+S]`.

History lookup uses absolute `positions`, shifts `0..max_ngram_size-1`, clamp at 0, and a cumulative `blocked` mask. At sequence beginning (`positions < shift`) or after a `DEAD` token, substituted token is `pad_id` (compressed id of `engram_pad_id`).

For future candidate transition prefill `[[0,3]]`, `start_pos=0`; first decode one token `15`, `start_pos=2`:

- shift 0: current decode token at absolute position 2.
- shift 1: prefill token at position 1 (`3`).
- shift 2: prefill token at position 0 (`0`).
- shift 3: pad, because `positions < 3` blocks the 4th slot.
- 2-gram hashes use `[current, pos1]`; 3-gram hashes use `[current, pos1, pos0]`; 4-gram hashes use `[current, pos1, pos0, pad]` structurally. No numeric hash values are claimed.

`ParallelEngramEmbedding` and `Engram.forward` have no temporal cache; they depend only on weights, current `hash_ids`, `x`, and optional mask. Incremental order is `input_ids -> engram_hash(...) -> embedding -> optional Engram at layers 1 and 14`; `layer_hash_index` mapping and checkpoint tensors are static.

## Attention cache and compressed/index/candidate/top-k lifetime

Every attention layer owns a persistent `window_kv_cache`. KV source layers 2, 8, 14, 20 additionally own persistent `compress_kv_cache`; their compressors/indexers own the persistent partial/index buffers described above. Names in official source are exactly `window_kv_cache`, `compress_kv_cache`, `kv_state`, `score_state`, `k_cache`, `shared_attn.compress_kv`, `shared_attn.index_k`, `shared_attn.candidates`, and `shared_attn.topk_idxs`.

Candidate/top-k values produced during prefill are not reused by first decode. During decode, source/index layers regenerate `topk_idxs`; candidate source layer 20 regenerates `candidates` from persistent `index_k`/compressed state, and layers after 20 consume the current-call candidate mask. Generation identity resets every `Transformer.forward` call even if bytes could coincidentally match.

## Window KV transition

For prefill `S=2,start_pos=0`, each layer computes local KV, writes slots 0 and 1 of its fixed-size ring, and attends over local `kv` with causal top-k rows. For first decode `S=1,start_pos=2`, each layer writes slot `2 % 128 = 2`, then attends over `window_kv_cache[:bsz]`. `get_window_topk_idxs` lists oldest-first ring slots with invalid future/unfilled slots as `-1`; for this case valid slots are 0, 1, and 2. Storage is fixed-size ring; overflow evicts by modulo overwrite.

## Compressor / Indexer mutable inventory

`Compressor`: static/config/weights are `compress_ratio`, `head_dim`, `norm`, `wkv`, and for ratio>1 `wgate`. Persistent inference state is `kv_state` and `score_state`; no counters/cursors are stored, all positions derive from `start_pos`.

`Indexer`: static/config/weights are ownership flags, ratio, candidate config, dims, `wq_b`, `weights_proj`, and if owner `wk`, `k_norm`. Persistent inference state is owner `k_cache`. `freqs_cis` is a mutable reference set lazily from the owning `Attention.freqs_cis`, but it is static positional table plumbing, not temporal history. No separate cursor/counter/history field exists.

## Hyper-Connection and main_hidden cross-call lifetime

`Transformer.forward` recreates `h` from current `input_ids` embedding every call, expands to HC copies, sets `main_hiddens = []`, and creates identity `pre_mix`. Block0 of an incremental call receives no `pre_mix` from prefill. Final `h` from prefill is not next call input. With config target layers `[37,38,39]`, each call appends three `[B,S,5120]` tensors and returns `main_hidden [B,S,15360]`; for first `S=1` decode the structural shape is `[B,1,15360]`. Prefill `main_hidden [B,2,15360]` is not persistent model state.

## Session reset semantics

Official source exposes no explicit official reset API for `NgramHashState`, window KV, compressed KV, index cache, compressor partial state, candidate/top-k slots, or other attention buffers. Lifecycle is tied to using the same model object and `start_pos` discipline. A new generation beginning at `start_pos=0` overwrites the visible prefix/ring portions needed by that session; stale data outside the source-visible ranges may remain allocated. Native runtime should therefore provide an explicit session abstraction/reset discipline even though official reference does not name one.

## Prefill-end persistent-state schema

```text
prefill_state_after = {
  start_pos_next: 2,
  official_model_state: {
    ngram_cache: {owner: Transformer.engram_hash, valid_positions: [0,1]},
    per_layer_attention: {
      layer_i: {window_kv_cache: {valid_ring_slots: [0,1]}},
      kv_source_layer_j: {compress_kv_cache, compressor_partial_state, indexer_k_cache}
    }
  },
  call_local_not_persisted: {pre_mix, h, main_hiddens, topk_idxs_value, candidates_value},
  target_runtime_rng_state: {next_session_key_digest: "175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4"}
}
```

## First-decode input-state schema and handoff

```text
decode_step0_input = {
  input_ids: [[15]],
  start_pos: 2,
  persistent_model_state: prefill_state_after.official_model_state,
  target_runtime_rng_state: prefill_state_after.target_runtime_rng_state
}
```

Handoff is one-to-one for `ngram_cache`, every layer's `window_kv_cache`, source-layer `compress_kv_cache`, source-layer `compressor_partial_state`, owner `indexer_k_cache`, and target runtime RNG key. `topk_idxs`/`candidates`/`pre_mix`/`h`/`main_hiddens` do not hand off as persistent model state.

## First-decode mutation table

| persistent field | first S=1 classification |
|---|---|
| `NgramHashState.cache` | READ_THEN_UPDATE / OVERWRITE_POSITION at absolute pos 2 |
| each `window_kv_cache` | READ_THEN_UPDATE / OVERWRITE_POSITION ring slot 2 |
| ratio>1 `Compressor.kv_state/score_state` at source layers 2,8,14 | OVERWRITE_POSITION slot 0; no completed group at pos2 |
| ratio1 source layer20 compressor | no partial state; NOT_ACCESSED |
| source `compress_kv_cache` ratio2 layers 2,8,14 | READ_ONLY for prior compressed length on first pos2 decode (no new latent) |
| source `compress_kv_cache` ratio1 layer20 | READ_THEN_UPDATE / OVERWRITE_POSITION index 2 |
| owner `Indexer.k_cache` ratio2 layers 2,8,14 | READ_ONLY on first pos2 decode unless no new latent; previous keys read |
| owner `Indexer.k_cache` ratio1 layer20 | READ_THEN_UPDATE / OVERWRITE_POSITION index 2 |
| `shared_attn.*` slots | APPEND_OR_ADVANCE as current-call assignments only; not semantic persisted values |
| target MLX RNG key | READ_ONLY by model; runtime sampling later READ_THEN_UPDATE outside official forward |

## Ownership partition

Official-model-owned persistent state: `NgramHashState.cache`, all per-layer `window_kv_cache`, KV source `compress_kv_cache`, ratio>1 compressor `kv_state/score_state`, index-owner `k_cache`, and static weights/config/buffers.

Official-call-local state: current `input_ids` slices, embeddings, `h`, `pre_mix`, `main_hiddens`, logits/output_ids return, current-call `topk_idxs`, current-call `candidates`, and shared-attention handoff values as values.

Target-runtime-owned session state: MLX RNG `session_key/draw_key/next_session_key`, SSD/offload/mmap/bookkeeping, and any native session container/reset machinery. The RNG key is not a `Transformer.forward` tensor state or return member.

## Recommended next boundaries

1. Persistent-state container / prefill-end snapshot contract.
2. `NgramHashState` first incremental update.
3. Window KV + rotary/position first incremental update.
4. Compressed/index/candidate/top-k first incremental lifecycle.
5. Engram@1/@14 on first incremental token.
6. Full connected deterministic first decode logits.
7. Sampling + main_hidden + return/session-state advancement.

## Non-claims

No incremental numerical authority; no first-decode logits/sample/main_hidden authority; no incremental Ngram/compressed/KV numerical values; no long-context or wraparound qualification; no image-mask DEAD crossing numerical authority; no world_size>1, full-model, production, or performance qualification.
