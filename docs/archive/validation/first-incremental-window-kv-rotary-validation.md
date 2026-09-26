# Boundary13c: first incremental positional / rotary / layer0 window-KV validation

Status: **PASS**.

Scope:

```text
Boundary13a prefill state
-> Boundary13b incremental Ngram replay in memory
-> first decode input_ids [[15]], start_pos=2, S=1
-> current-call embedding / HC expansion / identity pre_mix
-> Block0 attention prelude / attn_norm
-> layer0 Attention Q rotary and window-KV rotary at absolute position 2
-> layer0 window_kv_cache slot2 publication
-> get_window_topk_idxs(start_pos=2)
STOP before sparse_attn
```

No compressed KV, Compressor, Indexer, candidate selection, compressed top-k, `Engram.forward`, sparse attention output, attention output projection, Block completion, logits, sampling, main_hidden, or generation-loop advancement is executed.

## Handoff regressions

Boundary13a manifest:

```text
311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301
```

Boundary13b handoff, recomputed in memory and not artifact-injected:

```text
post-Ngram cache digest: 04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c
full incremental hash:   09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530
layer1 hash:             4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373
layer14 hash:            5d93f09bfecb5b8a3a722603bb5e1cd44710849cc4729223df92953386bf3ad7
```

## Source spans

Official pinned `model.py` identity is regressed. Narrow spans include:

- `Transformer.forward` entry lines 1242-1267
- Block attention prelude lines 957-986
- `Attention.forward` Q/window call path lines 765-780
- `_window_kv` lines 700-721
- `apply_rotary_emb` lines 390-406
- `get_window_topk_idxs` lines 409-427

## Transformer entry / Block0 attention input

For the first incremental call, prefill `h`, `pre_mix`, and `main_hiddens` are not reused. They are recreated from `input_ids [[15]]`.

```text
embedding digest:             75fc2a390b761925546540151bdf4735e79289d75e8a4d234d1ac0dc2ae7ba06
Block0 attention input digest: 9f3d5433e680798d5f32fe10b8384476664078668b234c0780c7bce20ef775be
```

## Absolute position 2 evidence

Rotary uses:

```text
start_pos = 2
end_pos = 3
freqs_cis[2:3]
rope_theta = 10000
original_seq_len = 0
```

Frequency digests:

```text
cos(position2): 08014ebec49df6b660eef9154f73ef63d91cb5d75f860f64523aecd0289e0a45
sin(position2): 1665e30c199582ec502e76320ff61d1c628212766584b38afd060cf2585fe62f
```

Position-0 controls differ for both Q and KV, so position 0 is not substituted.

## Q / KV rotary digests

Layer0 Q:

```text
post-rotary digest: 7ba5035746cd43e25016dd0069c2bb5b4bc64ccb229e35e89741e8d88cd6cab9
```

Layer0 KV before window quantization:

```text
post-rotary digest: eb0d334e615e729d6e9e5765e1ca352f5072067048dcfc32f6384484b4f3afd1
```

Vectorized source-order rotary and independent arithmetic rotary match exactly.

## Window-KV transition

Boundary13a layer0 visible history is preserved in slots 0 and 1. Decode position 2 publishes to slot `2 % 128 = 2`.

```text
slot0 before: 1c6a13138e16d4abcf19c1c93313ce8da7f41adac3a31035f83132f2b2edf524
slot0 after:  1c6a13138e16d4abcf19c1c93313ce8da7f41adac3a31035f83132f2b2edf524

slot1 before: 5cdf50ce91d245204f50db483466d8aee56795efc1313da868c4ea4c4f204e30
slot1 after:  5cdf50ce91d245204f50db483466d8aee56795efc1313da868c4ea4c4f204e30

slot2/new window KV: 636e9636e288fd3bac5e8cf5aa3095187f715dc0c71195d72283c086be00c12b
post-visible [slots0:3]: 9a39b3b26a0b352b067fd5fbac0487911478f335078041d84ce3ef47a309920a
```

Visible ring positions after Boundary13c:

```text
absolute positions: [0,1,2]
ring slots: [0,1,2]
```

Unused/stale capacity is not promoted to authority.

## Window top-k / index

For `start_pos=2, S=1, window_size=128`, source-order and independent index construction match exactly.

```text
shape: [1,1,128]
dtype: int32
digest: fae2c9f0a421368ce686c681d9d7c1ec97be4c343b8cc03c92df7d3e70fb2bc0
valid slots: [0,1,2]
last 10 entries: [-1,-1,-1,-1,-1,-1,-1,0,1,2]
```

Only positions 0, 1, and 2 are valid; future/unfilled slots are `-1`.

## Non-mutation

Unchanged through Boundary13c:

- Boundary13b post-Ngram cache and full incremental hashes
- layers 1-39 window state
- compressed KV visible prefixes
- Indexer `k_cache` visible prefixes
- Compressor partial-state classifications
- target runtime RNG state
- generation-loop control

## STOP flags

All are false/excluded:

```text
compressed_KV_path_executed
Compressor_executed
Indexer_executed
candidate_selection_executed
compressed_topk_executed
Engram_forward_executed
sparse_attn_executed
Attention_output_projection_executed
Block0_completion_executed
MoE_executed
layer1_plus_executed
final_RMSNorm_executed
logits_executed
sampling_executed
main_hidden_executed
generation_loop_advanced
```

## Non-claims

No compressed/index/candidate/top-k authority; no `Engram.forward`; no sparse attention numerical output; no Attention output projection; no Block0 completion; no MoE/layer1+/logits/sampling/main_hidden authority; no generation-loop advancement; no world_size>1, production, or performance qualification.
