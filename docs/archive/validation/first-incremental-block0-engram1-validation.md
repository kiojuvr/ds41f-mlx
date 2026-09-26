# Boundary13d: first incremental Block0 completion + Engram@1 handoff

Status: **PASS**.

Scope:

```text
Boundary13a regeneration
-> Boundary13b Ngram replay in memory
-> Boundary13c layer0 pre-sparse replay in memory
-> layer0 sparse_attn
-> inverse rotary
-> layer0 Attention output projection
-> Block0 HC attention post
-> Block0 FFN/MoE
-> Block0 completion
-> Engram@1
STOP before Block1
```

No upstream tensor payload is injected from Boundary13b/13c artifacts. Artifact digests are used only as regression gates.

## Handoff regressions

Boundary13b:

```text
post-Ngram cache: 04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c
full hash:        09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530
layer1 hash:      4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373
```

Boundary13c:

```text
Q post-rotary:       7ba5035746cd43e25016dd0069c2bb5b4bc64ccb229e35e89741e8d88cd6cab9
KV post-rotary:      eb0d334e615e729d6e9e5765e1ca352f5072067048dcfc32f6384484b4f3afd1
window visible[0:3]: 9a39b3b26a0b352b067fd5fbac0487911478f335078041d84ce3ef47a309920a
window top-k:        fae2c9f0a421368ce686c681d9d7c1ec97be4c343b8cc03c92df7d3e70fb2bc0
```

## layer0 sparse_attn

Inputs are carried directly from the Boundary13c in-memory replay:

```text
Q digest:          7ba5035746cd43e25016dd0069c2bb5b4bc64ccb229e35e89741e8d88cd6cab9
KV visible digest: 9a39b3b26a0b352b067fd5fbac0487911478f335078041d84ce3ef47a309920a
top-k digest:      fae2c9f0a421368ce686c681d9d7c1ec97be4c343b8cc03c92df7d3e70fb2bc0
```

Sparse output digest:

```text
4cff07ee5c4455690cc6b2d2139d4c815c5d0ade0897eb759020bdec44bedb69
```

Layer0 has no compressed path; compressed KV is not consumed. `-1` mask entries are present and handled by the sparse attention primitive.

## Attention output path

```text
inverse rotary digest:       a041e24df7c96e7ee4de7133ebe3019d91e221951d07ee4f6b2ade415d9752a7
wo_a grouped output digest:  23e1edca1ef7db213475b23113c09703efac93c851392b389ae1f222e3b62422
Attention output digest:     0814f25097ba620c760e3db144f55d25b0f9434d03d8edd291a7a81a19105add
```

## Block0 completion

```text
x_after_attn digest: 9f230b41307ccff6fe28d99cbb8550860c726b2687830e14be3b768160eeb95f
MoE input digest:    7510cf767ee815ddf37f15c2b48eaed1c479cdca788721a36880017eb637bb28
MoE output digest:   ea6fefed4813adbefb8aba2e49b54a0fc1847a06ae45ea0f6b1cde0efe654ab7
Block0 x_out digest: ecbf76fb8c272dc20399cf4e9dc839e291adeac096d05556891fc30c3169740f
Block0 ffn_pre:      036a63ede630537263cbe5125569fc36680270c7bd35e841bc789705d992cf74
```

MoE routing:

```text
indices: [[206, 278, 208, 105, 307, 38]]
selected expert set: [38, 105, 206, 208, 278, 307]
```

## Engram@1

Engram@1 consumes the actual first-incremental `Block0 x_out` plus the recomputed Boundary13b incremental layer1 hash.

```text
incoming Block0 x_out: ecbf76fb8c272dc20399cf4e9dc839e291adeac096d05556891fc30c3169740f
layer1 hash digest:   4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373
post-Engram@1 h:      b651ee96bcf7b1f82a5ea5f18678efc85668b4766a69c2fc2b117ee75be0794b
```

Engram sparse row lookup, wkv projection, gate, and residual update digests are recorded in the JSON artifact.

## Persistent state invariants

Unchanged during Boundary13d:

- Ngram cache `[[0,3,15]]`
- layer0 window cache after Boundary13c
- layers1-39 window cache
- compressed KV caches
- Indexer `k_cache`
- Compressor partial-state classifications
- target runtime RNG state
- generation-loop control

`Block0 x_out`, `Block0 ffn_pre`, and `post-Engram@1 h` are call-local handoff values, not official persistent cache state.

## STOP

Stopped after `post-Engram@1 h` and `Block0 ffn_pre` handoff. Block1 is not executed.

Non-claims: no Block1, layer2, Compressor/Indexer/candidate/compressed-topk, Engram@14, logits, sampling, main_hidden, generation-loop advancement, world_size>1, production, or performance authority.
