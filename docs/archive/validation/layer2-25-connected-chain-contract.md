# Boundary 7f: layer2 -> layer25 connected residual/HC + shared-state generation chain

Status: PASS for the bounded prefill fixture.

Boundary 7f executes Blocks 2 through 25 in one connected dataflow.  The only external residual inputs are bounded `x2` and deterministic synthetic incoming `pre_mix2`; shared-attention state starts empty because source/config review shows layer2 is a self-contained KV/index source for this region.

## Role summary

Pinned config/source review:

- layer2: `compress_ratio=2`, KV source, Indexer/key owner; publishes generation@2 `compress_kv`, `index_k`, and `topk_idxs`.
- layers3-7: consume generation@2 `compress_kv` and `topk_idxs`.
- layer8: KV/index source; overwrites `compress_kv`, `index_k`, `topk_idxs`.
- layers9-13: consume generation@8.
- layer14: KV/index source; overwrites `compress_kv`, `index_k`, `topk_idxs`.
- layers15-19: consume generation@14.
- layer20: KV/index/candidate source; overwrites `compress_kv`, `index_k`, publishes `candidates`, and overwrites `topk_idxs`.
- layers21-23: consume generation@20 `compress_kv`/`topk_idxs`.
- layer24: index source/candidate consumer; consumes generation@20 `compress_kv`, `index_k`, `candidates`, and publishes topk.
- layer25: consumes `compress_kv` and generation@24 `topk_idxs`.

Window KV remains layer-local.

## Key digests

Initial:

```text
x2:       9b84710eb3107f0da3bcc1770ede41513af4b8fb6df9850819203e1fa06da02c
pre_mix2: c4a9b0e568a1fae40750696c6e213b4dca41550d000d4aca6e9e50d016d45f1c
```

Generation@2:

```text
Block2 Attention input: 9c2604dd5be4f3175eee1a68684e639079078e49c6596e70357aed7f32500edb
compress_kv@2:          e41a9774c76857e12dca3738627354a4a40bc605b8ed46a2a007a67d3c2ea9a8
index_k@2:              7b24801d9cefb6879715ba97d7207627be2385b8546da577b9322fa3bad41c3f
topk@2:                 baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424
```

Generation@8:

```text
Attention input@8: 70f7734699e77cf880b1229cb1f78cce373004a026f01eb2f3076936c1a07424
compress_kv@8:    6e36aa58e068046427e50e8a4bc69e419c9dfcad65324861c1454b7280066b3a
index_k@8:        14121ee441019d279779c7247104d5aaf04ecdf91626b3127592291f70483100
```

Generation@14:

```text
Attention input@14: df248fcaacdbb00ab3bff7f0d50a5181a25a93157e698045db039c316c0d2a96
compress_kv@14:    1d9ee825b5a60399b54def3ed43501b59e3a2c4652c0a8548c06bd947f7ed8e7
index_k@14:        be07614fa208aacb7d8119d5d032d4589f93443fdfd2b626b123ff76ab989850
```

Generation@20 / candidates:

```text
Attention input@20: 08396090ad833d9943bd441eab10ac5d284fd7aa89a95b9d3d111063e9517319
compress_kv@20:    25b790d9a60d051269df3a8fd1e995c76f8685b4790f23d91039aaea76b1bdda
index_k@20:        ee3390b4e1920bda01cd0f0b5674fd56f125b024abd5df370f2ebe51b855a213
candidates@20:     27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1
```

Generation@24 topk:

```text
topk@24: f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

Main 7e seam removed:

```text
x7_out -> Block8 x:          5780c1938fedc169a2512f7937cafe086b5b6298625239cf1be89e8fa68542fa
ffn_pre7 -> Block8 pre_mix:  33fafbcd03776cab43d7a4f2ccba2cc7eb23c490dcff96c25e9a03851367766f
```

All 23 adjacent x/pre_mix carry gates pass for 2->3 through 24->25.

Final:

```text
x25_out:   1c7fab0573e587b4aa5444313acf865cf17d52dcaaafb53153db7aed537da140
ffn_pre25: 0160a6d4d015a859eab80b6cc4408e16aa75334d9719c0d101d44d731164dc10
```

Safe claim:

> For the bounded prefill fixture, layers 2 through 25 execute as one connected Block chain. Residual and Hyper-Connection carry flow directly between every adjacent layer, while the reviewed compressed-attention shared-state generations and overwrites from layer 2 onward persist through the same execution.

Non-claims: no layers0-1 residual-stream execution, no production provenance for x2, no production provenance for incoming pre_mix2, no production-scale candidate pruning, no decode/ring/partial compression group, no world_size > 1 expert parallelism, no layer26+, no Transformer final norm/logits, no full-model correctness, and no performance/fusion claim.
