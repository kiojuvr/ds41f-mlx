# Boundary 7e: layer8 -> layer25 connected residual/HC + shared-state generation chain

Status: PASS for the bounded prefill fixture.

Boundary 7e executes Blocks 8 through 25 in one connected dataflow.  The only external residual inputs are bounded `x8` and deterministic synthetic incoming `pre_mix8`; shared-attention state starts empty because layer8 is source-reviewed as a self-contained KV/index source for this region.

## Role summary

Pinned config/source review:

- layer8: `compress_ratio=2`, KV source, Indexer/key owner; publishes `compress_kv`, `index_k`, and `topk_idxs`.
- layers9-13: `compress_ratio=2`, not KV/index/candidate sources; consume generation@8 `compress_kv` and `topk_idxs`.
- layer14: KV/index source; overwrites `compress_kv`, `index_k`, and `topk_idxs`.
- layers15-19: consume generation@14 `compress_kv` and `topk_idxs`.
- layer20: KV/index/candidate source; overwrites `compress_kv`, `index_k`, publishes `candidates`, and overwrites `topk_idxs`.
- layers21-23: consume generation@20 `compress_kv` and `topk_idxs`.
- layer24: index source/candidate consumer; consumes generation@20 `compress_kv`, `index_k`, `candidates`, and overwrites `topk_idxs`.
- layer25: consumes `compress_kv` and generation@24 `topk_idxs`.

Window KV remains layer-local and is not shared cross-layer state.

## Key digests

Initial:

```text
x8:       9b84710eb3107f0da3bcc1770ede41513af4b8fb6df9850819203e1fa06da02c
pre_mix8: c4a9b0e568a1fae40750696c6e213b4dca41550d000d4aca6e9e50d016d45f1c
```

Layer8 actual HC-derived Attention input and generation@8:

```text
attention input: aa3dfe2f32e2411f5cba352a76f402ec4754011d3f39d88e85472f9abbdc2fbe
compress_kv@8:   693211519acd2db84a7ea707579d72176e9171bfdf0b43d20bff6be650d1f2bd
index_k@8:       a410e1d05fd50770363ae1a5ec2721e13a3c33cf34c6bc2c20301076c70b1bab
topk@8:          baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424
```

Layer14 overwrite generation:

```text
attention input: 09163b89dcfa87df2e7a601fcc0008a65e2807861b442e2a91478a89c8289167
compress_kv@14:  eddf8f3d3ab803085a37293791d44d63b8f4408e0122934e6d930446cf90d1ba
index_k@14:      ad7f95d35cb2c6e30499aa66edb6ca391af24383a437d5c1b9bb14b958c913ae
topk@14:         baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424
```

Layer20 overwrite / candidate generation:

```text
attention input: e7d2fbda90a3052a117de0885760271a85fb4ec95f48b487685c9911c7690ef2
compress_kv@20:  58aea1910be679f1b7629c9578db400f2e6eb41f18f79068c05285b03d30b9b0
index_k@20:      52e3ec7a6edef690e25e078ba0bfaa015146142603c65953ce04811fba78450c
candidates@20:   27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1
topk@20:         f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

Layer24 topk generation:

```text
topk@24: f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

The value can match an earlier short-fixture generation, but artifact records each source branch generation separately.

## Main seam removal

Boundary 7d external seam is removed:

```text
x13_out -> Block14 x:        e85584ba89cbc283b0fb4014bedd4d4521504eba0a62ca93f430c7a005c93025
ffn_pre13 -> Block14 pre_mix: 0381fb5678b6ba64d656c1b59a57ed1d342feab2aaccf46b89981238ad4e13cf
```

All 17 adjacent x/pre_mix carry gates pass for 8->9 through 24->25.

Final:

```text
x25_out:   dabc711009c2dfebb94afbc34a306325db329a0d7faf5f978607d13841a8fdc8
ffn_pre25: 7ad40b9f8892e84bb1a552b0202bb59cf8cc2ec97b733081609d06884d3ed9ca
```

Safe claim:

> For the bounded prefill fixture, layers 8 through 25 execute as one connected Block chain. Residual and Hyper-Connection carry flow directly between every adjacent layer, while source-backed shared-attention state generations and overwrites persist through the same execution.

Non-claims: no layers0-7 residual-stream execution, no production provenance for x8, no production provenance for incoming pre_mix8, no production-scale candidate pruning, no decode/ring/partial compression group, no world_size > 1 expert parallelism, no layer26+, no Transformer final norm/logits, no full-model correctness, and no performance/fusion claim.
