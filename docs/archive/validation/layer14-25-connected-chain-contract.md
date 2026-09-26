# Boundary 7d: layer14 -> layer25 connected residual/HC + shared-state generation chain

Status: PASS for the bounded prefill fixture.

Boundary 7d executes Blocks 14 through 25 in one connected dataflow.  The only external residual inputs are bounded `x14` and deterministic synthetic incoming `pre_mix14`; shared-attention state starts empty because layer14 is source-reviewed as a self-contained KV/index source for this region.

## Role summary

Pinned config/source review:

- layer14: `compress_ratio=2`, KV source, Indexer/key owner, not candidate source; publishes `compress_kv`, `index_k`, and `topk_idxs`.
- layers15-19: `compress_ratio=2`, not KV/index/candidate sources; consume layer14 `compress_kv` and `topk_idxs`; no candidates are used.
- layer20: `compress_ratio=1`, KV source, Indexer/key owner, candidate source; overwrites `compress_kv`, `index_k`, publishes `candidates`, and overwrites `topk_idxs`.
- layers21-23: consume layer20 `compress_kv` and `topk_idxs`.
- layer24: index source/candidate consumer; consumes layer20 `compress_kv`, `index_k`, `candidates`, and overwrites `topk_idxs`.
- layer25: consumes `compress_kv` and layer24 `topk_idxs`.

Window KV remains layer-local for each layer and is not shared cross-layer state.

## Initial digests

```text
x14:       9b84710eb3107f0da3bcc1770ede41513af4b8fb6df9850819203e1fa06da02c
pre_mix14: c4a9b0e568a1fae40750696c6e213b4dca41550d000d4aca6e9e50d016d45f1c
```

## Key generation digests

Layer14:

```text
attention input: 93eb76f4c8e8c643617746eb57f970f10b62e7d76ec6ed2bd46be7e33ebfe198
compress_kv:     7cfcf06073dd7db12c1704aa2abc328d2d807945712bb7b88f4f851e8f564fd7
index_k:         86732b09f341a7b1b0b94d69094e32f835310ee81dec0e160eb808ff34aaa955
topk_idxs:       baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424
```

Layer20 overwrites / candidate publication:

```text
attention input: 778f7d7e2ab428f28ae9e8ed1f52ca86abad6ddc3ca46eeafa8c72afcfa0ce9a
compress_kv:     8a883f46484455132641e96bd3935164ecc53a3d6527fb3705545965b963ee7a
index_k:         df8b734cb7ba97d5f057c116280ef3a7262f580509e98fb9f84ae5f7b26315fa
candidates:      27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1
topk_idxs:       f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

Layer24 topk overwrite generation:

```text
topk_idxs: f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

The layer24 top-k value happens to equal layer20 in this short fixture, but it is recorded as a separate source-branch generation.

## Carry and final output

All adjacent `x_out -> next x` and `ffn_pre -> next pre_mix` gates pass for 14->15 through 24->25.  The main 7c artificial seam is removed:

```text
x19_out -> Block20 x:
0dad3f18af935ac9b27e11a1684cc1278bbf51578f8e8cbb3efd1bd5be25a370

ffn_pre19 -> Block20 pre_mix:
5ed1dd16e3470454e787bc46d269eaa433fd136ae9e87a34b26eb44dd6740836
```

Final:

```text
x25_out:   9f0126201045fe2ac89ab75f35092ec19fb9a308993c214e858ebfd836a0dbc4
ffn_pre25: 9b400664a67dfd42066c9f00607a86a5b052b81207d54a6fa3dbdc43aa5ea80a
```

Safe claim:

> For the bounded prefill fixture, layers 14 through 25 execute as one connected Block chain. Residual and Hyper-Connection carry flow directly between every adjacent layer, while shared-attention state generations and source-backed overwrites persist through the same execution.

Non-claims: no layers0-13 residual-stream execution, no production provenance for x14, no production provenance for incoming pre_mix14, no production-scale candidate pruning, no decode/ring/partial compression group, no world_size > 1 expert parallelism, no layer26+, no Transformer final norm/logits, no full-model correctness, and no performance/fusion claim.
