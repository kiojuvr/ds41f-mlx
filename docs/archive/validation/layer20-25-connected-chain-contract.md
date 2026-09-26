# Boundary 7c: layer20 -> layer25 connected residual/HC carry + shared-attention state chain

Status: PASS for the bounded prefill fixture.

Boundary 7c executes layers 20 through 25 as one connected Block chain.  The only external starts are bounded `x20` and deterministic synthetic incoming `pre_mix20`; all later Block inputs, HC carries, and shared-attention state are generated within the same execution.

Scope: layers `[20,21,22,23,24,25]`, B=1, S=2, start_pos=0, prefill, world_size=1, hc=4.  STOP before layer26.

## Source/config roles

Pinned config/source roles:

- layer20: `compress_ratio=1`, KV source, Indexer/key owner, candidate source; publishes `compress_kv`, `index_k`, `candidates`, and top-k.
- layers21-23: `compress_ratio=1`, not KV/index/candidate sources; consume layer20 `compress_kv` and `topk_idxs`.
- layer24: `compress_ratio=1`, index source but not KV source; consumes layer20 `compress_kv`, `index_k`, and `candidates`; publishes/overwrites `topk_idxs`.
- layer25: not KV/index/candidate source; consumes `compress_kv` and layer24 `topk_idxs`.

Window KV remains layer-local for each layer and is not cross-layer shared state.

## Key digests

Initial:

```text
x20:       9b84710eb3107f0da3bcc1770ede41513af4b8fb6df9850819203e1fa06da02c
pre_mix20: c4a9b0e568a1fae40750696c6e213b4dca41550d000d4aca6e9e50d016d45f1c
```

Layer20 actual HC-derived Attention input and producer publications:

```text
attention input: b530547bcbff7a41654688eee80b3ec95b448ec745fab95e06354646ab7ece28
compress_kv:     e122f4af67355e6ac8dc9bb3a6a7d3c21dac5f0c2f23cad3f8fc746bab0dd703
index_k:         b52aa5bde59e1fe94144decf63d32d6397cf0c873372d495acae3658bf4e23a5
candidates:      27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1
topk_idxs:       f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
x20_out:         e4ad1d7c11bd59a8e4bbff1c8d9a298ca5266a8575f4173db899454b45d221ec
ffn_pre20:       a43c436b4965db35db796efad5120e8f5a79b5e3eab6d552fd990d2e3902102a
```

Final layer25:

```text
x25_out:   ea5ea84839f988092c6b942bd0432a662b60f9aa066ca47b1c11335b9164c81c
ffn_pre25: 4ab9739ab524c0069f2a8720f1572094ec5724925224b4edf15eaca61833557b
```

All adjacent carry gates pass for `20->21`, `21->22`, `22->23`, `23->24`, and `24->25`: each `xN_out` is the next layer input and each `ffn_preN` is the next layer incoming pre_mix.

## State generation / overwrite

The state generation table in `artifacts/native-layer20-25-connected-chain-validation.json` records:

- layer20 source-backed publications for `compress_kv`, `index_k`, `candidates`, `topk_idxs`
- layer21/22/23 consumption of layer20 `compress_kv` and `topk_idxs`
- layer24 consumption of layer20 `compress_kv`, `index_k`, and `candidates`
- layer24 source-backed `topk_idxs` overwrite generation; for this short fixture its digest equals layer20 topk but it is still a distinct source branch generation
- layer25 consumption of layer24 `topk_idxs` and layer20 `compress_kv`

Safe claim:

> For the bounded prefill fixture, layers 20 through 25 execute as one connected Block chain: each Block output and HC carry feeds the next Block directly, while the reviewed shared-attention state produced by layer 20 and subsequently updated by later source layers persists through the same execution.

Non-claims: no layers0-19 residual-stream execution, no production provenance for x20, no production provenance for incoming pre_mix20, no production-scale candidate pruning, no decode/ring/partial compression group, no world_size > 1 expert parallelism, no layer26+, no Transformer final norm/logits, no full-model correctness, and no performance/fusion claim.
