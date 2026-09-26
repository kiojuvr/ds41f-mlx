# Boundary 7a: layer-24 -> layer-25 HC carry and shared-attention state integration

Status: PASS for the bounded prefill fixture.

Scope:

```text
layers = [24, 25]
B = 1
S = 2
start_pos = 0
prefill
world_size = 1
hc = 4
```

The runner starts from the same bounded initial authority as Boundary 6e: checkpoint-derived `x_hc`, deterministic synthetic incoming `pre_mix`, and bounded external shared-attention state.  It executes Block24 and then Block25 in one connected dataflow.  Boundary 6e artifacts are used as digest gates only; Block24 outputs are not loaded from artifacts and injected into Block25.

## Source/config roles

From pinned `config.json` and `inference/model.py`:

| layer | compress_ratio | kv source | index source / has Indexer | candidate source | candidate consumer | shared topk behavior |
| --- | ---: | --- | --- | --- | --- | --- |
| 24 | 1 | no | yes | no | yes, because candidate source layer is 20 | produces/updates `shared_attn.topk_idxs` |
| 25 | 1 | no | no | no | no Indexer branch | consumes existing `shared_attn.topk_idxs` |

Layer 24 and 25 both read the bounded external `shared_attn.compress_kv`.  Layer 24 additionally reads bounded external `shared_attn.index_k` and `shared_attn.candidates` to run its Indexer.  Layer 25 does not run an Indexer, so it does not read `index_k` or `candidates` in this seam.

Window KV is layer-local cache/publication and is not a cross-layer shared-attention carry:

```text
layer24 window KV = layer24 local publication
layer25 window KV = layer25 local publication
```

## HC carry seam

Block24 returns:

```text
x24_out
ffn_pre24
```

The connected runner passes them directly as:

```text
Block25 x input      = x24_out
Block25 incoming mix = ffn_pre24
```

Validated carry digests:

```text
x24_out / layer25 x:        a596b0585c9702257b730d81ccc9bd8eac03df53404d64d8be83c2dd8625ae63
ffn_pre24 / layer25 pre_mix: 8fcc739b02187a2a805bc06ab26e8c66d5cec632520cab47f7ade518bd0b0d2d
```

## Shared-attention state seam

The source-backed cross-layer publication/consumption seam in this adjacent pair is:

```text
layer24 Attention._compress_topk_idxs publishes shared_attn.topk_idxs
layer25 Attention._compress_topk_idxs returns/consumes shared_attn.topk_idxs
```

Digest:

```text
shared_attn.topk_idxs layer24 publication / layer25 consumption:
f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

Other shared fields are classified as bounded external initial input and unchanged/read-only across this two-block seam unless source-backed otherwise:

- `compress_kv`: read by layer24 and layer25, unchanged
- `index_k`: read by layer24 Indexer only, unchanged
- `candidates`: read by layer24 Indexer only, unchanged

## Layer25 outputs

Layer25 routing and output are recorded in `artifacts/native-layer24-25-connected-validation.json`.

Key layer25 digests:

```text
attention input:  e44da41161f2298df30c333d8b59d2d6c0ba433bc0115e7d7bf7cecb955c0b80
attention output: 7ffc06f69241802ea9a64caf519a0f57c2cf788d6edd0f995a276a80b779285d
MoE input:        d3b054dfa8382a5b9ec1e2bd0f43e813dc742a075cd0e13fc811cf23a04ad676
full MoE output:  0e74d7e06d22489330c94854d481fc7938dd48ec0607ab21eb17b0cf20869101
x25_out:          628aaa78c458fd316e069589d58fa89319cf7bd5076060899a2485ccec5206ea
ffn_pre25:        f4efaa2df89d7f61febc1166d44ad04e425c8f8a3390a4783cbe825e227d3d9a
```

Safe claim:

> For the bounded layer-24 -> layer-25 prefill fixture, layer-24 Block outputs and HC carry are consumed directly by layer 25 in one connected execution, and the declared shared-attention state persists across the two Blocks according to the reviewed official source semantics.

Non-claims: no production provenance for the synthetic incoming `pre_mix` entering layer24, no layers 0-23 execution, no candidate/state production before the bounded initial shared state unless executed here, no production-scale candidate pruning, no decode/ring/partial compression group, no world_size > 1 expert parallelism, no Transformer final norm/logits, no full-model correctness, and no performance/fusion claim.
