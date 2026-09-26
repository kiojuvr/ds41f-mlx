# Boundary 8 closeout

Status: PASS  
Current integrated numerical authority: `artifacts/native-layer0-39-all-blocks-connected-prefill-validation.json` at commit `dae92ab0d9b7cbc6c77d8dac878d3e81b22492ab`.

## Closed scope

Fixture scope:

```text
tokens [[0,3]]
B=1, S=2, start_pos=0, prefill, world_size=1
Transformer entry -> Blocks 0..39
STOP after Block39.forward returns x39_out, ffn_pre39
before h = layer.hc_pre(h, pre_mix)
```

Boundary 8 closes all 40 Blocks as one connected bounded prefill execution. It does **not** close Transformer post-loop output processing.

## Authority chain

- Boundary0-5: primitive / Attention / compressed-state semantic authorities.
- Boundary6: HC + MoE + single Block semantics.
- Boundary7g: closed subscope authority, Transformer entry -> Blocks0..25.
- Boundary8a: closed subscope/regression authority 0..28, post-layer25 consumer region and `topk@28`.
- Boundary8b: closed subscope/regression authority 0..32, `topk@28` consumers and `topk@32`.
- Boundary8c: closed subscope/regression authority 0..36, `topk@32` consumers and final configured index source `topk@36`.
- Boundary8d: current integrated numerical authority 0..39, `topk@36` consumers and final Block boundary.

Older Boundary8a-8c artifacts are not invalidated; they remain exact prefix regression evidence. They do not override Boundary8d for this identical fixture/scope.

## Source topology

```text
num_hidden_layers = 40
kv_source_layer_ids = [2,8,14,20]
index_source_layer_ids = [2,8,14,20,24,28,32,36]
candidate_source_layer_id = 20
```

Late lifecycle:

```text
compress_kv@20 persists through Block39
index_k@20 actual later consumers: 24,28,32,36
candidates@20 actual later consumers: 24,28,32,36
topk@24 consumers: 25,26,27
topk@28 consumers: 29,30,31
topk@32 consumers: 33,34,35
topk@36 consumers: 37,38,39
```

Digest equality between topk generations does not collapse generation identity.

## Top-level digests

```text
x36_out:
499ab386c1c3a547619c1fec569b74712dcc68804148975ba8439bcc35377c05

ffn_pre36:
cb4af2a88e07151442d5fb92ccdebac977b8223f3e80a8a47200af64f0593170

x39_out:
c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3

ffn_pre39:
8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc
```

Final shared-state snapshot:

```text
compress_kv: fdb027edf978cebd05926802259c639f106f673c997da85129dbaa5ce3f0df41
index_k:     70b8711d0fdf54876d56ff9bd993b7504f5c78463b1bf912634803c194bff1f5
candidates:  27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1
topk_idxs:   f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

## Carry closeout

All 39 adjacent Block seams, `0->1` through `38->39`, are exact:

```text
all x carries exact = true
all pre_mix carries exact = true
```

## Transformer post-loop boundary

Reviewed source order after the Block loop:

```text
last Block returns h, pre_mix
-> h = layer.hc_pre(h, pre_mix)
-> logits = self.head(self.norm(h))
-> output_ids = sample(logits, self.temperature)
-> main_hidden assembly
-> return
```

Boundary 8 stop:

```text
after Block39.forward returns x39_out, ffn_pre39
before h = layer.hc_pre(h, pre_mix)
```

## Closed claims

For the stated fixture scope only:

- Transformer token embedding entry
- initial HC residual expansion
- initial identity `pre_mix`
- all 40 Blocks 0..39 execute in one connected dataflow
- all 39 adjacent residual `x` carries exact
- all 39 adjacent HC `pre_mix` carries exact
- all bounded Attention paths 0..39
- all bounded FFN/MoE paths 0..39
- compressed-attention state generation lifecycle `2 -> 8 -> 14 -> 20`
- candidate publication at 20
- index/topk generations `24 -> 28 -> 32 -> 36`
- generation@20 compressed/index/candidate state lifetime through final Block
- generation@36 topk consumption by 37/38/39
- final Block39 `x_out` and `ffn_pre39`

## Still open / non-claims

Open:

- Transformer post-loop HC collapse: `h = layer.hc_pre(h, pre_mix)`
- final RMSNorm
- ParallelHead
- connected logits
- sampling
- production-scale candidate pruning
- decode start_pos > 0
- window-KV ring/wrap semantics
- compressed decode partial-group accumulation/publication
- multi-call cache persistence
- world_size > 1 / expert parallel / all-reduce
- Engram
- MTP / DSpark
- long-context numerical qualification
- full Transformer output correctness
- full-model correctness
- performance/fusion/production qualification

Non-claims: no Transformer post-loop HC collapse, no final RMSNorm, no connected ParallelHead/logits, no sampling correctness, no decode/ring/partial compression-group semantics, no full Transformer-output correctness, and no full-model correctness.

## Authority guard

Semantic correctness authority is official local source + official checkpoint raw bits + independent/native arithmetic contracts. `not_omlx_derived = true`. oMLX / old `deepseek-v41-flash-mlx` evidence must not enter the semantic correctness authority chain.

Safe closeout claim:

> For B=1, S=2, start_pos=0, world_size=1 and token fixture [[0,3]], the official-reference-derived native execution from Transformer entry through all 40 Blocks (0–39) is closed as one connected bounded prefill validation. Every adjacent residual/Hyper-Connection carry between Blocks is connected directly, and the reviewed compressed-attention shared-state lifecycle persists through the final Block.
>
> This closeout stops at the Block39 return boundary and does not validate the Transformer post-loop Hyper-Connection collapse, final RMSNorm, ParallelHead, logits, sampling, or full Transformer output.

Next candidate: Boundary 9a, post-loop HC collapse only (`Block39 x_out + ffn_pre39 -> h = layer.hc_pre(h, pre_mix)`, stop before `self.norm`).
