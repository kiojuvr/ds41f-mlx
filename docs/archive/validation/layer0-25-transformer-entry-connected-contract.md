# Boundary 7g: Transformer entry -> layers0-25 connected bounded prefill

Status: PASS for the bounded prefill fixture.

Boundary 7g starts from official Transformer entry semantics, not external `x`/`pre_mix` fixtures.  It executes token embedding, HC residual expansion, official identity initial `pre_mix`, then Blocks 0 through 25 as one connected dataflow.  STOP before layer26, final norm, and logits.

## Transformer entry contract

Source-reviewed order:

```text
input_ids -> ParallelEmbedding
h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)
pre_mix = make_identity_pre_mix(h, hc_mult)  # float32, copy 0 = 1, others = 0
Block0(h, start_pos=0, pre_mix, image_mask=None)
```

Tokens:

```text
[[0, 3]]
```

Entry digests:

```text
tokens:             96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a
embedding output:   e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1
initial HC x:       5d0d812064ce2fc25ea149ef428b80354094b902d0212b9077e33d8c3ff087cb
initial pre_mix:    56e94d4f8d9e543d1260250ae1fdc346aea6fc5468f1a7865e77534e2fce98a1
```

## Layer0/1 regime

Config/source review shows layers0/1 have `compress_ratio=0`, no KV source and no Indexer/compressed-attention branch.  They use the window-only Attention path and the same `MoE` FFN source class as later Blocks.  They do not consume layer2 shared state before it exists; shared state remains empty until layer2 publishes.

Layer0:

```text
attention input: 055ee38bab468da2470854d34f70b857b873abed595fbe0d57f8880e74abd231
x0_out:          ba2e6acdac3178115513c81e871f0106541cba2810f7dbc5c4a0c8c309dbf938
ffn_pre0:        44c4d2aa90451df19a44e79788b72ca65d296b55e46129a8e944ee2e08406cc1
```

Layer1:

```text
attention input: d4e311d2180b8076789ba6b066df9f6142422ccc7683b6a12666cb0b40a7b8a1
x1_out:          a9abb13060e3b5986de38d653b16d9e91189f68c8815bbaeba6d8cd647fca8a9
ffn_pre1:        393d4e5fd0d9209d0374f1896d2504e008d806ede1ec4b23b4260fba3816b4b3
```

Main seam closed:

```text
x1_out -> Block2 x:          a9abb13060e3b5986de38d653b16d9e91189f68c8815bbaeba6d8cd647fca8a9
ffn_pre1 -> Block2 pre_mix:  393d4e5fd0d9209d0374f1896d2504e008d806ede1ec4b23b4260fba3816b4b3
```

## Shared-state generations

Shared state is source-backed empty at Transformer entry.  Layers0/1 leave compressed-attention shared state empty.  New upstream inputs recompute all later generations:

```text
generation@2 attention input:  1ba0219f5bdf7dc18d27728ec131c02dcd2a106e94aec6494a821453ef0bce7f
compress_kv@2:                 b0344a39d07f2603a5c05828987655cd23a1d5173c7373d1a7522825c2c96d44
index_k@2:                     e619fe853544e92679d9705aa2cef4c3199d02d84c2d946e924f2365fc76fcaf
topk@2:                        baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424

compress_kv@8:                 9b94dcb8f668c171fcbff6fbe62ec04f53289a800aa10909db1ea9df0ff181dd
index_k@8:                     e14360a0265915b6b292ffca18d678e034f0dbf9de05b1fb59e20b2b871fc725

compress_kv@14:                a582748a87b83769b08b0ec08fcd40c4a72407ffb2ce794830e65ab9c0eac5ed
index_k@14:                    51ccd331ca8f5e2bd92471d9c8cc24de087d1623307b192441ba20a0b95a23db

compress_kv@20:                fdb027edf978cebd05926802259c639f106f673c997da85129dbaa5ce3f0df41
index_k@20:                    70b8711d0fdf54876d56ff9bd993b7504f5c78463b1bf912634803c194bff1f5
candidates@20:                 27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1

topk@24:                       f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

All 25 adjacent x/pre_mix carry gates pass for 0->1 through 24->25.

Final stopped output:

```text
x25_out:   703ad30300805df5266836a8a57427409853deb3402973bab703977fab1cea4b
ffn_pre25: 41cd033507e0dcc64e47d1375d3c6c5dea7a90abce1515de4c6ef05bcac33504
```

Safe claim:

> For the bounded prefill fixture, official Transformer entry semantics from token embedding and initial Hyper-Connection state through Blocks 0–25 execute as one connected dataflow. Every adjacent residual/HC carry is connected directly, and compressed-attention shared-state generations from layer 2 onward follow the reviewed source-backed lifecycle.

Non-claims: no layer26+, no Transformer final norm, no logits/ParallelHead end-to-end, no production-scale candidate pruning, no decode/ring/partial compression-group semantics, no world_size > 1 expert parallelism, no full-model correctness, and no performance/fusion claim.
