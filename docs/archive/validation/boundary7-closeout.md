# Boundary 7 closeout

Status: PASS.

Current integrated numerical authority:

```text
artifacts/native-layer0-25-transformer-entry-validation.json
commit: 4accce4b1c11cd38e5a9f60e948decff99d69b6b
```

Scope:

```text
tokens = [[0,3]]
B = 1
S = 2
start_pos = 0
prefill
world_size = 1
layers = 0..25
STOP before layer26
```

Boundary 7g supersedes 7a-7f as the integrated numerical authority.  Boundary 7a-7f artifacts remain semantic/regional evidence for carry semantics, source-backed producer/consumer ownership, and state lifetime/overwrite contracts, but their intermediate/output digests must not constrain Boundary 7g values because upstream fixture provenance differs.

## Source authority

```text
official inference/model.py SHA256:
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65

official inference/kernel.py SHA256:
1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455

source identity hash method:
raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1
```

Authority discipline: official-reference-derived native arithmetic is the semantic authority.  oMLX / old `deepseek-v41-flash-mlx` evidence is not promoted into the semantic correctness chain.

## Boundary 7g digest summary

```text
embedding:
e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1

initial HC x:
5d0d812064ce2fc25ea149ef428b80354094b902d0212b9077e33d8c3ff087cb

initial pre_mix:
56e94d4f8d9e543d1260250ae1fdc346aea6fc5468f1a7865e77534e2fce98a1

Block1 -> Block2 x:
a9abb13060e3b5986de38d653b16d9e91189f68c8815bbaeba6d8cd647fca8a9

Block1 -> Block2 pre_mix:
393d4e5fd0d9209d0374f1896d2504e008d806ede1ec4b23b4260fba3816b4b3
```

Shared-state generations:

```text
compress_kv@2:  b0344a39d07f2603a5c05828987655cd23a1d5173c7373d1a7522825c2c96d44
index_k@2:      e619fe853544e92679d9705aa2cef4c3199d02d84c2d946e924f2365fc76fcaf
compress_kv@8:  9b94dcb8f668c171fcbff6fbe62ec04f53289a800aa10909db1ea9df0ff181dd
index_k@8:      e14360a0265915b6b292ffca18d678e034f0dbf9de05b1fb59e20b2b871fc725
compress_kv@14: a582748a87b83769b08b0ec08fcd40c4a72407ffb2ce794830e65ab9c0eac5ed
index_k@14:     51ccd331ca8f5e2bd92471d9c8cc24de087d1623307b192441ba20a0b95a23db
compress_kv@20: fdb027edf978cebd05926802259c639f106f673c997da85129dbaa5ce3f0df41
index_k@20:     70b8711d0fdf54876d56ff9bd993b7504f5c78463b1bf912634803c194bff1f5
candidates@20:  27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1
topk@24:        f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f
```

Final stop:

```text
x25_out:
703ad30300805df5266836a8a57427409853deb3402973bab703977fab1cea4b

ffn_pre25:
41cd033507e0dcc64e47d1375d3c6c5dea7a90abce1515de4c6ef05bcac33504
```

## Closed claims, fixture-scoped

For bounded B=1, S=2, start_pos=0, world_size=1, token fixture `[[0,3]]`, and Transformer entry through layer25 only:

- Transformer token embedding entry
- initial HC residual expansion
- identity initial pre_mix
- layers0/1 no-compress window-only path
- Blocks 0..25 connected residual stream
- adjacent HC pre_mix carries 0->1 through 24->25
- bounded Attention paths 0..25
- bounded FFN/MoE paths 0..25
- shared state initially empty
- generation@2 publication
- generation@8 overwrite
- generation@14 overwrite
- generation@20 overwrite + candidates
- generation@24 topk overwrite
- producer->consumer shared-state lifecycle through layer25
- layer-local window KV distinguished from cross-layer shared state

## Still open

- layer26+
- Transformer final norm
- ParallelHead / logits connected end-to-end
- production-scale candidate pruning
- decode start_pos>0
- window KV ring / wrap semantics
- compressed decode partial-group accumulation/publication
- multi-step cache persistence across calls
- world_size > 1 and distributed expert parallel/all-reduce behavior
- Engram
- MTP / DSpark
- long-context numerical behavior
- performance/fusion/production runtime qualification
- full-model correctness

Safe closeout claim:

> For B=1, S=2, start_pos=0, world_size=1 and token fixture [[0,3]], the official-reference-derived native execution from Transformer token embedding and initial Hyper-Connection state through Blocks 0–25 is closed as one connected bounded prefill validation. Residual/HC carry and the reviewed compressed-attention shared-state lifecycle are connected across the validated scope.
