# Boundary 12b2: ParallelEngramEmbedding + connected Engram@layer1 bounded validation

Status: PASS.

Classification: `official-reference-derived bounded connected Engram@layer1 numerical authority`.

Scope:

```text
tokens [[0,3]]
B=1, S=2, start_pos=0
prefill, world_size=1, engram_mask=None
STOP after post_engram1_h
```

Connected path closed:

```text
tokens
-> fresh NgramHashState
-> embedding / HC repeat
-> Block0
-> Engram@layer1
STOP
```

Block1, Engram@14, sampling, and main_hidden are not executed.

## Hash regression

Hashes are regenerated from token IDs inside the runner. Boundary12b1 is used as a regression authority, not as a tensor source.

- full `engram_hashes` digest: `f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d`
- layer1 selected hashes digest: `8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5`

## pre_engram1_h

Produced by embedding -> HC repeat -> Block0 only, with no prior Engram contribution.

- shape: `[1,2,4,5120]`
- dtype: BF16
- digest: `ba2e6acdac3178115513c81e871f0106541cba2810f7dbc5c4a0c8c309dbf938`
- min: `-7.53125`
- max: `9.375`
- mean: `-0.006265745590280858`

The upstream embedding digest regresses to `e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1`.

## Sparse ParallelEngramEmbedding

Only layer1 official tensors are used. The giant Engram embedding table is not scanned.

- logical rows requested: `48`
- unique rows: `48`
- embedding weight bytes read: `12288`
- embedding scale bytes read: `384`
- full giant embedding table read: `false`
- sparse random-access rows only: `true`
- ordered sparse row payload digest: `13274474cc2cc860c598b6f238e3b2c4c84a9263378d916ed1be9d89879d4fdf`

ParallelEngramEmbedding output:

- shape: `[1,2,24,256]`
- dtype: BF16
- digest: `7a203d55dd97c4d0398bb81b70409d299a246289f60e8a2ecb209305cc180ed4`
- source-order dequantization vs independent explicit FP8/E8M0 decode: byte-exact

## flatten -> wkv seam

- flattened shape: `[1,2,6144]`
- byte-preserving reshape: PASS

## wkv

Read in full because payload is bounded and consumed by this boundary.

- raw `wkv.weight` digest: `44d799d6444a3f728ae96c873c8199bdff49cafbc0b5d703a0bf07be948d0447`
- raw `wkv.scale` digest: `2c395fc76e65fad2f33e5a3d8f532f40dc232754a0e637e42ca89a674e3b44e0`
- flattened input digest: recorded in JSON
- activation FP8 digest: `b6279f2d20dab04c953cc8ed88aeafafaa0fa83bc645515169e3b0d5443517d7`
- activation E8M0 scale digest: `ec3abae604bf0f0888638de597a70b8dd7e6eea88e883523bda23c28023c8136`
- wkv output digest: `0d2fb1ad830f2c0e9c8ee6407bc22dc06fc172a361665ea833880efaacb7907d`
- output shape: `[2,25600]` BF16

Predeclared anchor rows all matched independent FP32 accumulation with BF16 ULP 0.

## key/value split

- key `[1,2,20480]` digest: `a38eeaf361caf47b12802f77d2932c1c5f800421ace0cef810fae69eff0dfb0a`
- key reshaped `[1,2,4,5120]` digest: `a38eeaf361caf47b12802f77d2932c1c5f800421ace0cef810fae69eff0dfb0a`
- value `[1,2,5120]` digest: `3d0617311e3dec2726b699741e812f6271fb641e76259131c394d580baa945e6`

Split/reshape is byte-preserving.

## q/k provenance and gate

- `q_weight` digest: `0aca22a679f1a2479e8065a7b3c64e035b13fe1fa63189644e3556b0a333da56`
- `k_weight` digest: `61152efdabab20ca8c29fe2e087d7de9c5c1684bfef930fa00541efe58fff422`

Gate arithmetic uses the source order from `Engram.forward`. Independent gate reconstruction covers all 8 `(B,S,HC)` gate values with fixed increasing-D FP32 reductions.

- dot max abs difference: `0.0` <= `2e-5`
- gate max abs difference: `0.0` <= `1e-5`

The full gate records are in `artifacts/native-engram-layer1-validation.json`.

## post_engram1_h

- shape: `[1,2,4,5120]`
- dtype: BF16
- digest: `3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9`
- BF16 max ULP vs independent reconstruction: `0`

## I/O accounting

- embedding unique rows requested: `48`
- embedding weight bytes read: `12288`
- embedding scale bytes read: `384`
- wkv weight bytes read: `157286400`
- wkv scale bytes read: `153600`
- q_weight bytes read: `40960`
- k_weight bytes read: `40960`
- total checkpoint bytes read: `157534592`

Diagnostic only; not performance authority.

## STOP

Stopped after `post_engram1_h`.

Next operation would be:

```text
Block1.forward
```

Not executed:

- Block1
- Engram@14
- sampling
- main_hidden

## Safe claim

For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded text fixture `[[0,3]]`, the connected path from token IDs through fresh-prefill NgramHashState, embedding, Block0, sparse ParallelEngramEmbedding lookup, layer1 FP8 wkv projection, source-defined gate arithmetic, and the Engram@layer1 residual update agrees with the independent bounded arithmetic contracts through the final BF16 post-Engram1 residual stream.

This closes Engram@layer1 only. It does not validate Engram@layer14, Block1-and-later connected replay with Engram state, main_hidden, incremental/decode hashing, distributed Engram, or full Transformer.forward behavior.

## Next boundary

Boundary 12b3: Engram@layer14 with upstream connected Engram@1 state.

12b3 must start from the connected residual stream:

```text
tokens -> Block0 -> Engram@1 -> Blocks1..13 -> Engram@14
```

It must not use a no-Engram historical layer14 input.

## Non-claims

- no Engram@layer14 numeric authority
- no Block1-and-later Engram-connected authority
- no main_hidden numeric authority
- no Transformer.forward return correctness
- no incremental/decode NgramHashState correctness
- no False/image-mask DEAD crossing numeric authority
- no distributed Engram correctness
- no SSD/offload semantic qualification
- no full-model correctness
- no performance/production qualification
