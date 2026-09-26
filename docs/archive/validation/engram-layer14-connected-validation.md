# Boundary 12b3: Engram@layer14 with upstream connected Engram@layer1 state

Status: PASS.

Classification: `official-reference-derived bounded connected Engram@layer14 numerical authority with upstream connected Engram@layer1 state`.

Scope:

```text
tokens [[0,3]]
B=1, S=2, start_pos=0
prefill, world_size=1, engram_mask=None
STOP after post_engram14_h and preservation of ffn_pre13/shared state
```

Connected path closed:

```text
tokens [[0,3]]
-> NgramHashState
-> embedding / initial HC
-> Block0
-> Engram@1
-> Blocks1..13
-> Engram@14
STOP before Block14
```

No historical/no-Engram layer14 input is used.

## Hash and Engram@1 regressions

Hashes are regenerated from token IDs in the runner:

- full `engram_hashes`: `f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d`
- layer1 selected hashes: `8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5`
- layer14 selected hashes: `33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80`

Engram@1 is regenerated in the same execution:

- `post_engram1_h`: `3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9`

## Blocks1..13 connected carries

After Block0, Engram@1 modifies only `h`; `pre_mix` remains Block0 `ffn_pre`. Block1 consumes:

```text
x = post_engram1_h
pre_mix = ffn_pre0
```

Blocks1..13 then execute on the Engram-connected residual trajectory. The artifact records per-layer:

- x/pre_mix inputs
- attention input/output
- x_after_attn
- MoE input/output
- x_out
- ffn_pre

All adjacent x carries and pre_mix carries are exact.

## Shared attention state

One persistent shared attention state object is used through Blocks0..13 and is not reset after Engram@1.

Connected generation digests:

Layer2:

- `compress_kv`: `838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae`
- `index_k`: `0a02a69899257836865cacec8a1a6d0d1bfb590cf07b8a3f602300e02ee7875d`
- `topk_idxs`: `baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424`

Layer8:

- `compress_kv`: `29f479332ce47e4429d7c46d4fc1a952efa16f74ca110b1a9a025f032a9e2213`
- `index_k`: `cd1c51ca26f6404bde0eb2230908af462e9440a923885236f35ba92df6542d87`
- `topk_idxs`: `baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424`

These are new Engram-connected digests, not regressions to old no-Engram numerical values.

## pre-Engram14 seam

`pre_engram14_h = Block13 x_out` on the Engram@1-connected trajectory:

- shape: `[1,2,4,5120]`
- dtype: BF16
- digest: `063e002c44f246805e4630678b705cb7987c9064693605fba9297b385e8bb931`
- min: `-145.0`
- max: `516.0`
- mean: `-0.1582478321623057`

`pre_block14_pre_mix = Block13 ffn_pre`:

- shape: `[1,2,4]`
- dtype: FP32
- digest: `950063b739a1c4bb0e75c14850d1cb4921ce27cc366878ebe0b9d1d92e9c1bec`
- min: `0.0002218989102402702`
- max: `0.9330078959465027`
- mean: `0.23340286824350187`

Engram@14 changes `h` only. The `ffn_pre13` pre_mix carry is preserved for future Block14.

## Shared state across Engram@14

The shared attention state before and after Engram@14 is recorded and all fields are unchanged. Engram.forward does not mutate `compress_kv`, `index_k`, `candidates`, or `topk_idxs`.

## Layer14 sparse embedding

Layer14 selected hashes contain 48 logical rows and 48 unique rows. The giant layer14 embedding table is not scanned.

- weight bytes read: `12288`
- scale bytes read: `384`
- full table read: `false`
- sparse random-access rows only: `true`
- ordered sparse-row payload digest: `8d1bbfa560fb056ce548c6044bc5a19bab573fedd33dc0b5ae6d3ecfe2301470`

Layer14 ParallelEngramEmbedding output:

- shape: `[1,2,24,256]`
- dtype: BF16
- digest: `15536731a8a4241b2a7c525111fe0aac17b444c57b68af2f12a1b36bad23534f`
- independent explicit byte-decoder match: byte-exact

## Layer14 wkv

- raw `wkv.weight` digest: `9abdfe920afa0528c9987a7d01450f5022cc765fa8b115fb132597ddb25cb429`
- raw `wkv.scale` digest: `8c8d7aa1ccc5a6c24f3e454de2e884db32890874263567889a0e182b3584642a`
- activation FP8 digest: `fb455db73774caf9124ff2f2093be89338a86ef35ebf8ab325cf7ff04898b3bc`
- activation E8M0 scale digest: `284e9f6e7e9e32e47c4fd049da557ae3c72854e843ffc7f18e451837e0f29339`
- output digest: `4a8e2062d543cba990a51e49b5106be2820b6bcd8cf0adb60f1d6565681aa7ec`
- predeclared anchor rows max BF16 ULP: `0`

## Key/value split

- key `[1,2,20480]`: `409b22feb44ea7b1154352677c77f185410e4eec1c3bb79886e8d23cf6bda169`
- reshaped key `[1,2,4,5120]`: same digest, byte-preserving
- value `[1,2,5120]`: `bb4641ddceb3abfe82cbb2af9197b222798bff8e2539d2faf4efb06dabb10142`

## q/k and gate

Layer14 q/k digests:

- `q_weight`: `a49e74aa912b88f56df4183862c729bc8539ce4a7b76e6107da858b59e782e28`
- `k_weight`: `fa3128a62fef3630e32a0953e2dc32eb6f6870492ba447bb1877ff37f4f6cd93`

All 8 gates are reconstructed independently with explicit increasing-D FP32 reductions.

- dot max abs diff: `0.0` <= `2e-5`
- gate max abs diff: `0.0` <= `1e-5`

## post_engram14_h

- shape: `[1,2,4,5120]`
- dtype: BF16
- digest: `ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd`
- min: `-145.0`
- max: `528.0`
- mean: `-0.1554516457952559`
- BF16 max ULP vs independent reconstruction: `0`

This is the primary handoff to Boundary12b4.

## Block14 handoff

Future Block14 inputs are fixed as:

```text
x = post_engram14_h
pre_mix = ffn_pre13
shared_state = persistent state from Blocks0..13
```

Machine-readable handoff:

- `post_engram14_h_digest`: `ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd`
- `block14_pre_mix_digest`: `950063b739a1c4bb0e75c14850d1cb4921ce27cc366878ebe0b9d1d92e9c1bec`
- `pre_mix_unchanged_across_engram14`: `true`

## I/O accounting

- layer14 embedding logical rows: `48`
- unique rows: `48`
- embedding weight bytes: `12288`
- embedding scale bytes: `384`
- wkv weight bytes: `157286400`
- wkv scale bytes: `153600`
- q_weight bytes: `40960`
- k_weight bytes: `40960`
- total layer14 Engram checkpoint bytes read: `157534592`

No giant table scan.

## STOP

Stopped after `post_engram14_h` and after preserving `ffn_pre13` plus shared attention state.

Not executed:

- Block14
- Block15+
- main_hidden
- sampling

Next operation:

```text
Block14.forward(h=post_engram14_h, pre_mix=ffn_pre13, shared_state=persistent_state_from_Blocks0_13)
```

## Safe claim

For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded `[[0,3]]` prefill fixture, the connected residual trajectory from the previously validated Engram@layer1 update through Blocks1-13 reaches Engram@layer14 without resetting Hyper-Connection carries or compressed attention shared state.

The actual connected pre-Engram14 residual, official layer14 sparse Engram lookup, FP8 wkv projection, source-defined gate arithmetic, and residual update agree with the bounded independent arithmetic contracts through the final BF16 post-Engram14 residual stream.

This does not yet validate Block14-or-later execution on the Engram-modified trajectory, main_hidden, Transformer.forward return behavior, incremental/decode Engram state, or full-model correctness.

## Next boundary

Boundary 12b4: replay the connected model path with both Engram insertions from post-Engram14 through Block39 / deterministic logits seam.

12b4 must carry forward the same connected execution state:

```text
post_engram14_h + ffn_pre13 + shared attention state after Block13
```

Block14 must generate its KV/index state from the Engram14-modified hidden; old no-Engram generation14 digests must not be used as expected values.

## Non-claims

- no Block14-and-later Engram-connected authority
- no main_hidden numeric authority
- no Transformer.forward return correctness
- no incremental/decode NgramHashState correctness
- no False/image-mask DEAD crossing numeric authority
- no distributed Engram correctness
- no SSD/offload semantic qualification
- no full-model correctness
- no performance/production qualification
