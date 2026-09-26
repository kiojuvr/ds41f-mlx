# Boundary13b: first incremental NgramHashState numerical validation

Status: **PASS**.

Scope is strictly:

```text
Boundary13a prefill snapshot
-> input_ids [[15]], start_pos=2, S=1
-> NgramHashState cache position2 update
-> history gather
-> 2/3/4-gram signed-int64 hashing
-> layer1/layer14 hash publication
STOP before embedding
```

No first-decode `Transformer.forward`, embedding, `Engram.forward`, Block0, Attention, logits, sampling, main_hidden, or return packaging is executed.

## Source identities

- `engram.py`: `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`
- Boundary13 broad `NgramHashState` span lines 118-184: `4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe`
- Narrow `NgramHashState.forward` path lines 146-184: `ad57108ba22f67076c84fc11646ab730a783f27e17d8e48184c5082205e145a3`

Boundary13a manifest digest hard-regressed:

```text
311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301
```

Boundary13a state is regenerated in memory; no serialized cache tensor artifact is consumed.

## Token map

Two independent paths agree exactly.

```text
shape: [129280]
dtype: int64
digest: 26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496
compressed_vocab_size: 99092
compressed pad id: 2
token_map[0]: 0
token_map[3]: 3
token_map[15]: 15
```

## Prefill cache regression

Boundary13a Ngram visible prefix remains:

```text
cache[0,0:2] = [[0,3]]
digest: 96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a
```

The original Boundary13a snapshot object/manifest is not mutated. Boundary13b operates on a decode working-state clone.

## Position2 write

For decode input `[[15]]`, `token_mask=None`:

```text
compressed_decode_token = 15
cache[0,2] = 15
cache[0,0:3] = [[0,3,15]]
digest: 04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c
no_dead_written = true
```

Positions 0 and 1 remain unchanged and keep the Boundary13a prefix digest.

## Cache read-position evidence

Source-order history gather reads only:

```text
{0,1,2}
```

No unspecified capacity at position `>=3` is read.

Shift evidence:

```text
shift0: gather pos2, raw 15, blocked false, final 15
shift1: gather pos1, raw 3,  blocked false, final 3
shift2: gather pos0, raw 0,  blocked false, final 0
shift3: gather pos0 per source clamp, blocked true by sequence boundary, final pad 2
```

Final semantic history:

```text
[[[15,3,0,2]]]
shape: [1,1,4]
digest: cf5abf85fe2bdf166063c0b32e83f38c962d3b19bed561606d97f1271b8f378f
```

Source-order and independent history reconstruction are byte-exact.

## Event order

```text
compress token15
< write cache position2
< construct positions
< history gather
< hash arithmetic
```

This proves shift0 observes the newly written decode token.

## Hash constants and signed-int64 semantics

Regressed from Boundary12b0/12b1:

```text
Engram layer order: [1,14]
max_ngram_size: 4
n_heads: 8
n_hash_cols: 24
multipliers digest: 7345f44ec93e965df6581af59c78a5dd6efba7be2e855486450a6594d7ebd7c6
primes digest: ed542f58c4b3593c4120e899620231785b695dfc61bcc3d04b61b113abd9cf9e
offsets digest: edf229962df441c86f9c20cbc126579dd548421e231250b146ee6f5bea0b2f8b
```

Signed int64 multiply-with-wrap, XOR over wrapped bits, and positive-prime modulo self-check PASS.

## Published hashes

Full output:

```text
shape: [1,1,2,24]
dtype: int64
digest: 09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530
```

Layer1 values:

```text
[[[10394440, 25597646, 41525500, 56949077, 72088738, 87803199,
   103660714, 117890322, 133679554, 149048834, 164909235,
   179450177, 194623690, 210212541, 225598121, 241122337,
   264857356, 275121728, 291772318, 316098564, 326982803,
   351309675, 366718203, 378263177]]]
```

Layer1 digest:

```text
4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373
```

Layer14 values:

```text
[[[14784224, 31005239, 34470870, 63449341, 69938491, 90531711,
   100952646, 114166730, 131094349, 150341785, 173304480,
   179798762, 202529617, 215756271, 232472038, 248712503,
   267471672, 277953512, 294997123, 309159487, 329088915,
   350071831, 354793593, 378402873]]]
```

Layer14 digest:

```text
5d93f09bfecb5b8a3a722603bb5e1cd44710849cc4729223df92953386bf3ad7
```

Source-order and independent explicit-loop signed-int64 hash implementations agree byte-exact for all 48 IDs. Bucket legality and layer table bounds PASS.

## Equivalence and isolation

- `token_mask=None` and `token_mask=[[True]]` from independent clones match exactly for compressed token, cache slice, history, layer hashes, and full hash.
- Replay from an independent clone of the same Boundary13a prefill snapshot is exact.
- Boundary13a source-visible snapshot remains unchanged.

## Other persistent state non-mutation

Unchanged by Boundary13b:

```text
all 40 window_kv visible prefixes
compressed_kv visible prefixes
Indexer k_cache visible prefixes
Compressor partial-state classifications
```

## STOP flags

```text
embedding_executed = false
Block0_executed = false
Attention_executed = false
Engram_forward_executed = false
generation_loop_control_advanced = false
decode_forward_executed = false
```

## Artifact policy

The artifact contains bounded evidence only: `cache[0,0:3]`, history `[1,1,4]`, and final 48 hash IDs. It does not serialize mutated cache tensors as future-boundary oracle payloads.

```text
incremental_state_tensor_artifact_injection_for_future_boundaries = false
```

## Non-claims

No first-decode embedding authority; no decode `Engram.forward`; no decode window KV, rotary/query, compressed/index/candidate/topk, Block, logits, sample, or main_hidden authority; no wraparound/capacity, False/image-mask DEAD crossing, world_size>1, production, or performance qualification.
