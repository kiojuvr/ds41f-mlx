# Boundary 12b1: NgramHashState bounded numerical validation

Status: PASS.

Classification: `official-reference-derived bounded NgramHashState numerical authority`.

Scope only:

- tokens: `[[0,3]]`
- `B=1`, `S=2`
- `start_pos=0`
- fresh prefill state/cache
- `world_size=1`
- `engram_mask=None`
- stop immediately after `engram_hashes` generation

No Engram checkpoint embedding weights or Engram arithmetic tensors are read in this boundary.

## Authority baseline

Boundary12b0 contract: `artifacts/engram-semantic-foundation-contract.json`.

Reconfirmed source/tokenizer identities:

- `inference/engram.py`: `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`
- `NgramHashState` lines 129-184: `698c03c890bc8e1b55e93cd540a678f68845d7541c9da1d3d8e9a41f1cdde64d`
- `tokenizer.json`: `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`
- `tokenizer_config.json`: `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`

## Token-map authority

The full tokenizer-derived token map was constructed by a source-order implementation and independently reconstructed without calling the source helper.

Recorded identity:

- tokenizer length: `129280`
- token_map shape: `[129280]`
- dtype: `int64`
- digest: `26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496`
- compressed vocab size: `99092`
- unique compressed ids: `99092`
- `engram_pad_id`: `2`
- `token_map[0]`: `0`
- `token_map[3]`: `3`
- `token_map[2]` / compressed pad id: `2`

The independent token-map digest matches exactly.

## Static constants

Hard-gated from Boundary12b0/local source:

- layer order: `[1,14]`
- layer hash index: `1 -> 0`, `14 -> 1`
- max ngram size: `4`
- heads: `8`
- n_hash_cols: `24`
- multipliers:
  - layer1: `[76632096046245, 4839876093313, 35959672319349, 73987337458391]`
  - layer14: `[67716810739261, 51510806800915, 30921347202721, 82619226485591]`

Primes and offsets are recorded fully in `artifacts/native-ngram-hash-state-validation.json` with digests.

## Signed int64 semantics

The independent oracle uses explicit two's-complement signed-int64 wrapping for multiplication and XOR. A tiny overflow self-check with actually overflowing products verifies agreement with NumPy int64 wrapping for the operations used by the hash. Python arbitrary-precision products are not used silently for hash arithmetic.

## Fixture execution evidence

Input:

```text
input_ids = [[0,3]]
compressed = [[0,3]]
```

Fresh cache relevant slice after write:

```text
cache[0,0:2] = [0,3]
```

No `DEAD=-1` value is written because `engram_mask=None`.

History tensor after source blocked/pad rules:

```text
shape [1,2,4]
[[[0,2,2,2],
  [3,0,2,2]]]
```

The independent history reconstruction matches exactly.

## Final hash evidence

Final `engram_hashes`:

- shape: `[1,2,2,24]`
- dtype: `int64`
- digest: `f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d`

Layer digests:

- layer1 selected hashes `[1,2,24]`: `8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5`
- layer14 selected hashes `[1,2,24]`: `33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80`

Source-order output and independent explicit-loop output are byte-exact. All final hashes are within their legal per-column prime buckets and layer table bounds.

`engram_mask=None` vs fresh all-True mask also matches exactly for this text-only fixture.

## Authority promotion

Boundary12b1 validates, for this bounded scope only:

- token compression for the pinned tokenizer
- fresh-prefill cache/history construction
- 2/3/4-gram signed-int64 rolling hash arithmetic
- layer1 and layer14 hash publication
- final `[1,2,2,24]` Engram hash tensor

Safe claim:

For the pinned DeepSeek-V4.1-Flash tokenizer/config/source and the bounded text fixture `[[0,3]]` at `B=1`, `S=2`, `start_pos=0` with a fresh `NgramHashState`, the source-derived and independently reconstructed token compression, history construction, signed-int64 rolling hash arithmetic, prime bucketing, and per-column offset publication agree exactly through the final `[1,2,2,24]` Engram hash tensor.

This validates only the bounded fresh-prefill hash-state fixture. It does not validate incremental/decode cache behavior, masked image/dead-token crossing, Engram embedding lookup, Engram.forward numerical behavior, or connected residual-stream correctness.

## Next boundary

Boundary 12b2: `ParallelEngramEmbedding + Engram@layer1`.

12b2 may use the Boundary12b1 layer1 selected hashes `[1,2,24]` as authority input. It should avoid full 98GB table loads and random-access only the referenced official safetensors rows where possible.

## Non-claims

- no incremental/decode NgramHashState correctness
- no False/image-mask DEAD crossing numeric authority
- no ParallelEngramEmbedding numeric authority
- no Engram.forward numeric authority
- no Engram checkpoint tensor arithmetic
- no connected Engram residual-stream correctness
- no main_hidden numeric authority
- no Transformer.forward return correctness
- no distributed Engram correctness
- no SSD/offload semantic qualification
- no full-model correctness
- no performance/production qualification
