# Boundary13a: bounded prefill-end persistent-state container / numerical snapshot

Status: **PASS**.

Boundary13a promotes only a bounded prefill-end persistent-state snapshot for the Engram-connected native fixture:

```text
tokens = [[0,3]]
B = 1
S = 2
start_pos = 0
world_size = 1
engram_mask = None
```

No first-decode `Transformer.forward` is executed.

## Container

Reusable validation helper: `tools/native_decode_session_state.py`.

The in-memory validation container is `NativeDecodeSessionState` with explicit partitions:

```text
official_model_persistent_state
generation_loop_control
target_runtime_session_state
```

It is a validation helper, not a production API and not an mmap/offload/session-performance policy.

## Source identity gates

Boundary13 source identities are hard-regressed:

- `model.py`: `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`
- `engram.py`: `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`
- `config.json`: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`
- `generate.py`: `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0`

Boundary13 canonical spans for NgramHashState, Compressor, Indexer, Attention, SharedAttentionRuntime, Transformer.forward, and the generate loop are also regressed in the artifact/checker.

## Official-model persistent snapshot

Only source-proven persistent state is represented:

- `NgramHashState.cache`
- all 40 per-layer `Attention.window_kv_cache` visible slots
- KV-source `Attention.compress_kv_cache` for layers `[2,8,14,20]`
- owner `Indexer.k_cache` for layers `[2,8,14,20]`
- ratio>1 `Compressor.kv_state` / `score_state` allocations, classified by first-read semantics

No whole unused cache capacity is hashed.

## Ngram visible slice

```text
allocated cache shape: [4,4096]
dtype: int64
valid batch range: [0,1]
valid absolute position range: [0,2)
valid slice values: [[0,3]]
valid slice digest: 96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a
```

Unused `torch.empty` capacity is unspecified and excluded from authority.

## Window KV snapshot

All 40 layers are captured. For each layer:

```text
allocated shape: [4,128,512]
dtype: BF16(uint16)
visible absolute positions: [0,1]
ring slots: [0,1]
visible slice shape: [1,2,512]
visible digest == producer window_kv digest
```

Layer 0 example digest:

```text
11432f637842b3698f05ce050c563a38130fe9ae6ba822755ca0e8eeb1e50592
```

All per-layer digests are in `artifacts/native-prefill-end-persistent-state-validation.json`.

## Compressed KV owner snapshots

KV source topology is exactly `[2,8,14,20]`.

```text
layer 2, ratio 2, valid compressed range [0,1): 838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae
layer 8, ratio 2, valid compressed range [0,1): 29f479332ce47e4429d7c46d4fc1a952efa16f74ca110b1a9a025f032a9e2213
layer 14, ratio 2, valid compressed range [0,1): e2c045f500a5776d647fc22ebc99faf2bb2f7131ba2e332e256fed3aadfa5edb
layer 20, ratio 1, valid compressed range [0,2): 17eacfb671aa2a37c302b5e6f097346b7951f7bd3ff23bd0ed5193c00695900d
```

These digests are over visible prefixes only.

## Indexer owner snapshots

Actual persistent key owners are layers `[2,8,14,20]`; not every index-source layer owns independent `k_cache`.

```text
index_k@2:  0a02a69899257836865cacec8a1a6d0d1bfb590cf07b8a3f602300e02ee7875d
index_k@8:  cd1c51ca26f6404bde0eb2230908af462e9440a923885236f35ba92df6542d87
index_k@14: 82f4b94a61be422936f51e142f786be31a37df14412ac700ef804a98f56f5c85
index_k@20: a2ec36dab41f04ef5f7ab63fabcaa823a22232b9a2fdb3d28d990dfa75e4a896
```

## Compressor partial-state classification

For ratio>1 source layers `[2,8,14]`, prefill `S=2` leaves no semantically required partial bytes for first decode step0. At decode absolute position 2, official source overwrites slot `2 % 2 == 0` before completing/reading a new ratio2 group.

Therefore ratio>1 `kv_state` / `score_state` are represented as persistent allocations but their prefill-end bytes are classified as:

```text
slot0: OVERWRITTEN_BEFORE_READ
slot1: UNOBSERVED
read_by_decode_step0_bytes: 0
visible_digest_recorded: false
```

No meaningless overwrite-before-read bytes are promoted.

## Explicit call-local exclusions

The persistent snapshot excludes:

```text
h
pre_mix
main_hiddens
candidates
topk_idxs
shared_attn.compress_kv
shared_attn.index_k
shared_attn.candidates
shared_attn.topk_idxs
```

Machine-readable gates:

```text
call_local_values_excluded_from_persistent_snapshot = true
candidates_in_persistent_state = false
topk_idxs_in_persistent_state = false
```

`shared_attn.*` physical attributes may exist after prefill, but their semantic cross-call value requirement is false.

## Generation-loop handoff

Deterministic control handoff only:

```text
prefill output token: 15
next_input_ids: [[15]]
next_start_pos: 2
next_sequence_length: 1
```

This is generation-loop control state, not official model cache state.

## Target runtime RNG partition

Boundary12e stochastic runtime handoff remains out-of-band:

```text
next_session_key digest:
175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4
```

It is not official model persistent state, not a deterministic decode fixture requirement, and not a `Transformer.forward` return member. No new RNG draw is generated.

## Snapshot manifest digest

```text
311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301
```

This is a digest over canonical sorted JSON metadata for the semantic bounded snapshot. It is not a hash of giant unused cache allocations.

## Upstream prefill regressions

Unchanged:

```text
full Ngram hash: f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d
post_engram1_h: 3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9
post_engram14_h: ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd
final logits: 7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd
output_ids: [15]
```

## Non-claims

No first-decode numerical execution; no incremental Ngram hash authority; no decode window KV update authority; no decode compressed/index authority; no decode Engram authority; no first-decode logits/sample/main_hidden authority; no wraparound/capacity, long-context, world_size>1, production, or performance qualification.
