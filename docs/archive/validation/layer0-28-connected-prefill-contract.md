# Layer0-28 connected prefill contract (Boundary 8a)

Artifact: `artifacts/native-layer0-28-connected-prefill-validation.json`  
Runner: `tools/run_native_layer0_28_connected_prefill_validation.py`  
Status: PASS

## Scope

For B=1, S=2, `start_pos=0`, `world_size=1`, token fixture `[[0,3]]`:

```text
Transformer entry -> Blocks 0..28
STOP before Block29
```

This endpoint was chosen from local source/config review, not from the initial candidate scope. The reviewed checkpoint has no layer26 KV/index source. Layer28 is the first post-layer25 source boundary and publishes a new `topk_idxs` generation only.

## Boundary7 regression

Boundary7g remains the closed authority for exact scope 0..25. Boundary8a reruns from token IDs in one connected execution and gates the Boundary7 subset exactly:

- embedding output: PASS
- initial HC `x`: PASS
- initial `pre_mix`: PASS
- generation@2, @8, @14, @20 and topk@24: PASS
- `x25_out`: PASS
- `ffn_pre25`: PASS

No Boundary7 tensors or shared-state snapshots are injected.

## New seams

The main new production carry is exact:

```text
Block25 x_out     == Block26 input x
Block25 ffn_pre   == Block26 incoming pre_mix
```

Adjacent carries 26->27 and 27->28 are also exact for both residual stream and HC `pre_mix`.

## Shared-state lifetime

A single logical shared state is carried from Transformer entry through Block28:

```text
generation@2 -> generation@8 -> generation@14 -> generation@20
candidate generation@20
topk generation@24
topk generation@28
```

Layer26 and layer27 consume `compress_kv@generation@20` and `topk_idxs@generation@24`. Layer28 consumes `compress_kv@generation@20`, `index_k@generation@20`, and `candidates@generation@20`, then overwrites `topk_idxs` with generation@28. Window KV remains layer-local.

## Non-claims

No layers beyond Block28, no final norm, no ParallelHead/logits, no decode, no cache persistence across calls, no world_size > 1/distributed behavior, no Engram/MTP/DSpark, no long-context qualification, no full-model correctness, and no performance/fusion/production qualification.
