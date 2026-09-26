# Layer0-36 connected prefill contract (Boundary 8c)

Artifact: `artifacts/native-layer0-36-connected-prefill-validation.json`  
Runner: `tools/run_native_layer0_36_connected_prefill_validation.py`  
Status: PASS

## Scope

For B=1, S=2, `start_pos=0`, `world_size=1`, token fixture `[[0,3]]`:

```text
Transformer entry -> Blocks 0..36
STOP before Block37
```

Boundary 8c starts from token IDs and reruns the full connected execution. It does not inject Boundary8b `x32_out`, `ffn_pre32`, or shared-state snapshots.

## Source-confirmed region

Layers33-35 consume `compress_kv@20` and `topk@32`. Layer36 is the final configured index source for the local config. It consumes `compress_kv@20`, `index_k@20`, and `candidates@20`, then publishes/overwrites `topk@36`.

No new `compress_kv` or `index_k` owner exists after layer20 in this checkpoint.

## Boundary8b regression

Boundary8b remains exact through Block32. The runner gates embedding, initial HC state, generation@2/@8/@14/@20, `topk@24`, `topk@28`, `topk@32`, `x32_out`, and `ffn_pre32`.

## New carries

The following adjacent carries are exact:

```text
32->33 x / pre_mix
33->34 x / pre_mix
34->35 x / pre_mix
35->36 x / pre_mix
```

## Shared-state timeline

```text
compress_kv@20 -> consumed through 36
index_k@20     -> consumed by 24, 28, 32, 36
candidates@20 -> consumed by 24, 28, 32, 36
topk@24       -> consumed by 25, 26, 27
topk@28       -> consumed by 29, 30, 31
topk@32       -> consumed by 33, 34, 35
topk@36       -> published for future 37, 38, 39
```

Digest equality between topk generations is not treated as generation identity.

## Authority relationship

- Boundary7g: closed authority for exact scope 0..25
- Boundary8a: closed regression/subscope authority for exact scope 0..28
- Boundary8b: closed regression/subscope authority for exact scope 0..32
- Boundary8c: current integrated numerical authority for exact scope 0..36

## Non-claims

No layers37-39, no Transformer final norm, no connected ParallelHead/logits, no production-scale candidate pruning, no decode, no ring/wrap or partial compressed decode semantics, no multi-call cache persistence, no world_size > 1 distributed behavior, no Engram, no MTP/DSpark, no long-context qualification, no full-model correctness, and no performance/fusion/production qualification.
