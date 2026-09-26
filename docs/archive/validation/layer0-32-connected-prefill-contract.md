# Layer0-32 connected prefill contract (Boundary 8b)

Artifact: `artifacts/native-layer0-32-connected-prefill-validation.json`  
Runner: `tools/run_native_layer0_32_connected_prefill_validation.py`  
Status: PASS

## Scope

For B=1, S=2, `start_pos=0`, `world_size=1`, token fixture `[[0,3]]`:

```text
Transformer entry -> Blocks 0..32
STOP before Block33
```

Boundary 8b starts from token IDs and reruns the full connected execution. It does not inject Boundary8a `x28_out`, `ffn_pre28`, or shared-state snapshots.

## Source-confirmed region

Layers29-31 are consumers of `compress_kv@20` and `topk@28`. Layer32 is the next configured index source after layer28. It consumes `compress_kv@20`, `index_k@20`, and `candidates@20`, then publishes/overwrites `topk@32`.

No new `compress_kv` or `index_k` owner exists after layer20 in this checkpoint.

## Boundary8a regression

Boundary8a remains exact through Block28. The runner gates:

- embedding output
- initial HC `x`
- initial `pre_mix`
- generation@2, @8, @14, @20
- `topk@24`
- `topk@28`
- `x28_out`
- `ffn_pre28`

## New carries

The following adjacent carries are exact:

```text
28->29 x / pre_mix
29->30 x / pre_mix
30->31 x / pre_mix
31->32 x / pre_mix
```

## Shared-state timeline

```text
compress_kv@20 -> consumed through 32
index_k@20     -> consumed by 24, 28, 32
candidates@20 -> consumed by 24, 28, 32
topk@24       -> consumed by 25, 26, 27
topk@28       -> consumed by 29, 30, 31
topk@32       -> published for future 33+
```

Digest equality between topk generations is not treated as generation identity.

## Authority relationship

- Boundary7g: closed authority for exact scope 0..25
- Boundary8a: closed regression/subscope authority for exact scope 0..28
- Boundary8b: current integrated numerical authority for exact scope 0..32

## Non-claims

No layers33+, no Transformer final norm, no connected ParallelHead/logits, no production-scale candidate pruning, no decode, no ring/wrap or partial compressed decode semantics, no multi-call cache persistence, no world_size > 1 distributed behavior, no Engram, no MTP/DSpark, no long-context qualification, no full-model correctness, and no performance/fusion/production qualification.
