# Layer0-39 all Blocks connected prefill contract (Boundary 8d)

Artifact: `artifacts/native-layer0-39-all-blocks-connected-prefill-validation.json`  
Runner: `tools/run_native_layer0_39_all_blocks_connected_prefill_validation.py`  
Status: PASS

## Scope

For B=1, S=2, `start_pos=0`, `world_size=1`, token fixture `[[0,3]]`:

```text
Transformer entry -> Blocks 0..39
STOP immediately after Block39 returns x39_out, ffn_pre39
```

The runner starts from token IDs and executes one connected dataflow. It does not inject Boundary8c `x36_out`, `ffn_pre36`, `topk@36`, or any shared-state snapshot.

## Transformer post-loop boundary

Reviewed `Transformer.forward` order after the Block loop:

```text
last Block returns h, pre_mix
-> h = layer.hc_pre(h, pre_mix)
-> logits = self.head(self.norm(h))
-> output_ids = sample(logits, self.temperature)
-> main_hidden assembly
-> return output_ids, logits, main_hidden
```

Boundary 8d stops before the first post-loop operation, `layer.hc_pre(h, pre_mix)`. It does not validate post-loop HC collapse, final RMSNorm, ParallelHead, logits, or sampling.

## Source-confirmed final region

`num_hidden_layers == 40`, so Block39 is the final Block. Layers37-39 have `compress_ratio=1`, are not KV or index sources, and consume `compress_kv@20` plus `topk@36`. They do not consume `index_k@20` or `candidates@20` on their non-indexer branch.

## Boundary8c regression

Boundary8c remains exact through Block36. The runner gates embedding, initial HC state, generation@2/@8/@14/@20, `topk@24`, `topk@28`, `topk@32`, `topk@36`, `x36_out`, and `ffn_pre36`.

## Carries

New carries are exact:

```text
36->37 x / pre_mix
37->38 x / pre_mix
38->39 x / pre_mix
```

All 39 adjacent Block seams, `0->1` through `38->39`, are exact in the same execution.

## Final shared-state lifecycle

```text
compress_kv@20 -> persists and is consumed through layer39
index_k@20     -> actual consumers: 24, 28, 32, 36; preserved at final snapshot
candidates@20 -> actual consumers: 24, 28, 32, 36; preserved at final snapshot
topk@36       -> actual consumers: 37, 38, 39; no later Block overwrite
```

The final shared object snapshot is a Block-loop boundary observation only; no claim is made that Transformer post-loop output processing consumes it.

## Authority relationship

- Boundary7g: closed subscope authority 0..25
- Boundary8a: closed subscope authority 0..28
- Boundary8b: closed subscope authority 0..32
- Boundary8c: closed subscope authority 0..36
- Boundary8d: current integrated numerical authority for Transformer entry through all Blocks 0..39 only

Boundary8d is not full Transformer-output authority.

## Non-claims

No Transformer post-loop HC/output processing, no final norm, no ParallelHead/logits, no production-scale candidate pruning, no decode, no ring/wrap or partial compressed decode semantics, no multi-call cache persistence, no world_size > 1 distributed behavior, no Engram, no MTP/DSpark, no long-context qualification, no full Transformer correctness, no full-model correctness, and no performance/fusion/production qualification.
