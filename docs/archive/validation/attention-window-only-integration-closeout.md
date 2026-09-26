# Layer-0 window-only Attention integration closeout

This closeout composes already validated Boundaries 1-4 into one native dataflow. It adds no new model semantics.

## Scope

```text
layer = 0
B = 1
S = 2
tokens = [0, 3]
start_pos = 0
compress_ratio = 0
world_size = 1
```

Integrated path:

```text
x
-> Q prelude
-> window-KV prelude
-> sparse_attn
-> inverse rotary
-> grouped wo_a
-> wo_b
-> Attention output
STOP before Block / HC
```

Authority fixtures reused:

- Boundary 1 Q prelude: `artifacts/attention-q-prelude-official-reference-fixture.json`
- Boundary 2 window-KV prelude: `artifacts/window-kv-prelude-official-reference-fixture.json`
- Boundary 3 sparse attention: `artifacts/sparse-attn-official-reference-fixture.json`
- Boundary 4 output projection: `artifacts/attention-output-projection-official-reference-fixture.json`

Validation artifact:

```text
artifacts/native-attention-window-only-integration-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_attention_window_only_integration.py \
  --out artifacts/native-attention-window-only-integration-validation.json
```

## Gates

The native script executes a single Python-orchestrated native primitive dataflow and compares every boundary digest to existing validated artifacts:

| Stage | Result |
| --- | --- |
| Q rotary output | bit-exact |
| window-KV | bit-exact |
| topk indices | int32 bit-exact |
| sparse-attn output | bit-exact |
| inverse rotary | bit-exact |
| grouped `wo_a` | bit-exact |
| final `wo_b` / Attention output | bit-exact |

Final Attention output digest:

```text
49afde916b9fb6eb507f9e56159312f5e813ff9da99897ba6b72720e884c9820
```

Native Metal-backed primitives are executed for FP8 linears, RMSNorm, rotary helpers, act-quant, sparse attention, grouped BF16 `wo_a` linears, and final FP8 `wo_b`.

## Non-claims

This closeout does not validate:

- compressed KV, Compressor, Indexer, or candidate selection;
- Block or HC residual integration;
- MoE;
- logits, full layer, full prefill, or full model correctness;
- performance or fusion.

## Next planning note

After this pass, the two remaining high-risk semantic areas are:

1. Boundary 5: compressed KV / Indexer / candidate semantics;
2. Boundary 6: Block / Hyper-Connections.

Boundary 5 should normally come first if the next goal is attention coverage beyond the window-only path. Boundary 6 should come first only if the immediate goal is validating HC/residual orchestration independent of compressed attention state.
