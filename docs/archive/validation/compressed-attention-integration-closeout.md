# Boundary 5 closeout: compressed Attention bounded prefill path

Status: PASS for the bounded layer-24 prefill path only.

## Scope validated

Configuration was read from the pinned local checkpoint. The first candidate-consumer layer is layer 24, with `compress_ratio = 1`, `B = 1`, `S = 2`, `start_pos = 0`, prefill, `world_size = 1`.

Validated chain:

| Boundary | Scope | Artifact |
|---|---|---|
| 5a | Compressor + compressed KV publication | `artifacts/native-compressed-kv-official-reference-validation.json` |
| 5b | Indexer score/top-k publication | `artifacts/native-indexer-topk-official-reference-validation.json` |
| 5c | candidate block publication | `artifacts/native-candidate-block-official-reference-validation.json` |
| 5d | candidate consumer masking + top-k publication | `artifacts/native-candidate-consumer-official-reference-validation.json` |
| 5e | window + compressed KV/index assembly -> `sparse_attn` | `artifacts/native-compressed-sparse-attn-validation.json` |
| 5f | Boundary 5e sparse output -> inverse rotary -> grouped `wo_a` -> `wo_b` -> Attention output | `artifacts/native-compressed-attention-integration-validation.json` |

The 5f final Attention output digest is recorded in the validation artifact as `998305d9e51d2bb3200f0a83675d3cd9aa184961ff6f6d444f97be1935298163`.

## Layer-24 output projection tensors

Layer 24 checkpoint tensors were explicitly inspected and are not assumed to match layer 0:

- `layers.24.attn.wo_a.weight`: shard `model-00027-of-00048.safetensors`, dtype `F8_E4M3`, shape `[8192, 4096]`
- `layers.24.attn.wo_a.scale`: shard `model-00027-of-00048.safetensors`, dtype `F8_E8M0`, shape `[256, 128]`
- `layers.24.attn.wo_b.weight`: shard `model-00027-of-00048.safetensors`, dtype `F8_E4M3`, shape `[5120, 8192]`
- `layers.24.attn.wo_b.scale`: shard `model-00027-of-00048.safetensors`, dtype `F8_E8M0`, shape `[160, 256]`

## Claims

For this bounded prefill path, validated components are:

```text
Compressor
Indexer
candidate publication/consumer wiring
window + compressed KV/index assembly
sparse_attn
inverse rotary
grouped wo_a / flattened wo_a
wo_b
Attention output
STOP before Block / HC
```

The native validation runs a single Python-orchestrated native dataflow from the Boundary 5e sparse-attn output through inverse rotary, native grouped `wo_a`, native `wo_b`, and final Attention output.

## Non-claims

- decode, ring-buffer behavior, or partial compression groups;
- production-scale candidate pruning;
- short fixture candidate mask retains all reachable positions, so candidate machinery wiring is validated but production pruning effect is not;
- Block / Hyper-Connections;
- MoE;
- logits, full layer, or full model correctness;
- performance or fusion.
