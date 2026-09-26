# ParallelHead.forward semantics contract

This document closes only the bounded output-head primitive seam for the reviewed official `ParallelHead.forward` implementation. It does **not** claim full-model logits correctness.

## Authority

Official reference source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

File SHA-256:

```text
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
```

Reviewed spans:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `ParallelHead` | 997-1017 | `6247ecdd05c6a9abae3c53eb58bb1154a88e56caa1d149ec71554c405b18afa0` |
| `ParallelHead.forward` | 1008-1017 | `2537fe7c5c90707fb39100f54ef9559efed9979513f5f0f0939e063742ceaa52` |

Reviewed source behavior:

```python
if not full_logits:
    x = x[:, -1]
logits = F.linear(x.float(), self.weight)
if world_size > 1:
    all_logits = [torch.empty_like(logits) for _ in range(world_size)]
    dist.all_gather(all_logits, logits)
    logits = torch.cat(all_logits, dim=-1)
return logits
```

## Checkpoint tensor provenance

Official checkpoint tensor:

| Tensor | Shard | Checkpoint dtype | Checkpoint shape | Provenance |
| --- | --- | --- | --- | --- |
| `head.weight` | `model-00043-of-00048.safetensors` | `BF16` | `[129280, 5120]` | official checkpoint raw safetensors bits |

The official `ParallelHead.__init__` allocates `self.weight` as `torch.float32` and comments that the checkpoint is BF16 but kept as FP32 so logits are FP32. Contractually, checkpoint `head.weight` BF16 raw values are converted to their exact FP32 numeric representation before `forward` math.

## Operation contract

For world size 1:

- input `x` has shape `[batch, sequence, dim]` with `dim=5120`;
- `x.float()` is applied before the linear operation;
- `F.linear(x.float(), weight_f32)` is interpreted for this bounded fixture as `input_f32 @ weight_f32.T`;
- accumulation and output are FP32;
- `full_logits=False` first selects the last position (`x[:, -1]`) and returns shape `[batch, vocab]` for the full primitive;
- `full_logits=True` computes all sequence positions and returns shape `[batch, sequence, vocab]` for the full primitive;
- therefore `forward(x, False) == forward(x, True)[:, -1]` under identical weights and arithmetic.

For tensor parallelism (`world_size > 1`), each rank owns `part_vocab_size = vocab_size // world_size`; after per-rank `F.linear`, the official contract gathers rank-local logits with `dist.all_gather` and concatenates them along the last dimension. This repository has not executed that TP path in this fixture; it is documented as contract only.

Predeclared native validation tolerance:

```text
max_abs_error_lte = 1e-4
max_relative_error_lte = 1e-6
```

The generated fixture and current native run are bit-exact inside the selected-row bounded scope, which is stricter than the tolerance.

## Bounded fixture

Fixture:

```text
artifacts/parallel-head-official-reference-fixture.json
```

Scope:

- input tensor: `embed.weight` rows for tokens `[0, 3, 42]`, used only as bounded FP32 input data;
- head weight rows: `head.weight` rows `[0, 1, 2, 3, 42, 128799, 129264, 129279]`;
- sequence shape: `[1, 3, 5120]`;
- selected vocabulary output only: 8 rows, not the full 129280-row logits tensor.

Expected provider:

```text
weight_f32 = BF16 checkpoint bits -> FP32
input_f32 = input -> FP32
expected = input_f32 @ weight_f32.T
```

The fixture is not oMLX-derived, does not execute PyTorch/MLX/native as expected-value providers, and records `not_omlx_derived=true`.

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_parallel_head_fixture.py \
  --out artifacts/parallel-head-official-reference-fixture.json
```

Observed digests:

```text
expected_full_logits_true_f32_sha256: 7d8ab29bd64ff7181789fe21531906a9ee60addb94367282ba4ff73fed9a6384
expected_full_logits_false_f32_sha256: fab5be3a8a72c7d76304fe24a1b24824c044f833a93a9f516e2b31f2389b90ab
```

## Native validation

Validation artifact:

```text
artifacts/native-parallel-head-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_parallel_head_against_official_reference.py \
  --out artifacts/native-parallel-head-official-reference-validation.json
```

Observed result:

```text
full_logits_true: bit_exact=true, max_abs_error=0.0
full_logits_false: bit_exact=true, max_abs_error=0.0
false_output_equals_true_last_position: bit_exact=true
native Metal executed: true
```

## Non-claims

This contract does not validate:

- full-model logits correctness;
- Transformer hidden-state production before the head;
- full-vocabulary native output allocation/readback beyond the selected-row fixture;
- tensor-parallel execution, only its gather/concat contract is documented;
- FP8, FP4, Attention, MoE, Hyper-Connections, Engram, cache/state publication, sampling, DSpark/MTP, or performance optimization;
- oMLX behavior as an official correctness oracle.
