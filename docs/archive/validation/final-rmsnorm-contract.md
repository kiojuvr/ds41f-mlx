# Boundary 9b: final RMSNorm contract

Authority artifact: `artifacts/native-final-rmsnorm-validation.json`.

## Scope

Validated bounded dataflow:

```text
tokens [[0,3]]
-> Transformer entry
-> Blocks 0..39
-> post-loop h = layer.hc_pre(h, pre_mix)
-> normalized_h = self.norm(h)
STOP before self.head(normalized_h)
```

Parameters: `B=1`, `S=2`, `start_pos=0`, prefill, `world_size=1`.

Boundary 9a remains the closed exact subscope authority through the post-loop HC collapse. Boundary 9b promotes authority only through final RMSNorm and still stops before ParallelHead/logits.

## Source authority

Pinned official source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/config.json
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/model.safetensors.index.json
```

Canonical identity method:

```text
raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1
```

Reviewed RMSNorm implementation:

```text
inference/model.py lines 281-293
SHA256 ac829397ad0c5f99412def7adb54ba0334397baa5c2ab795d4531f072fc47ecc
```

Official arithmetic:

```python
dtype = x.dtype
x = x.float()
var = x.square().mean(-1, keepdim=True)
x = x * torch.rsqrt(var + self.eps)
return (self.weight * x).to(dtype)
```

Transformer construction and forward order reviewed:

```text
self.norm = RMSNorm(args.dim, self.norm_eps)
h = layer.hc_pre(h, pre_mix)
logits = self.head(self.norm(h))
```

Boundary 9b executes only `self.norm(h)` after the post-loop HC collapse.

## Checkpoint tensor provenance

The final norm tensor is resolved from `model.safetensors.index.json`, not assumed from a hard-coded shard:

```text
tensor name: norm.weight
shard: model-00043-of-00048.safetensors
dtype: BF16
shape: [5120]
```

Hidden dimension is cross-checked across source/config/checkpoint/native constants: `5120`.

## Arithmetic contract

Actual connected input is produced inside the runner from token IDs; the Boundary9a artifact tensor is not loaded as input.

```text
post_loop_h: [1,2,5120] BF16
norm.weight: [5120] BF16
normalized_h: [1,2,5120] BF16
```

Independent reconstruction records digests for input, FP32 input, square, mean-square, rsqrt, normalized pre-weight tensor, final norm weight, weighted FP32 output, and final BF16 output. Mean-square and rsqrt values are also recorded per token.

## Producer -> consumer seam

The audited seam is:

```text
Boundary9a post_loop_h publication in the same runner
-> Transformer self.norm consumption in the same runner
```

Artifact records matching producer and consumer digests with `same_tensor_dataflow = true` and `artifact_tensor_injection = false`.

## STOP and non-claims

Next source operation recorded but not executed:

```text
self.head(...)
```

Not validated by Boundary 9b: ParallelHead, connected logits, sampling, decode, ring/partial compression-group semantics, multi-call cache persistence, `world_size > 1`, Engram, MTP/DSpark, long-context qualification, full Transformer-output correctness, full-model correctness, or performance/fusion/production qualification.

Next boundary candidate: **Boundary 9c: ParallelHead / logits**, after reviewing official `ParallelHead.forward` and checkpoint provenance. Sampling must not be included in 9c.
