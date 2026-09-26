# Boundary 9a: Transformer post-loop Hyper-Connection collapse contract

Authority artifact: `artifacts/native-post-loop-hc-collapse-validation.json`.

## Scope

Validated bounded dataflow:

```text
tokens [[0,3]]
-> Transformer entry
-> Blocks 0..39
-> Block39 returns h = x39_out and pre_mix = ffn_pre39
-> h = layer.hc_pre(h, pre_mix)
STOP before self.norm(h)
```

Parameters: `B=1`, `S=2`, `HC=4`, `start_pos=0`, prefill, `world_size=1`.

Boundary 8d remains the closed exact subscope authority through Block39 return. Boundary 9a promotes the integrated authority only through the post-loop HC collapse and still stops before final RMSNorm.

## Source authority

Pinned official source: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py`.

Canonical identity method:

```text
raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1
```

Boundary-requested `Block.hc_pre` canonical span:

```text
inference/model.py lines 960-963
SHA256 103935a48b2cba8d9e34507bafa52db0f21e2a84448a1a83cdd4f3c5a7e727a4
```

Executable `Block.hc_pre` function reviewed locally:

```python
def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
    """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)
```

Transformer post-loop source order reviewed:

```python
h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
h = layer.hc_pre(h, pre_mix)
logits = self.head(self.norm(h))
```

Boundary 9a executes only the `layer.hc_pre(h, pre_mix)` line after the block loop.

## Arithmetic contract

Actual connected inputs are produced in the same execution from token IDs, not loaded from Boundary 8 artifacts:

```text
x39_out:   [1,2,4,5120] BF16
ffn_pre39: [1,2,4] FP32
```

Independent reconstruction:

```text
x_f32 = BF16 -> FP32
weighted = ffn_pre39[..., None] * x_f32
collapsed_f32 = sum(weighted, axis=HC, dtype=FP32)
collapsed_bf16 = official BF16 rounding/cast semantics
```

Expected output:

```text
post_loop_h: [1,2,5120] BF16
```

The post-loop call does not compute a new `hc_mixes()`, does not create a layer40, and does not introduce any virtual HC parameter. It consumes the `ffn_pre39` returned by Block39 directly.

## STOP and non-claims

Next source operation recorded but not executed:

```text
self.norm(h)
```

Not validated by Boundary 9a: final RMSNorm, ParallelHead, logits, sampling, decode, ring/partial compression-group semantics, multi-call cache persistence, `world_size > 1`, Engram, MTP/DSpark, long-context qualification, full Transformer-output correctness, full-model correctness, or performance/fusion/production qualification.

Next boundary: **Boundary 9b: final RMSNorm only** (`post-loop collapsed h -> self.norm(h)`, STOP before `self.head`).
