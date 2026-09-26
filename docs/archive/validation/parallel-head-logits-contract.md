# Boundary 9c: ParallelHead / connected final-position logits contract

Authority artifact: `artifacts/native-parallel-head-logits-validation.json`.

## Scope

Validated bounded dataflow:

```text
tokens [[0,3]]
-> Transformer entry
-> Blocks 0..39
-> post-loop HC collapse
-> final RMSNorm
-> self.head(normalized_h)
-> final-position generation logits
STOP before sample(logits, self.temperature)
```

Parameters: `B=1`, `S=2`, `start_pos=0`, prefill, `world_size=1`.

Boundary 9b remains the closed exact subscope authority through final RMSNorm. Boundary 9c promotes authority only through the default `ParallelHead` path for final-position generation logits.

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

Reviewed source identities:

```text
ParallelHead.__init__/forward: inference/model.py lines 997-1031
SHA256 673f828a099975e80e8524b8a13ababd456151fe9f588a4f14b45c8ece1b74df

Transformer.__init__ head construction: inference/model.py lines 1201-1206
SHA256 61744fe50e430431914908fe4196f57aa267edfa0725ba8d94f49a93cd9466a5

Transformer.forward head call / sample order: inference/model.py lines 1268-1270
SHA256 bbdcbbf784a2f261ec02408f90c0b97be0d6a1054e2ec83eca63819efe25f1b7
```

Reviewed `ParallelHead` contract:

```python
self.part_vocab_size = vocab_size // world_size
self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

def forward(self, x, full_logits=False):
    if not full_logits:
        x = x[:, -1]
    logits = F.linear(x.float(), self.weight)
    if world_size > 1:
        all_gather then cat along vocab dimension
    return logits
```

For Boundary 9c, `Transformer.forward` calls `self.head(self.norm(h))` without `full_logits=True`, so only the default `full_logits=False` final-position path is validated.

## Checkpoint tensor provenance

The head tensor is resolved from `model.safetensors.index.json`:

```text
tensor name: head.weight
shard: model-00043-of-00048.safetensors
checkpoint dtype: BF16
shape: [129280, 5120]
```

Source declares the runtime `ParallelHead.weight` parameter as FP32. The validation therefore converts checkpoint BF16 rows to FP32 before the F.linear-equivalent projection and records both raw BF16 and runtime FP32 digests.

Source/checkpoint tying classification: `embed.weight` and `head.weight` are separate source modules/parameters and separate checkpoint index entries; Boundary 9c treats them as not tied.

## Shape contract

With `world_size=1`:

```text
input normalized_h:       [1,2,5120] BF16
head selected input:      [1,5120] BF16, normalized_h[:, -1, :]
rank-local logits:        [1,129280] FP32
connected returned logits [1,129280] FP32
```

The all-gather branch is reviewed but not executed or validated for `world_size > 1`.

## Independent validation

Native/source-order path:

```text
selected hidden -> x.float() -> FP32 chunked matrix-vector projection
```

Independent path:

```text
same runtime FP32 weight semantics, but separate chunked elementwise multiply + FP32 row sums
```

Predeclared anchor rows are fixed before execution:

```text
[0, 1, 2, 3, 4096, 8192, 16384, 32768, 65536, 98304, 123456, 129279]
```

Each anchor records BF16 row digest, explicit independent FP32 dot-product, native logit, and difference.

The comparison contract is predeclared with absolute tolerance for all logits and relative tolerance only where `|independent logit| >= 1.0`; near-zero logits are governed by max-absolute error.

## STOP and non-claims

Strict STOP:

```text
after:  logits = self.head(normalized_h)
before: output_ids = sample(logits, self.temperature)
```

Not executed: `sample()`, temperature sampling, argmax-as-substitute sampling, random draw, or output-id generation.

Not claimed: sampling correctness, full-sequence logits, `world_size > 1` vocabulary gathering, decode/cache/ring/distributed semantics, Engram, MTP/DSpark, long-context, full-model correctness, or performance/production qualification.

Next recommendation: perform **Boundary 9 closeout** before any sampling boundary.
