# Boundary12d: integrated deterministic `Transformer.forward` return packaging

Status: **PASS / authority promoted**.

Boundary12d validates an explicit-temperature-zero, bounded, source-order-equivalent native execution of `Transformer.forward` return packaging for:

```text
tokens = [[0,3]]
B = 1
S = 2
start_pos = 0
prefill = true
world_size = 1
engram_mask = None
full_logits = false
temperature = 0.0
```

This boundary adds no new model arithmetic. It integrates already-current authorities in one connected execution:

- Boundary12b4: Engram-connected deterministic logits
- Boundary12b5-A: explicit `temperature == 0` sampling result
- Boundary12c: Engram-connected `main_hidden` capture/mean/concat

## Source identity

Pinned source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

Recorded gates:

```text
model.py SHA256:
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65

Transformer.forward lines 1242-1272:
6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c

main_hidden span lines 1259-1272:
56ba7227f4202c0791c33e08300943a0675ef07b62ee824af92889d1e7dd57a3
```

Boundary12d also records a derived return-packaging span covering:

```python
logits = ...
output_ids = sample(...)
main_hidden = ...
return output_ids, logits, main_hidden
```

from local pinned source lines 1269-1272.

## Integrated source order

The validated event order is:

```text
capture37
capture38
capture39
logits
sample
main_hidden_concat
return
```

Hard ordering includes:

```text
event(sample) < event(main_hidden_concat)
```

Captures happen during the layer loop. Final `main_hidden` concatenation happens only after deterministic sampling, matching the reviewed source order.

## Upstream regressions in the same run

```text
full Ngram hash:
f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d

post_engram1_h:
3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9

post_engram14_h:
ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd

pre-Block37:
0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18

pre-Block38:
0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9

pre-Block39:
1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e
```

## Captures

Boundary12d executes the Boundary12c-qualified BF16 mean at the source target-layer capture position:

```text
BF16 h -> explicit FP32 HC=4 accumulation -> *0.25 -> BF16 RNE
```

Append order:

```text
[37,38,39]
```

Capture digests:

```text
37: dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb
38: 8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c
39: b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80
```

## Logits and deterministic sample

Logits:

```text
shape: [1,129280]
dtype: FP32
digest: 7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd
argmax: 15
max: 13.22089958190918
```

Sampling:

```text
temperature: 0.0
source branch: temperature == 0 -> logits.argmax(dim=-1)
output_ids: [15]
shape: [1]
dtype: int64
tie_count: 1
```

RNG flags:

```text
stochastic_rng_executed = false
mlx_rng_executed = false
pytorch_rng_executed = false
```

## main_hidden after sample

Final concat is performed after `output_ids` exists. No cast or arithmetic occurs during concat.

```text
shape: [1,2,15360]
dtype: BF16(uint16)
digest: 4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997
```

Segments:

```text
0:5120:
dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb

5120:10240:
8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c

10240:15360:
b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80
```

## Return packaging

The native validator constructs a source-order-equivalent return tuple:

```python
(output_ids, logits, main_hidden)
```

Validated:

```text
container: tuple
length: 3
index 0: output_ids
index 1: logits
index 2: main_hidden
```

Packaging performs no numeric cast/mutation:

```text
return[0] == output_ids byte-exact
return[1] == logits byte-exact
return[2] == main_hidden byte-exact
```

Native representation classification:

```text
source-order and source-dtype-equivalent native return packaging
```

The native objects are not claimed to be PyTorch Tensor objects.

## Authority classification

Boundary12d promotes:

```text
current official-reference-derived bounded Engram-connected deterministic Transformer.forward return authority
```

Scope explicitly includes:

```text
temperature = 0.0
```

It closes the bounded connected deterministic forward dataflow through Engram@1/14, Blocks0..39, target captures, logits, explicit temperature-zero sample, post-sample `main_hidden` concat, and return tuple ordering/member identity.

## Limits / non-claims

This is an explicit-temperature-zero deterministic bounded return authority. It does not validate default-temperature PyTorch stochastic returns, PyTorch RNG bitstreams, PyTorch/MLX RNG parity, backend-independent stochastic output identity, decode/incremental `NgramHashState`, False/image-mask DEAD crossing numerical authority, distributed/world_size>1 correctness, long-context qualification, full-model correctness, or performance/production qualification.

Next boundary:

```text
Boundary12e: target-MLX-runtime stochastic integrated return
```

using Boundary12b5-C as a separate runtime/key-specific stochastic authority.
