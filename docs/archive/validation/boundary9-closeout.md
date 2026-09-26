# Boundary 9 closeout

Boundary 9 fixes the deterministic connected pre-sampling forward authority for the bounded fixture:

```text
tokens [[0,3]], B=1, S=2, start_pos=0, prefill, world_size=1, full_logits=False
```

Closed path:

```text
token IDs
-> Transformer entry
-> Blocks 0..39
-> post-loop HC collapse
-> final RMSNorm
-> ParallelHead
-> final-position generation logits
STOP before sample(logits, self.temperature)
```

Current integrated numerical authority:

```text
artifacts/native-parallel-head-logits-validation.json
commit acf587776f7b9b98ccc6a60b4b34eafeb1ce744e
```

## Authority chain

- Boundary 9a: `artifacts/native-post-loop-hc-collapse-validation.json`, commit `ccedd09360aa3dce48e98a2861e588a0b5652ab2`
  - closed exact subscope authority through post-loop HC collapse
  - stops before final RMSNorm
- Boundary 9b: `artifacts/native-final-rmsnorm-validation.json`, commit `caec1425b26edad090e5415b3dd65c43951eecb4`
  - closed exact subscope authority through final RMSNorm
  - stops before ParallelHead
- Boundary 9c: `artifacts/native-parallel-head-logits-validation.json`, commit `acf587776f7b9b98ccc6a60b4b34eafeb1ce744e`
  - current integrated numerical authority through final-position logits
  - stops before sampling

Boundary8 closeout remains the all-Block subscope foundation and is not invalidated.

## Source authority

```text
official inference/model.py SHA256: 4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
official inference/kernel.py SHA256: 1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455
source identity method: raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1
```

Boundary 9 source identities are recorded in `artifacts/boundary9-closeout.json` for `Block.hc_pre`, `RMSNorm`, `Transformer` final norm/head/sample order, and `ParallelHead.__init__/forward`.

## Fixed facts

### Boundary 9a

```text
x39_out: c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3
ffn_pre39: 8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc
post_loop_h BF16 [1,2,5120]: 6b99fb26048a577ce78756397101be15441d1ddb7fa9c7aa205007e10869061a
```

HC collapse arithmetic: BF16 `x` plus FP32 `pre_mix`, FP32 weighted reduction over `HC=4`, then BF16 collapsed hidden. Native and independent reconstruction are exact.

### Boundary 9b

```text
norm.weight BF16 [5120]: 9cd3b57cd9513541b9771bf66c9b356bf1a7b20ff050ed69f7e97cff9fedd428
normalized_h BF16 [1,2,5120]: 00186e76a7a7bd78ce40de4a9a912025c9d6724b1600a30f171eca9a2cb65eb5
```

RMSNorm source order: `x.float()`, square, `mean(-1)`, `rsqrt(var + eps)`, multiply `x`, multiply norm weight, cast to original dtype. Independent arithmetic is exact under the Boundary9b contract.

### Boundary 9c

ParallelHead source contract:

```text
full_logits default = False
x = x[:, -1]
F.linear(x.float(), self.weight)
world_size=1: no all_gather
```

Head provenance:

```text
head.weight checkpoint BF16 [129280,5120]
raw BF16 digest: 68f446ddda4243d5c8d57d2a9729c125f7fb8b2ee050c78ac6ff5e0cacde6789
runtime FP32 digest: ea729fc899cda136019ea91bbd85502ccddcf58bab8c7482029d8834415eb8cd
not tied to embedding
```

Input/logits:

```text
normalized_h: 00186e76a7a7bd78ce40de4a9a912025c9d6724b1600a30f171eca9a2cb65eb5
selected final-position hidden: 79999590fe7867d7f398be3cce6cd4bda7dac7e2fcdf8c1571019175ffb9d746
logits FP32 [1,129280]: b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d
argmax diagnostic: token 372, logit 12.943254470825195
```

The argmax is diagnostic only and is not a sampling result.

## ParallelHead comparison policy

Boundary9c is not bit-exact across independent FP32 reduction implementations. It passes the predeclared bounded contract:

```text
full_vocab_max_abs_lte: 0.005
full_vocab_max_rel_lte where |independent logit| >= 1: 0.0001
anchor_max_abs_lte: 0.005
```

Actual:

```text
max_abs: 1.9073486328125e-06
max_rel meaningful: 1.2064747352269478e-06
anchor_max_abs: 1.1920928955078125e-05
```

Predeclared anchor rows `[0,1,2,3,4096,8192,16384,32768,65536,98304,123456,129279]` all PASS for explicit independent FP32 dot versus native logit.

## Closed claims

For the stated fixture only, Boundary9 closes:

- Transformer entry
- all 40 Blocks connected
- post-loop HC collapse
- final RMSNorm
- ParallelHead default `full_logits=False` path
- last-position hidden selection
- `world_size=1` no-gather head path
- checkpoint BF16 -> runtime FP32 head parameter semantics
- final-position full-vocabulary logits
- bounded numerical agreement of ParallelHead under the predeclared comparison contract

## Still open

Sampling, temperature semantics, RNG/random draw, output IDs, `full_logits=True` full-sequence logits, `world_size > 1` ParallelHead all-gather, production candidate pruning, decode/cache/ring/partial compression-group semantics, distributed Block/MoE semantics, Engram, MTP/DSpark, `main_hidden` assembly, long-context qualification, full `Transformer.forward`, full-model correctness, and performance/fusion/production qualification remain open.

## Safe closeout claim

For B=1, S=2, start_pos=0, world_size=1, full_logits=False and token fixture [[0,3]], the official-reference-derived connected native deterministic pre-sampling forward path from Transformer entry through all 40 Blocks, post-loop Hyper-Connection collapse, final RMSNorm, and ParallelHead final-position full-vocabulary logits is closed as one bounded validation.

ParallelHead numerical agreement is bounded by the predeclared FP32 comparison contract rather than claimed bit-exact across independent reduction implementations. This closeout stops before sample() and does not validate output_ids generation, RNG/temperature semantics, decode, full-sequence logits, distributed vocabulary gathering, or full model behavior.
