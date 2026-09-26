# Boundary12e: target-MLX-runtime stochastic integrated return

Status: **PASS / authority promoted**.

Authority classification:

```text
qualified target-MLX-runtime runtime/key-specific stochastic integrated return authority
```

Boundary12e does **not** supersede Boundary12d. Boundary12d remains the deterministic source-order-equivalent return authority at explicit `temperature=0`. Boundary12e validates the target-MLX runtime/key-specific stochastic return at `temperature=1` and explicit seed `289513473`.

## Fixture

```text
tokens = [[0,3]]
B = 1
S = 2
start_pos = 0
prefill = true
world_size = 1
engram_mask = None
full_logits = false
temperature = 1.0
MLX = 0.32.2
device = Metal GPU / mx.gpu
seed = 289513473
```

The runner starts from token IDs and regenerates the connected trajectory. No logits, output_ids, captures, main_hidden, or post-Engram tensors are injected from artifacts.

## Source order

Validated event order:

```text
capture37: 42
capture38: 44
capture39: 46
logits: 50
mlx_sampling: 51
main_hidden_concat: 52
return: 53
```

Hard ordering:

```text
mlx_sampling < main_hidden_concat
```

## Upstream connected regressions

```text
full Ngram hash:
f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d

post_engram1_h:
3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9

post_engram14_h:
ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd
```

Captures and append order:

```text
append_order = [37,38,39]

37: dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb
38: 8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c
39: b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80
```

## Logits and host-to-MLX seam

```text
shape: [1,129280]
dtype: FP32
digest: 7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd
```

Host-to-MLX FP32 roundtrip digest remains:

```text
7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd
```

## Sampling authority split

Official source-derived sampling arithmetic:

```text
probs = softmax(logits / max(temperature, 1e-5), dim=-1, dtype=float32)
scores = probs / Exp1_noise
output_ids = argmax(scores)
```

Runtime-specific substituted component:

```text
rng_provider = qualified target MLX runtime
```

Recorded non-claims:

```text
official_pytorch_rng_executed = false
pytorch_rng_parity_claimed = false
pytorch_rng_bitstream_parity = false
pytorch_stochastic_output_identity_claimed = false
```

## RNG state and noise

Initial session key:

```text
values: [0,289513473]
digest: 39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630
```

Split policy:

```text
session_key -> mx.random.split(session_key, 2) -> draw_key, next_session_key
```

```text
draw_key digest:
4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1

next_session_key digest:
175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4
```

Noise policy:

```text
U = mx.random.uniform(low=2^-24, high=1, shape=[1,129280], dtype=mx.float32, key=draw_key, stream=mx.gpu)
noise = -log1p(-U)
```

Noise digest:

```text
38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4
```

Classification remains target-runtime Exp(1)-compatible noise under the qualified FP32 endpoint policy.

## MLX stochastic sampling result

```text
target MLX sampled token: 9468
independent log-domain token: 9468
tie_count: 1
source score gap: 3.0003824830055237
independent log-score gap: 1.7397182473531227
```

MLX probs:

```text
digest: 722acab2f31289fc2d2de5cc8ed588a94efd7a45bae12879a4b0ed08df3072f0
max_abs <= 1e-6: PASS
sum_abs_error <= 1e-6: PASS
```

Token `9468` is specific to the pinned target MLX runtime, explicit seed/key state, current Engram-connected logits, and qualified Boundary11a noise policy. It is not an official PyTorch or backend-independent stochastic token.

## output_ids representation seam

MLX sampled value is transferred/represented as native return `int64`:

```text
shape: [1]
dtype: int64
value: [9468]
digest: 08d628a36758c2a7f574ca12b076ed1b29b84c8d12b7e9f08c9a19d697774bbd
```

This is classified as a runtime representation seam, not sampling arithmetic.

## main_hidden after stochastic sample

Final concat occurs after stochastic `output_ids` exists:

```text
shape: [1,2,15360]
dtype: BF16(uint16)
digest: 4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997
```

Segments:

```text
0:5120: dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb
5120:10240: 8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c
10240:15360: b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80
```

## Return packaging and runtime state

Returned source-order-equivalent tuple:

```text
(output_ids, logits, main_hidden)
```

Validated:

```text
container: tuple
length: 3
member order: output_ids, logits, main_hidden
member shapes: [1], [1,129280], [1,2,15360]
member dtypes: int64, FP32, BF16(uint16)
```

Return member digests:

```text
return[0]: 08d628a36758c2a7f574ca12b076ed1b29b84c8d12b7e9f08c9a19d697774bbd
return[1]: 7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd
return[2]: 4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997
```

The `next_session_key` is stored out-of-band in `runtime_state_after`. It is **not** a fourth `Transformer.forward` return tuple member.

## Same-key return-tail reproducibility

Repeating the sampling/return tail from the same regenerated logits/captures and same initial session key produced identical:

```text
noise
sampled output_ids
next_session_key
main_hidden
returned tuple member digests
```

This is same-key stochastic return-tail reproducibility, not a second full-model execution.

## Limits / non-claims

No official PyTorch stochastic return authority; no PyTorch RNG bitstream authority; no PyTorch/MLX RNG parity; no backend-independent stochastic output identity; no decode/incremental `NgramHashState` correctness; no False/image-mask DEAD crossing numerical authority; no distributed/world_size>1 correctness; no long-context qualification; no full-model correctness; no performance/production qualification.

Next major phase:

```text
Boundary13: decode / incremental-state source audit
```

Start with an audit before implementation; do not jump directly to long decode execution.
