# Boundary 11b: connected target-MLX-runtime stochastic sampling

Base commit: `cb286d836d910d256c8b76a2d04f0e1fcc415586`.

Boundary11b is the first bounded execution that connects:

```text
actual connected model logits
+ actual MLX-generated sampling noise
+ source-defined nonzero sampling arithmetic
```

The result is classified only as a target-MLX-runtime sampled token under a pinned explicit RNG key/state. It is not an official PyTorch sampled token, backend-independent sampled token, or canonical DeepSeek sampled token.

## Authority dependencies

Boundary11b depends on, and does not supersede:

- Boundary9: deterministic connected model forward through final-position logits.
- Boundary10: source-defined sampling arithmetic branch semantics.
- Boundary11a: pinned MLX 0.32.2 / Metal GPU sampling-noise distribution and reproducibility semantics.

The machine-readable validation artifact is:

```text
artifacts/native-connected-mlx-stochastic-sampling-validation.json
```

## Connected model execution

The connected execution starts from token IDs:

```text
[[0, 3]]
```

and executes:

```text
tokens -> embedding -> Blocks0..39 -> post-loop HC collapse -> final RMSNorm -> ParallelHead -> final-position logits
```

No Boundary9 logits artifact is injected. The Boundary9 logits regression is exact:

```text
shape: [1,129280]
dtype: FP32
digest: b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d
```

## Host/native to MLX logits seam

Boundary11b records the seam where native FP32 logits become an MLX tensor for sampling.

Hard gate:

```text
producer native FP32 logits digest == MLX -> host roundtrip FP32 digest
```

This passed with digest:

```text
b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d
```

The MLX consumer tensor is `mlx.core.float32` with shape `[1,129280]`.

## Temperature

Boundary11b uses:

```text
temperature = 1.0
effective_temperature = max(1.0, 1e-5) = 1.0
temperature_floor_reviewed = true
temperature_floor_exercised = false
```

## Boundary11a RNG policy preserved exactly

Boundary11b does not redesign the generator. It uses the Boundary11a construction:

```text
session_key
-> mx.random.split(session_key, 2)
-> [draw_key, next_session_key]

U = mx.random.uniform(
    low=2^-24,
    high=1,
    shape=[1,129280],
    dtype=mx.float32,
    key=draw_key,
    stream=mx.gpu,
)

noise = -log1p(-U)
```

The primary seed is `289513473`. The key regression is exact:

```text
session_key_before: [0, 289513473]
digest: 39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630

draw_key: [2308264947, 2363689365]
digest: 4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1

next_session_key: [613528892, 572711044]
digest: 175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4
```

The generated noise exact-regresses to Boundary11a production fixture:

```text
digest: 38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4
min: 8.482521479891147e-06
max: 13.234334945678711
mean: 0.9979646651521974
all_positive: true
all_finite: true
```

Classification remains target-runtime Exp(1)-compatible sampling noise with the Boundary11a predeclared FP32 endpoint policy. This is not strengthened to a mathematically exact continuous Exp(1) claim.

## MLX sampling arithmetic

The installed MLX API was inspected. Boundary11b uses:

```text
mx.softmax(scaled_logits, axis=-1, stream=mx.gpu)
```

Then:

```text
scaled_logits = logits_mx / 1.0
probs = MLX FP32 softmax over vocab
scores = probs / noise
sampled_token = argmax(scores, axis=-1)
```

MLX softmax is validated against an independent FP64 reconstruction using Boundary10 tolerances:

```text
softmax_max_abs_lte: 1e-6
softmax_sum_abs_error_lte: 1e-6
```

Boundary11b also computes an independent FP64 log-domain winner with the exact generated MLX noise:

```text
log_score_i = float64(logit_i) - log(float64(noise_i))
```

Hard gate:

```text
MLX sampled token == independent log-domain token
```

## Result classification

The recorded token is:

```text
target_mlx_sampled_token_id: 9468
```

Classification:

```text
sampled token from Boundary9 connected logits at temperature=1.0
using Boundary11a pinned MLX RNG stream seed/session key 289513473
on the pinned MLX/runtime/device
```

Boundary10b token `795` is explicitly not an expected value for Boundary11b because Boundary10b used a different deterministic supplied-noise fixture. The shared authority is only sampling arithmetic semantics.

## Stop boundary

Boundary11b stops after:

```text
target_mlx_sampled_token_id
next_session_key
```

The next source operation is recorded but not executed:

```python
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

Boundary11b does not execute main_hidden assembly, `Transformer.forward` return packaging, decode, or a second model-token step.

## Safe claim on PASS

For B=1, S=2, start_pos=0, world_size=1, full_logits=False, token fixture [[0,3]], temperature=1.0, and the pinned explicit MLX session key derived from seed 289513473, the connected deterministic Boundary9 logits are consumed by the Boundary10 source-defined sampling arithmetic using the Boundary11a-qualified MLX sampling-noise generator.

The resulting target-MLX-runtime sampled token agrees with an independent FP64 log-domain winner reconstruction using the exact generated MLX noise, and the explicit RNG state advances to the recorded next session key.

This is a pinned target-MLX-runtime sampling result. It does not claim PyTorch RNG bitstream parity, equal sampled tokens for equal numeric seeds across backends, or backend-independent canonical token identity.

## Non-claims

- no PyTorch/MLX RNG bitwise parity
- no cross-backend sampled-token parity
- no CPU/GPU RNG stream parity
- no cross-version RNG stream guarantee
- no mathematical claim of an exact continuous Exp(1) beyond the Boundary11a declared FP32 endpoint/distribution contract
- no main_hidden assembly
- no `Transformer.forward` return correctness
- no second-step/decode semantics
- no cache persistence qualification
- no full model correctness
- no performance/production qualification
