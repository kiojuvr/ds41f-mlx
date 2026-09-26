# Boundary 11 closeout

Base commit: `f05144ea1911ee999d4502cdd90e092a09c725d2`.

Boundary11 closeout fixes the Boundary11a/Boundary11b authority chain. It adds no new model math, RNG, or sampling implementation.

Machine-readable closeout: `artifacts/boundary11-closeout.json`.

## Authority hierarchy

Boundary11 does not supersede existing authorities:

- Boundary9: current integrated deterministic model-forward authority through final-position logits. Boundary9 is not promoted to stochastic model-forward authority.
- Boundary10: source-defined sampling arithmetic branch authority.
- Boundary11a: pinned target-MLX RNG distribution/reproducibility authority.
- Boundary11b: connected target-MLX-runtime stochastic sampling authority for the pinned fixture, temperature, and explicit RNG state.

## Boundary11a summary

Artifact: `artifacts/mlx-exp1-rng-validation.json` at commit `cb286d836d910d256c8b76a2d04f0e1fcc415586`.

Runtime:

```text
MLX: 0.32.2
device: Metal GPU / mx.gpu
```

RNG policy:

```text
session_key -> mx.random.split(session_key, 2) -> [draw_key, next_session_key]
```

Generator:

```text
U ~ uniform(low=2^-24, high=1, dtype=float32, key=draw_key)
noise = -log1p(-U)
```

Classification:

```text
FP32 endpoint-policy Exp(1)-compatible target-runtime sampling-noise generator
```

This is not strengthened to a mathematically exact continuous Exp(1) claim because the lower endpoint is deliberately truncated at `2^-24`.

Statistical authority:

```text
N: 1,048,576
alpha: 1e-6
DKW epsilon: 0.0026302598969264743
```

Run A:

```text
mean: 1.0003840149697363
variance: 1.0011172490328388
KS: 0.0005064618640895358
PASS
```

Run B:

```text
mean: 0.9999476762247645
variance: 1.0032118197697908
KS: 0.0010346903184289546
PASS
```

Reproducibility:

```text
same key + same shape -> exact same output
same key -> exact same next key
different key -> different output
```

## Boundary11b summary

Artifact: `artifacts/native-connected-mlx-stochastic-sampling-validation.json` at commit `f05144ea1911ee999d4502cdd90e092a09c725d2`.

Scope:

```text
tokens: [[0,3]]
B=1, S=2, start_pos=0, prefill
world_size=1
full_logits=False
temperature=1.0
MLX 0.32.2 / Metal GPU / mx.gpu
session seed: 289513473
```

Connected path:

```text
tokens
-> embedding
-> Blocks0..39
-> post-loop HC collapse
-> final RMSNorm
-> ParallelHead
-> final-position logits
-> host/native FP32 -> MLX FP32
-> MLX softmax
-> Boundary11a MLX sampling noise
-> probs / noise
-> argmax
```

Boundary11b stops after `target_mlx_sampled_token_id` and `next_session_key`, before `main_hidden` assembly and `Transformer.forward` return packaging.

## Boundary9 regression anchor

```text
logits shape: [1,129280]
dtype: FP32
digest: b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d
artifact_tensor_injection: false
```

The logits are regenerated from token IDs inside Boundary11b.

## Host -> MLX seam

```text
native logits digest:
b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d

MLX dtype: mlx.core.float32
MLX shape: [1,129280]

MLX -> host roundtrip digest:
b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d

exact FP32 bytes preserved: true
```

## Primary RNG state

```text
seed: 289513473

session key values: [0,289513473]
session key digest: 39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630

draw key values: [2308264947,2363689365]
draw key digest: 4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1

next session key values: [613528892,572711044]
next session key digest: 175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4
```

The `next_session_key` is part of the sampling result/state transition.

## Actual MLX noise

```text
digest: 38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4
shape: [1,129280]
dtype: FP32
min: 8.482521479891147e-06
max: 13.234334945678711
mean: 0.9979646651521974
all_positive: true
all_finite: true
```

Classification remains target-runtime Exp(1)-compatible sampling noise under the Boundary11a FP32 endpoint policy.

## MLX sampling arithmetic

Actual API:

```text
mx.softmax(scaled_logits, axis=-1, stream=mx.gpu)
```

Probability tensor:

```text
digest: e08896fb5461b28e4019cdf978f6e91db521b94fee3b91e757fdf25bb8366ea6
sum: 1.000000051814112
min: 4.329862172741585e-15
max: 0.0354914627969265
```

Independent FP64 comparison:

```text
max_abs: 1.924732130620388e-09
sum_abs_error: 5.1814111667880525e-08
```

Predeclared contract:

```text
max_abs <= 1e-6
sum_abs_error <= 1e-6
```

PASS. No softmax bit-exactness claim is made.

## Runtime sampled-token authority

```text
target_mlx_sampled_token_id: 9468
independent_log_domain_token: 9468
tie_count: 1
```

Runtime score-domain:

```text
top1 token: 9468
prob: 0.00020057029905728996
noise: 3.877706330968067e-05
score: 5.1723952293396

top2 token: 1610
score: 1.0868216753005981
gap: 4.0855735540390015
```

Independent log-domain:

```text
top1 token: 9468
log-score: 17.925053292457022

top2 token: 1610
log-score: 16.364974861388077
gap: 1.5600784310689448
```

Classification: pinned target-MLX-runtime sampled token.

Forbidden classifications: official PyTorch sampled token, canonical DeepSeek token, backend-independent sampled token.

## Boundary10b separation

Boundary10b token `795` is not an expected value for Boundary11b.

```text
Boundary10b used a different deterministic supplied-noise fixture.
Boundary11b uses actual Boundary11a MLX-generated noise.
Common authority: sampling arithmetic semantics only.
Noise realization and sampled token are not shared.
```

## Reproducibility authority

Primary fixture rerun from identical tokens, temperature, MLX version/device, and session key:

```text
same logits: true
same generated noise: true
same next_session_key: true
same sampled token: true
```

Secondary seed:

```text
seed: 289517570
noise digest: 245a54e0bd3bef90f644457cf32d137dd7d804febbe233ac44eaf8aaf7d91265
noise differs: true
sampled token: 74294
```

Different token is diagnostic only. Different noise is the required gate. Different keys are not required to always produce different sampled tokens.

## PyTorch compatibility classification

```text
pytorch_noise_bitstream_used: false
pytorch_seed_parity_required: false
pytorch_sampled_token_parity_required: false
```

Boundary11 validates target-runtime semantic/reproducibility behavior, not PyTorch RNG bitstream compatibility. Optional future exact PyTorch parity remains compatibility-only and outside Boundary11 semantic authority.

## Closed claims

Boundary11 closes, for the exact pinned fixture/runtime scope:

- target MLX sampling-noise construction contract
- explicit session-key state transition
- same-key pinned-runtime reproducibility
- declared distributional qualification
- connected Boundary9 logits -> MLX FP32 seam
- connected MLX nonzero-temperature sampling arithmetic
- actual MLX noise consumption
- runtime sampled-token result
- independent log-domain winner agreement

## Still open

- PyTorch/MLX RNG bitwise parity
- cross-backend sampled-token parity
- CPU/GPU RNG sequence parity
- cross-version RNG sequence guarantee
- mathematically exact continuous Exp(1) beyond the declared Boundary11a endpoint policy
- main_hidden assembly
- `Transformer.forward` return packaging
- second model-token step
- decode start_pos>0
- cache persistence
- window/ring decode semantics
- compressed decode partial-group semantics
- distributed semantics
- Engram
- MTP/DSpark
- long-context qualification
- full `Transformer.forward` correctness
- full-model correctness
- performance/production qualification

## STOP boundary

```text
stopped_after:
  target_mlx_sampled_token_id = 9468
  next_session_key = [613528892,572711044]

next_source_operation:
  main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None

main_hidden assembly: not executed
Transformer.forward return: not executed
```

## Authority contamination guard

- Boundary9 remains deterministic model-forward authority through logits.
- Boundary10 remains sampling arithmetic semantics authority.
- Boundary11a remains RNG distribution/reproducibility authority.
- Boundary11b is only the connected target-runtime sampling authority for the pinned fixture/runtime/key.
- Boundary11b does not transform runtime-specific RNG output into backend-independent model semantic authority.
- oMLX / old deepseek-v41-flash-mlx evidence remains excluded from semantic correctness authority.

## Safe closeout claim

Boundary 11 closes the target-MLX-runtime stochastic sampling layer for the pinned MLX 0.32.2 / Metal GPU environment.

Boundary11a qualifies the explicit-key FP32 sampling-noise generator under its declared endpoint, distributional, and reproducibility contract.

Boundary11b connects that qualified generator to the exact Boundary9 final-position logits through the Boundary10 sampling arithmetic at temperature=1.0. For token fixture [[0,3]] and session seed 289513473, the pinned runtime produces target-MLX sampled token 9468 and advances the explicit session state to the recorded next key. An independent FP64 log-domain reconstruction using the exact generated noise selects the same token.

This is a pinned target-runtime result, not a PyTorch bitstream-parity or backend-independent token claim. Boundary11 stops before main_hidden assembly and Transformer.forward return packaging.

## Non-claims

- no PyTorch/MLX RNG bitwise parity
- no cross-backend sampled-token parity
- no CPU/GPU RNG sequence parity
- no cross-version RNG guarantee
- no exact continuous Exp(1) claim beyond Boundary11a contract
- no main_hidden assembly
- no `Transformer.forward` return correctness
- no second-step/decode semantics
- no cache persistence qualification
- no full-model correctness
- no performance/production qualification
