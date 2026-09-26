# Boundary 11a: MLX Exp(1) RNG contract

Base commit: `c9c8ab1de87c69e1af7e8cb74b2bdff528ec75a1`.

Boundary 11a qualifies only native target-runtime Exp(rate=1) sampling-noise semantics for the DeepSeek source expression:

```text
torch.empty_like(probs).exponential_(1)
```

It does not reimplement model forward, sampling arithmetic, logits connection, or token sampling.

## Authority classification

Machine-readable artifact: `artifacts/mlx-exp1-rng-validation.json`.

- `semantic_requirement`: positive independent/pseudorandom samples from `Exp(rate=1)` suitable for the source-defined sampling arithmetic.
- `not_semantic_requirement`: same bit sequence as PyTorch `exponential_(1)`.
- `optional_compatibility_goal`: reproduce a pinned PyTorch version/device/seed draw stream.

PyTorch bitwise parity is not a Boundary 11a PASS condition. If ever pursued, it belongs to a compatibility-only boundary such as Boundary 11p.

Boundary relationships after PASS:

- Boundary9: integrated deterministic model-forward authority through logits.
- Boundary10: source-defined sampling arithmetic branch authority.
- Boundary11a: target MLX runtime Exp(1) RNG distribution/reproducibility authority.

Boundary11a does not supersede Boundary9/10.

## Target runtime inventory

Recorded target:

- MLX version: `0.32.2`
- Python: `3.13.15`
- macOS: `26.5.2`
- target stream/device: Metal GPU / `mx.gpu`, default `Device(gpu, 0)`
- available `mx.random` APIs: `bernoulli`, `categorical`, `gumbel`, `key`, `laplace`, `multivariate_normal`, `normal`, `permutation`, `randint`, `seed`, `split`, `truncated_normal`, `uniform`
- explicit key API present: `mx.random.key`, `mx.random.split`
- global seed API present: `mx.random.seed`
- uniform API present: `mx.random.uniform`, documented half-open `[low, high)`
- random integer API present: `mx.random.randint`
- native exponential API: absent

## RNG state policy

Production contract uses explicit keys, not hidden global RNG state:

```text
session_key -> mx.random.split(session_key, 2) -> [draw_key, next_session_key]
draw_key -> mx.random.uniform(..., key=draw_key, stream=mx.gpu)
next_session_key becomes the session state for the next sample
```

`mx.random.seed(...)` is inventoried but is not the production contract while explicit key/split APIs are available.

## Exp(1) construction

Because `mx.random.exponential` is absent, Boundary11a uses inverse-CDF construction:

```text
U ~ uniform(low=2^-24, high=1, dtype=float32, key=draw_key)
E = -log1p(-U)
E dtype = FP32
```

Endpoint policy is fixed before execution:

- `uniform` is documented as `[low, high)`.
- `low=2^-24` avoids `U=0`, so `E` is strictly positive.
- `high=1` and half-open semantics avoid `U=1`, so `E` is finite.
- This tiny lower truncation is recorded as the target-runtime discretization policy and is not adjusted after seeing results.

The theoretical authority is `Exp(rate=1)`, with `F(x)=1-exp(-x)`, mean `1`, variance `1`, and `Q(p)=-log(1-p)`.

## Qualification gates

Primary statistical fixture uses `N=1,048,576` with seeds/keys A and B. DKW is predeclared:

```text
alpha = 1e-6
epsilon = sqrt(log(2/alpha)/(2N)) = 0.0026302598969264743
PASS: KS <= epsilon
```

Mean sanity gate is independent:

```text
abs(sample_mean - 1) <= 6/sqrt(N) = 0.005859375
```

Production-shape fixture is `[1,129280]`, FP32, same shape as DeepSeek `probs/noise` path.

## Safe claim on PASS

For the pinned target MLX runtime and device, the native sampling-noise generator is qualified for the DeepSeek sampling contract as an FP32, strictly-positive Exp(rate=1) generator under a predeclared distributional and reproducibility test suite. Its empirical CDF, mean, positivity, finiteness, and same-key reproducibility satisfy the declared bounds.

This is a target-runtime distributional/reproducibility qualification. It does not claim bitwise equivalence with PyTorch `exponential_(1)`, cross-backend RNG parity, or equality of sampled tokens for equal numeric seeds.

## Non-claims

- no PyTorch/MLX RNG bitwise parity
- no CPU/GPU RNG sequence parity
- no cross-version RNG sequence guarantee
- no actual sampled-token authority
- no sampling distribution qualification beyond tested MLX runtime/version/device
- no main_hidden assembly
- no full `Transformer.forward` return
- no decode/cache qualification
- no full-model correctness
- no performance/production qualification
