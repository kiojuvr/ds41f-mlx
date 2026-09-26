# Boundary 10 closeout

Status: PASS.

Boundary10 closes sampling arithmetic branch semantics layered on Boundary9 logits. It does not add new model math or official RNG implementation authority.

## Authority hierarchy

- Boundary9 remains the current integrated deterministic model-forward authority through final-position logits.
- Boundary10a validates `Boundary9 logits -> explicit temperature=0 argmax branch`.
- Boundary10b validates `Boundary9 logits -> temperature=1.0 arithmetic -> supplied-noise conditional result`.

Boundary10 does not supersede Boundary9.

## Source identity

Sampling source: `inference/model.py` lines 1285-1292

- SHA256: `da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a`
- method: `raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1`

Contract:

```text
if temperature == 0:
    argmax(logits)
else:
    logits / max(temperature, 1e-5)
    -> FP32 softmax
    -> Exp(1) noise in official source
    -> probs / noise
    -> argmax
```

## Boundary10a summary

Artifact: `artifacts/native-sampling-temperature-zero-validation.json`  
Commit: `e2f88cb2c9cc78765c6084a589b5da741e31a3ca`

Classification: validated explicit parameterized deterministic sampling branch.

Source/default temperature remains `1` / `1.0`; the executed fixture was explicit `temperature = 0.0`. Therefore this is not default `Transformer.forward` sampling authority.

Facts:

- input logits digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`
- max logit: `12.943254470825195`
- argmax: `372`
- tie count: `1`
- output ids: `[372]`
- dtype: `int64`
- shape: `[1]`
- native argmax and independent scan agree

## Boundary10b summary

Artifact: `artifacts/native-sampling-supplied-noise-validation.json`  
Commit: `a05d8bacc4017b2b836a30055408c9dd13cd2b92`

Classification: official-reference-derived sampling arithmetic conditional on supplied deterministic positive noise.

Not classified as official RNG authority or default sampled-token authority.

Temperature:

- input: `1.0`
- effective: `1.0`
- temperature floor reviewed: true
- temperature floor exercised: false

Supplied-noise fixture:

```text
V = 129280
j_i = (65537 * i + 17) mod V
u_i = (j_i + 0.5) / V      # float64
noise_i = float32(-log1p(-u_i))
```

Noise facts:

- shape: `[1, 129280]`
- dtype: FP32
- digest: `fe4cf22185a06020968728fcd88706683ac8c16a6f1b0b4f1f561893ee5a629f`
- min: `3.8675816540489905e-06`
- max: `12.462882995605469`
- mean: `0.9999973190020374`
- all positive: true
- all finite: true

Arithmetic evidence:

- scaled logits digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`
- softmax digest: `be8593220e2e73092a1a129875f8d331fea446af127b1ef6704aee607f2cca26`
- softmax sum: `0.9999999429814543`
- softmax min: `4.329862596258059e-15`
- softmax max: `0.0354914627969265`
- conditional scores digest: `5bdcb0b61b5a72eba6ed429b091e4144356b0b335a4f3df3d5f0fc48f451c564`
- conditional output id: `795`

Softmax comparison policy:

- predeclared `softmax_max_abs_lte = 1e-6`
- predeclared `softmax_sum_abs_error_lte = 1e-6`
- actual max abs: `1.9247323179705234e-09`
- actual sum abs error: `5.701854566275699e-08`
- PASS

No bit-exact softmax claim is made.

Independent winner check:

- source-order path: FP32 softmax -> supplied-noise division -> argmax
- source-order winner: `795`
- independent path: `argmax(float64(logit_i)/T - log(float64(noise_i)))`
- independent winner: `795`

Winner margins:

- source top1 token: `795`, score `1.047795057296753`
- source top2 token: `45039`, score `0.7252915501594543`
- source gap: `0.3225035071372986`
- independent top1 token: `795`, log-score `16.328405665299115`
- independent top2 token: `45039`, log-score `15.960536056701747`
- independent gap: `0.3678696085973687`
- tie count: `1`

General tie semantics remain open.

## RNG boundary

Machine-readable closeout fixes:

```text
official_rng_call_in_source = torch.empty_like(probs).exponential_(1)
official_rng_call_executed = false
rng_draw_replaced_by_supplied_fixture_for_arithmetic_validation = true
supplied_noise_is_not_claimed_as_official_rng_output = true
```

Still open:

- torch exponential implementation
- RNG state
- seed semantics
- manual_seed interaction
- device/backend RNG algorithm
- CPU/CUDA differences
- MLX RNG algorithm
- PyTorch/MLX RNG parity
- exact draw sequence
- cross-backend sampled-token equality
- distributional qualification
- reproducibility across devices

## STOP

For both 10a and 10b, the next model source operation after sampling is:

```python
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

Boundary10 keeps:

- main_hidden assembly: not executed
- Transformer.forward return: not executed

For 10b, official RNG was replaced by supplied noise, so it is not a literal source execution reaching `main_hidden`.

## Closed claims

Closed for scope `B=1`, `S=2`, tokens `[[0,3]]`, `start_pos=0`, prefill, `world_size=1`, `full_logits=False`:

- sample source contract review
- temperature==0 branch arithmetic
- temperature==0 connected argmax result for explicit fixture
- nonzero temperature scaling at `T=1`
- FP32 softmax arithmetic
- probability/noise division arithmetic
- conditional argmax semantics
- supplied-noise producer/consumer arithmetic contract
- independent log-domain winner equivalence for the supplied-noise fixture

10b additionally requires supplied noise digest `fe4cf22185a06020968728fcd88706683ac8c16a6f1b0b4f1f561893ee5a629f`.

## Remaining open claims

Not closed:

- official RNG generation
- actual default stochastic sampled token
- full default sample path execution
- sampling distribution correctness
- temperature floor exercised behavior
- RNG seed/reproducibility
- cross-backend RNG equivalence
- main_hidden assembly
- Transformer.forward return
- decode

Boundary10b token `795` is a conditional output id, not an official model output token.

## Safe closeout claim

Boundary 10 closes two source-defined sampling arithmetic branches on top of the existing Boundary9 connected logits authority:

1. an explicit temperature=0 deterministic argmax fixture, producing token `372`; and
2. the temperature=1.0 nonzero sampling arithmetic conditional on a predeclared deterministic positive-noise fixture, for which the source-order FP32 softmax/division path and an independent FP64 log-domain reconstruction both select conditional token `795`.

Boundary 10 does not validate torch exponential RNG generation, RNG state/seed/device/backend behavior, MLX/PyTorch RNG parity, the actual default stochastic sampled token, or sampling distribution correctness. Boundary9 remains the integrated deterministic model-forward authority through logits; Boundary10a/10b are branch-specific sampling authorities.
