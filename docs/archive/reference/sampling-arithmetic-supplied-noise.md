# Boundary 10b: nonzero-temperature sampling arithmetic with supplied noise

Status: PASS for source-defined nonzero-temperature sampling arithmetic at `temperature=1.0`, conditional on an explicit deterministic positive supplied-noise fixture.

This is not official RNG generator validation.

## Authority relationship

- Boundary9: current integrated deterministic model-forward authority through final-position logits.
- Boundary10a: validated explicit `temperature=0` deterministic sampling branch.
- Boundary10b: validates `temperature>0` arithmetic conditional on supplied noise.

Boundary10b does not supersede Boundary10a and is not default stochastic sampling authority.

## Source contract

Pinned local source: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.

Hash method: `raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1`.

`inference/model.py` lines 1285-1292:

- SHA256: `da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a`

Reviewed contract:

```python
if temperature == 0:
    return logits.argmax(dim=-1)

logits = logits / max(temperature, 1e-5)
probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

Boundary10b factors the source RNG call into an explicit supplied fixture:

```text
official source: torch.empty_like(probs).exponential_(1)
Boundary10b:  deterministic supplied-noise fixture
```

## Temperature

Primary fixture:

```text
temperature = 1.0
effective_temperature = max(1.0, 1e-5) = 1.0
```

Reason: `ModelArgs.temperature` source default is `1`, and `generate.py` default is `1.0`.

Classification: source-default-temperature arithmetic conditional on explicit supplied noise.

Temperature floor was reviewed but not exercised.

## Supplied-noise fixture

For `V = 129280` and token index `i = 0..V-1`:

```text
j_i = (65537 * i + 17) mod V
u_i = (j_i + 0.5) / V      # float64
noise_i = float32(-log1p(-u_i))
```

Shape: `[1, 129280]`  
Dtype: FP32

Classification:

- deterministic permutation of Exp(1) quantile representatives
- positive supplied-noise arithmetic fixture
- not an RNG draw
- not a distributional sampling test

Observed fixture:

- digest: `fe4cf22185a06020968728fcd88706683ac8c16a6f1b0b4f1f561893ee5a629f`
- min: `3.8675816540489905e-06`
- max: `12.462882995605469`
- mean: `0.9999973190020374`
- all positive: true
- all finite: true

## Connected execution

Runner: `tools/run_native_sampling_supplied_noise_validation.py`  
Artifact: `artifacts/native-sampling-supplied-noise-validation.json`

Execution starts from token IDs `[[0, 3]]` and regenerates connected Boundary9 logits. No logits artifact tensor is loaded.

Boundary9 logits regression:

- shape: `[1, 129280]`
- dtype: FP32
- digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`

Boundary10a regression remains PASS:

- temperature=0 output token: `372`

## Source-order arithmetic evidence

Source-order path:

```text
scaled_logits = logits / effective_temperature
probs = FP32 softmax(scaled_logits, dim=-1)
scores = probs / supplied_noise
conditional_output_id = argmax(scores, dim=-1)
```

Observed:

- scaled logits digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`
- softmax probability digest: `be8593220e2e73092a1a129875f8d331fea446af127b1ef6704aee607f2cca26`
- softmax sum: `0.9999999429814543`
- softmax min: recorded in artifact
- softmax max: recorded in artifact
- conditional scores digest: `5bdcb0b61b5a72eba6ed429b091e4144356b0b335a4f3df3d5f0fc48f451c564`
- conditional output token id: `795`

## Independent validation

Independent winner oracle uses log-domain FP64 reconstruction:

```text
argmax(softmax(logits / T) / noise)
== argmax(float64(logit_i) / T - log(float64(noise_i)))
```

Result:

- source-order conditional output token: `795`
- independent log-domain output token: `795`
- exact winner match: PASS

Softmax reconstruction uses a separate FP64 max-subtracted exp and independent reduction.

Predeclared contract:

```text
softmax_max_abs_lte = 1e-6
softmax_sum_abs_error_lte = 1e-6
```

Observed:

- native-vs-independent max abs: `1.9247323179705234e-09`
- softmax sum abs error: `5.701854566275699e-08`
- PASS

Winner margin:

- source top1 token: `795`
- source top1 score: `1.047795057296753`
- source top2 token: `45039`
- source top2 score: `0.7252915501594543`
- source top1-top2 gap: `0.3225035071372986`
- independent top1 log-score: `16.328405665299115`
- independent top2 token: `45039`
- independent top2 log-score: `15.960536056701747`
- independent log-score gap: `0.3678696085973687`
- tie count: `1`

General tie semantics remain a non-claim.

## Producer -> consumer seam

```text
Boundary9 connected logits -> Boundary10b source-order nonzero sampling arithmetic
```

- producer logits digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`
- consumer observed logits digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`
- same tensor dataflow: true
- artifact tensor injection: false

## RNG boundary

Required guard values:

- `official_rng_call_executed = false`
- `rng_draw_replaced_by_supplied_fixture_for_arithmetic_validation = true`
- `supplied_noise_is_not_claimed_as_official_rng_output = true`

Unverified and explicitly out of scope:

- `torch.empty_like(probs).exponential_(1)`
- RNG state
- seed semantics
- device/backend RNG algorithm
- PyTorch <-> MLX RNG parity
- actual random draw

## STOP

Stopped after computing:

```text
conditional_output_id = argmax(probs / supplied_noise)
```

Exact next source operation not executed:

```python
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

## Safe claim

For the existing B=1, S=2, start_pos=0, world_size=1, full_logits=False token fixture `[[0,3]]`, the connected Boundary9 final-position logits are validated through the source-defined nonzero-temperature sampling arithmetic at temperature `1.0` conditional on a predeclared supplied positive-noise fixture. The source-order FP32 softmax/division path and an independent log-domain winner reconstruction select the same conditional output token.

conditional output token id = `795`.

This does not validate torch exponential RNG generation, RNG state, seed/device/backend behavior, MLX/PyTorch RNG parity, or the actual default stochastic sampled token. The supplied noise fixture is an arithmetic test input, not an asserted official RNG draw.

## Non-claims

No official exponential RNG correctness, no torch RNG state validation, no manual_seed semantics, no cross-backend RNG parity, no actual default stochastic sampled-token equality, no sampling distribution qualification, no temperature-floor execution qualification, no main_hidden assembly, no full-sequence logits, no world_size>1 head/all_gather, no decode/cache/ring/partial compression-group semantics, no multi-call cache persistence, no distributed Block/MoE semantics, no Engram, no MTP/DSpark, no long-context qualification, no full `Transformer.forward` correctness, no full-model correctness, and no performance/production qualification.

Next step: Boundary 10 closeout before any actual exponential RNG generation boundary.
