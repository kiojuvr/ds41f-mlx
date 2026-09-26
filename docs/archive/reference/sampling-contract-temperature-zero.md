# Boundary 10a: sampling contract temperature==0

Status: PASS for the explicit parameterized `temperature=0` deterministic branch only.

## Authority relationship

- Boundary9 remains the integrated deterministic model-forward authority through final-position logits.
- Boundary10a validates only `Boundary9 connected logits -> sample(logits, temperature=0)`.
- This is not default full `Transformer.forward` authority and does not execute `main_hidden` assembly or return.

## Source identity

Hash method: `raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1`.

Pinned local source: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.

Key spans:

- `inference/model.py` `sample()` lines 1285-1292: `da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a`
- `inference/model.py` `ModelArgs` temperature default lines 45-54: `4ec99aba20c3ceb13e4ad5a8a5c6cad7e89930237dfdb73179df3a87f9be2fd8`
- `inference/model.py` `Transformer.__init__` temperature assignment lines 1187-1195: `fa7046a56f703828fe98e5f7d3ab00b291e508b912f2eeabe4bca286aa386bae`
- `inference/model.py` `Transformer.forward` logits/sample/main_hidden order lines 1242-1272: `6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c`
- `inference/generate.py` load/config temperature override lines 105-123: `7bb85f4fc58aaea4d1acec2f78f592472f62756fd852a3d77e7ee16090a30fe9`
- `inference/generate.py` CLI temperature path lines 205-216: `6ad035e65d3adf322839cbef91c21fefcafd2c0368bc030df2a6a1d90083283a`

## Sampling contract

Local source defines:

```python
if temperature == 0:
    return logits.argmax(dim=-1)

logits = logits / max(temperature, 1e-5)
probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
return probs.div_(
    torch.empty_like(probs).exponential_(1)
).argmax(dim=-1)
```

Boundary10a executed only the first branch. The `temperature>0` branch was source-reviewed but not validated.

## Temperature provenance

- Source dataclass default: `ModelArgs.temperature = 1`.
- `inference/config.json`: no `temperature` key.
- top-level `config.json`: no `temperature` key.
- `generate.py`: CLI `--temperature` default is `1.0`; `main()` assigns `args.temperature = temperature` before constructing `Transformer`.
- Boundary10a executed fixture value: explicit `0.0`.

Classification: explicit parameterized deterministic branch fixture; not default `Transformer.forward` sampling validation.

## Validation evidence

Runner: `tools/run_native_sampling_temperature_zero_validation.py`.
Artifact: `artifacts/native-sampling-temperature-zero-validation.json`.

Connected execution starts from token IDs `[[0, 3]]`, regenerates Boundary9 logits, and does not inject logits from an artifact.

Boundary9 regression:

- logits shape: `[1, 129280]`
- logits dtype: FP32
- logits digest: `b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d`

Temperature-zero sampling result:

- max logit: `12.943254470825195`
- argmax token: `372`
- tie count: `1`
- `output_ids` shape: `[1]`
- `output_ids` dtype: `int64`
- `output_ids` value: `[372]`

Native `argmax(dim=-1)` and an independent first-maximum scan agree. Because this fixture has `tie_count = 1`, no general tie-branch claim is made.

## STOP

Stopped after:

```python
output_ids = sample(logits, self.temperature)
```

Exact next source operation not executed:

```python
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

## Non-claims

No validation is claimed for `temperature>0` sampling correctness, exponential RNG correctness, cross-backend RNG parity, distributional sampling, default-temperature sampling, `main_hidden` assembly, full-sequence logits, `world_size>1`, decode/cache/ring semantics, multi-call cache persistence, distributed Block/MoE semantics, Engram, MTP/DSpark, long-context behavior, full `Transformer.forward`, full-model correctness, or performance/production use.

Next candidate: Boundary 10b, temperature>0 sampling arithmetic conditional on explicit supplied exponential noise, keeping RNG generation separate.
