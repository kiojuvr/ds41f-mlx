# Rotary official-reference-derived fixture

This contract covers the reviewed official `precompute_freqs_cis` and
`apply_rotary_emb` source spans, records a bounded expected-value fixture, and
validates the native rotary data path against bounded no-YaRN and YaRN fixtures.

## Authority

Official reference source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

File SHA-256:

```text
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
```

Reviewed spans:

| Target | Lines | Source SHA-256 |
| --- | ---: | --- |
| `precompute_freqs_cis` | 369-389 | `19cf7246ca3e6c479158e68d4ab7825d078aa788d3ce494b0c90649bb40117fc` |
| `apply_rotary_emb` | 392-406 | `1c47da553d29f41c798a2eda82b72476473913d1201e32283de0e4977aa30b1a` |

## Bounded fixture contract

Fixture classification:

```text
official_reference_derived_independent_arithmetic_contract
```

Declared arithmetic contract:

```text
freqs = 1 / (base ** (arange(0, dim, 2, float32) / dim))
if original_seq_len > 0: apply the reviewed YaRN frequency interpolation branch
angles = outer(arange(seqlen), freqs)
freqs_cis = polar(ones_like(angles), angles)

apply_rotary_emb:
- view adjacent f32 element pairs as complex values
- multiply by per-position freqs_cis
- conjugate freqs_cis first when inverse=True
- view real/imag pairs and flatten back to the original last dimension
```

Current fixture scopes:

```text
no-YaRN fixture:
  shape: [batch=1, seqlen=4, heads=1, dim=8]
  original_seq_len: 0
  artifact: artifacts/rotary-official-reference-fixture.json

YaRN fixture:
  shape: [batch=1, seqlen=4, heads=1, dim=8]
  original_seq_len: 16
  artifact: artifacts/rotary-yarn-official-reference-fixture.json

input dtype: float32
expected output: float32 raw bit hex
```

Regenerate:

```sh
python3 tools/run_official_rotary_fixture.py \
  --out artifacts/rotary-official-reference-fixture.json

python3 tools/run_official_rotary_fixture.py \
  --original-seq-len 16 \
  --out artifacts/rotary-yarn-official-reference-fixture.json
```

## Native validation

Validation artifacts:

```text
artifacts/native-rotary-official-reference-validation.json
artifacts/native-rotary-yarn-official-reference-validation.json
```

Regenerate:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rotary_against_official_reference.py \
  --out artifacts/native-rotary-official-reference-validation.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rotary_against_official_reference.py \
  --reference artifacts/rotary-yarn-official-reference-fixture.json \
  --out artifacts/native-rotary-yarn-official-reference-validation.json
```

Observed no-YaRN result:

```text
bit_exact: false
max_abs_error: 1.1920928955078125e-07
max_f32_ulp_error: 2
within_predeclared_tolerance: true
native_version: ds41f-prefill-native-official-rotary-v7
```

Observed YaRN result:

```text
bit_exact: false
max_abs_error: 5.960464477539063e-08
max_f32_ulp_error: 2
within_predeclared_tolerance: true
native_version: ds41f-prefill-native-official-rotary-v7
```

The native path executes `precompute_freqs_cis`-equivalent frequency generation,
then `apply_rotary_emb` forward, then inverse application through Metal-backed
buffers. Bit-exact equality is not required because backend trigonometric
implementations may differ from the pure-Python fixture. The predeclared
tolerance is `max_abs_error_lte = 1e-6` and `max_f32_ulp_error_lte = 64`; the
observed maximum is 2 F32 ULP.

## Non-claims

This contract does not validate:

- YaRN behavior beyond the bounded `original_seq_len=16` fixture;
- attention, Engram, MoE, HC, DSpark/MTP;
- logits, cache/state, full layer, or full prefill correctness;
- oMLX behavior;
- full-model native correctness beyond the bounded rotary helper contracts.
