# Boundary 12b5: Engram-connected sampling rebind

Status: PASS.

This boundary rebinds existing Boundary10 sampling arithmetic contracts and Boundary11a MLX RNG/key semantics to the current Boundary12b4 Engram-connected logits. It does not design a new sampling algorithm.

## Current logits regression

The runner starts from `tokens [[0,3]]` and regenerates the Engram-connected path through logits. No logits artifact tensor is injected.

- logits digest: `7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd`
- shape: `[1,129280]`
- dtype: FP32
- max: `13.22089958190918`
- argmax token: `15`

## Branch A: temperature=0

Source argmax branch selected.

- output_ids: `[15]`
- native token: `15`
- independent full-vocab scan token: `15`
- max logit: `13.22089958190918`
- tie count: `1`

Classification: current Engram-connected explicit-temperature-zero sampling result.

## Branch B: supplied-noise arithmetic

Temperature `1.0`; official RNG is not executed. The deterministic supplied-noise fixture is regenerated.

- noise digest: `fe4cf22185a06020968728fcd88706683ac8c16a6f1b0b4f1f561893ee5a629f`
- all positive/finite: true
- softmax digest: `4262dddbd4432beb9aeaab953b6b8435f5ccbfd34d345028ddd611f181d43e4d`
- conditional token: `795`
- independent log-domain token: `795`
- score gap: `0.06666707992553711`
- log-score gap: `0.060778134064285894`
- tie count: `1`

This is conditional arithmetic evidence, not an official RNG draw.

## Branch C: target MLX stochastic sampling

Runtime:

- MLX: `0.32.2`
- device: Metal GPU / `mx.gpu`

Primary seed/key:

- seed: `289513473`
- session key digest: `39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630`
- draw key digest: `4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1`
- next key digest: `175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4`

MLX noise:

- digest: `38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4`
- all positive/finite: true

Logits seam:

- host digest: `7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd`
- MLX roundtrip digest: `7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd`

Sampling result:

- target MLX token: `9468`
- independent log-domain token: `9468`
- score gap: `3.0003824830055237`
- log-score gap: `1.7397182473531227`
- tie count: `1`

Same-key repeat is exact for logits, noise, next key, and sampled token.

Secondary seed `289517570`:

- secondary noise digest: `245a54e0bd3bef90f644457cf32d137dd7d804febbe233ac44eaf8aaf7d91265`
- secondary sampled token: `74294`
- only noise difference is gated; token difference is not required.

## Historical token non-use guard

Old no-Engram tokens `372`, `795`, `9468`, and `74294` are historical/provenance evidence only and are not expected values for this Engram-connected trajectory. The Branch B/C tokens happen to match prior IDs for this fixture, but gates are winner-agreement gates against freshly regenerated current logits/noise, not historical-token expected-value gates.

## STOP

Stopped after target MLX sampled token and next session key. Next source operation is:

```text
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

Not executed:

- main_hidden concat
- Transformer.forward return packaging
- decode

## Non-claims

- no PyTorch/MLX RNG bitwise parity
- no backend-independent sampled-token identity
- no main_hidden concat authority
- no Transformer.forward return correctness
- no incremental/decode NgramHashState correctness
- no False/image-mask DEAD crossing authority
- no distributed/world_size>1 correctness
- no long-context qualification
- no full-model correctness
- no performance/production qualification

## Next boundary

Boundary 12c: main_hidden captures at pre-Block37/38/39 -> `mean(dim=2)` -> `concat(dim=-1)`.
