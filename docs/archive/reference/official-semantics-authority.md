# Official DeepSeek semantics authority pin

This document records the candidate official semantics authority to replace the superseded oMLX-derived correctness contracts.

## Status

Classification: **CLEAN provenance only; not numerical validation**.

This pin records source identity. It does not execute model math, validate logits/cache/state, or authorize additional native model-math implementation by itself.

## Candidate official reference source

Local official checkpoint snapshot:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
```

M1 checkpoint provenance recorded Hugging Face revision:

```text
dba1be0a40aa45a94ad051997016db3960a90277
```

The checkpoint snapshot includes a readable reference implementation under `inference/`. The important pinned file identities are:

| File | SHA-256 |
| --- | --- |
| `inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `inference/generate.py` | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` |
| `inference/config.json` | `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809` |
| top-level `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |

Full machine-readable record:

```text
artifacts/official-semantics-authority.json
```

Function-level review metadata:

```text
artifacts/official-reference-review.json
```

See `docs/official-reference-review.md`.

Regenerate with:

```sh
python3 tools/record_official_semantics_authority.py \
  --out artifacts/official-semantics-authority.json
```

## Use rules

- Official checkpoint raw tensors/config/tokenizer remain the model-data authority.
- The pinned official reference implementation / published architecture is the model-semantics authority.
- oMLX remains compatibility/performance/implementation-donor material only.
- DwarfStar remains execution-architecture authority only.
- Future semantic fixtures must state exactly which official reference function/file and arithmetic contract produced the expected result.
- A fixture that compares against oMLX logits/cache/state remains `not_official_qualification` unless separately revalidated against this authority.

## Remaining gap

The local files still need an explicit review step before becoming executable semantic oracle code for broader native stages:

1. verify the local snapshot and M1 HF revision against the official DeepSeek release source;
2. map the minimal reference functions needed for the next native stage;
3. define arithmetic/dtype contracts independent of oMLX;
4. generate only bounded fixtures from those contracts.
