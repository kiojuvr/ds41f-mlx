# M0 implementation plan

> Correctness-authority repair note: this document is historical. M0 oMLX output agreement is now classified as compatibility/reproduction evidence only, not official DeepSeek model correctness. See `docs/correctness-authority-audit.md`.

M0 reproduces the known-good oMLX DeepSeek-V4.1-Flash runtime in this repository without beginning new optimization.

## Source identities

- oMLX upstream baseline: clean `b390b31e0c6831225fed0f24d278eb1db7fcb68b`.
- oMLX local known-good: `$HOME/omlx-0.7.0.dev2`, same commit plus recorded DeepSeek-V4.1 image-token parser patch.
- DwarfStar: `$HOME/ds4`, performance architecture donor only; no M0 code import.
- Qualification archive: `/Volumes/SDXC-512/deepseek-v41-flash-mlx`, historical contract/artifact source requiring per-artifact provenance classification before reuse as an oracle.

## M0 deliverables

1. Source and checkpoint provenance artifacts.
2. Prompt/tokenizer fixtures proving the local known-good image-token parser behavior.
3. Thin oMLX runtime wrapper using the official checkpoint directly.
4. API/server contract migrated from the qualification archive.
5. `/health`, `/v1/models`, `/v1/chat/completions` passing through the new runtime core.
6. Short oMLX compatibility comparison artifacts.

## What M0 keeps from oMLX

- DeepSeek-V4.1 direct checkpoint loading.
- Processor/tokenizer behavior including local image-token parser patch.
- Language model execution path: CSA2, mHC, MoE, Engram, packed attention and grouped expert fast paths.
- Existing MLX execution topology and memory behavior.
- SSD-backed Engram embeddings by default on the M3 Ultra / 512 GB target; resident Engram loading is not the M0 default.

## What M0 imports from the oracle archive

- Provenance/checkpoint manifest methodology.
- API smoke contract.
- Historical correctness/invalid-request atomicity rules, subject to authority reclassification.
- Qualification record format.
- Negative-result knowledge.

## What M0 does not import from DwarfStar

- Metal kernels.
- command-buffer topology.
- expert-major scheduling.
- GGUF/quantized weight layout.
- SSD expert streaming.
- decode/prefill graph redesign.

DwarfStar comparison begins after M0 and after short-context performance viability is established.
