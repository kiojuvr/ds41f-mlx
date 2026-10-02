# M26 — DSpark/MTP performance-gap attribution

## Result

The large M25-vs-published gap is explained primarily by **missing full DSpark prompt priming in the ds41f M25 diagnostic path**.

M25 deliberately kept production P7 unchanged: MTP was OFF during `DENSE_P0_P7`, and upstream generation saw only the held-out terminal token.  Stock oMLX does something materially different: with MTP active during normal prompt prefill, DeepSeek V4.1 captures selected target-layer hidden states across the prompt into `DSparkContextCache`; `take_primed()` transfers that draft context into generation after the first main token.  M25 artifacts show no retained host prime context at generation start.

This milestone does **not** change production MTP support, P7, persistence, serving, or the M25 lifecycle rejection.

## Key measurements

All new runs are diagnostic-only, sequential, direct upstream `BatchGenerator` runs with Engram SSD, prefix cache disabled, model load excluded, Python/code-style prompt, and 128 generated tokens except the quick release OFF 4K check (32 tokens).  Artifacts are under `artifacts/m26/`.

| Configuration | Context | OFF tok/s | MTP tok/s | Acceptance | Accepted/cycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| M25 ds41f P7, terminal-only priming | 2K | 19.93 | 24.69 | 70.6% | 1.38 |
| M25 ds41f P7, terminal-only priming | 200K | 19.08 | 19.98 | 48.2% | 0.79 |
| Stock oMLX 0.7.0, official checkpoint, full prompt priming | 4K | 19.94 | 42.68 | 94.5% | 4.48 |
| Stock oMLX 0.7.0, official checkpoint, full prompt priming | 64K | 19.60 | 41.51 | 94.5% | 4.48 |
| Stock oMLX dev2, official checkpoint, full prompt priming | 4K | 20.03 | 43.99 | 94.5% | 4.48 |

Published oQ4e-mtp was ~34-36 tok/s with MTP ON and ~19.6-20 tok/s OFF.  The official checkpoint under stock full prompt priming already exceeds that MTP throughput in this diagnostic, while OFF remains the same class.  Therefore the oQ4e checkpoint is not required to explain the M25 gap.

## Attribution

- **Priming:** dominant.  Changing from M25 terminal-only priming to stock full prompt priming changes the speculative path from modest/near-flat to ~2.1x over OFF on the same official checkpoint.
- **Checkpoint representation:** not reproduced for oQ4e because no local `DeepSeek-V4.1-Flash-oQ4e-mtp` checkpoint was available.  Available evidence says it is not the dominant cause of M25's low MTP result.
- **oMLX revision:** not dominant.  dev2 and final 0.7.0 are very close on the measured official-checkpoint path.
- **Workload/sampling:** secondary.  Stock full-primed greedy at 4K still reached 39.96 tok/s with 90.9% acceptance, far above M25.

## Decision

**A. MTP remains worth pursuing.**  A correctly configured native upstream path demonstrates a substantial practical speculative gain.  The next MTP milestone should address the no-replay committed-idle transition, while also preserving or recreating the upstream full DSpark prompt-priming state contract.

Production remains MTP/DSpark/speculative decode OFF; M25's lifecycle rejection is preserved unchanged.
