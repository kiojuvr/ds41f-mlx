# Known oMLX 0.7.0.dev2 baseline

User-provided oMLX 0.7.0.dev2 benchmark for the official checkpoint:

`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`

| Context | Prefill | Decode | TTFT | Peak Mem |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 193.6 tok/s | 35.5 tok/s | 169 s | 292.82 GB |
| 64K | 190.8 tok/s | 36.3 tok/s | 344 s | 292.85 GB |
| 128K | 176.4 tok/s | 29.0 tok/s | 743 s | 292.90 GB |
| 200K | 183.9 tok/s | 36.9 tok/s | 1088 s | 292.96 GB |

These values are the known long-context production baseline to reproduce/compare against for corresponding long-context fixtures. Do not treat the 29-37 tok/s range as a universal short-context decode target; prompt length, generated length, content, and DSpark/MTP acceptance materially affect short bounded runs.

For M0.5 short-context reconciliation, see `docs/m0_5-closeout.md` and `artifacts/m0_5/historical-reconciliation.json`. The naive direct `mlx_lm.generate_step` path measured in `docs/m0_5-observation-short-128.md` remains rejected as non-representative.
