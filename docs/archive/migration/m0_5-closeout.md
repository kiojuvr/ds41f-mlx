# M0.5 closeout

M0.5 is closed as a short-context oMLX production-server baseline reconciliation milestone. It does not qualify long context and does not introduce native/runtime optimizations.

## Question reconciled

The earlier M0.5 production-server measurements around `19.5 tok/s` were configuration-negative fixtures: they used a temporary minimal per-model settings file that omitted `mtp_enabled=true`, so DeepSeek-V4.1 embedded DSpark/MTP was inactive.

After copying the historical known-good per-model settings from `$HOME/.omlx/model_settings.json` into an isolated temporary server base, the same oMLX production-server topology activated embedded DSpark/MTP and measured:

| Fixture | Prompt tokens | Generation cap / output | Decode | Prefill | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| short prompt | 46 | 128 / 128 | 27.17 tok/s | 20.41 tok/s | `artifacts/m0_5/short-perf-omlx-server-config-check/short-128-known-good-settings.json` |
| ~2K prompt | 2030 | 128 / 128 | 24.46 tok/s | 167.56 tok/s | `artifacts/m0_5/short-perf-omlx-server-config-check/prompt2k-128-known-good-settings.json` |

Both runs used the official checkpoint, SSD-backed Engram, `omlx serve`, `--memory-guard off`, `--no-cache`, and known-good per-model settings including `mtp_enabled=true`, `deepseek_v41_engram_ssd_offload=true`, `dflash_enabled=false`, `vlm_mtp_enabled=false`, and `moe_expert_offload_enabled=false`.

## Historical reconciliation

Historical source:

```text
/Users/kioju/.omlx/logs/server.log.2026-09-12
artifacts/m0_5/historical-reconciliation.json
```

The historical 29-37 tok/s range was not a universal short-context baseline. It came from admin benchmark and server runs with different prompt lengths, generation lengths, and MTP acceptance rates.

Relevant historical points:

| Historical fixture | Prompt tokens | Generation | Approx decode | Notes |
| --- | ---: | ---: | ---: | --- |
| manual short server | 17 | 446 | 25.0, 29.0 tok/s | DSpark/MTP active; longer generation than M0.5 short |
| admin PP1024/TG128 | 1024 | 128 | ~24.82 tok/s | DSpark/MTP active; MTP accept 77/99 |
| admin PP4096/TG128 | 4096 | 128 | ~26.35 tok/s | DSpark/MTP active; MTP accept 83/101 |
| admin PP8192/TG128 | 8192 | 128 | ~26.11 tok/s | DSpark/MTP active; MTP accept 83/96 |
| admin PP16K+/TG128 | 16384+ | 128 | ~29-37 tok/s | Different context regime and often higher MTP acceptance |

The M0.5 short fixture at 27.17 tok/s is bracketed by the historical prompt-17 server runs. The M0.5 ~2K fixture at 24.46 tok/s matches the nearby historical PP1024/TG128 point and is within normal variance of PP4096/TG128 once prompt length/content and MTP acceptance are considered.

## Sampling and measurement definitions

- Historical admin benchmark single-request path used `temperature=0.0`, `top_p=1.0`, `max_tokens=128`, skip cache store, and reports decode separately from prefill.
- M0.5 server runner used `temperature=0.0` and records the server usage fields `generation_duration` / `generation_tokens_per_second`, which are separate from `prompt_eval_duration` / `prompt_tokens_per_second`.
- Current known-good-setting runs did not send an explicit request `top_p`, but with `temperature=0.0` and `force_sampling=false`, sampling is greedy and `top_p` is not a performance-relevant difference.
- The long-context 29-37 tok/s values must be compared only to corresponding long-context fixtures, not used as a short-context universal target.

## Decision

The current 27.17 / 24.46 tok/s measurements match the corresponding historical short-context oMLX fixtures within normal run/content/MTP-acceptance variance.

M0.5 is closed. Do not optimize further for M0.5. Future native/runtime candidates should compare against the reconciled M0.5 fixtures first; long-context qualification may proceed only under the existing correctness/provenance gates and with matching configuration recorded.
