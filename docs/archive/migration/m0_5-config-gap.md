# M0.5 decode configuration gap

This investigation compared the known-good oMLX per-model configuration in
`$HOME/.omlx/model_settings.json` against the temporary M0.5 server base in
`artifacts/m0_5/omlx-server-base/model_settings.json`.

## Finding

The ~19.5 tok/s M0.5 production-server decode runs were not using the same
DeepSeek-V4.1 decode path as the known-good oMLX environment.

The temporary base wrote only:

```json
{
  "deepseek_v41_engram_ssd_offload": true,
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 128
}
```

The known-good per-model settings include:

```json
{
  "deepseek_v41_engram_ssd_offload": true,
  "max_context_window": 1048576,
  "mtp_enabled": true,
  "vlm_mtp_enabled": false,
  "moe_expert_offload_enabled": false,
  "dflash_enabled": false
}
```

The material decode-path difference is `mtp_enabled=true`.  For the official
source checkpoint this causes oMLX's DeepSeek-V4.1 loader to preserve the
DSpark draft weights and enables embedded DSpark speculative decoding.  With the
minimal temporary settings, `mtp_enabled` defaults false, the loader drops the
DSpark draft path, and decode falls back to ordinary single-token decoding.

## Log evidence

Minimal temporary M0.5 base (`~19.5 tok/s`) has no speculative-backend or MTP
activation log:

```text
VLMBatchedEngine loaded: .../DeepSeek-V4.1-Flash
Chat completion: model=DeepSeek-V4.1-Flash, 128 tokens in 18.63s (19.5 tok/s), prompt: 2030
```

Known-good per-model settings copied into the same temporary server topology
activate DSpark/MTP:

```text
Speculative backend selected for .../DeepSeek-V4.1-Flash: embedded DSpark (model_type=deepseek_v41, active)
MTP path activated for uid=0 (model has mtp_forward, batch=1, primed=46)
MTP[0] finish=length tokens=128 cycles=57 tok/cycle=2.25 accept=70/101 (69.3%) depth[d1=44/57,d2=19/34,d3=7/10]
Chat completion: model=DeepSeek-V4.1-Flash, 128 tokens in 6.97s (27.2 tok/s), prompt: 46
```

## Reproduction artifacts

The runner now accepts `--model-settings-source` to copy an existing
`model_settings.json` into an isolated `--base-path` without modifying the user
base.

Artifacts:

```text
artifacts/m0_5/short-perf-omlx-server-config-check/short-128-known-good-settings.json
artifacts/m0_5/short-perf-omlx-server-config-check/short-128-known-good-settings.json.server.log
artifacts/m0_5/short-perf-omlx-server-config-check/prompt2k-128-known-good-settings.json
artifacts/m0_5/short-perf-omlx-server-config-check/prompt2k-128-known-good-settings.json.server.log
```

Results with the same `omlx serve --memory-guard off --no-cache` topology but
known-good per-model settings:

| Workload | Decode | Prefill | Notes |
| --- | ---: | ---: | --- |
| short + 128 | 27.17 tok/s | 20.41 tok/s | DSpark/MTP active, 69.3% accept |
| ~2K + 128 | 24.46 tok/s | 167.56 tok/s | DSpark/MTP active, 65.0% accept |

The short 128-token result removes the M0.5 gap once it is compared to the corresponding historical short-context fixtures rather than to the long-context 29-37 tok/s range. See `docs/m0_5-closeout.md` and `artifacts/m0_5/historical-reconciliation.json`.

## M0.5 status

Closed. The primary cause of the 19.5 tok/s result was the temporary base omitting the saved `mtp_enabled=true` setting. With known-good settings, the M0.5 short-context measurements match corresponding historical short-context oMLX fixtures within normal run/content/MTP-acceptance variance. Do not optimize further for M0.5.
