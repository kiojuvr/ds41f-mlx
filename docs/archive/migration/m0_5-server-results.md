# M0.5 oMLX production-server short performance results

M0.5 is now closed after configuration reconciliation. See `docs/m0_5-closeout.md`.

The diagnostic direct `generate_step` path was rejected as non-representative. M0.5 measurement was moved to the oMLX production server topology via `tools/run_short_perf_omlx_server.py`.

The runner starts:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/omlx serve \
  --model-dir /Volumes/KIOXIA-PRO-1/models/deepseek-ai \
  --memory-guard off --no-cache
```

and writes a per-model `model_settings.json` forcing `deepseek_v41_engram_ssd_offload=true`.

## short prompt + 128 decode

Artifact:

```text
artifacts/m0_5/short-perf-omlx-server/short-128-count.json
```

Prompt: `Count upward from 1, one number per line. Do not stop early.`

Result:

- prompt tokens: 46
- completion tokens: 128
- model load duration: 84.16 s
- prefill: 2.17 s / 21.18 tok/s
- decode: 6.51 s / 19.65 tok/s / 0.0509 s TPT
- swap used delta: 0 bytes
- swapins/swapouts delta: 0 / 0
- compression/decompression delta: 0 / 0

## approximately 2K prompt + 128 decode

Artifact:

```text
artifacts/m0_5/short-perf-omlx-server/prompt2k-128-count-r125.json
```

Prompt: same line repeated 125 times.

Result:

- prompt tokens: 2030
- completion tokens: 128
- model load duration: 82.46 s
- prefill: 12.06 s / 168.27 tok/s
- decode: 6.57 s / 19.49 tok/s / 0.0513 s TPT
- swap used delta: 0 bytes
- swapins/swapouts delta: 0 / 0
- compression/decompression delta: 0 / 0

## Interpretation

The production-server path is much faster than the rejected direct `generate_step` diagnostic path and reproduces the expected order of magnitude for bounded prefill. Decode is stable around 19.5 tok/s on these short 128-token runs, below the user-provided long-context oMLX baseline of ~29-37 tok/s.

Follow-up investigation found that these two runs used the temporary minimal per-model settings and therefore omitted the known-good `mtp_enabled=true` setting. They did **not** activate the DeepSeek-V4.1 embedded DSpark/MTP fast decode path. See `docs/m0_5-config-gap.md`.

These artifacts remain useful as the non-MTP production-server baseline and swap sanity check, but they are not configuration-identical to the known-good oMLX decode baseline.

Configuration-identical follow-up runs using the historical per-model settings are recorded in:

```text
artifacts/m0_5/short-perf-omlx-server-config-check/short-128-known-good-settings.json
artifacts/m0_5/short-perf-omlx-server-config-check/prompt2k-128-known-good-settings.json
```

Those runs activate DeepSeek-V4.1 embedded DSpark/MTP and reconcile against historical short-context oMLX fixtures; see `docs/m0_5-closeout.md`.
