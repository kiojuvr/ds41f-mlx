# M0.5 observation: direct generate_step short-128

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx.py \
  --prompt ping --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx/short-128.json
```

Result artifact:

```text
artifacts/m0_5/short-perf-omlx/short-128.json
```

Summary:

- load: 80.285 s
- prompt tokens: 31
- completion tokens: 128
- prefill: 7.895 s / 3.93 tok/s
- decode: 365.903 s / 0.350 tok/s / 2.859 s TPT
- peak MLX memory: 303,509,389,780 bytes
- active MLX memory: 301,129,104,944 bytes
- system swap used delta: 0 bytes
- swapins/swapouts delta: 0 / 0
- compression/decompression delta: 0 / 0

Decision:

This is not acceptable as a production-performance baseline and should not be used to advance long-context qualification. The result is materially inconsistent with the known oMLX 0.7.0.dev2 production benchmark reported for the same official checkpoint, where decode is ~29-37 tok/s at long contexts.

The likely cause is that `tools/run_short_perf_omlx.py` drives `mlx_lm.generate_step` directly over `model.language_model`, bypassing the oMLX production generation/scheduler path and any relevant production settings. Treat this run as a diagnostic negative result for the naive direct path, not as the pinned oMLX production baseline.

Action:

Stop additional M0.5 canonical measurements, including the 2K+128 fixture, until the measurement path is changed to the actual known-good oMLX production benchmark/generation topology or an equivalent path proven to match it.
