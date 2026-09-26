# M0.5 short-context performance gate

M0.5 decides whether the current execution core is fast enough to justify spending resources on long-context qualification.

## Rule

Correctness and API success are not sufficient. Do not start 32K / 64K / 128K / 200K / 256K validation if short-context performance is materially and unexplainedly worse than the pinned oMLX known-good baseline.

## Initial workload

Use the official checkpoint and SSD-backed Engram. Run bounded short prompts only. The canonical M0.5 fixture has two points:

1. short chat prompt + 128 decode tokens, e.g. `ping`, for steadier decode/TPT than smoke-length generation.
2. approximately 2K-token prompt + 128 decode tokens, for bounded prefill plus decode.

Do not add more points unless a concrete decision requires it. These are not long-context qualification runs.

Example commands:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx.py \
  --prompt ping --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx/short-128.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx.py \
  --prompt ping --target-prompt-tokens 2048 --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx/prompt2k-128.json
```

## Required measurements

Record:

- source and checkpoint identities
- model load time
- prompt token count
- generated token count
- prefill seconds and prefill tokens/s
- decode seconds, decode tokens/s, and TPT
- peak MLX memory
- active/cache memory after run
- process RSS/footprint if available
- system swap used delta
- vm_stat swapins/swapouts and compression/decompression deltas
- Engram mode

## Decision

- If current runtime is materially slower than pinned oMLX and the gap is unexplained, block long-context qualification.
- If the gap is caused by the temporary M0 API boundary but core direct oMLX is acceptable, record that and keep production API architecture deferred.
- If direct oMLX baseline itself cannot be reproduced, stop and fix baseline reproduction first.

## Tools

- `tools/run_short_perf_omlx.py`: direct diagnostic short performance record. It is useful for bounded measurement mechanics, memory/swap telemetry, and negative results, but it must not be assumed to represent the oMLX production benchmark unless it is shown to match the production generation topology.
- `tools/compare_short_perf.py`: compares baseline and candidate JSON records and blocks long-context qualification on material regressions.

## Current status

The first short-128 diagnostic direct run was materially slower than the known oMLX production baseline. See `docs/m0_5-observation-short-128.md`.

The measurement path was then moved to the oMLX production server topology. The first two bounded canonical fixtures used a temporary minimal per-model settings file and measured the non-MTP path; see `docs/m0_5-server-results.md` and `docs/m0_5-config-gap.md`.

M0.5 was then rerun with historical known-good per-model settings, reconciled against corresponding historical short-context oMLX fixtures, and closed. See `docs/m0_5-closeout.md`.
