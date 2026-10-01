# Milestone 6 performance qualification status

Status: **qualified through 200K** (`M6_PERFORMANCE_QUALIFIED_200K`).

## Scope

Milestone 6 qualifies the existing dense P0-P7 runtime as a practical production-prefill/decode candidate at realistic context sizes. It is a measurement and qualification gate, not an optimization task and not a serving-integration task.

Production candidate under measurement:

- dense P0-P7 prefill path;
- `P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM`;
- `P7_ENGRAM_TILE = 2048`;
- oMLX `GenerationBatch` MTP-OFF decode;
- same live `DeepseekV41Cache` handoff through P5.

Explicitly excluded for M6:

- `DS41F_P8_TILE_NATIVE_CARRY=1`;
- selector promotion;
- Attention/MoE/HC profiling or optimization;
- `mx.compile` additions;
- custom Metal;
- MTP/DSpark.

P8 is closed with `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`. Tile-native code is retained experimental/default OFF and is not production-selected.

## Environment

Required qualification environment:

- Python 3.13.15;
- MLX 0.32.2;
- NumPy 2.3.5;
- oMLX 0.7.0.dev2, revision `b390b31e0c6831225fed0f24d278eb1db7fcb68b`;
- `preserve_mtp=False`;
- `engram_ssd_offload=True`;
- `moe_expert_offload_resident_fraction=None`.

The M6 result artifact records exact observed versions, git commit, checkpoint path/revision/digests where available, and hardware.

## Methodology

Canonical artifact:

```text
artifacts/m6/performance-qualification/result.json
```

Harness:

```text
tools/run_m6_performance_qualification.py
```

Long cases must be launched under:

```text
tools/qualification_supervisor.py
```

Required ladder:

```text
2048
8192
16384
32768
65536
131072
200000
```

For each case record:

- deterministic tokenizer-free valid token fixture and terminal holdout;
- serving-path prefill wall time from request-local prefill start through committed live `DeepseekV41Cache` ready for P5;
- effective prefill tok/s;
- P5 admission/bootstrap time;
- first generated token latency and total TTFT;
- bounded decode throughput, preferably 32 generated tokens;
- MLX active/cache/peak memory and process `ru_maxrss` before prefill, after prefill, after decode, and peak;
- P6 segment/state/lifecycle evidence;
- P7 Engram background reads and foreground fallback;
- P5 same-cache zero-replay/repack/export evidence;
- Engram history/frontier progression and donor cleanup evidence;
- watchdog result and stop reason, if any.

Stop the ladder on correctness failure, foreground Engram fallback, P5 replay/repack/export, decode `<15 tok/s` for long contexts, no semantic progress, memory/swap pathology, or worker failure. A stopped watchdog is a result, not an instruction to optimize.

## Comparison baseline

Recorded target-machine oMLX 0.7.0.dev2 baseline:

| Context | oMLX prefill tok/s | oMLX decode tok/s | oMLX TTFT | oMLX peak |
|---:|---:|---:|---:|---:|
| 32768 | ~193.6 | ~35.5 | ~169 s | ~292.82 GB |
| 65536 | ~190.8 | ~36.3 | ~344 s | ~292.85 GB |
| 131072 | ~176.4 | ~29.0 | ~743 s | ~292.90 GB |
| 200000 | ~183.9 | ~36.9 | ~1088 s | ~292.96 GB |

M6 does not require ds41f to beat every oMLX cell. Success means practical, stable, correct, memory-safe, no replay, and usable long-context operation.

## Results

Canonical artifact: `artifacts/m6/performance-qualification/result.json`.

Short contexts were run in one warm-model process. Major long contexts were run as separate supervised worker processes under `tools/qualification_supervisor.py`. Decode bound: 16 generated tokens for all cases.

| Context | Prefill wall | Prefill tok/s | P5/bootstrap | First token | TTFT | Decode tok/s | MLX active/cache/peak after decode | P7 bg/fg | Segments |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2.832 s | 723.27 | 4.917 s | 0.051 s | 7.800 s | 19.84 | 301.22 / 10.32 / 304.11 GB | 2 / 0 | 1 |
| 8192 | 9.303 s | 880.59 | 3.200 s | 0.057 s | 12.560 s | 18.65 | 301.44 / 75.85 / 311.10 GB | 8 / 0 | 1 |
| 16384 | 16.537 s | 990.73 | 3.231 s | 0.052 s | 19.820 s | 18.84 | 301.67 / 90.35 / 319.87 GB | 16 / 0 | 1 |
| 32768 | 34.732 s | 943.45 | 5.511 s | 0.056 s | 40.299 s | 18.71 | 301.35 / 85.18 / 319.64 GB | 32 / 0 | 2 |
| 65536 | 70.857 s | 924.90 | 5.601 s | 0.057 s | 76.515 s | 18.86 | 301.39 / 85.36 / 319.67 GB | 64 / 0 | 4 |
| 131072 | 149.494 s | 876.77 | 5.802 s | 0.058 s | 155.354 s | 18.90 | 301.47 / 86.00 / 319.73 GB | 128 / 0 | 8 |
| 200000 | 241.463 s | 828.29 | 9.364 s | 0.056 s | 250.883 s | 19.01 | 301.54 / 95.35 / 319.79 GB | 196 / 0 | 13 |

All cases preserved P5 zero replay/repack/export, final pre-handoff all-40 cache frontiers equal to the intended prefix length, Engram history present, P7 foreground fallback 0, and decode throughput above the `>=15 tok/s` practical gate.

## Failures / limitations

No M6 ladder case failed through 200K. Limitations remain:

- decode bound was 16 generated tokens for all cases, not 32;
- short contexts shared one warm-model process; long contexts used separate supervised processes;
- swap was not reported by invasive tooling; no memory/swap pathology was observed by the harness/supervisor boundaries;
- long-session robustness is out of M6 scope;
- many-hour sessions, repeated tool calls, many append cycles, save/restore, fork under live scheduler, and hundreds-of-turn memory stability remain later work;
- selector promotion is deferred to a separate finalization task.

## Qualification decision

Current decision: **M6_PERFORMANCE_QUALIFIED_200K**.

Recommended next frontier: production-prefill selector finalization followed by Milestone 7 serving qualification. Do not implement those in the M6 task.

Allowed final decisions after measurement:

```text
M6_PERFORMANCE_QUALIFIED_200K
M6_PERFORMANCE_QUALIFIED_TO_128K
M6_PERFORMANCE_QUALIFIED_TO_64K
M6_PERFORMANCE_QUALIFIED_TO_32K
M6_NOT_QUALIFIED
```
