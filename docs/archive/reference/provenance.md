# Provenance contract

Every run must record enough identity to reproduce and audit the result.

## Required identities

- ds41f-mlx git commit and dirty status.
- oMLX upstream commit: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`.
- oMLX local known-good patch diff and SHA-256.
- DwarfStar commit, when used for comparison.
- Qualification archive commit and artifact identities, when used.
- Official checkpoint path, config/index identity, shard count, tensor count, payload size, and read-only status.
- Python, MLX, OS, hardware and environment relevant to execution.

## Local oMLX patch

`$HOME/omlx-0.7.0.dev2` intentionally includes the DeepSeek-V4.1 image-token parser fix. Treat this as a recorded local patch over upstream baseline, not as an untracked mystery dirty state.

M0 artifacts:

```text
artifacts/m0/source-identity.json
artifacts/m0/omlx-local-patch.diff
artifacts/m0/omlx-local-patch.sha256
artifacts/m0/checkpoint-identity.json
```

M1 reconstructed contract artifacts:

```text
artifacts/m1/checkpoint-provenance.json
artifacts/m1/runtime-identity.json
artifacts/m1/api-atomicity/run-*/result.json
artifacts/m1/summary.json
```

## Checkpoint policy

The official checkpoint directory is read-only source of truth. Runtime cache, generated artifacts, converted files, or repair outputs must not be written into it.
