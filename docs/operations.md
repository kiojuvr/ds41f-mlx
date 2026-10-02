# Operations and configuration

This is the operator-facing path for the scoped text runtime. Historical milestone documents are not required for normal startup.

## Dependencies

Required local components:

- official DeepSeek-V4.1-Flash checkpoint;
- oMLX checkout used by the qualified GenerationBatch decode path;
- DeepSeek `deepseek-recipe` checkout containing the V4.1 tokenizer and git revision evidence;
- Python environment with FastAPI/uvicorn, MLX, mlx-lm, and an importable `deepseek-recipe` package/native extension available;
- SSD-backed KV artifact location for persistence.

The runtime does not fall back to another checkpoint, quantized model, recipe implementation, or backend if these are missing.

## Configuration contract

All machine-specific paths are overrideable without source edits:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Checkpoint path | `DS41F_CHECKPOINT` | `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash` |
| oMLX checkout | `DS41F_OMLX_PATH` | `~/omlx-0.7.0.release` |
| deepseek-recipe checkout | `DS41F_RECIPE_PATH` | `/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe` |
| KV persistence root | `DS41F_KV_ROOT` | `/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv` |
| Host | `DS41F_HOST` | `127.0.0.1` |
| Port | `DS41F_PORT` | `8000` |
| Maximum live sessions | `DS41F_MAX_LIVE_SESSIONS` | `4` |
| Trace history limit | `DS41F_TRACE_HISTORY_LIMIT` | `32` |
| Diagnostic endpoints | `DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS` | `0` |
| Model id | `DS41F_MODEL_ID` | `deepseek-v4.1-flash` |
| Python used by Rust-owned server spawn | `DS41F_PYTHON` (or legacy alias `DS41F_RUNTIME_PYTHON`) | caller's `python3` unless set |

The single configuration authority is `ds41f_mlx.config.RuntimeConfig`. Server startup, provenance inspection, and unified qualification resolve settings through this seam.

## Qualified dependency and operational checkout

M20 promotes clean upstream `v0.7.0` at
`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, with Python 3.13.15,
MLX 0.32.2 and mlx-lm `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`
(distribution version `0.31.4.dev132+g94cdcae13`). Use
`~/.venvs/omlx-0.7.0.release/bin/python` for the commands below.

`~/omlx-0.7.0.release` remains the versioned qualification authority/default;
`~/omlx-0.7.0.dev2` remains the unchanged historical rollback checkout.
`~/omlx` is the intended normal operational/current path, but was not constructed
as part of this promotion. Once separately constructed, select it with
`DS41F_OMLX_PATH=$HOME/omlx` and verify against the M20 artifact:

```bash
python3 -m ds41f_mlx.qualify --check-artifact artifacts/m20/promotion.json
```

The check compares revision/local content, production GLM native binary hashes,
package versions, ds41f runtime identity, recipe, checkpoint and selectors—not
an oMLX directory name. Rebuilding kernels, advancing a release or adding local
code does not silently inherit M20 qualification. Full build/package/native
records are in `artifacts/m20/release-environment.json`; see the
[M20 report](m20-omlx-release-migration.md) for evidence scope.

## Inspect provenance without loading the model

```bash
python3 -m ds41f_mlx.provenance
```

This reports ds41f commit/dirty state, Python/platform, MLX and mlx-lm versions, configured paths, checkpoint fingerprint, oMLX and recipe revisions, pinned revision checks, production selector, and MTP/DSpark/speculation OFF state.

Outcome meanings:

- `PASS`: configured environment matches required paths and pinned release revisions.
- `WARNING`: environment is usable enough to inspect, but a revision/dirty-state/path condition changed and requires review or requalification before claiming the same release evidence.
- `FAIL`: a required configured dependency/path is missing or invalid.

## Launch the server

```bash
python3 -m ds41f_mlx.serve
```

The launcher resolves configuration, adds the configured oMLX checkout to `sys.path`, validates paths, prints resolved production identity, imports the existing FastAPI app, and starts uvicorn. It prefers an installed `deepseek-recipe` package with its native extension; the configured recipe checkout remains the tokenizer/provenance source. No manual `PYTHONPATH` export is required for the configured oMLX checkout.

Useful variants:

```bash
python3 -m ds41f_mlx.serve --print-config
python3 -m ds41f_mlx.serve --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Unified qualification

Inspect-only:

```bash
python3 -m ds41f_mlx.qualify --mode inspect
```

Quick operational qualification:

```bash
python3 -m ds41f_mlx.qualify --mode quick
```

Full requalification mode, including the configured real-model gate:

```bash
python3 -m ds41f_mlx.qualify --mode full
```

Generated artifacts are written under `artifacts/m18/` unless `--output` is supplied. The repository ignores generated artifacts by default; a release manager may force-add a specific artifact when it is intended to become canonical release evidence. Qualification artifacts contain a tested runtime identity, not just a Git HEAD, so committing the artifact does not by itself invalidate the run.

Check whether current runtime contents still match a prior artifact without rerunning expensive model gates:

```bash
python3 -m ds41f_mlx.qualify --check-artifact artifacts/m18/<artifact>.json
```

Check whether current runtime contents are still covered by a migrated historical expensive-evidence attestation:

```bash
python3 -m ds41f_mlx.qualify --check-evidence artifacts/m18/<attestation>.json
```

Canonical server acceptance through the documented launcher:

```bash
python3 -m ds41f_mlx.acceptance
```

Rust boundary build/test:

```bash
cargo test
```

Rust clients should use `ds41f_api::Ds41fClient` to connect to the documented local server or `ds41f_api::RuntimeProcess` to spawn `python3 -m ds41f_mlx.serve`. Set `DS41F_PYTHON=$HOME/.venvs/omlx-0.7.0.release/bin/python` (or pass that path explicitly) for production use so the Rust-owned process uses the qualified M20 environment rather than system/Xcode Python. `RuntimeProcess::spawn*` waits only for process-alive `/health`; use `wait_model_ready()` when the application requires model-loaded inference readiness. `shutdown()` reports whether SIGTERM completed gracefully or a forced-kill fallback was needed. This does not change runtime ownership: model/session state remains in the server process.

Runtime identity invalidation rules:

- stale: ds41f runtime source digest changes;
- stale: oMLX base revision, local patch/content identity, production GLM native hashes, or runtime package versions change;
- stale: deepseek-recipe base revision or approved local patch/content identity changes;
- stale: checkpoint fingerprint or production selector/MTP/DSpark/speculation state changes;
- not stale by itself: generated qualification artifacts, documentation, or unrelated non-runtime repository state changes.

Outcome meanings:

- `ENVIRONMENT_VALID`: inspect-only config/provenance checks passed.
- `QUICK_RUNTIME_QUALIFIED`: provenance plus cheap/static/runtime gates passed; this is not a fresh long-context benchmark.
- `FULL_RELEASE_REQUALIFIED`: quick gates plus configured real-model qualification gates passed on this machine.
- `FAILED`: one or more required gates failed.

## Supported runtime behavior

See `api.md` and `release-qualification.md` for the exact supported API and release scope. The operations workflow does not broaden scope: vision, batching, MTP, DSpark, speculative decode, sessionized Responses/Messages, server-side tool execution, distributed serving, authentication, cross-runtime KV portability, and arbitrary stateful stop strings remain outside the release contract.
