# Operations and configuration

This is the operator-facing path for the scoped text runtime. Historical milestone documents are not required for normal startup.

## Dependencies

Required local components:

- official DeepSeek-V4.1-Flash checkpoint;
- oMLX checkout used by the qualified GenerationBatch decode path;
- DeepSeek `deepseek-recipe` checkout containing the V4.1 tokenizer;
- Python environment with FastAPI/uvicorn, MLX, mlx-lm, and recipe dependencies available;
- SSD-backed KV artifact location for persistence.

The runtime does not fall back to another checkpoint, quantized model, recipe implementation, or backend if these are missing.

## Configuration contract

All machine-specific paths are overrideable without source edits:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Checkpoint path | `DS41F_CHECKPOINT` | `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash` |
| oMLX checkout | `DS41F_OMLX_PATH` | `~/omlx-0.7.0.dev2` |
| deepseek-recipe checkout | `DS41F_RECIPE_PATH` | `/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe` |
| KV persistence root | `DS41F_KV_ROOT` | `/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv` |
| Host | `DS41F_HOST` | `127.0.0.1` |
| Port | `DS41F_PORT` | `8000` |
| Maximum live sessions | `DS41F_MAX_LIVE_SESSIONS` | `4` |
| Trace history limit | `DS41F_TRACE_HISTORY_LIMIT` | `32` |
| Diagnostic endpoints | `DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS` | `0` |
| Model id | `DS41F_MODEL_ID` | `deepseek-v4.1-flash` |

The single configuration authority is `ds41f_mlx.config.RuntimeConfig`. Server startup, provenance inspection, and unified qualification resolve settings through this seam.

## Inspect provenance without loading the model

```bash
python3 -m ds41f_mlx.provenance
```

This reports ds41f commit/dirty state, Python/platform, MLX and mlx-lm versions, configured paths, checkpoint fingerprint, oMLX and recipe revisions, pinned revision checks, production selector, and MTP/DSpark/speculation OFF state.

Outcome meanings:

- `PASS`: configured environment matches required paths and pinned release revisions.
- `WARNING`: environment is usable enough to inspect, but a revision/path condition changed and requires requalification before claiming the same release evidence.
- `FAIL`: a required configured dependency/path is missing or invalid.

## Launch the server

```bash
python3 -m ds41f_mlx.serve
```

The launcher resolves configuration, adds the configured oMLX and recipe checkouts to `sys.path`, validates paths, prints resolved production identity, imports the existing FastAPI app, and starts uvicorn. No manual `PYTHONPATH` export is required for the configured checkouts.

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

Generated artifacts are written under `artifacts/m16/` unless `--output` is supplied. The repository ignores generated artifacts by default; a release manager may force-add a specific artifact when it is intended to become canonical release evidence.

Outcome meanings:

- `ENVIRONMENT_VALID`: inspect-only config/provenance checks passed.
- `QUICK_RUNTIME_QUALIFIED`: provenance plus cheap/static/runtime gates passed; this is not a fresh long-context benchmark.
- `FULL_RELEASE_REQUALIFIED`: quick gates plus configured real-model qualification gates passed on this machine.
- `FAILED`: one or more required gates failed.

## Supported runtime behavior

See `api.md` and `release-qualification.md` for the exact supported API and release scope. The operations workflow does not broaden scope: vision, batching, MTP, DSpark, speculative decode, sessionized Responses/Messages, server-side tool execution, distributed serving, authentication, cross-runtime KV portability, and arbitrary stateful stop strings remain outside the release contract.
