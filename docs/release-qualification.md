# Release qualification: DS41F_TEXT_RUNTIME_RELEASE_QUALIFIED

## Status

`DS41F_TEXT_RUNTIME_RELEASE_QUALIFIED` for the scoped local text runtime below. This is not a universal project-complete claim.

## Supported scope

- Hardware: Mac Studio M3 Ultra 512 GB class target.
- Model: official DeepSeek-V4.1-Flash checkpoint.
- Input: text only.
- Prefill: `DENSE_P0_P7`, P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM`.
- Decode: oMLX `GenerationBatch`, MTP OFF, DSpark OFF, speculative decode OFF.
- Serving: local single-flight HTTP.
- Stateless API: Chat Completions, Responses, Messages in the already qualified text scope.
- Stateful API: Chat Completions sessions only.
- Agent/tool behavior: ordinary client function tools, repeated tool/result loops; server does not execute tools.
- Persistence: same-backend idle artifact save/restore.
- Termination: DeepSeek V4.1 EOS token semantics, length, and cancellation cleanup.

## Public contract closure

See `docs/api.md`. Key release decisions:

- maximum live sessions: `DS41F_MAX_LIVE_SESSIONS`, default `4`;
- one backend request at a time;
- unknown session, overlap conflict, invalid tool result, restore mismatch/corruption, and invalid request fail before mutation;
- stateful streaming emits official chunks only after a committed session boundary;
- non-empty/non-null `stop` on stateful Chat Completions is rejected before mutation; arbitrary KV rollback is not part of the release.

## Canonical startup

From a clean shell with documented local dependencies available:

```bash
export PYTHONPATH=/Users/kioju/omlx-0.7.0.dev2:$PYTHONPATH
export DS41F_MAX_LIVE_SESSIONS=4
export DS41F_TRACE_HISTORY_LIMIT=32
python3 -m uvicorn ds41f_mlx.serving.server:app --host 127.0.0.1 --port 8000
```

Required default paths in the current release build:

- checkpoint: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`;
- oMLX checkout: `/Users/kioju/omlx-0.7.0.dev2`;
- deepseek-recipe checkout/tokenizer: configured by `DEFAULT_RECIPE` in `ds41f_mlx.serving.deepseek_recipe_backend`;
- persistence root default: `DEFAULT_KV_ROOT` in `ds41f_mlx.runtime.kv_persistence`.

Health lifecycle: `/health` returns `alive` before model load, `ready` after model load, and `unavailable` with `fatal_error` on fatal backend error. Graceful shutdown/close must release live sessions, GenerationBatch ownership, and the single-thread executor. Persisted artifacts remain on disk; live sessions do not remain active after shutdown.

## Regression matrix

The release matrix is bounded and composes the previously qualified production architecture without inventing a new runtime:

| Case | Evidence | Result |
| --- | --- | --- |
| Plain stateless text | `artifacts/m7/deepseek-recipe-serving/result.json`, EOS recheck in `artifacts/m14/termination-qualification.json` | PASS |
| Plain stateful text | `artifacts/m12/sessionized-http-qualification.json`, `artifacts/m14/termination-qualification.json` | PASS |
| Natural EOS termination | `artifacts/m14/termination-qualification.json` | PASS |
| Multi-turn continuation | `artifacts/m8/long-session-qualification.json`, restored evidence | PASS |
| Tool call -> result -> final | `artifacts/m11/tool-boundary-qualification.json`, termination recheck | PASS |
| Repeated tool loop | `artifacts/m13/repeated-tool-agent-qualification.json`, termination recheck | PASS |
| Persistence -> teardown -> restore -> continue | `artifacts/m9/kv-persistence-qualification.json`, `artifacts/m10/restored-long-session-qualification.json` | PASS |
| Streaming committed boundary | `artifacts/m12/sessionized-http-qualification.json` | PASS |
| Cancellation/recovery | `artifacts/m7/deepseek-recipe-serving/result.json`, restored long-session evidence | PASS |
| Invalid request before mutation | tool invalid-result evidence plus stateful stop policy unit test | PASS |
| Overlap conflict | sessionized HTTP evidence | PASS |
| Long-context representative case | `artifacts/m6/performance-qualification/result.json` through 200K | PASS |

## Invariants retained

Official checkpoint only; no requantized substitute; dense P0-P7 selector; P7 SSD Engram; P5 terminal holdout exactly once; one executable cache authority; prompt replay and full-cache repack zero on stateful continuation; no hidden fresh-prefill fallback; exact-prefix continuation; all 40 cache offsets equal frontier at idle boundaries; MTP/DSpark/speculation OFF; backend-local determinism; EOS token semantics; fail-closed restore validation.

## Performance and safety sanity

Performance class remains the established dense P0-P7 class: practical prefill through 200K, practical GenerationBatch decode above the floor, EOS avoiding post-answer over-generation, and persistence/restore small relative to load/inference. Recorded target evidence reports no unexpected swap in the qualified workloads and bounded diagnostics/session traces. Resource cleanup evidence covers cancellation, session close, and request cleanup; memory growth/leak stress beyond representative release workloads remains outside this scoped claim.

## Provenance

Release evidence records:

- ds41f commit(s) in artifact metadata; M15 documentation/code closure at current HEAD;
- oMLX revision `b390b31e0c6831225fed0f24d278eb1db7fcb68b` in production qualification artifacts;
- deepseek-recipe revision `8cadfede7063c896b944e7bae05daa3549ae97ea` in session/tool/termination artifacts;
- MLX/mlx-lm/Python/hardware versions where recorded by the individual artifacts;
- target checkpoint path and checkpoint identity in performance/provenance artifacts.

## Cheap gates

Canonical cheap gate command set:

```bash
python3 -m unittest tests/test_stateful_request_policy.py
python3 tools/check_legacy_import_integrity.py
python3 tools/check_native_import_dependencies.py
python3 tools/check_repository_self_containment.py
cmake -S native -B native/build
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

## Explicitly unsupported/unqualified

Vision, batching, MTP, DSpark, speculative decode, sessionized Responses/Messages, server-side tool execution, MCP/plugins, web search, shell tools, distributed serving, authentication, cross-runtime KV portability, arbitrary stateful stop strings, and new kernel optimization are not release features.

## Recommended next milestone

Post-release operational hardening: packaging/configuration cleanup and a fresh one-command production qualification runner that records a single combined provenance artifact on the target machine.
