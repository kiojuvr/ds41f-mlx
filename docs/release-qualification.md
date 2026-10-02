# Release qualification: DS41F_TEXT_RUNTIME_RELEASE_QUALIFIED

## Status

`DS41F_TEXT_RUNTIME_RELEASE_QUALIFIED` for the scoped local text runtime below. This is not a universal project-complete claim.

M20 promotes exact upstream oMLX `v0.7.0` (`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`) after fresh bounded dependency-reachable gates, A/B, persistence, HTTP/tool/EOS, repeated-session and targeted 200K endpoint checks. See [M20](m20-omlx-release-migration.md) and `artifacts/m20/promotion.json`. The old M18 whole-runtime migration attestation is stale for this dependency; it is not reclassified to preserve inheritance. Unchanged source/prefill evidence is retained explicitly, not represented as fresh release timing.

M21 adds the Rust-facing `ds41f_api` boundary over the unchanged local HTTP/SSE server. It does not alter model runtime bytes, prompt/protocol semantics, prefill/decode selectors, persistence format, or session ownership. Boundary-specific evidence is in `artifacts/m21/rust-boundary-qualification.json`; closeout real-server evidence is in `artifacts/m21/real-rust-boundary-acceptance.log`. Real-model endpoint behavior remains qualified by M20; M21 closeout only proves the Rust consumer boundary can drive that server with the qualified Python environment.

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
python3 -m ds41f_mlx.provenance
python3 -m ds41f_mlx.serve
```

Machine-specific paths are configured through `DS41F_CHECKPOINT`, `DS41F_OMLX_PATH`, `DS41F_RECIPE_PATH`, and `DS41F_KV_ROOT`; host/port/session/diagnostic settings are documented in `docs/operations.md`. The launcher applies configured oMLX and recipe checkout paths before importing the FastAPI app, so manual `PYTHONPATH` setup is not the canonical path.

Health lifecycle: `/health` returns `alive` before model load, `ready` after model load, and `unavailable` with `fatal_error` on fatal backend error. Rust `RuntimeProcess::spawn*` treats `alive` only as process startup; `wait_model_ready()` is required for inference readiness. Graceful shutdown/close must release live sessions, GenerationBatch ownership, and the single-thread executor. Rust `shutdown()` sends SIGTERM first and truthfully reports `Graceful`, `AlreadyExited`, or `Forced` if a kill fallback was required. Persisted artifacts remain on disk; live sessions do not remain active after shutdown.

## Regression matrix

The release matrix is bounded and composes the previously qualified production architecture without inventing a new runtime:

| Case | Evidence | Result |
| --- | --- | --- |
| Plain stateless text | `artifacts/m20/release-tool-eos.json`; retained protocol/source evidence | PASS |
| Plain stateful text | `artifacts/m20/release-tool-eos.json` | PASS |
| Natural EOS termination | `artifacts/m20/release-tool-eos.json` | PASS |
| Multi-turn continuation | `artifacts/m20/release-bounded-baseline.json`, repeat and persistence evidence | PASS |
| Tool call -> result -> final | `artifacts/m20/release-sessionized-http.json`, tool/EOS recheck | PASS |
| Repeated tool loop | `artifacts/m20/release-tool-eos.json` | PASS |
| Persistence -> teardown -> restore -> continue | `artifacts/m20/release-persistence.json` | PASS |
| Streaming committed boundary | `artifacts/m20/release-sessionized-http.json` | PASS |
| Cancellation/recovery | `artifacts/m20/release-bounded-baseline.json`, cheap failure/ownership tests; Rust SSE drop/recovery in `artifacts/m21/real-rust-boundary-acceptance.log` | PASS |
| Invalid request before mutation | tool invalid-result evidence plus stateful stop policy unit test | PASS |
| Overlap conflict | `artifacts/m20/release-sessionized-http.json` | PASS |
| Long-context representative case | `artifacts/m20/release-200k-endpoint.json`; retained unchanged M6 ladder | PASS |

## Invariants retained

Official checkpoint only; no requantized substitute; dense P0-P7 selector; P7 SSD Engram; P5 terminal holdout exactly once; one executable cache authority; prompt replay and full-cache repack zero on stateful continuation; no hidden fresh-prefill fallback; exact-prefix continuation; all 40 cache offsets equal frontier at idle boundaries; MTP/DSpark/speculation OFF; backend-local determinism; EOS token semantics; fail-closed restore validation.

## Performance and safety sanity

Performance class remains the established dense P0-P7 class: practical prefill through 200K, practical GenerationBatch decode above the floor, EOS avoiding post-answer over-generation, and persistence/restore small relative to load/inference. Recorded target evidence reports no unexpected swap in the qualified workloads and bounded diagnostics/session traces. Resource cleanup evidence covers cancellation, session close, and request cleanup; memory growth/leak stress beyond representative release workloads remains outside this scoped claim.

## Provenance

Release evidence records:

- ds41f commit(s) in artifact metadata; M15 documentation/code closure at current HEAD;
- promoted oMLX `v0.7.0` revision `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, clean upstream source, native/package identity in M20;
- preserved dev2 rollback revision `b390b31e0c6831225fed0f24d278eb1db7fcb68b` and original local identity in historical artifacts;
- deepseek-recipe revision `8cadfede7063c896b944e7bae05daa3549ae97ea` in session/tool/termination artifacts;
- MLX/mlx-lm/Python/hardware versions where recorded by the individual artifacts;
- target checkpoint path and checkpoint identity in performance/provenance artifacts.

## Evidence validity commands

```bash
python3 -m ds41f_mlx.qualify --check-artifact artifacts/m20/promotion.json
python3 -m ds41f_mlx.acceptance
```

`--check-artifact` verifies the promoted content/dependency identity without rerunning expensive gates. Checking the old M18 attestation correctly reports stale evidence for the new dependency.

## Cheap gates

Canonical cheap gate command set:

```bash
python3 -m unittest tests/test_stateful_request_policy.py tests/test_runtime_config.py
cargo test
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

Package/operational hardening for mixed Python/Rust local applications, including versioned release packaging and one-command acceptance orchestration. MTP and other acceleration remain deferred.
