# M21 — Production Rust API Boundary

## Decision

M21 selects a **Rust client/process-control boundary over the existing local HTTP server**.

Rust-facing applications integrate through the crate `ds41f_api`.  The crate can either connect to an already running `ds41f_mlx.serve` instance or spawn it as a child process and wait for `/health`.  All generation, session, prompt/protocol conversion, persistence, streaming formatting, and tool-call semantics remain inside the already-qualified Python/deepseek-recipe/oMLX runtime.

This is intentionally not a native Rust embedding layer, a C ABI, a Python-extension ABI, or a second model runtime.

## Why this boundary

M20 qualified the production path as:

```text
official checkpoint
  -> DENSE_P0_P7 production prefill
  -> P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM
  -> P5 terminal holdout/handoff exactly once
  -> oMLX GenerationBatch decode
  -> MTP OFF / DSpark OFF / speculative decode OFF
  -> official deepseek-recipe local HTTP serving
```

The durable correctness property is that there is one executable cache/session authority after P5: oMLX `GenerationBatch`, mediated by the recipe-backed server.  A Rust embedding, FFI, or reimplementation boundary would have to import Python/MLX/oMLX object lifetimes or duplicate protocol/session semantics to look convenient.  That would increase the risk of prompt replay, second cache ownership, independent tool/thinking parsing, or partial mutation after client failure.

The loopback HTTP boundary is already the qualified production seam.  A Rust crate over that seam gives local applications and agent harnesses a typed, stable Rust surface while preserving failure isolation: if Rust client code panics or is dropped, the model process remains the only runtime owner; if the model process fails, Rust observes an HTTP/process error rather than corrupting in-process model state.

## Alternatives considered and rejected

- **Rust FFI directly into Python/oMLX objects.** Rejected for M21 because it couples Rust lifetimes to Python interpreter, MLX allocator, `GenerationBatch`, and recipe internals. It offers no correctness benefit over the qualified HTTP seam and makes cancellation/drop semantics harder to make fail-closed.
- **Rust reimplementation of DeepSeek request/tool/thinking semantics.** Rejected because `deepseek-recipe` is the authority. M21 must not create a competing protocol implementation.
- **Rust-owned KV/session handles.** Rejected because cross-runtime KV portability is explicitly outside the release scope and would introduce a second executable cache authority.
- **A new IPC protocol.** Rejected because no current evidence shows HTTP is the limiting factor, while a new protocol would require fresh protocol, streaming, cancellation, and operational qualification.
- **Rust model runtime rewrite or native C++ binding.** Rejected as outside M21 and contrary to the M20 selected production path.

## Ownership model

| Responsibility | Owner |
| --- | --- |
| Model loading/execution | Python ds41f runtime + oMLX/MLX |
| Prompt rendering, request conversion, response/tool/thinking semantics | `deepseek-recipe` through the server |
| Live executable cache/session state | Server-side stateful session / `GenerationBatch` authority |
| Persistence/restore validation | Server/runtime |
| Rust request construction and transport | `ds41f_api` |
| Optional server child-process lifetime | `ds41f_api::RuntimeProcess`, when the caller elects to spawn |
| Cancellation | HTTP stream/request drop from Rust; server cleanup remains authoritative |
| Diagnostics/provenance | Bounded server endpoints and health/models/session metadata; no KV export |

## Cross-boundary invariants

- Rust never receives KV tensors, cache offsets as authority, oMLX objects, or prompt-token replay hooks.
- Rust sends ordinary JSON requests to documented endpoints and receives JSON/SSE results.
- Stateful continuation uses `/v1/sessions/{id}/chat/completions`; Rust does not reconstruct history into a fresh stateless request as a hidden fallback.
- Client-side tool loops are represented as normal Chat Completions messages/tool results sent back to the same server session.
- MTP, DSpark, speculative decode, CED prefill, batching, vision, and cross-runtime KV portability remain OFF/out of scope.

## Failure and cancellation

Transport failures return `Ds41fError` and do not imply any Rust-owned session mutation.  HTTP 4xx/5xx responses are surfaced with status and body.  Dropping a streaming iterator closes the TCP connection; the server remains responsible for cleanup and for not exposing protocol chunks beyond committed session boundaries.  If `RuntimeProcess` owns a child process, `shutdown()` sends termination and waits; `Drop` is best-effort cleanup only and must not be used as the sole qualification signal.

## M21 implementation scope

Implemented in M21:

- Rust crate `rust/ds41f_api` using only the Rust standard library;
- startup/readiness helper for optionally spawned local server processes;
- synchronous request methods for health, models, stateless Chat Completions, stateful sessions, persistence/restore, and raw endpoint access;
- SSE streaming iterator for stateless and stateful Chat Completions;
- explicit error/status handling and timeout configuration;
- repository tests with mock HTTP/SSE servers proving request paths, cancellation/drop, chunked streaming, session lifecycle calls, and failure propagation.

Explicitly outside M21:

- direct Rust model execution;
- Rust prompt/protocol semantics beyond minimal JSON request helpers;
- server-side tool execution;
- Rust KV/session serialization;
- performance optimization or alternate transports.

## Qualification summary

M21 changes no Python/oMLX/deepseek-recipe runtime implementation.  Real-runtime M20 evidence therefore remains authoritative for model execution, P5 handoff, stateful continuation, streaming semantics, EOS/tool behavior, persistence, and performance.  M21 adds boundary-specific Rust tests and an inspect-only qualification artifact.  A bounded real-model acceptance run through the same HTTP endpoints is the correct operational gate when local model dependencies are available.
