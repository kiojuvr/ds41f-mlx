# M0 runtime boundary

The current API server is a thin developer boundary for using the runtime from outside the process. It is not the production architecture.

## Current purpose

- Exercise request validation and OpenAI-compatible response shape.
- Keep backend unavailable/error behavior explicit.
- Provide a narrow bridge to the local known-good oMLX runtime for M0 reproduction.
- Use SSD-backed Engram embeddings by default; resident Engram loading is not appropriate for the M3 Ultra / 512 GB M0 path.
- Capture provenance and smoke artifacts.

## Non-goals for this boundary

- It is not the final serving architecture.
- It is not a scheduler design.
- It is not a performance architecture.
- It is not a substitute for native core correctness or short-context performance gates.
- It should not force the future core implementation into this process/worker topology.

## Production architecture timing

Production server architecture decisions are deferred until after the core runtime path is established and measured. If the thin boundary introduces measurable overhead or awkward semantics, it can be replaced. The hard invariants are official checkpoint data, pinned official semantics authority, provenance, authority-classified gates, and qualification policy, not the current API process layout.
