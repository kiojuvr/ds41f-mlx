# M0 API contract

The API layer is a non-release developer interface until final qualification. In M0 it is only a thin boundary for using the runtime externally; it is not the production serving architecture.

## Required endpoints

| Endpoint | M0 behavior |
| --- | --- |
| `GET /health` | Returns JSON with runtime identity and `release: false`. |
| `GET /v1/models` | Returns OpenAI-style model list containing the configured DeepSeek-V4.1-Flash model id. |
| `POST /v1/chat/completions` | Validates request, calls the M0 oMLX runtime bridge when enabled, and returns explicit errors for unsupported/unavailable runtime states. |

## Rules

- Unknown model must not silently route to another model.
- Empty messages and invalid generation options must fail before runtime execution.
- Backend errors must not be converted into normal stop finishes.
- Unsupported features must be explicit errors unless they are actually connected and tested.
- The server is not release-qualified merely because smoke tests pass.

## M0 bridge

The first bridge may be simple and conservative. It should keep oMLX execution ownership intact and focus on request/response API behavior, provenance, and oMLX compatibility comparison. Performance work begins only after baseline reproduction. The current process/worker topology is not a hard invariant and must not constrain the later production core design.

Current implementation modes:

- default / `DS41F_BACKEND=unconnected`: valid generation requests return explicit `501 runtime_unavailable`.
- `DS41F_BACKEND=omlx-worker`: starts a line-oriented Python worker that owns the oMLX model process and maps successful output into an OpenAI-style response. Loading the full model can use hundreds of GiB and is not exercised by the default smoke test. The worker Python is selected by `DS41F_WORKER_PYTHON`, then `DS41F_PYTHON`, then `$HOME/.venvs/omlx-0.7.0.dev2/bin/python` if present.
