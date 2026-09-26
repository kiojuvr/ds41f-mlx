# API status

The intended external HTTP surface is OpenAI-compatible chat completion.

## Endpoints

### `GET /health`

Reports service health for the active backend.

### `GET /v1/models`

Returns the available model identifiers exposed by the server.

### `POST /v1/chat/completions`

Accepts chat-completion requests for the target model path.  Supported request options depend on the currently wired backend.

## Backend/runtime ownership

The canonical production architecture is the local native model core under `native/`.  The Python/oMLX bridge and `ds41f_mlx` modules remain compatibility and tooling scaffolds.  They are not the future architecture authority and should not be duplicated into new production model-core code.

A fully connected native HTTP serving path is not claimed until explicitly validated.  Until then, API behavior should be described per backend used by a deployment.

## Error behavior

Invalid requests should fail atomically with respect to model/session state: no partial token commit, KV mutation, publication, or generation-state advance should survive a rejected request.

## Unsupported/unqualified options

Do not assume support for vision, DSpark/MTP, long-context production serving, streaming parity, or PyTorch RNG parity unless a current qualification entry states it.

## Development guidance

New API work should bind to the native text/generation session rather than creating a second production session-state implementation.  Compatibility shims may remain for tools, but canonical runtime ownership belongs to `TextBackboneState` and generation state.
