# M0 closeout

M0 is closed as a baseline reproduction milestone. It does not claim production architecture, long-context qualification, or performance superiority.

## Source baseline

- upstream oMLX baseline: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
- local known-good oMLX: same commit plus recorded DeepSeek-V4.1 image-token parser patch
- official checkpoint: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`
- Engram mode for M3 Ultra / 512 GB: SSD-backed by default

## Completed gates

### Provenance

Artifacts:

```text
artifacts/m0/source-identity.json
artifacts/m0/omlx-local-patch.diff
artifacts/m0/omlx-local-patch.sha256
artifacts/m0/checkpoint-identity.json
```

Checkpoint identity recorded 96,085 tensors, 48 shards, total payload `510286023000`.

### Prompt/parser fixture

Artifact:

```text
artifacts/m0/prompt-encoding/fixture-20260922-210238.json
```

The local image-token parser patch is represented by the fixture: literal `<｜deepseek_image｜>` text produces zero image tokens, while structured image content produces the image token.

### Direct oMLX runtime smoke

Artifacts:

```text
artifacts/m0/omlx-direct-smoke/result.json
artifacts/m0/omlx-direct-smoke/chat-ping.json
```

Direct chat smoke passed with SSD-backed Engram. For the `ping` chat request, generated token id `[671]` and decoded text `"The"`.

### API bridge smoke

Artifacts:

```text
artifacts/m0/api-smoke/run-20260922-212213/result.json
artifacts/m0/api-smoke/run-20260922-212213/valid_connected.json
```

The thin M0 API boundary passed:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` via `DS41F_BACKEND=omlx-worker`
- expected invalid request behavior

The server bridge generated token id `[671]` and decoded text `"The"` for the same `ping` chat request.

### Direct vs server oracle comparison

Artifact:

```text
artifacts/m0/oracle-compare/direct-vs-server.json
```

The direct oMLX and server bridge paths matched generated ids, decoded text, prompt-id digest, and completion-token usage.

## Explicit non-claims

- M0 does not qualify long context.
- M0 does not prove production server architecture.
- M0 does not introduce DwarfStar kernels or topology.
- M0 does not claim performance superiority over oMLX.
- M0 does not validate DSpark, vision release behavior, streaming, or cancellation.

## Next phase

Proceed to M0.5: short-context production performance gate. Long-context qualification remains blocked until M0.5 passes.
