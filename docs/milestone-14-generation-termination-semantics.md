# Milestone 14 generation termination semantics

Status: **M14_TERMINATION_QUALIFIED** for DeepSeek V4.1 EOS token termination on the ds41f direct GenerationBatch path.

Primary evidence: `artifacts/m14/termination-qualification.json` (`schema: ds41f.m14.termination-qualification.v1`).

## Discovered termination architecture

Official `deepseek-recipe` defines the V4/V4.1 message terminator:

```text
EOS_TOKEN = "<｜end▁of▁sentence｜>"
```

The pinned tokenizer encodes it as the single canonical token id:

```json
{"token_ids": [1]}
```

`mlx-lm` `BatchGenerator` owns token-level stop semantics through `SequenceStateMachine`. Its constructor accepts `stop_tokens`; when a generated token sequence matches and transitions to `None`, `GenerationBatch.Response.finish_reason` becomes `"stop"`, and the response carries the final cache and `all_tokens` including the stop token.

Before M14, ds41f constructed direct `BatchGenerator` without `stop_tokens`. This preserved the no-replay cache admission topology but omitted the EOS termination configuration that the higher-level mlx-lm/oMLX server paths normally install from tokenizer/model EOS ids. Consequently token id `1` was decoded and sent to recipe as ordinary text, and generation continued until `max_tokens`.

## Implemented bridge

M14 adds a narrow explicit bridge instead of importing the full scheduler:

```text
deepseek-recipe tokenizer EOS text
  -> RecipePreparedRequest.stop_token_ids
  -> OMLXDecodeConfig.stop_token_ids
  -> OMLXGenerationSession BatchGenerator(stop_tokens=[[id]])
  -> SequenceStateMachine finish_reason="stop"
```

Protocol output suppresses the terminal EOS token chunk when it is the backend stop token, but the token remains in `GenerationBatch` cache/history. This is required because official complete conversation encodings include assistant message EOS markers; exact-prefix continuation after a naturally stopped turn therefore succeeds only if the committed `all_tokens` retains the EOS token.

M11 early parsed-tool boundaries remain separate. Tool-call completion can occur before natural EOS and is still handled by the official parser probe.

## Qualification closeout

Real HTTP/session qualification covered:

- plain assistant answer;
- single tool result -> final assistant answer;
- repeated-tool chain -> final assistant answer;
- fresh stateless production control over the repeated-tool final prompt;
- next-turn exact-prefix continuation after a plain naturally stopped answer.

Recorded results:

| Case | Finish | EOS tail | EOS leaked | Generated tokens |
| --- | --- | --- | --- | --- |
| plain answer | `stop` | tail token `1` | no | 8 |
| single-tool final | `stop` | tail token `1` | no | 10 |
| repeated-tool final | `stop` | tail token `1` | no | 47 |
| fresh stateless repeated control | `stop` | protocol EOS hidden | no | n/a |

The repeated-tool final response that previously ran to length now stops after the concise useful answer. The session frontier changed from the M13 length-run `564` to the M14 EOS-stopped `515`, avoiding 49 post-EOS generated/cache tokens in that fixture. The visible final content remains the useful answer prefix and no longer exposes `<｜end▁of▁sentence｜>` as ordinary text.

All qualified cases retained:

- prompt replay `0`;
- full-cache repack/reconstruction `0`;
- all cache offsets equal frontier;
- exact-prefix continuation after natural EOS;
- official recipe finish reason `stop` for final answer turns.

## Request stop sequences

Official request stop strings remain owned by `deepseek-recipe` stream parsing. M14 did not add cache rollback for detokenized stop strings; arbitrary stop-string qualification is deferred. EOS token termination is token-level and cache-consistent because the stop token is consumed into the cache/history exactly once.
