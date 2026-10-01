# Milestone 8 long-session architecture

Status: **M8_LONG_SESSION_QUALIFIED** for text-only single-session repeated DeepSeek-recipe continuation on the official checkpoint.

Primary evidence: `artifacts/m8/long-session-qualification.json` (`schema: ds41f.m8.long-session-qualification.v2`).

## Canonical production lifecycle

M8 keeps the M7 production path and extends it across turns without making a second executable state authority:

```text
recipe encoded conversation N
  -> exact token-prefix boundary check against completed session history
  -> append only new suffix[:-1] through DeferredPrefillAppend/P6/P7
  -> sealed live DeepseekV41Cache at frontier T-1
  -> P5 zero-replay handoff of suffix[-1]
  -> oMLX GenerationBatch MTP-OFF decode
  -> turn boundary: extract GenerationBatch cache + all_tokens
  -> repeat
```

The live authority is exactly one object at a time:

- idle between turns: the extracted request-local `DeepseekV41Cache[40]`;
- appending: the P6/P7 append transaction privately owns that cache and keeps the public frontier frozen until seal;
- decoding: `BatchGenerator/GenerationBatch` owns the scheduler cache;
- diagnostics: scalar traces, token history, segment records, and digests are evidence only.

Full replay/fresh prefill remains a diagnostic oracle only. It is not a production continuation fallback.

## Boundary decision

The completed assistant generation is part of the next canonical recipe history only if the next recipe encoding is an exact token extension of the session's completed `all_tokens`. The production boundary rule is therefore:

```text
next_recipe_tokens[:len(session_history)] == session_history
suffix = next_recipe_tokens[len(session_history):]
append suffix[:-1]
decode starts with suffix[-1]
```

If the prefix does not match, M8 rejects the turn before mutating cache. It does not search for a longest suffix, silently retokenize, rebuild, or replay the conversation. This makes token-history divergence diagnosable instead of hidden.

## Implementation

New runtime seam: `ds41f_mlx.runtime.continuation_session.M8LiveContinuationSession`.

Key behavior:

- `ensure_idle()` extracts `(cache, all_tokens)` from the active `OMLXGenerationSession` via `GenerationBatch.extract_cache`; no repack/export is introduced.
- `begin_turn_from_recipe_tokens()` enforces the exact prefix-extension contract.
- `begin_turn_from_suffix()` supports already-derived suffixes for internal tools/tests.
- prompt suffix prefill uses `DeferredPrefillAppend.create(... committed_frontier=current_frontier ...)` against the existing live cache.
- the terminal suffix token is supplied once to `OMLXGenerationSession.start()`.
- cancellation calls the same extraction seam and leaves an idle, reusable live cache if extraction succeeds.
- diagnostics retain bounded scalar/token evidence (`turn_records[-16:]`) including replay/repack counts and frontier movement.

`OMLXGenerationSession` now exposes `extract_final_state()` and `current_token_history()` so completed generation can be handed back to the append lifecycle without reconstructing cache tensors.

## Failure atomicity and recovery

- Rejected next-turn input is checked before append and leaves the idle cache unchanged.
- Failed append marks the cache invalid through existing P6 failure state; the session must be closed or rebuilt diagnostically from a fresh reference path.
- Failed handoff burns the live handoff result, matching P5 one-shot ownership.
- Cancellation is a turn boundary, not a heuristic text stop: the scheduler cache is extracted and the exact token frontier is recorded.

## Diagnosability, not loop suppression

M8 does not add content heuristics to suppress model loops. It records enough bounded evidence to distinguish:

- model behavioral repetition with coherent frontiers and zero replay;
- token-history divergence at the recipe boundary;
- accidental prompt replay (`prompt_replay_count > 0`);
- cache reconstruction/repack (`full_cache_repack_count > 0`);
- stale/invalid append ownership or unsealed P6 state.

## KV save/restore/resume

KV persistence is **deferred to the next milestone**. M8 defines the stable seam: the idle state `(DeepseekV41Cache[40], all_tokens, config/provenance, scalar diagnostics)` immediately after `extract_final_state()` and before any new append begins. Persisting there is natural, but qualifying it requires an explicit cache serialization format for packed slots, compressor pending rows, Engram history slot, owner/provenance metadata, and restore validation. For this environment, cache persistence artifacts should live under `/Volumes/USB-SSD-RAID-0`.

M8 deliberately does not introduce a second cache translation layer or portable state authority.

## Qualification closeout

The M8 real-checkpoint qualification rerendered each next complete conversation through the pinned DeepSeek-recipe V4.1 text prompt semantics, verified the completed GenerationBatch `all_tokens` were an exact prefix of the next encoding, appended only the new suffix minus its terminal token, consumed the terminal once through P5/GenerationBatch bootstrap, extracted the live cache and `all_tokens`, and repeated.

Closeout run summary:

- official checkpoint: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`;
- oMLX revision: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`;
- DeepSeek-recipe source revision: `8cadfede7063c896b944e7bae05daa3549ae97ea`;
- production selector: `DENSE_P0_P7`;
- turns: initial assistant turn plus 8 follow-up recipe turns;
- frontier growth: 45-token pre-terminal initial prefix to final idle frontier 169;
- exact-prefix-extension: true at every recipe turn;
- prompt replay: 0;
- full-cache repack/reconstruction: 0;
- cache offsets: all 40 layers equal final frontier at every idle boundary;
- cancellation: one follow-up turn cancelled after one generated token, returned to idle, and subsequent recipe turns continued successfully;
- invalid non-extension input: rejected before cache mutation;
- diagnostics: bounded scalar/token evidence; no hidden replay/fallback;
- memory: process resident high-water after model load remained approximately 16.9 GiB over the repeated-turn run;
- decode: per-turn measured decode throughput remained in the roughly 19.9-20.3 tok/s range for this short-context run, without progressive collapse.

The environment lacked the `deepseek_recipe._native` Python extension after dependency repair, so the runner used a text-only thinking-mode renderer transcribed from the pinned official `deepseek-recipe-encoding/src/v4` source plus the pinned official tokenizer JSON. The fallback reproduces the M7 minimal `hi` prompt length/terminal (`31`, terminal `<think>` token `128821`) and records its provenance in the artifact. M7 remains the direct Python-binding serving qualification; M8's boundary evidence is source-derived for the simple text chat subset and uses the same tokenizer and prompt semantics.

## Unqualified scope

M8 remains single-session/single-flight, text-only, MTP/DSpark OFF. It does not qualify multimodal, tool execution, batching, speculative decode, length-finish cache extraction, or cross-process KV restore.
