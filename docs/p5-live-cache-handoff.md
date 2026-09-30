# P5 live-cache handoff from 3115674

Status: **P5 PASS within the new connected path**, including real short and wide MLX handoff/bootstrap/decode. Stop after P5. Production runtime selection is unchanged; P6/P7 and benchmarking remain out of scope.

## API and terminal-token contract

Production module: `ds41f_mlx/prefill_fp8_mlx/handoff.py`.

```python
# Request ownership supplies the COMPLETE prefix history, not arena.tokens.
# full_prompt = prefix_token_ids + [terminal_prompt_token]
result = LivePrefillResult.from_committed(
    setup, prefix_token_ids=prefix_token_ids,
)
session = handoff_to_generation(
    result, model,
    terminal_prompt_token=terminal_prompt_token,
    config=decode_config,
    max_tokens=4,
)
# handoff_to_generation has already called start(terminal) exactly once.
first = session.next_token()
```

P3/P4 prefill **only** `prefix_token_ids`. P5 passes that complete history as the `all_tokens` seed and inserts **only** `[terminal_prompt_token]` into BatchGenerator. It does not need final-prefix logits. The terminal token must not have been prefilled as the final prompt position and then sent to `start()` again. Repeated token *values* earlier in the prefix are legitimate; holdout refers to the terminal position, not token-value uniqueness.

Before bootstrap all 40 frontiers equal `len(prefix_token_ids)`. After bootstrap all active scheduler frontiers equal `len(prefix_token_ids)+1`. The terminal token is not replay: it was deliberately held out. Length mismatch fails closed, including when the last sweep's token slice is mistakenly supplied instead of complete request history. No P6 deferred/multi-sweep executor was added.

## Admission and ownership

`validate_committed_cache(setup, prefix_token_ids)` performs scalar/metadata checks only:

- arena transaction begun, committed, valid and not failed;
- publication transaction not failed and no pending publications;
- existing cache list has exactly 40 layers;
- all frontiers equal complete prefix length and the committed plan's base+count;
- no continuation-state export or cache repack;
- serving prefix logits suppressed, with no final-prefix logits tensor;
- singleton int32 offsets; packed uint8 window/KV/index geometries;
- configured KV/index source lengths match frontier/compression ratio;
- ordinary/refresh layers may retain valid empty packed source representations;
- compressor pending KV/gates have expected shape and compatible float dtype;
- cache compression layout matches loaded model configuration;
- layer0 tokenizer-derived Engram history exists when configured. The already-qualified history may be host int64; it is not translated.

No tensor contents, digests, NumPy tensor conversion, tensor copy, cache allocation, cache merge/repack, or reconstruction from publication metadata is performed. Request-owned Python token history is immutable metadata in the result.

Creating the result reserves/freezes the prefill runner. A second claim fails. Handoff invokes only existing `OMLXGenerationSession.from_prefilled_cache`; it does not invoke `from_prefill_state`, `OMLXDecodeStateAdapter`, or `PrefillContinuationState`. Admission must retain the identical cache list. Before start, the runner's cache reference is detached and execution authority revoked; result cache access is also consumed. After start, the session clears its admitted singleton handle and BatchGenerator/GenerationBatch owns active decode cache state. Scheduler cache wrapping/mutation after start is normal oMLX lifecycle, not a P5 repack.

A failed transfer/bootstrap burns the result: potentially-mutated state cannot be handed off again. Existing GenerationSession now marks a start attempt before insertion (including failed insertion), records its singleton inserted prompt, and verifies actual scheduler frontier advancement instead of merely inferring it. Repeated start fails closed.

Passive Python diagnostic aliases can still exist; they are not additional executable cache authorities. The old runner cannot execute commands after reservation/transfer. P5 does not pretend Python object aliases can be erased from external callers.

## Real qualification

Target interpreter and runtime are the same as P3/P4 qualification:

- `/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3`;
- oMLX 0.7.0.dev2 at b390b31, MLX 0.32.2;
- imported `language.py` SHA256 `2c64bef36c9fc4a6007cd34e7b120142ced072a8d491cf1f83f2c5b1a956763f`;
- existing external encoding/processing changes are recorded in the earlier runtime-authority evidence; no installed package changes were made here;
- official DeepSeek-V4.1-Flash checkpoint, `preserve_mtp=False`, `engram_ssd_offload=True`.

Tool: `tools/run_prefill_fp8_mlx_p5_smoke.py`. Fixed prefix is `[1]` repeated; the distinct terminal token is `3`. It explicitly records provenance before loading, drives SweepPlan commands only for prefill, then uses the existing generation session. No runtime-selector change or alternate decode implementation.

```bash
export PATH=/Users/kioju/.venvs/omlx-0.7.0.dev2/bin:$PATH
python3 tools/run_prefill_fp8_mlx_p5_smoke.py \
  --full-prompt-tokens 2049 --mtp-off --no-benchmark \
  --out artifacts/p5-live-cache-handoff/short.json
python3 tools/run_prefill_fp8_mlx_p5_smoke.py \
  --full-prompt-tokens 8193 --mtp-off --no-benchmark \
  --out artifacts/p5-live-cache-handoff/wide.json
```

The short passed before the wide was run. Complete JSON, logs and exit statuses are retained in `artifacts/p5-live-cache-handoff/`.

| Check | Short full prompt 2049 | Wide full prompt 8193 |
|---|---|---|
| Real status | **PASS** | **PASS** |
| Prefix frontiers | 40 x 2048 | 40 x 8192 |
| Post-terminal bootstrap frontiers | 40 x 2049 | 40 x 8193 |
| Two generated-step frontiers | 40 x 2050, then 40 x 2051 | 40 x 8194, then 40 x 8195 |
| Generated greedy tokens | 305, 270 | 305, 270 |
| Terminal bootstrap forwards | exactly one `[[3]]` | exactly one `[[3]]` |
| Handoffs | 1 | 1 |
| Prefix prompt replay | 0 | 0 |
| Handoff/cache repacks | 0 | 0 |
| Continuation-state exports | 0 | 0 |
| Adapter full-source prepare count | 0 (ordinary prefix path) | 1 (qualified exact suffix path) |
| Serving final-prefix logits | suppressed | suppressed |

### Identity and exactly-once evidence

Each smoke records IDs of the cache list, layer-cache objects at layers0/20/39, and their slots0..6:

1. after P3/P4 prefill;
2. after constructing LivePrefillResult, immediately before transfer;
3. inside actual `from_prefilled_cache` admission.

All recorded identities match. The admitted session's initial cache is the same list. No persistent tensor is copied for P5. This diagnostic object inspection is not production tensor-content inspection.

A process-local wrapper around actual loaded `LanguageModel._forward` records bootstrap inputs and frontiers. At handoff completion there is exactly **one** call: input `[[3]]`, with all incoming offsets equal to the prefix length. Prefix seed equals the full request-owned prefix; BatchGenerator inserted prompt is `[3]`; replay accounting is zero. Each subsequent bounded decode call is one token at uniformly advancing frontiers. This is stronger than just a zero replay marker.

### Behavioral boundary

Both real bootstraps produced a first generated token and two consistent decode steps using the existing greedy sampler. No independently reproduced direct-`_forward` distribution comparison was performed (optional in P5). The identical displayed token pair is observed fixture behavior, not a cross-topology equality gate or broad numerical/determinism claim. Ordinary full-decoder prefill is not used as a universal equality oracle for the DwarfStar dependency cone.

## Tests and telemetry

`python3 -m unittest discover -s tests -q`: **51 tests passed** in the target environment, retaining all original 34. New tests cover malformed/mismatched/divergent cache admission, uncommitted/failed/pending transactions, export/repack rejection, missing Engram history, suppressed prefix logits, complete history independent of last arena slice, same object identity, held-out terminal exactly once, zero replay, producer revocation, repeated and failed handoff/start, actual GenerationSession seed/insertion logic, and the retained complete real short/wide accounting. Unit tests do not reload a 475-GiB model; the separate real smoke commands above did.

Compile and whitespace checks passed. No performance result is emitted; existing session-internal step accounting is not used as a benchmark.

`one_live_cache_handoff_no_replay` is now **qualified within the new connected path**, backed by both real smokes. The P5 smoke emits that scoped field; the summary records the updated interpretation. Production/global ArchitectureCompletion flags are **not** promoted because selection is unchanged.

Remaining false/unqualified **within the new path**:

- `dwarfstar_carry_lifetime`: partial request arena/ping-pong evidence, not the full lifetime gate;
- `deferred_decoder_suffix_lifetime`: P6 deferred/resume/full retention requirements remain;
- `scheduling_hooks_effective`: P7 remains a known blocker;
- `no_cpu_hot_path_roundtrip`: scalar admission/frontier reads and official host/disk Engram handling are not globally qualified.

Planner ownership, no independent prefill layer loop, frontier-driven visibility, explicit diagnostic materialization and zero intermediate repack retain their prior new-path qualification. Materialization qualification does not imply P7 grouping. The complete production structural gate remains **false**.

## Stop

P5 is implemented and bounded-real-qualified. No semantic/admission failure occurred in either real P5 smoke. Stop here: no production-selector change, P6, P7, benchmark, or optimization.
