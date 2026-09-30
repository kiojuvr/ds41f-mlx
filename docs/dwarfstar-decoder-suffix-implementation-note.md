# DwarfStar decoder-suffix implementation note

Pinned source audited: `/Users/kioju/ds4/ds4.c`, `antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac`.

Relevant functions:

- `ds41_graph_prefill_sweep(...)`
- `ds41_decoder_prepare(...)`
- `ds41_carry_copy(...)`

Wide-prefill decoder path:

1. Encoder layers run full span.
2. At decoder entry (`il >= 20`) the sweep uses DwarfStar's exact dependency cone:
   `needed(layer) = 1 + (DS4_N_LAYER - 1 - layer) * 127`.
3. For layer 20 only, `ds41_decoder_prepare(... publish=true, offset=0, rows=total_count)` publishes compressed/global source state from the full encoder-final hidden state.
4. For each decoder suffix layer, `ds41_decoder_prepare(... publish=false, offset=first-127, rows=127)` warms the local-window KV dependency rows immediately preceding the suffix query cone.
5. Suffix query execution then processes only `first = total_count - needed(layer)` through the final row, chunked by DwarfStar's suffix chunk policy.
6. Warmup rows are dependency state only; they must not advance the public committed token frontier as prompt rows.

`ds41f_mlx.prefill_fp8_mlx` mapping:

- `RequestArena.prepare_decoder_suffix(...)` records a distinct prepare state, separate from query spans and source rows.
- `RequestArena.require_decoder_prepared(...)` fails decoder-suffix query execution if prepare state is absent.
- `OfficialFP8MLXBlockRunner` uses `absolute_start = arena.base_frontier + command.offset` for query execution.
- Layer-20 suffix execution exposes the full encoder-final source state to the block runner path; this is a structural seam for official/oMLX source-publication math and is not a claim of numerical equivalence.
- `PublicationTopology.from_model_config(...)` derives publication/consumer topology from official/oMLX config and validates it against expected V4.1 topology.

This is still pre-P5/P6 architecture work. It does not claim full real-checkpoint numerical qualification or final decode handoff readiness.
