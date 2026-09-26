# M2 minimal layer-major/state-publication prototype plan

> Correctness-authority repair note: this document is historical. Its oMLX `_forward`, logits, cache, and state comparisons are now classified as oMLX-compatibility/regression diagnostics only, not official DeepSeek qualification. See `docs/correctness-authority-audit.md`.

This plan follows the M2 evidence gathered in `docs/m2-architecture-comparison.md`.
It does not reopen M0/M0.5/M1 and does not authorize long-context qualification.

## Current evidence

The following hypotheses were tested and rejected as sufficient explanations for
the DwarfStar-vs-oMLX prefill scaling-shape gap:

1. API/server overhead: traced scheduler overhead was about 0.1%.
2. Paged prefix-cache boundary snapshot overhead: disabling the prefix cache did
   not change 2,048-token chunking or throughput.
3. Simple scheduler chunk-size increase: diagnostic 4,096-token and 8,192-token
   chunks did not improve throughput; 8,192 regressed throughput and increased
   memory.

The remaining plausible structural difference is deeper than chunk size:

- oMLX production prefill executes repeated full-model chunks through the MLX
  module graph.
- DwarfStar's V4.1 path uses layer-major processing, explicit carry/state
  buffers, explicit raw/compressed/index cache publication, and tighter lifetime
  control.

## Non-goals

- No checkpoint conversion, quantization, GGUF requirement, or DwarfStar Q4 path.
- No DSpark replacement.
- No broad profiler.
- No 32K/64K/128K qualification until a bounded prototype wins.
- No optimization of the rejected direct `mlx_lm.generate_step` path.
- No production server redesign.

## Prototype objective

Build the smallest text-only prefill prototype that changes execution topology
rather than simply changing scheduler chunk size.

The prototype should answer one question:

> Can layer-major state publication over an 8K-ish prompt produce a bounded
> prefill throughput win over the current oMLX full-model chunked production
> path while preserving historical oMLX adapter compatibility?

## Scope constraints

Initial scope should be deliberately narrow:

- text-only prompt;
- single request;
- no vision;
- no multi-session batching;
- no DSpark acceptance path change;
- generation length 1 for timing;
- official checkpoint weights loaded by the known-good oMLX loader;
- historical oMLX adapter precision/behavior preserved for compatibility diagnostics only;
- target fixture: 4K/8K first, 16K only if bounded win appears.

## Minimal topology experiment

The prototype should not try to port all DwarfStar kernels.  It should adapt the
state-publication structure around the existing oMLX DeepSeek-V4.1 modules.

Candidate shape:

1. Load the official checkpoint through the existing oMLX DeepSeek-V4.1 loader.
2. Produce token embeddings for an entire bounded prompt once.
3. Carry `h` and `pre` as full-prompt arrays between layers instead of driving
   repeated full-model scheduler chunks.
4. For each layer:
   - apply Engram if present;
   - run HC/attention/MoE for the layer over a bounded slice or full prompt;
   - publish that layer's cache/state explicitly;
   - release temporary arrays before the next layer.
5. After the final layer, compute logits only for the final prompt position and
   compare the greedy next token/logit boundary against the historical oMLX compatibility reference.

This is a prototype of layer-major/state-publication topology.  It may still use
existing oMLX kernels inside each layer.  It should not introduce new kernels as
its first step.

## Expected hard part

DeepSeek-V4.1 attention state is not a simple dense KV cache. Historical oMLX-compatible
publication must preserve:

- 128-token local window state;
- compressed KV source layers;
- index source layers;
- candidate block/index state;
- Engram history;
- hyper-connection pre/post state;
- BF16/FP8/FP4 activation rounding already encoded in the oMLX modules.

If preserving these through a standalone layer-major prototype requires large
rewrites, stop and record that evidence rather than creating an approximate
path.

## oMLX compatibility gates before any performance claim

A prototype may be timed only after these bounded compatibility checks pass:

1. Prompt tokenization digest matches the production server fixture.
2. For a small prompt, final next-token id matches the M0 direct/server oMLX compatibility artifact where applicable.
3. For 1K/2K text prompts, final greedy next token matches the current oMLX
   production path.
4. No official checkpoint files are modified.
5. Runtime/source/checkpoint identity is recorded.

For early prototype work, exact full-logit equality was desirable as an oMLX
regression signal but not official correctness. Top-token agreement at the final
prompt position is only a minimum compatibility smoke gate. Any promotion beyond
exploratory prototype requires a stronger authority-classified contract.

## Performance gates

Only after oMLX compatibility diagnostics pass:

1. Compare against the recorded production oMLX fixtures:
   - `artifacts/m2/omlx-prefill-shape/shape-4k-8k-16k.json`
   - `artifacts/m2/omlx-prefill-trace/trace-4k-8k.json`
2. Use 4K and 8K first.
3. Require a meaningful bounded win over ~190 tok/s uncached prefill without
   unexplained memory/swap regression.
4. If 4K/8K does not win, reject or redesign; do not run 16K/32K.
5. If 4K/8K wins, run 16K once as bounded evidence, not qualification.

## Promotion/rejection criteria

Promote to an implementation candidate only if all are true:

- bounded oMLX compatibility diagnostics pass;
- 4K/8K prefill improves materially over production oMLX;
- peak memory/RSS/swap remain plausible;
- improvement cannot be reproduced by simple scheduler step changes;
- provenance is recorded.

Reject or pause if any are true:

- cache/state publication cannot be preserved without approximation;
- output behavior diverges from the historical oMLX adapter before a performance comparison;
- memory grows materially without explanation;
- performance is flat or worse than current oMLX;
- the implementation starts turning into a broad DwarfStar port.

## Layer-state inspection harness

Added:

```text
tools/inspect_m2_v41_layer_state.py
```

This harness loads the official checkpoint through the known-good oMLX loader,
wraps the 40 DeepSeek-V4.1 layers for a tiny text-only forward pass, and records
layer/cache shapes.  It is not a performance benchmark.

Smoke run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/inspect_m2_v41_layer_state.py \
  --target-tokens 8 \
  --out artifacts/m2/layer-state/inspect-8tok.json
```

Summary:

- load: 90.10 s
- tiny forward: 2.48 s
- layers observed: 40
- logits shape: `[1, 8, 129280]`
- config: `dim=5120`, `hc_mult=4`, `n_heads=64`, `head_dim=512`, `window_size=128`
- Engram layers: `[1, 14]`
- KV source layers: `[2, 8, 14, 20]`
- index source layers: `[2, 8, 14, 20, 24, 28, 32, 36]`
- candidate source layer: `20`, with block size `8`

Representative cache/state observations:

| Layer | Engram | Shared keys after layer | Non-empty cache slots after layer |
| ---: | --- | --- | --- |
| 0 | no | `[]` | window slot 1: `[1, 8, 528] uint8` |
| 1 | yes | `[]` | window slot 1 |
| 2 | no | `idx`, `index_k`, `kv` | window slot 1, compressed slot 2 `[1, 4, 288] uint8`, index slot 3 `[1, 4, 68] uint8` |
| 8 | no | `idx`, `index_k`, `kv` | same ratio-2 source pattern |
| 14 | yes | `idx`, `index_k`, `kv` | same ratio-2 source pattern |
| 20 | no | `candidates`, `idx`, `index_k`, `kv` | window slot 1, compressed slot 2 `[1, 8, 288] uint8`, index slot 3 `[1, 8, 68] uint8` |
| 24/36/39 | no | `candidates`, `idx`, `index_k`, `kv` | window slot 1 only for non-KV-source layers |

Implication for the prototype:

A layer-major path must explicitly preserve more than per-layer local windows.
It needs source-group shared state (`kv`, `index_k`, `idx`, and from layer 20
onward `candidates`) plus Engram history and per-layer cache slots.  This makes a
trivial wrapper around the current `Block.__call__` insufficient for a real
layer-major path unless the source-group shared-state publication contract is
made explicit.

## Next concrete engineering step

Do not start by writing kernels.  The next useful prototype step is to factor a
small, text-only state-publication contract around the observed source groups:

1. define the layer groups implied by `kv_source_layers` and `index_source_layers`;
2. define which shared values are produced/consumed by each group;
3. build an oMLX-compatibility skeleton that can replay the existing oMLX layer
   calls for a tiny prompt while publishing this state through an explicit object
   instead of an ad-hoc `shared` dict;
4. compare the tiny prompt's final next token against the unwrapped oMLX path.

Only after that skeleton is exact should any 4K/8K timing be attempted.

## Historical oMLX-compatibility state-publication skeleton

Added:

```text
ds41f_mlx/m2_state_publication.py
tools/run_m2_state_publication_fixture.py
```

The skeleton is deliberately not a scheduler or performance path.  It defines a
layer/frontier-oriented `ExplicitStatePublication` mapping that remains
compatible with existing oMLX layer calls but records explicit producer
frontiers.  Frontiers are derived from the loaded DeepSeek-V4.1 config:

- Engram layers publish `engram` frontiers;
- KV source layers publish `kv`;
- index source layers publish `index_k` and `idx`;
- the candidate source layer publishes `candidates`.

The fixture performs a tiny text-only two-chunk replay and compares three paths:

1. accepted oMLX `LanguageModel._forward`;
2. an oMLX-compatibility replay using the current implicit plain-dict `shared` state;
3. the same replay using `ExplicitStatePublication`.

The replay copies oMLX control flow only enough to expose state publication.  It
reuses the current oMLX embeddings, Engram calls, HC, attention, compressor,
indexer/candidate logic, MoE/FFN, cache objects, activation packing, and logits
projection.  It does not change checkpoint representation, precision,
quantization semantics, kernels, DSpark settings, or production scheduling.

Smoke artifact:

```text
artifacts/m2/state-publication/fixture-16tok-2chunk.json
```

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_state_publication_fixture.py \
  --target-tokens 16 \
  --chunk-tokens 8 \
  --out artifacts/m2/state-publication/fixture-16tok-2chunk.json
```

Observed frontiers:

```text
1:  engram
2:  idx, index_k, kv
8:  idx, index_k, kv
14: engram, idx, index_k, kv
20: candidates, idx, index_k, kv
24: idx, index_k
28: idx, index_k
32: idx, index_k
36: idx, index_k
```

Correctness gates in the smoke artifact passed:

- implicit replay final greedy token matched accepted `_forward`;
- explicit replay final greedy token matched accepted `_forward`;
- implicit and explicit final logits digests matched accepted `_forward`;
- explicit publication-frontier digests matched implicit-dict frontier digests;
- final explicit cache digest matched accepted `_forward`;
- two chunks were exercised;
- final explicit cache digest matched implicit replay after chunk-1 to chunk-2
  propagation.

This proves only the smallest explicit-publication contract over a tiny bounded
fixture.  It does not justify 4K/8K timing by itself if later fixtures expose a
semantic mismatch.

## Historical loop-inversion oMLX-compatibility prototype

Added:

```text
tools/run_m2_layer_major_correctness_fixture.py
```

This fixture performs the first actual loop inversion, still at tiny scale and
still oMLX-compatibility-only.  It compares:

1. accepted chunk-major oMLX `_forward` over two chunks;
2. chunk-major explicit-publication replay;
3. layer-major explicit-publication replay, ordered as layer 0 over all chunks,
   then layer 1 over all chunks, etc.

The layer-major replay keeps per-chunk hidden/pre tensors and per-chunk explicit
publication objects, while each layer's existing oMLX cache object advances
across chunks before the next layer starts.  Engram hashes/history are computed
when layer 0 processes each chunk, preserving chunk-to-chunk Engram history for
later Engram producer layers.  The replay still uses existing oMLX operations
for all math and cache contents.

Prototype module/tool:

```text
ds41f_mlx/m2_layer_major.py
tools/run_m2_layer_major_correctness_fixture.py
```

Smoke artifacts:

```text
artifacts/m2/layer-major-correctness/fixture-16tok-2chunk.json
artifacts/m2/layer-major-correctness/fixture-32tok-4chunk.json
artifacts/m2/layer-major-correctness/fixture-32tok-4chunk-final-logits.json
```

Runs:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_correctness_fixture.py \
  --target-tokens 16 \
  --chunk-tokens 8 \
  --out artifacts/m2/layer-major-correctness/fixture-16tok-2chunk.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_correctness_fixture.py \
  --target-tokens 32 \
  --chunk-tokens 8 \
  --out artifacts/m2/layer-major-correctness/fixture-32tok-4chunk.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_correctness_fixture.py \
  --target-tokens 32 \
  --chunk-tokens 8 \
  --final-logits-only \
  --out artifacts/m2/layer-major-correctness/fixture-32tok-4chunk-final-logits.json
```

Observed gates passed:

- two chunks exercised;
- chunk-major explicit replay logits/cache matched accepted `_forward`;
- layer-major explicit replay logits/cache matched accepted `_forward`;
- layer-major greedy token matched accepted `_forward`;
- publication-frontier digests matched between chunk-major explicit replay and
  layer-major explicit replay.

A diagnostic `final_logits_only` mode was also added and immediately tested. It
projects only the final prompt-position logits after all hidden/cache/frontier
state has been computed.  The 32-token / 4-chunk diagnostic preserved final
cache, publication frontiers, and greedy token, but failed exact final-logits
digest comparison against accepted `_forward`:

```text
artifacts/m2/layer-major-correctness/fixture-32tok-4chunk-final-logits.json
ok: false
layer_major_logits_exact: false
layer_major_token_exact: true
layer_major_cache_exact: true
publication_frontiers_exact: true
```

This means "project only the last position" is not an oMLX-exact optimization
under the current oMLX projection/kernel behavior, even though the greedy token
matched in this tiny fixture. Do not promote final-logits-only projection as an
oMLX-compatible optimization. A bounded performance prototype must preserve the
historical oMLX projection behavior used by accepted `_forward`, or first explain and gate any numerical
mismatch.

The current layer-major prototype still stores per-chunk hidden state and is not
a performance design.  Do not treat it as evidence for 4K/8K speed until a
separately bounded performance prototype removes fixture-only storage and
preserves the oMLX compatibility gates.

## Bounded 4K full-projection performance prototype

Added:

```text
tools/run_m2_layer_major_perf.py
```

This harness is a bounded single-request prototype, not production
qualification.  It preserves historical oMLX full-chunk/full-position projection
behavior and explicitly disables the rejected `final_logits_only` path.  Timed
layer-major execution skips publication digest materialization, but keeps the
explicit shared-state object and existing oMLX model/cache math. Compatibility
checks compare greedy token and final cache digest against accepted chunk-major
`_forward`.

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_perf.py \
  --target-tokens 4096 \
  --chunk-tokens 2048 \
  --out artifacts/m2/layer-major-perf/perf-4k-full-projection.json
```

Result artifact:

```text
artifacts/m2/layer-major-perf/perf-4k-full-projection.json
```

Initial observed result:

| Path | Chunks | Seconds | Tokens/s | Greedy | Peak MLX active |
| --- | --- | ---: | ---: | ---: | ---: |
| accepted chunk-major `_forward` | `[2048, 2048]` | 20.652 | 198.34 | 201 | 316.36 GB |
| layer-major prototype | `[2048, 2048]` | 20.727 | 197.62 | 201 | 321.98 GB |

Follow-up bounded lifetime/materialization isolation was then made inside the
same harness:

- compute accepted greedy/cache digest;
- delete accepted logits/cache Python references before layer-major timing;
- run `gc.collect()`, `mx.clear_cache()`, and `mx.reset_peak_memory()` before the
  layer-major timed section;
- keep historical oMLX full-projection behavior and final-cache compatibility checks.

Artifact:

```text
artifacts/m2/layer-major-perf/perf-4k-full-projection-lifetime-isolated.json
```

Observed result after isolation:

| Path | Chunks | Seconds | Tokens/s | Greedy | Peak MLX active |
| --- | --- | ---: | ---: | ---: | ---: |
| accepted chunk-major `_forward` | `[2048, 2048]` | 20.674 | 198.12 | 201 | 316.36 GB |
| layer-major prototype | `[2048, 2048]` | 20.704 | 197.83 | 201 | 319.79 GB |

Gates passed:

- two or more chunks exercised;
- historical oMLX full-projection behavior fixed;
- greedy token exact;
- final cache digest exact.

A second bounded change targeted layer-major path materialization directly:

- `eval_boundary=layer` and `eval_boundary=chunk` materialize retained h/pre/cache
  state during the layer-major sweep to cut long MLX graph chains;
- `output_mode=full_chunks_last` still projects every chunk at full length, but
  returns only the final chunk logits instead of concatenating a full-prompt
  logits tensor.

Correctness artifact for the output-retention change:

```text
artifacts/m2/layer-major-correctness/fixture-32tok-4chunk-full-chunks-last.json
```

It passed exact last-chunk logits, final cache, frontier, and greedy-token gates.

Performance artifacts:

```text
artifacts/m2/layer-major-perf/perf-4k-full-projection-eval-layer.json
artifacts/m2/layer-major-perf/perf-4k-full-projection-eval-chunk.json
artifacts/m2/layer-major-perf/perf-4k-full-chunks-last.json
```

| Layer-major variant | Seconds | Tokens/s | Peak MLX active | Result |
| --- | ---: | ---: | ---: | --- |
| lifetime-isolated baseline | 20.704 | 197.83 | 319.79 GB | flat |
| `eval_boundary=layer` | 21.097 | 194.15 | 319.66 GB | slower |
| `eval_boundary=chunk` | 20.911 | 195.88 | 319.66 GB | slower |
| `output_mode=full_chunks_last` | 20.636 | 198.49 | 314.38 GB | memory lower, speed flat |

Decision after these bounded 4K changes:

- The oMLX-compatibility-preserving Python/MLX layer-major prototype still does not
  produce a 4K speed win.
- Explicit MLX eval boundaries reduce neither time nor peak memory enough; they
  slow the prototype.
- Avoiding full-prompt logits concatenation is useful for memory and preserves
  historical oMLX full-chunk projection behavior, but it does not create throughput
  separation.
- Per the M2 promotion criteria, do not proceed to 8K/16K timing from this
  version.
- The useful evidence is now narrower: loop inversion, explicit state
  publication, historical oMLX full-projection behavior, benchmark-lifetime isolation,
  graph-boundary materialization, and output-logits retention have all been
  tested in bounded form.  The remaining likely gap is not a shallow Python loop
  inversion; it would require a real carry-buffer/static-lifetime implementation
  or lower-level scheduling/buffer ownership change, which is outside this
  skeleton's scope.
