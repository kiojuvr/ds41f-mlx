# M2 DwarfStar-authoritative prefill engine

M2 treats a pinned DwarfStar DeepSeek-V4.1 prefill graph as the architecture
authority for a new prefill engine.  oMLX is retained only as a historical
adapter, compatibility reference, implementation donor, and performance
baseline.  It is not an official DeepSeek semantics/logits/cache/state authority.

Authority refresh note: the fetched public upstream revision is pinned as
`0aaea5a238fb41a35106a551e73c8409dfb751ac` and does contain the current V4.1
`ds41_graph_prefill_sweep` / deferred-decoder symbols.  See
`docs/m2-dwarfstar-v41-prefill-provenance.md`.  The native C code below remains
a generic scaffold until it is reconciled to that V4.1 sweep authority.

This deliberately permits an asymmetric runtime:

```text
prefill: DwarfStar-derived architecture
decode:  oMLX-derived DSpark/MTP architecture
```

The objective is no longer to ask an LLM to discover which shallow oMLX loop
change might become fast.  The objective is to implement toward the known
DwarfStar graph structure while preserving official checkpoint data and avoiding
model-semantics claims until the official DeepSeek reference is pinned.

## DwarfStar authority points

Pinned revision:

```text
0aaea5a238fb41a35106a551e73c8409dfb751ac
```

Reviewed source at that revision:

```text
$HOME/ds4/ds4.c
$HOME/ds4/ds4_deepseek41_gpu.h
```

Relevant authority functions/comments:

- `metal_graph_prefill_layer_major`
- `metal_graph_encode_layer_batch`
- `metal_graph_dspark_capture_prefill_layer`
- `metal_graph_capture_prefill_seed_router_selected`
- `metal_graph_seed_streaming_expert_cache_layer_from_mapped_hotlist`
- `metal_graph_seed_streaming_expert_cache_from_hotlist`
- `metal_graph_seed_streaming_expert_cache_from_prefill`
- `metal_graph_encode_output_head`

DwarfStar prefill structure:

1. upload prompt tokens;
2. upload prompt embeddings into HC/carry buffer;
3. for each layer:
   - prepare/map/readahead layer weights where applicable;
   - begin layer command batch;
   - encode full layer batch;
   - capture DSpark prefill state seam;
   - publish layer state/frontier;
   - seed router/expert-cache state;
   - end layer command batch;
4. release prefill mask cache;
5. seed streaming expert cache from hotlist/prefill;
6. select final output row;
7. encode output head;
8. read logits.

The first implementation in this repo captures that graph shape as an explicit
plan and executes it through the current oMLX-operation adapter.  That adapter
is now classified as an oMLX-compatibility diagnostic, not official correctness
evidence.  It does not import DwarfStar kernels, GGUF/Q4 formats, or DwarfStar
precision semantics.

## Added engine seam

```text
ds41f_mlx/dwarfstar_prefill.py
tools/run_m2_dwarfstar_prefill_smoke.py
```

`DwarfStarPrefillEngine` builds a `DwarfStarPrefillPlan` with the graph stages
above, including DeepSeek-V4.1 publication frontiers derived from the loaded
oMLX config.  Its current adapter executes through `ds41f_mlx.m2_layer_major` so
correctness remains pinned to existing oMLX operations.

This is the new seam where a real static carry-buffer / command-buffer / buffer
ownership implementation should replace the adapter internals later.  Decode is
not changed.

## Superseded oMLX-compatibility bridge

Added historically:

```text
ds41f_mlx/dwarfstar_semantics.py
tools/run_m2_dwarfstar_semantics_contract.py
```

The M0-M2 correctness-authority audit supersedes this as an official-semantics
contract. It records how each DwarfStar graph step was bound to the historical
oMLX adapter. It may be used as compatibility/regression documentation only and
must not be used as official DeepSeek correctness evidence. Historical examples:

- `upload_embeddings_hc` is bound to oMLX `LanguageModel.embed` plus HC
  repeat/pre-mask behavior;
- `prepare_layer_weights` is bound to the read-only official safetensors loaded
  by oMLX, not DwarfStar GGUF/Q4 layer packs;
- `encode_layer_batch` is bound to oMLX `Block.__call__` semantics for Engram,
  attention, HC, MoE/FFN, activation packing, dtype behavior, and current kernel
  selection;
- `publish_state_frontier` is bound to explicit `kv`, `index_k`, `idx`,
  `candidates`, and Engram-history frontiers;
- `select_output_row` records the final-logits-only negative result against the
  historical oMLX adapter and requires no official semantics inference from that
  mismatch;
- decode policy remains oMLX DSpark/MTP.

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_dwarfstar_semantics_contract.py \
  --out artifacts/m2/dwarfstar-prefill/official-semantics-contract.json  # historical filename; superseded
```

## Smoke gate

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_dwarfstar_prefill_smoke.py \
  --target-tokens 32 \
  --chunk-tokens 8 \
  --out artifacts/m2/dwarfstar-prefill/smoke-32tok-4chunk.json
```

Historical gates:

- at least two chunks exercised;
- DwarfStar authority recorded in the plan;
- plan is layer-major (`encode_layer_batch` once per layer);
- greedy token exact against accepted oMLX chunk-major prefill;
- last-chunk logits exact against accepted oMLX full-chunk projection;
- final cache digest exact against oMLX.

This smoke proves the new engine seam and DwarfStar architecture plan only. The
logits/cache checks are oMLX-compatibility diagnostics and are not official model
correctness evidence.

## Native C/Metal-leaning planner scaffold

Added:

```text
ds41f_mlx/native/ds41f_prefill_native.h
ds41f_mlx/native/ds41f_prefill_native.c
ds41f_mlx/native/ds41f_prefill_kernels.metal
ds41f_mlx/native_prefill.py
tools/run_m2_native_prefill_plan.py
```

This is the first step toward changing prefill internals boldly toward the
DwarfStar C/Metal architecture without changing model math yet.  The native C
ABI owns graph/arena planning for the DwarfStar-authoritative prefill shape:

- layer-major step count;
- split command-batch count;
- publication frontier count;
- static ping-pong HC carry-buffer estimate;
- last-chunk logits retention estimate instead of full-prompt logits retention.

The Metal file currently defines only minimal copy/zero ownership kernels.  It
is a placeholder vocabulary for future carry-buffer and command-buffer work, not
a replacement for official DeepSeek-V4.1 math.

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_plan.py \
  --target-tokens 4096 \
  --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-plan-4k.json
```

Result artifact:

```text
artifacts/m2/dwarfstar-prefill/native-plan-4k.json
```

Observed 4K native plan:

```text
n_tokens: 4096
max_chunk_tokens: 2048
layer_steps: 40
command_batches: 42
publication_frontiers: 9
carry_buffer_bytes: 335,544,320
full_prompt_hidden_bytes: 167,772,160
last_chunk_logits_bytes: 1,059,061,760
full_prompt_logits_bytes: 2,118,123,520
```

Gates passed:

- native library compiled/loaded;
- native layer-step count matched the Python DwarfStar plan;
- token count exact;
- publication frontier count exact;
- historical oMLX-compatibility bridge present;
- native plan uses full-prompt current/next ping-pong carry buffers;
- native plan avoids full-prompt logits retention for the final readout path.

This does not claim speed.  It establishes a native ABI seam where subsequent
work can move carry buffers, command submission, and then selected official-math
kernels into C/Metal while the semantics contract remains fixed.

## Native arena / carry-buffer ownership

Added native context ownership APIs:

```text
ds41f_prefill_native_context_create
ds41f_prefill_native_context_destroy
ds41f_prefill_native_context_info
ds41f_prefill_native_upload_tokens
ds41f_prefill_native_swap_carry
ds41f_prefill_native_current_carry
ds41f_prefill_native_next_carry
```

and Python smoke:

```text
tools/run_m2_native_prefill_arena.py
```

This is the first concrete move of prefill lifetime ownership into C.  The
native context now owns:

- uploaded prompt token buffer;
- static ping-pong HC carry buffers sized from the historical oMLX adapter config;
- carry-buffer current/next swap state.

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_arena.py \
  --target-tokens 4096 \
  --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-arena-4k.json
```

Result artifact:

```text
artifacts/m2/dwarfstar-prefill/native-arena-4k.json
```

Observed 4K arena:

```text
carry_buffer_bytes: 335,544,320
token_buffer_bytes: 16,384
total_owned_bytes: 335,560,704
n_tokens_uploaded: 4096
carry swap toggled current index 0 -> 1
```

Gates passed:

- native library loaded;
- arena allocated expected carry bytes;
- token buffer sized exactly;
- token upload count exact;
- carry swap toggled;
- current/next carry addresses swapped.

This still does not execute model math.  It moves DwarfStar-style static
carry-buffer ownership behind the native prefill ABI so later work can wire
Metal command submission and official-math kernels into the same seam.

## Native command stream / no-op graph submission

Added native command stream APIs:

```text
ds41f_prefill_native_build_commands
ds41f_prefill_native_command_count
ds41f_prefill_native_command_at
ds41f_prefill_native_execute_noop_graph
```

and Python smoke:

```text
tools/run_m2_native_prefill_commands.py
```

Authority check before Metal connection: DwarfStar `metal_graph_encode_layer_batch`
processes `n_tokens` for the layer, writes `metal_graph_batch_next_hc(g)`, and
then swaps `batch_cur_hc_by_tier` / `batch_next_hc_by_tier` once at the end of
the layer.  Therefore this native scaffold uses full-prompt current/next
ping-pong carry buffers and swaps once per layer, not once per layer/chunk.
Chunks in this scaffold are slices inside the full prefill-call carry buffer.

The native context now owns a DwarfStar-style layer-major command stream:

```text
upload_tokens
upload_embeddings_hc
for each layer:
    begin_layer
    for each chunk:
        encode_layer_chunk into next-carry slice
    swap_carry once for the full layer
    publish_frontier
    end_layer
encode_output_head
read_logits
```

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_commands.py \
  --target-tokens 4096 \
  --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-commands-4k.json
```

Result artifact:

```text
artifacts/m2/dwarfstar-prefill/native-commands-4k.json
```

Observed 4K command stream:

```text
command_count: 244
upload_tokens: 1
upload_embeddings_hc: 1
begin_layer: 40
encode_layer_chunk: 80
swap_carry: 40
publish_frontier: 40
end_layer: 40
encode_output_head: 1
read_logits: 1
```

Gates passed:

- command count exact;
- one encode per layer/chunk;
- one carry swap per layer, after all chunks for that layer have written their next-carry slices;
- one begin/end and one publish command per layer;
- output head and logits read once;
- no-op graph submitted every command;
- uploaded token count remained exact.

This is still a skeleton, but command ownership is now native and layer-major.

## Native command-driven carry mutation

The previous no-op command stream was advanced so `encode_layer_chunk` now
performs a deterministic native write into that chunk's slice of the next carry
buffer.  After all chunks for a layer have written next-carry slices, one
`swap_carry` command publishes the full-prompt next buffer as current.  This is
not DeepSeek math; it is the first proof that the C-owned command stream can
mutate and publish C-owned carry buffers.

Added smoke:

```text
tools/run_m2_native_prefill_carry_mutation.py
```

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_carry_mutation.py \
  --target-tokens 4096 \
  --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-carry-mutation-4k.json
```

Result artifact:

```text
artifacts/m2/dwarfstar-prefill/native-carry-mutation-4k.json
```

Observed gates passed:

- all commands submitted;
- 80 encode-layer/chunk commands executed for 40 layers × 2 chunks;
- 40 layer-level carry swaps executed;
- last encoded layer/chunk was `39/1`;
- carry checksums changed;
- final carry index returned to the initial index after an even number of layer swaps.

The follow-up command smoke with token upload also passed:

```text
artifacts/m2/dwarfstar-prefill/native-commands-4k-carry-mutation.json
```

The next native step can replace the deterministic carry stamp with Metal-buffer
copy/upload-embedding staging, then progressively bind official math kernels
behind the same command stream.

## M2 planner reconciliation: sweep topology

Added:

```text
ds41f_mlx/dwarfstar_v41_sweep.py
tools/run_m2_ds41_sweep_planner.py
```

This is a planner-contract transcription of `ds41_graph_prefill_sweep`; it does
not do Metal/data-plane work.  It replaces the generic layer-major mental model
with the V4.1 authority control flow:

- `ds41_prefill_limit(ctx)` chooses 2K/4K/8K graph caps;
- `ds41_encoder_chunk_cap(g, total_count)` keeps short appends at 2K, 8K-ish
  appends at 4K, and larger appends at the graph cap;
- wide sweeps at 8K+ enable decoder suffix mode;
- encoder layers 0..19 process all rows;
- decoder layers 20..39 process only `needed = 1 + (39 - layer) * 127` rows;
- encoder-only/resume sweeps are represented as invalid partial sweeps;
- structured carry rows are recorded from `DS41_CARRY_ROWS`;
- Engram table 0/table 1 prefetch ordering and SSD read-ahead events are
  recorded as scheduling events;
- checkpoint validity is false during the sweep and true only after a completed
  non-encoder-only sweep plus final logits.

Runs:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_ds41_sweep_planner.py \
  --ctx 32768 --remaining 8192 \
  --out artifacts/m2/dwarfstar-prefill/ds41-sweep-plan-8k.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_ds41_sweep_planner.py \
  --ctx 32768 --remaining 16384 \
  --out artifacts/m2/dwarfstar-prefill/ds41-sweep-plan-16k.json
```

Observed 8K plan:

```text
count: 8192
encoder_chunk: 4096
decoder_suffix: true
encoder row-layer work: 163,840
decoder suffix row-layer work: 24,150
total row-layer work: 187,990
```

This matches the architecture-level explanation for the DwarfStar +8K throughput
jump: token count doubles from +4K, but row-layer work grows only modestly
because layers 20..39 reconstruct a shrinking dependency suffix.

## Native C state/lifetime topology reconciliation

Added:

```text
tools/run_m2_native_sweep_reconciliation.py
```

The native C planner now has a V4.1 sweep-topology API that records the pinned
`ds41_graph_prefill_sweep` state/lifetime contract before any Metal/model-math
work resumes.  It is separate from the older generic command-stream smoke ABI so
prior scaffolding remains inspectable.

The reconciled planner records:

- pinned DwarfStar remote/SHA;
- 2K/4K/8K effective chunk policy;
- encoder full-row work and shrinking decoder suffix work;
- encoder-only and deferred-decoder invalid partial states;
- checkpoint-invalid / may-commit transitions distinct from publication
  frontiers;
- per-layer HC swap after all row chunks for that layer;
- explicit allocation records with semantic role, byte size, alignment,
  representation, first/last use, owner, alias/reuse class, persistence class,
  and encoder-only/deferred-decoder survival;
- structured `DS41_CARRY_ROWS` roles: `residual`, `pre`, `ffn_split`,
  `selected_comp`, and `block_mask`;
- Engram table 0/table 1 prefetch and SSD/read-ahead scheduling events;
- final output-head/read-logits ordering before checkpoint commit.

Run:

```sh
python3 tools/run_m2_native_sweep_reconciliation.py \
  --out artifacts/m2/dwarfstar-prefill/native-sweep-reconciliation.json
```

Bounded fixture gates passed for short non-wide, +8K wide, +16K/deferred
planning, encoder-only, and resume-style plans.  The +8K fixture records:

```text
encoder_chunk: 4096
encoder row-layer work:       163,840
decoder suffix row-layer work: 24,150
total row-layer work:         187,990
swap_hc_after_layer:          40
```

This satisfies the state/lifetime topology reconciliation stage.  It still does
not implement attention math, MoE math, HC replacement, model-weight conversion,
quantization, or Metal performance kernels.

## Native Metal-backed buffer ownership / submission scaffold

Added:

```text
tools/run_m2_native_buffer_submission.py
```

The native ABI now creates C-owned buffer slots directly from the reconciled
V4.1 sweep allocation table and submits the planned command stream against those
slots.  On Darwin these slots are backed by `MTLBuffer` objects and submission
commits real Metal command buffers that touch the planned buffers through blit
fill operations.  Command-buffer granularity is now recorded and batched at a
DwarfStar-style logical execution boundary instead of committing once per buffer
touch.  This is deliberately a bounded scaffold and still does not execute model
math or bind real checkpoint weights.

The submission context records:

- one owned buffer slot per planned arena allocation;
- total owned bytes and memory-budget enforcement for bounded fixtures;
- buffer role, size, persistence class, allocation state, touched state, and
  checksum;
- submitted sweep commands and synthetic command-buffer count;
- deterministic buffer touches for HC ping-pong, structured carry rows,
  decoder-suffix slots when present, stage scratch, and final logits;
- checkpoint validity before and after submission;
- Metal backend availability, allocated Metal buffer count, committed Metal
  command-buffer count, encoder count, completion waits, and commit breakdowns
  by phase/layer/category.

Run:

```sh
python3 tools/run_m2_native_buffer_submission.py \
  --out artifacts/m2/dwarfstar-prefill/native-metal-submission.json
```

The default fixture is a 4K bounded topology fixture with small synthetic
dimension/vocabulary values because this stage validates ownership/submission
plumbing, not logits semantics.  The artifact gates require all planned buffers
to be allocated, structured carry and HC buffers to be touched, all logical
commands to be submitted, checkpoint validity to remain false before submission
and become true only after the final commit command, owned bytes to stay within
the fixture budget, and on Darwin the Metal backend to allocate one `MTLBuffer`
per slot and commit batched Metal command buffers for the encoded operations.

Current corrected +4K bounded submission artifact (`ctx=32768`,
`remaining=4096`, bounded synthetic `dim/vocab`):

```text
logical planner commands: 285
encoded buffer ops:       562
Metal command buffers:     81
Metal blit encoders:       81
completion waits:          81
commits by phase:          encoder=40, decoder_full=40, final_output=1
commits by category:       encode_rows=80, decoder_prepare_suffix=0,
                           swap_hc_layer=0, final_output_read=1
```

The earlier 41-command-buffer observation was a short 2K effective-count fixture
caused by using `ctx=4096`; the default fixture now uses `ctx=32768` so
`remaining=4096` plans a real +4K sweep.

A bounded +8K wide fixture also passes and exercises decoder suffix planning:

```sh
python3 tools/run_m2_native_buffer_submission.py \
  --ctx 32768 --remaining 8192 --dim 16 --hc-mult 4 --vocab-size 256 \
  --memory-budget-mib 128 \
  --out artifacts/m2/dwarfstar-prefill/native-metal-submission-batching-8k.json
```

Observed +8K topology:

```text
logical planner commands: 289
encoded buffer ops:       464
Metal command buffers:     85
Metal blit encoders:       85
completion waits:          85
commits by phase:          encoder=40, decoder_full=1,
                           decoder_suffix=43, final_output=1
commits by category:       encode_rows=63, decoder_prepare_suffix=21,
                           swap_hc_layer=0, final_output_read=1
```

Wait/drain authority was checked against pinned DwarfStar source before moving to
model math.  `ds41_graph_prefill_sweep` opens command batches around row/chunk
staging and row/chunk encode work, and drains active command batches with
`ds4_gpu_end_commands()` at the staging/Engram boundary and after encode/carry
copy.  In `ds4_metal.m`, `ds4_gpu_end_commands()` closes the batch and calls
`ds4_gpu_finish_command_buffer()`, which commits, waits pending command buffers,
waits the current command buffer, and releases transient buffers.  `ds4_gpu_flush_commands()`
is the non-blocking commit path, but it is not the normal V4.1 prefill sweep
drain in the inspected path.  Therefore the current synchronous completion wait
per submitted batch is a faithful correctness scaffold for the modeled batches;
Engram prefetch and streaming read-ahead overlap remain recorded as scheduling
events outside this dummy data plane until real I/O/model work is introduced.

`swap_hc_layer` is intentionally not a separate Metal commit: it is modeled as
native ownership/pointer state, matching the DwarfStar layer-boundary HC swap
contract.  The artifact records separate semantic status fields:

```text
ownership_correct: true
metal_submission_correct: true
model_semantics_validated: false
```

No attention math, MoE math, HC replacement, model-weight conversion,
quantization, or Metal performance kernels have been started.

## First official model-data-plane stage: embedding gather

Added:

```text
tools/run_m2_native_official_embedding.py
```

This is the first deliberately tiny official model-math/data-plane connection.
It reads a bounded BF16 `embed.weight` slice from the official read-only
safetensors checkpoint, passes the raw BF16 bits into the native library, runs a
Metal compute kernel that performs token-row gather into a native output buffer,
and checks the result bit-exactly against the reference slice gather.  It does
not load or convert the full checkpoint, does not touch attention/MoE/HC math,
and does not claim full native prefill correctness.

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_official_embedding.py \
  --out artifacts/m2/dwarfstar-prefill/native-official-embedding.json
```

Observed bounded fixture:

```text
source tensor: embed.weight
source shard:  model-00002-of-00048.safetensors
representation: BF16 raw uint16 bits
vocab rows: 16
dim: 128
tokens: [0, 3, 7, 1, 15, 4]
Metal command buffers: 1
Metal compute encoders: 1
completion waits: 1
embedding lookup bit exact: true
```

The artifact keeps the scope explicit:

```text
official_model_data_used: true
embedding_lookup_bit_exact: true
model_semantics_validated: false
validated_stage: embed.weight token gather only
```

This establishes that official checkpoint data can enter the native Metal data
plane without changing representation for this bounded stage.  The next model
math steps should continue stage-by-stage, preserving exactness gates before
widening scope.

## Official projection primitive

Added:

```text
tools/run_m2_native_official_projection.py
```

This validates the first arithmetic projection primitive using a real
DeepSeek-V4.1 checkpoint projection tensor and shape, before attempting attention
or a complete layer.  The fixture uses:

```text
weight tensor: layers.0.ffn.gate.weight
weight shard:  model-00003-of-00048.safetensors
weight shape:  [384, 5120]
weight dtype:   BF16 raw uint16 bits
input source:   embed.weight gathered rows for tokens [0, 3]
operation:      output = input_bf16_as_f32 @ weight_bf16_as_f32.T
output dtype:   F32
accumulation:   F32, k increasing in the native kernel
```

Run:

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_official_projection.py \
  --out artifacts/m2/dwarfstar-prefill/native-official-projection.json
```

The artifact records source weight identity, scale identity (`null` for this
BF16 projection), input digest, reference/native output digests, output dtype,
accumulation contract, bit-exactness, mismatch count, and max absolute/relative
error.  A tolerance contract is declared before evaluating the result rather than
chosen afterward:

```text
bit_exact_required: false
max_abs_error_lte: 1.0e-4
max_rel_error_lte: 1.0e-5
```

Observed bounded result:

```text
Metal command buffers: 1
Metal compute encoders: 1
completion waits: 1
bit_exact: true
mismatch_count: 0
max_abs_error: 0.0
max_rel_error: 0.0
```

The scope remains explicit:

```text
official_model_data_used: true
official_weight_representation_preserved: true
projection_primitive_validated: true
model_semantics_validated: false
validated_stage: layers.0.ffn.gate.weight BF16 linear projection only
```

This proves one official real-shape linear primitive on the native Metal data
plane.  It does not validate quantized expert projections, attention, MoE, HC,
complete-layer execution, or full native prefill correctness.
