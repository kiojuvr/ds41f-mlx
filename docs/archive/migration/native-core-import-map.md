# Native core import map — Phase 2A

Status: dependency-closure analysis only.  No legacy native source has been copied in this commit, no runtime implementation has changed, and no model math / numerical Boundary / benchmark has run.

Machine-readable closure: `artifacts/provenance/native-core-closure.json`.

## Source identity

Import source is frozen to:

```text
repo: kiojuvr/deepseek-v41-flash-mlx
branch: main
commit: 1b7d0a2c7d33602437dffd44e26a33f39f189661
```

Do not import newer or older legacy native state unless a future manifest explicitly changes this source identity.

## Phase 2 principle

Do not import state/attention as isolated fragments.  The legacy native runtime has a coherent C++/MLX dependency closure.  Phase 2 should import that closure with minimal source changes, then rebind authority to ds41f semantic contracts.

Correctness authority remains:

```text
ds41f official semantic validators/contracts = authority
imported native core = implementation
```

## Entry points traced

Phase 2A traces from:

```text
TextBackbone
TextEncoder
TextDecoder
TextGeneration
Block
CompressedBlock
ReusedBlock
SWA layer
Compressed producer layer
Reused consumer layer
```

Expected dependency families:

```text
checkpoint / weights
linear
mHC
MoE / routing / expert storage
SWA/window attention
compressed RoPE
Compressor
GlobalKVState
IndexKey
IndexQuery
SharedAttention
CompressedLayer
ReusedLayer
Engram
TextEncoder / TextDecoder / TextBackbone
generation / sampling
runtime/session state
Metal kernels
```

## Target layout

Recommended initial layout:

```text
native/
  CMakeLists.txt
  include/dsv41/        # initially retain legacy namespace
  src/
    attention/
    cache/
    engram/
    mhc/
    model/
    moe/
    runtime/
  metal/
    attention/
    engram/
    mhc/
    moe/
  tests/
```

Keep `dsv41` namespace initially.  A later wrapper or mechanical namespace rename can happen after build/test parity.

## Build target classification

Initial import should include only CORE_REQUIRED and TEST_REQUIRED.

### CORE_REQUIRED

```text
dsv41_checkpoint
dsv41_storage
dsv41_engram_mlx
dsv41_model_mlx
dsv41_text_pair_mlx
dsv41_runtime_bridge_mlx
```

### TEST_REQUIRED

```text
dsv41-swa-attention-test
dsv41-test-swa
dsv41-block-test
dsv41-text-encoder-test
dsv41-text-decoder-test
dsv41-text-backbone-test
dsv41-sampling-test
dsv41-generation-loop-test
dsv41-route-policy-test
dsv41-test-engram
dsv41-bridge-lifecycle-test
```

### DIAGNOSTIC / BENCHMARK / OBSOLETE

Trace/probe binaries, context ladder, performance measurement scripts, repair probes, and rejected-candidate benchmarks should not be imported into the default build in Phase 2.

## Attention source risk map

Legacy attention source mixes accepted production paths and historical diagnostics.

### `src/attention/compressed_layer.cpp`

Classification: mixed.

Contains:

- PRODUCTION_REQUIRED token/chunk compressed producer/consumer dataflow.
- PRODUCTION_REQUIRED accepted fixed-tile path and request-boundary compact topology.
- DIAGNOSTIC_OPTIONAL telemetry and fixed-tile attribution diagnostics.
- REJECTED/OBSOLETE wide/older packed switches unless kept unreachable.

Import decision:

- Prefer byte-preserving initial copy only if rejected branches are not production-selectable in ds41f build/config.
- Otherwise split/disable rejected switches in a dedicated adaptation commit with provenance.

### `src/attention/swa_attention.cpp`

Classification: mixed.

Contains:

- PRODUCTION_REQUIRED batched split-K, ragged tail, width-one, fixed-tile core, packed work-list.
- DIAGNOSTIC/REJECTED older `packed_chunk` / `wide_chunk` entry points.

Risk: this file currently includes generated headers for `packed_chunk_attention` and `wide_chunk_attention`.  If copied unchanged, those kernels may be needed for compilation even if not production.  They must not become selectable production paths.

### Rejected kernels

Do not import as production:

```text
wide_chunk_attention.metal
old packed/direct/fused variants
rank-3 padded attention
DwarfStar-style row-serial attention
direct packed one-dispatch MMA
old rectangular packed work-list
```

If a rejected/diagnostic source must be present only to compile an otherwise unmodified file, mark it `DIAGNOSTIC_OPTIONAL_OR_REJECTED_SOURCE_DEPENDENCY` and make it unreachable from production.

## State mapping

| Legacy owner | Legacy representation | ds41f semantic owner | Classification | Decision |
|---|---|---|---|---|
| `SwaLayerState` / `SwaReferenceState` | position plus bounded window KV ring/arrays | per-layer `window_kv_cache` | persistent | preserve/adapt |
| `CompressorState` | pending `kv_state` / `score_state` slots plus position | `Compressor.kv_state` / `score_state` | persistent for ratio>1 source layers | preserve/adapt |
| `GlobalKVState` | packed main/index KV bytes/scales plus compressor pending | `compress_kv_cache` and `Indexer.k_cache` | persistent source-layer state | preserve/adapt |
| `SharedAttentionReference` / publications | cache pointers, topk/candidates, position/source metadata | `shared_attn.compress_kv/index_k/topk_idxs/candidates` | call-local publication over persistent caches | preserve/adapt |
| `TextBackboneState` | encoder/decoder/hash/window/compressed/reuse states, revision | session model state plus runtime session container | persistent session state | preserve/adapt |
| sampling/generation RNG | native runtime seed/state | target-runtime RNG session state | runtime-owned persistent state | preserve/adapt with authority split |

## Existing ds41f scaffold classification

```text
ds41f_mlx/native/ds41f_prefill_native.c
  ARCHITECTURE_DONOR_ONLY or SUPERSEDED_BY_IMPORTED_CORE after native import

ds41f_mlx/native/ds41f_prefill_kernels.metal
  ARCHITECTURE_DONOR_ONLY; compare with imported Metal core later

ds41f_mlx/native_prefill.py
  INTEGRATE_LATER as Python binding/prototype or SUPERSEDED_BY_IMPORTED_CORE

ds41f_mlx/m2_layer_major.py and dwarfstar_* modules
  ARCHITECTURE_DONOR_ONLY / historical compatibility diagnostics, not canonical runtime
```

Do not delete these in Phase 2A.  Avoid permanent duplication after the imported core builds.

## Dependency checker policy

Future Phase 2 import must verify:

```text
no native source/build/test file references /Volumes/SDXC-512/deepseek-v41-flash-mlx
no include path points at old repo
no build path points at old repo
no generated kernel path points at old repo
```

Historical provenance strings in artifacts are allowed.

## Recommended commit structure from here

1. `Import legacy native core foundation`
   - CMake foundation
   - checkpoint/weights/linear/common model infrastructure
   - required generated/Metal plumbing
   - core headers

2. `Import proven attention and session-state runtime`
   - SWA
   - Compressor
   - GlobalKV
   - Indexer
   - SharedAttention
   - compressed/reused layers
   - production attention kernels
   - state tests

3. `Adapt imported native core to ds41f repository paths`
   - repository-local include/build wiring only
   - no semantic rewrite

If the mixed attention files make byte-preserving import unsafe, add a small dedicated adaptation commit that disables rejected branches while preserving accepted production lineage.

## Phase 2B/2C import result

Phase 2B imported a local native foundation under `native/`:

```text
native/CMakeLists.txt
native/include/dsv41/{checkpoint_atlas,weights,swa_state,engram}.hpp
native/src/model/{checkpoint_atlas,weights}.cpp
native/src/engram/{index,store}.cpp
native/src/attention/swa_state.cpp
native/third_party/nlohmann/json.hpp
```

The foundation builds as `dsv41_checkpoint` and `dsv41_storage`.

Phase 2C imported the attention/session subsystem and required generated Metal plumbing:

```text
SWA/window state and projection
compressed RoPE
Compressor
GlobalKVState
IndexKey
IndexQuery
KV quantization
SharedAttention publication
CompressedLayer / ReusedLayer attention implementation
accepted fixed-tile / packed work-list / batched split-K / ragged tail / width-one kernels
```

The imported MLX target is `dsv41_attention_session`.  It builds locally when configured with the local MLX CMake package:

```sh
cmake -S native -B native/build-mlx \
  -DDSV41_ENABLE_MLX=ON \
  -DCMAKE_PREFIX_PATH=$PWD/.venv/lib/python3.13/site-packages/mlx
cmake --build native/build-mlx -j 4
```

Checkpoint-free `dsv41-test-swa` passes in both foundation-only and MLX-enabled builds.

## Rejected selector handling

Legacy `compressed_layer.cpp` / `swa_attention.cpp` still require some diagnostic/superseded source dependencies to compile without a broad refactor.  In this import:

- `DSV41_RUNTIME_PACKED_CHUNK_ATTENTION` is disabled in `native/include/dsv41/execution_policy.hpp`.
- `DSV41_RUNTIME_WIDE_ATTENTION` is disabled in `native/include/dsv41/execution_policy.hpp`.
- Fixed-tile, ragged tail, width-one, and packed work-list accepted lineage remains available.

Thus `wide_chunk_attention` and old `packed_chunk_attention` source/kernel files may exist as diagnostic compile dependencies, but are not production-selectable.

## Existing ds41f scaffold classification after Phase 2C

```text
ds41f_mlx/native/ds41f_prefill_native.c
  ARCHITECTURE_DONOR_ONLY / SUPERSEDED_BY_IMPORTED_CORE once native bindings exist

ds41f_mlx/native/ds41f_prefill_kernels.metal
  ARCHITECTURE_DONOR_ONLY

ds41f_mlx/native_prefill.py
  INTEGRATE_LATER as possible Python binding facade, otherwise SUPERSEDED_BY_IMPORTED_CORE

ds41f_mlx/m2_layer_major.py and dwarfstar_* modules
  ARCHITECTURE_DONOR_ONLY / historical compatibility diagnostics, not canonical runtime
```

Do not grow a parallel runtime from these scaffolds.

## Dependency checks

Use:

```sh
python3 tools/check_legacy_import_integrity.py
python3 tools/check_native_import_dependencies.py
```

The second checker scans `native/` source/build/test files for active old-repo path dependencies and verifies rejected attention selectors remain disabled.

## Phase 2 status

The state + attention subsystem has crossed from historical implementation reference to local reusable native implementation source.  Phase 3 should import HC + MoE + Engram + full TextBackbone/generation using the same closure-import principle.
