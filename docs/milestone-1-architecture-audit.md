# Milestone 1 architecture restoration audit

Status: **COMPLETE**.

This audit satisfies Milestone 1 from `docs/implementation-plan.md`. It is an architectural inventory only. It does not promote a runtime selector, optimize model execution, start API work, or begin Milestone 2 implementation.

## Scope and inspected sources

Repository evidence inspected:

- `README.md`
- `docs/runtime-strategy.md`
- `docs/implementation-plan.md`
- `docs/architecture.md`
- `docs/correctness.md`
- `docs/session-state.md`
- `docs/performance.md`
- `docs/provenance.md`
- `docs/qualification.md`
- `docs/scaffold-classification.md`
- `docs/archive/migration/m2-architecture-comparison.md`
- `docs/archive/migration/m2-layer-major-prototype-plan.md`
- `docs/archive/migration/m2-dwarfstar-prefill-engine.md`
- `docs/archive/migration/m2-dwarfstar-v41-prefill-provenance.md`
- `ds41f_mlx/dwarfstar_prefill.py`
- `ds41f_mlx/dwarfstar_v41_sweep.py`
- `ds41f_mlx/m2_layer_major.py`
- `ds41f_mlx/m2_state_publication.py`
- `ds41f_mlx/native_prefill.py`
- `ds41f_mlx/native/ds41f_prefill_native.{h,c}`
- `ds41f_mlx/runtime/omlx_core.py`
- `ds41f_mlx/runtime/omlx_worker.py`
- `native/include`, `native/src`, `native/tests`, `native/metal`
- `artifacts/m2/dwarfstar-prefill/*`
- `artifacts/m2/layer-major-*/*`
- `artifacts/m2/omlx-prefill-*/*`
- `artifacts/performance/native-short-context-baseline.json`

Upstream/local architecture sources inspected:

- DwarfStar local clone: `$HOME/ds4`
  - HEAD: `0aaea5a238fb41a35106a551e73c8409dfb751ac`
  - source files: `ds4.c`, `ds4_metal.m`, `ds4_deepseek41_gpu.h`
- oMLX local known-good tree: `$HOME/omlx-0.7.0.dev2`
  - git HEAD: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
  - worktree has the previously known local DeepSeek V4.1 image-token/parser patch in `omlx/patches/deepseek_v41/encoding.py`, `processing.py`, and its test; this audit treats the model runtime architecture as the pinned oMLX tree plus that local patch context.
  - inspected model/runtime files include `omlx/patches/deepseek_v41/language.py`, `cache.py`, `dspark.py`, `mtp.py`, `engram.py`, `loading.py`, `moe_offload.py`, `packed_attention.py`, `kernels.py`, and `omlx/scheduler.py`.

## Component role classification

| Component | Current role | Intended future role | Retain/reuse/replace | Reason |
| --- | --- | --- | --- | --- |
| `native/*` | Current executable correctness/reference runtime | Reference/qualification plus selectively reusable production components | Retain; reuse selectively | Complete enough to load and execute official checkpoint smoke, with validators/tests/state machinery, but measured decode is about 0.31 tok/s and topology must not be assumed production. |
| `native/include` / `native/src/model` | Reference text runtime, checkpoint-backed model blocks, generation | Qualification/reference; possible reusable session/generation components | Retain; reuse by explicit seam only | Provides `TextFront`, `TextEncoder`, `TextDecoder`, `TextBackboneState`, `TextGenerationReference`, reset/fork/continuation behavior. |
| `native/src/attention`, `native/metal/attention` | Reference/native attention implementation and kernels | Reusable components or validation fixtures, not selected topology | Retain; do not locally optimize as architecture substitute | Encodes SWA, compressed producer, reused consumer, index/candidate behavior. Current topology is too slow to be production by default. |
| `native/src/moe`, `native/metal/moe` | MoE/HC/expert validation and native pipelines | Reusable production components where selected architecture needs them | Retain | Strong official semantics and grouped expert evidence; final expert scheduling/lifetime belongs to selected production architecture. |
| `native/src/engram`, `native/metal/engram` | SSD-backed Engram reference and MLX/Metal pieces | Reusable production component and qualification oracle | Retain | Engram semantics and SSD-backed behavior are required. Future architecture must preserve ordering and state behavior. |
| `native/tests`, `artifacts/validation`, `artifacts/provenance` | Correctness and provenance gates | Qualification machinery | Retain and extend | Correctness authority is implementation-scoped and must gate future runtime paths. |
| `ds41f_mlx/dwarfstar_*` | DwarfStar-derived prefill planning and adapter line | Intended production prefill architecture line | Retain and continue | Captures pinned V4.1 sweep/lifetime topology; incomplete full-model math connection. |
| `ds41f_mlx/native/*` | Native C/Metal DwarfStar-style ownership/submission/primitive scaffold | Milestone 2 prefill substrate frontier | Retain and continue from current ABI | Owns carry/sweep/buffer submission and bounded official primitives. Not full runtime yet. |
| `ds41f_mlx/native_prefill.py` | Python bridge to native prefill ABI | Tooling bridge for Milestone 2 | Retain | Exposes planner, submission, and official primitive ABI. |
| `ds41f_mlx/m2_state_publication.py` | Explicit oMLX-compatible frontier skeleton | State-publication reference/tool | Retain concepts; not production | Useful producer/consumer frontier inventory. |
| `ds41f_mlx/m2_layer_major.py` | oMLX-compatible loop inversion prototype | Negative/diagnostic evidence | Retain as tool; do not promote | Proved shallow Python/MLX layer-major inversion is speed-flat at 4K. |
| `ds41f_mlx/runtime/omlx_*` | Thin oMLX bridge/worker | Decode architecture comparison harness/source evidence | Retain | oMLX has strong practical decode baseline and model architecture evidence. |
| `ds41f_mlx/server.py` | API scaffold | Historical/current scaffold only | Retain, later replace/route through `deepseek-recipe` backend plan | Serving integration is later milestone. |

## DwarfStar V4.1 prefill source evidence

The relevant DwarfStar source is `antirez/ds4` at `0aaea5a238fb41a35106a551e73c8409dfb751ac`. Source inspection confirms the following concrete authority points:

| Evidence | Source location | Architectural fact |
| --- | --- | --- |
| `DS41_PREFILL_CAP`, `DS41_CARRY_ROWS`, `DS41_PREFILL_STORAGE`, `ds41_gpu_graph` | `$HOME/ds4/ds4.c` around lines 40086-40160 | V4.1 prefill has structured carry rows (`residual`, `pre`, `ffn_split`, `selected_comp`, `block_mask`), row storage, windows, compressed/index caches, previous KV/score, Engram tables/history, and prefill tokens in one graph object. |
| `ds41_prefill_limit` | `$HOME/ds4/ds4.c` around line 40133 | Prefill cap selects 2K/4K/8K by context/env policy. |
| `ds41_carry_cap` | `$HOME/ds4/ds4.c` around line 40145 | Carry capacity is compactable, memory-budgeted, and aligned to 2048 causal sweep boundaries. |
| `ds41_encoder_chunk_cap` | `$HOME/ds4/ds4.c` around line 41643 | Encoder chunk cap keeps short work at 2K, 8K-ish work at 4K, and larger work at graph cap. |
| `ds41_graph_prefill_sweep` | `$HOME/ds4/ds4.c` around lines 41660-42030 | Production prefill is a causal layer-major sweep with partial-sweep invalidity, encoder-only/deferred decoder handling, Engram prefetch, streaming layer/expert preparation, per-layer row chunks, publication/seeding, final state commit. |
| decoder suffix | `$HOME/ds4/ds4.c` around lines 41725-41736 | Wide sweeps at 8K+ process decoder layers 20..39 over shrinking dependency suffix rows, not full prompt rows. |
| Engram ordering | `$HOME/ds4/ds4.c` around lines 41690-41717 and 41762-41778 | Table 0 prefetch may start before sweep; table 1 starts at layer 2 for layer 14; pipeline/wait behavior overlaps disk reads and layer work. |
| state validity | `$HOME/ds4/ds4.c` around lines 41682-41685 and 41970-41975 | Graph/session is invalid during sweep and becomes valid only after successful non-encoder-only sweep and final state publication. |
| command drain semantics | `$HOME/ds4/ds4_metal.m` `ds4_gpu_end_commands`, `ds4_gpu_flush_commands` | Normal inspected prefill batches drain through `end_commands`; non-blocking flush exists but is not the usual modeled sweep completion path. |

Conclusion: the DwarfStar-derived prefill target is not a generic layer-major idea. It is specifically `ds41_graph_prefill_sweep` with compact structured carry, graph invalid/commit semantics, encoder/deferred-decoder split, shrinking decoder suffix, Engram/disk overlap, layer/expert streaming preparation, and explicit command-batch lifetime.

## Existing DwarfStar-derived implementation frontier

### Implemented as planner / topology contract

- `ds41f_mlx/dwarfstar_v41_sweep.py` transcribes:
  - pinned remote/SHA;
  - 2K/4K/8K prefill cap policy;
  - encoder full-row work;
  - wide decoder suffix;
  - encoder-only/resume constraints;
  - deferred decoder candidate flag;
  - checkpoint validity during/after sweep;
  - structured carry row names/roles;
  - Engram table prefetch and SSD/read-ahead scheduling events.
- `artifacts/m2/dwarfstar-prefill/ds41-sweep-plan-*.json` preserves concrete 8K/16K/32K planning artifacts.

### Implemented as native C ownership / lifetime scaffold

- `ds41f_mlx/native/ds41f_prefill_native.h` defines:
  - pinned DwarfStar authority constants;
  - generic prefill plan ABI;
  - V4.1 sweep config/allocation/command/result structs;
  - submission context ABI;
  - official primitive ABI for embedding and linear stages.
- `ds41f_mlx/native/ds41f_prefill_native.c` implements:
  - `ds41f_prefill_native_build_sweep_plan` with allocations and commands for invalidation, Engram prefetch, decoder suffix prepare, row encoding, HC swap, frontier publication, deferred decoder invalidity, output head, read logits, checkpoint commit;
  - submission context creation/destruction and bounded command submission over planned buffers;
  - older generic arena/token/carry/command scaffolds;
  - bounded official embedding gather and linear primitive functions.
- `ds41f_mlx/native_prefill.py` exposes these through ctypes and records source authority in JSON summaries.

### Implemented with real Metal submission

- `artifacts/m2/dwarfstar-prefill/native-metal-submission*.json` records bounded Metal-backed buffer allocation/submission.
- The 8K batching artifact records decoder suffix activity and command-buffer breakdown, e.g. `encode_rows=63`, `decoder_prepare_suffix=21`, `final_output_read=1`, all planned buffers allocated/touched, and checkpoint becoming valid only after final commit.
- This is real Metal buffer/submission plumbing, not model math.

### Connected to official checkpoint data

- `artifacts/m2/dwarfstar-prefill/native-official-embedding.json` validates a bounded BF16 `embed.weight` gather from the official safetensors checkpoint, bit-exact against reference, with one Metal compute submission.
- `artifacts/m2/dwarfstar-prefill/native-official-projection.json` validates one official real-shape BF16 linear projection (`layers.0.ffn.gate.weight`, `[384,5120]`) with F32 accumulation, bit-exact in the recorded fixture.

### Connected to real model math

- Bounded embedding gather and one BF16 projection primitive are connected to real official model data.
- The DwarfStar-derived prefill line is **not** connected to full attention, HC, MoE, Engram layer math, full output head, complete layer execution, full checkpoint prefill, or decode handoff.
- Additional primitive ABI names are present in `native_prefill.py`/C source, but without current milestone artifacts they are not treated as qualified implementation evidence.

### Performance measured

- oMLX-compatible layer-major Python prototype at 4K was about speed-flat with oMLX: roughly 198 tok/s, with compatibility fixtures passing. This is negative evidence for shallow loop inversion.
- Native C/Metal DwarfStar-derived scaffolds have no production throughput claim because they do not execute full model math.
- DwarfStar Q4 resident numbers remain architecture/scaling-shape evidence only, not official-checkpoint equivalence.

### Reusable directly for Milestone 2

- Pinned source/function map and `ds41_graph_prefill_sweep` topology.
- `ds41f_mlx/dwarfstar_v41_sweep.py` planner semantics.
- `ds41f_mlx/native/ds41f_prefill_native.{h,c}` sweep allocation/command/submission ABI.
- `ds41f_mlx/native_prefill.py` ctypes bridge.
- Official embedding/projection primitive loading/check patterns.
- `m2_state_publication.py` producer/consumer frontier inventory.
- Existing artifacts as regression checks for topology and bounded primitives.

### Remaining gap to real full-model official-checkpoint prefill

Milestone 2 must implement, behind the existing DwarfStar-derived sweep/lifetime seam:

1. full official embedding upload into the carry/HC representation used by the sweep;
2. HC pre/post math over sweep rows;
3. attention projections, RoPE, window KV, compressed KV, index K, candidate/index publication, and attention output under official semantics;
4. Engram row lookup/projection in the recorded prefetch order;
5. MoE gate/shared/routed expert math and selected expert scheduling under the selected residency model;
6. per-layer state publication and seed/prefetch side effects matching official contracts;
7. final norm/output head/read-logits path;
8. transaction semantics for invalid partial sweeps, encoder-only/deferred-decoder, commit, reset, fork, and continuation;
9. official checkpoint weight access without incompatible DwarfStar GGUF/quantization assumptions;
10. prefill-to-decode state handoff through a narrow production session boundary.

Milestone 2 should **not** reimplement a new planner. It should continue from the existing V4.1 sweep planner and native submission/primitive frontier.

## oMLX DeepSeek-V4.1 runtime architecture inventory

This inventory is for the later decode architecture decision. oMLX remains an architecture/implementation source and performance baseline, not a correctness authority.

### Source identity

- Runtime tree: `$HOME/omlx-0.7.0.dev2`
- Git HEAD: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
- Local patch context: DeepSeek V4.1 image-token/parser patch files are modified/untracked relative to the clean head. This affects processor/API behavior, not the model-core architecture facts below.

### Loading and residency

- `omlx/patches/deepseek_v41/loading.py:58` loads the official source checkpoint or a converted format, validates config/index identity, and builds `Model(config)`.
- Source checkpoint loading preserves MTP weights when active unless explicitly disabled.
- Engram can be SSD-offloaded through `DiskEngramEmbedding`; `EngramPrefetch` is installed to preserve GPU submission boundaries even when tables are resident.
- `moe_expert_offload_resident_fraction` can replace routed experts with `OffloadedExpert`, but the known baseline settings used no MoE expert offload.
- Shard loading calls `mx.eval(values)` and final load calls `mx.eval(model.parameters())`, so model residency is established before serving.

### Token execution topology and layer scheduling

- `LanguageModel._forward` in `language.py:833` is chunk-major/full-model: for each request row, it extracts per-layer cache rows, computes embeddings for the current input chunk, initializes HC streams, then iterates layers `0..39` for that chunk.
- Prefill is scheduler-chunked externally by `omlx/scheduler.py`; prior artifacts show 2,048-token chunks by default and negligible non-model scheduler overhead.
- Decode is the same `_forward` topology with `input_ids.shape[1] == 1` for target steps, plus DSpark/MTP wrapping when enabled.
- Layer loop uses an ad-hoc `shared` dictionary to pass `kv`, `index_k`, `idx`, and `candidates` frontiers inside one chunk.

### Graph/evaluation lifetime and synchronization

- oMLX relies primarily on MLX lazy graph construction/evaluation.
- Engram prefetch boundaries call `mx.async_eval(h, pre)` around Engram layers in `language.py:907` and `language.py:918`.
- Scheduler and cache-management paths use `mx.eval`, `mx.async_eval`, and `mx.synchronize` at request/cache/store boundaries; prior prefill traces show scheduler overhead around 0.1%, so the dominant cost is model/cache execution rather than API overhead.
- There is no DwarfStar-style explicit static C command stream for full prefill in oMLX; graph/lifetime ownership is by Python/MLX modules, cache objects, and scheduler chunk boundaries.

### State/cache ownership

- `DeepseekV41Cache` in `cache.py:10` is request-local and stores seven slots: offset, window KV, compressed KV, index K, partial KV, partial gates, and Engram history.
- Cache extract/merge/extend supports batching and boundary snapshots, with metadata versioning for compression ratios.
- In `_forward`, each layer gets an extracted row cache; after the row/chunk completes, layer caches are merged back and advanced.
- Attention updates `cache[1]` window KV, `cache[2]` compressed KV, `cache[3]` index K, `cache[4]/[5]` compressor tails, and `cache[6]` Engram history.
- DSpark verification uses `_mtp_verify_state` to capture temporary window/compressor state for partial rollback.

### Attention decode path

- `Attention.__call__` in `language.py:238` computes Q/KV projections, RoPE, packed local window KV, optional compressor/indexer paths, and sparse attention.
- Decode length 1 normally uses packed sparse attention paths; larger chunks can use `deepseek_v41_packed_attention` custom kernels when geometry matches.
- Source layers publish compressed KV and index data through shared state; consumers use selected local/compressed indices.
- The decode path updates the same request-local cache slots used by prefill rather than a separate static decode graph.

### MoE / routed expert scheduling

- `MoE.__call__` in `language.py:599` gates tokens to selected experts and runs shared/routed experts.
- For larger prefill (`x.shape[1] >= 32` and enough routed rows), oMLX sorts expert rows and calls grouped/sorted expert paths, then combines with shared expert output.
- For short decode/verification, it uses the shorter routed expert path; source comments note short DSpark verification blocks share decode execution geometry.
- Optional expert offload exists, but known good baseline did not use it.

### Engram behavior

- `Engram` and `NgramHash` in `engram.py` maintain token-normalized Engram lookup with request-local lookback.
- `_forward` computes Engram hashes/history once per chunk and stores history in cache slot 6 at layer 0.
- Engram layers submit prefetch/read work in layer order, with async eval boundaries around h/pre.
- SSD-offloaded Engram is supported and is part of the known baseline memory behavior.

### MTP / DSpark speculative decode

- `DSparkMixin` in `mtp.py:32` enables/disables DSpark/MTP, creates MTP cache, appends main hidden context, performs partial rollback, and wraps target verification.
- `dspark.forward_spec` in `dspark.py:140` requires preserved MTP weights, one committed target position after priming, aligned stage cache offsets, and returns draft proposals plus confidence from the DSpark stages.
- `proposal_forward` in `dspark.py:190` builds draft tokens from anchor/noise tokens through MTP stages and projects logits with the main head.
- `_forward` can capture target hidden at configured DSpark target layers; `DSparkMixin.__call__` primes DSpark cache on normal target forward and stashes verify state for rollback.
- This is a major reason oMLX decode cannot be compared only as target-token decode; its speculative architecture must be evaluated as part of the decode decision.

### Long-context/session implications

- Scheduler supports chunked prefill, boundary snapshots, prefix cache/paged cache, cache compaction/reconstruction, and cache-store admission control.
- Previous artifacts show oMLX sustained about 176-194 tok/s prefill and 29-37 tok/s decode at 32K-200K on the target official checkpoint baseline, with about 293 GB peak memory.
- Prefill scaling remains flat at about 190 tok/s for oMLX chunks; DwarfStar-derived prefill is still the intended prefill production direction because DwarfStar's V4.1 sweep has a different large-suffix topology.
- For decode, oMLX is a strong candidate because it already combines official checkpoint execution, request-local state, long-context cache behavior, SSD Engram, and DSpark/MTP speculative structure.

## Native/reference reusable production components

Reusable with explicit architecture fit:

- checkpoint atlas and `WeightCatalog` for official checkpoint identity/loading;
- Engram metadata/store/hash/layer components and SSD-backed behavior;
- official-source-derived validators and fixture contracts;
- `TextBackboneState` state taxonomy and reset/fork/continuation semantics;
- attention state contracts for window KV, compressed KV, index/candidate publications;
- MoE/HC arithmetic and routing validation code;
- sampling/generation arithmetic seams and lifecycle tests;
- runtime bridge/residency utilities where they fit the selected backend boundary.

Reference/qualification only unless redesigned:

- current `TextEncoder`/`TextDecoder` chunk/token execution topology;
- current native decode loop as a performance starting point;
- diagnostic performance harnesses and smoke tests;
- implementation-specific selectors/defaults from the current native path.

Replace or bypass for production architecture:

- prefill topology should be DwarfStar-derived sweep/lifetime topology, not current native reference topology;
- decode topology must be selected by DwarfStar-vs-oMLX architecture comparison, not assumed from current native code;
- API prompt/protocol parsing should later use official `deepseek-recipe`, not current scaffolds.

## Missing production seams

Milestone 2 and later work need these seams explicitly:

1. **Prefill backend seam:** DwarfStar-derived sweep ABI must become the production prefill executor while preserving official checkpoint data and correctness gates.
2. **Session transaction seam:** prefill invalid/commit/deferred states must map to `ds41f` reset/fork/continuation/publication contracts.
3. **State handoff seam:** full-model prefill output state must hand off into the selected decode architecture without recomputation or reference-topology leakage.
4. **Weight access seam:** official safetensors/atlas access must feed DwarfStar-derived scheduling without adopting incompatible DwarfStar checkpoint layout/quantization.
5. **Engram seam:** SSD-backed Engram lookup/prefetch must be ordered and observable across prefill/decode.
6. **Expert residency seam:** MoE expert scheduling/residency must be selected coherently with prefill/decode architecture, not piecemeal kernel replacement.
7. **Decode decision seam:** DwarfStar vs oMLX decode architecture comparison must decide graph lifetime, MTP/DSpark/speculation, cache ownership, and transaction semantics before implementation.
8. **Qualification seam:** validators must run against each implementation path separately; native reference qualification does not transfer automatically.
9. **Serving backend seam:** later `deepseek-recipe` integration must call a narrow backend interface independent of prompt/protocol internals.

## Explicit retain / reuse / replace decisions

- Retain `native/*`; reuse checkpoint, state, Engram, MoE/HC, sampling/generation, and validators only through documented seams.
- Do not optimize current native decode as a substitute for decode architecture selection.
- Retain and continue DwarfStar-derived prefill from `ds41f_mlx/dwarfstar_v41_sweep.py`, `ds41f_mlx/native_prefill.py`, and `ds41f_mlx/native/*`.
- Do not restart DwarfStar-derived prefill from scratch or replace it with a new generic layer-major experiment.
- Retain oMLX bridge and source inventory for decode architecture comparison.
- Do not treat oMLX logits/cache behavior as official correctness authority.
- Retain correctness validators as gates for every future production path.
- Replace current API-scaffold direction later with `deepseek-recipe` integration; no API work is part of Milestone 1 or 2.

## Milestone 1 completion determination

Milestone 1 is complete because:

- current component roles are classified with retain/reuse/replace decisions;
- the DwarfStar-derived V4.1 prefill implementation frontier is identified down to planner, native C ownership, Metal submission, official primitive, and missing full-model pieces;
- pinned DwarfStar source was inspected and concrete function/topology evidence is recorded;
- oMLX DeepSeek-V4.1 architecture relevant to decode selection is inventoried with source locations and architecture facts;
- native/reference components are separated into reusable production components versus qualification/reference machinery;
- missing production seams are explicit;
- no runtime behavior, selectors, optimization, API work, or multimodal work changed.

## Exact starting frontier for Milestone 2

Start Milestone 2 from the existing DwarfStar-derived prefill seam:

```text
ds41f_mlx/dwarfstar_v41_sweep.py
  + ds41f_mlx/native_prefill.py
  + ds41f_mlx/native/ds41f_prefill_native.{h,c}
  + artifacts/m2/dwarfstar-prefill/native-sweep-reconciliation.json
  + artifacts/m2/dwarfstar-prefill/native-metal-submission*.json
  + artifacts/m2/dwarfstar-prefill/native-official-embedding.json
  + artifacts/m2/dwarfstar-prefill/native-official-projection.json
```

The first Milestone 2 implementation work should connect additional official model math stages behind the existing sweep allocation/command/submission lifetime, beginning with full official embedding/carry initialization and then layer-local official primitives, while preserving invalid/commit/deferred transaction semantics. Do not create a new planner or switch to the current native reference topology.
