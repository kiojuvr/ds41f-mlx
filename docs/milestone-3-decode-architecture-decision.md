# Milestone 3 decode architecture decision

Status: **COMPLETE — oMLX-derived base decode selected; speculative DSpark/MTP staged after base decode admission.**

Milestone 3 decides the production decode architecture only. It does not implement the Milestone 4 runtime.

## 1. Source revisions inspected

- `ds41f-mlx`: current worktree after Milestone 2, especially `ds41f_mlx/prefill_session.py` and `PrefillContinuationState`.
- DwarfStar: `$HOME/ds4`, git `0aaea5a238fb41a35106a551e73c8409dfb751ac`.
  - Inspected `ds4.c`, `ds4_metal.m`, `ds4_deepseek41_gpu.h`.
  - Key locations: `ds4.c:17177` persistent graph caches; `ds4.c:19049` graph allocation; `ds4.c:24151` decode attention/body region; `ds4.c:25080` index/compressor update region; `ds4.c:41172` `ds41_graph_decode_layer`; `ds4.c:41297` `ds41_graph_step`; `ds4_metal.m:1282-1606` command-buffer creation/wait/finish.
- oMLX: `$HOME/omlx-0.7.0.dev2`, git `b390b31e0c6831225fed0f24d278eb1db7fcb68b`.
  - Known local image-token/parser patch is protocol/parser scope, not model-core decode architecture.
  - Key locations: `omlx/patches/deepseek_v41/language.py:238` `Attention`; `language.py:269` attention decode/cache update; `language.py:785` `LanguageModel`; `language.py:833` `_forward`; `cache.py:10` `DeepseekV41Cache`; `engram.py:1` Engram hash/history; `mtp.py:82` partial rollback; `mtp.py:131` DSpark wrapper; `dspark.py:140` `forward_spec`; `dspark.py:190` proposal execution; `scheduler.py:1789` scheduler.

## 2. DwarfStar architecture inventory

DwarfStar V4.1 decode is a C/Metal graph owned by `ds41_gpu_graph`. `ds41_graph_step` is the single-token target entry: it hashes the next token, reads SSD-backed Engram rows, opens Metal commands, embeds the token, executes layers 0..39, drains command buffers at queue boundaries, computes logits, then commits `history` and `pos` only on success. Per-layer execution is `ds41_graph_layer`; on non-Apple/TP paths `ds41_graph_decode_layer` can capture decode islands. State/cache ownership is inside graph tensors: raw sliding-window KV, attention compressed cache, index compressed cache, compressor frontier KV/score, index frontier KV/score, counters, Engram history, MTP/DSpark scratch, output-head buffers, and class-P reusable layer work buffers.

Concrete responsibilities:

| Responsibility | DwarfStar source evidence |
| --- | --- |
| Single-token target execution | `ds4.c:41297` `ds41_graph_step` |
| State/cache ownership | `ds4.c:17177-17210` graph raw/compressed/index/pending tensors and counters |
| Window KV update | decode attention body around `ds4.c:24151`, raw cache stores through `metal_graph_decode_kv_store` |
| Compressed KV/index/candidate state | compressor/indexer update around `ds4.c:25080`, `layer_n_comp`, `layer_n_index_comp`, `comp_selected` |
| Source/reuse scheduling | V4.1 source helpers `ds41_kv_source`, `ds41_index_source`; per-layer shared compressed/index selections in decode attention |
| Engram continuation | `ds41_graph_step` hashes into `next_history`, reads `g->table`, writes `g->engram_rows`, commits `g->history` after success |
| MoE/expert scheduling | `ds41_moe`, `ds41_moe_batch`, CPU/GPU router paths and streaming expert cache in `ds4_metal.m` |
| Graph/buffer lifetime | `metal_graph_alloc_raw_cap` and `ds41_gpu_graph` long-lived tensors |
| Metal submission/sync | `ds4_gpu_begin_commands`, `ds4_gpu_end_commands`; command creation/wait in `ds4_metal.m:1282-1606` |
| Commit/rollback | Base token commits by `g->history = next_history; g->pos++`; failure invalidates `g->valid`. Speculative scratch has saved frontiers/prefix slots. |
| Speculative/MTP | `ds4.c:17187` speculative scratch; `ds4.c:17300` MTP raw cache; `dspark_*` regions and verify functions around `ds4.c:38388` |
| Batching/concurrency | Single graph session primary; verify-block batching exists for speculative suffixes, not a request-local Python scheduler. |

DwarfStar is architecturally coherent for its own checkpoint/quantization/runtime, but its decode state is not exposed through a no-replay admission interface from neutral arrays.

## 3. oMLX architecture inventory

oMLX DeepSeek-V4.1 decode is request-local MLX lazy execution through `LanguageModel._forward`. Decode is the same chunk/full-model topology as prefill with `input_ids.shape[1] == 1` for target steps. For each request row, `_forward` extracts per-layer `DeepseekV41Cache` rows, computes embeddings, regenerates Engram hashes/history, iterates layers 0..39, merges rows back, advances offsets, and returns logits. `DeepseekV41Cache` has seven slots: offset, window KV, compressed KV, index K, partial KV, partial gates, Engram history.

Concrete responsibilities:

| Responsibility | oMLX source evidence |
| --- | --- |
| Single-token target execution | `LanguageModel._forward`, `language.py:833` |
| State/cache ownership | `DeepseekV41Cache`, `cache.py:10`; extract/merge/extend at `cache.py:50-79` |
| Window KV update | `Attention.__call__`, `language.py:282-288` |
| Compressed KV/index update | compressor/indexer paths, `language.py:303-323`; compressor verify capture at `language.py:143-154` |
| Source/reuse scheduling | per-forward `shared` dict carrying `kv`, `idx`, `candidates` between source and consumer layers |
| Engram continuation | `NgramHash` and `Engram`, `engram.py`; history in cache slot 6 at `language.py:927-928` |
| MoE/expert scheduling | `MoE.__call__`, grouped/sorted path for larger chunks and short-token path for decode/verification |
| Graph/eval/sync | MLX lazy graph; Engram prefetch uses `mx.async_eval`; scheduler/cache boundaries use `mx.eval`, `mx.synchronize` |
| Commit/rollback | Base decode mutates cache after successful `_forward`; DSpark verification snapshots cache and rolls back via `mtp_partial_rollback`, `mtp.py:82` |
| Speculative/MTP | DSpark target capture in `_forward`; `forward_spec` in `dspark.py:140`; `proposal_forward` in `dspark.py:190` |
| Batching/concurrency | Scheduler, request-local caches, cache extract/merge/extend, prefix/paged cache machinery in `scheduler.py` |

## 4. `PrefillContinuationState` mapping

Classification terms: `DIRECT`, `ZERO_OR_LOW_COPY_ADAPTER`, `TRANSFORM_REQUIRED`, `STATIC_MODEL_STATE`, `NOT_REQUIRED_BY_THIS_BACKEND`, `UNSUPPORTED`.

| M2 field | DwarfStar | oMLX |
| --- | --- | --- |
| `token_ids / token_frontier` | `TRANSFORM_REQUIRED`: graph `pos` plus Engram history; no import ABI | `ZERO_OR_LOW_COPY_ADAPTER`: set cache slot 0 offsets and request token history metadata |
| `ngram_hashes` | `TRANSFORM_REQUIRED`: DwarfStar stores `ds4_engram_history`, not exported hash arrays | `ZERO_OR_LOW_COPY_ADAPTER`: oMLX normally stores lookback history in slot 6; full hashes are validation/input to history seeding |
| Engram store identity/history | `STATIC_MODEL_STATE` + `TRANSFORM_REQUIRED`: SSD tables static, history internal | `STATIC_MODEL_STATE` + `ZERO_OR_LOW_COPY_ADAPTER`: `DiskEngramEmbedding` static, slot 6 history/request hasher state |
| `window_kv_by_layer[0..39]` | `TRANSFORM_REQUIRED`: must upload into `layer_raw_cache[*]` ring layout and counters | `ZERO_OR_LOW_COPY_ADAPTER`: maps to cache slot 1 per layer after pack/layout validation |
| `compressed_kv_by_source[2,8,14,20]` | `TRANSFORM_REQUIRED`: must upload to `layer_attn_comp_cache[source]` plus `layer_n_comp` | `ZERO_OR_LOW_COPY_ADAPTER`: maps to cache slot 2 on source layers |
| `index_k_by_source[2,8,14,20]` | `TRANSFORM_REQUIRED`: must upload to `layer_index_comp_cache[source]` plus counters | `ZERO_OR_LOW_COPY_ADAPTER`: maps to cache slot 3 on source layers |
| `candidates_by_source[20]` | `TRANSFORM_REQUIRED`: DwarfStar selected/candidate buffers are graph work/publication state | `NOT_REQUIRED_BY_THIS_BACKEND`: oMLX recomputes candidate blocks transiently in the same target forward from source@20 index state, no prompt replay |
| `topk_by_generation[2,8,14,20,24,28,32,36]` | `TRANSFORM_REQUIRED`: selected top-k graph tensors/counters need import | `NOT_REQUIRED_BY_THIS_BACKEND`: per-token top-k is transient from current index K; stored M2 values are provenance, not long-lived oMLX cache |
| `compressor_pending` | `TRANSFORM_REQUIRED`: graph frontier tensors `layer_attn_state_*`, `layer_index_state_*` | `ZERO_OR_LOW_COPY_ADAPTER`: maps to slots 4 and 5 on source layers |
| `shared_publications` | `TRANSFORM_REQUIRED`: DwarfStar graph publication tensors not importable | `NOT_REQUIRED_BY_THIS_BACKEND`: oMLX creates `shared` per `_forward`; durable inputs are cache slots 1-6 |
| `field_ownership` | `NOT_REQUIRED_BY_THIS_BACKEND`: unless building import verifier; graph source ids fixed in model | `NOT_REQUIRED_BY_THIS_BACKEND`: used by adapter validation, not runtime cache |
| `source_generation_order` | `NOT_REQUIRED_BY_THIS_BACKEND`: layer order fixed | `NOT_REQUIRED_BY_THIS_BACKEND`: layer order fixed in `_forward` |

No M2 state is allowed to disappear: fields marked not required are either static model facts, adapter validation facts, or per-forward transient values derived from durable admitted state without prompt replay.

## 5. Token/layer execution comparison

DwarfStar base decode is token-major: one token enters `ds41_graph_step`, then all 40 layers execute in C/Metal with explicit command batches. Layer work reuses named graph buffers and persistent graph tensors. Speculative verification can use small verify batches and prefix snapshots.

oMLX base decode is also one target token per call in normal generation, but execution is a lazy MLX full-model row/chunk pass: `_forward` extracts row caches, executes layers 0..39 in Python/MLX, then merges caches. Speculative DSpark changes the practical topology by adding proposal execution and target verification blocks, but the target verifier still reuses `_forward` with verification state capture.

## 6. Graph/lifetime comparison

DwarfStar has explicit C-owned graph and Metal buffer lifetime. It offers predictable buffer residency and command boundaries but requires an equally explicit state-import implementation before it can consume M2 state. Mutation boundary is manual: failures invalidate the graph or restore speculative frontiers.

oMLX has Python/MLX module lifetime, request-local cache objects, and MLX lazy graph evaluation. The cache object is the session authority, which aligns with M2 neutral arrays. Rollback is structurally available by cache snapshot/restoration in DSpark. Long-lived cache behavior is already present in scheduler/cache code.

## 7. Attention and persistent state

DwarfStar stores raw SWA rings, compressed attention cache, index cache, and pending compressor state as graph tensors with separate counters. It is efficient but opaque to `PrefillContinuationState` without a new native import ABI.

oMLX stores the same semantic categories directly in `DeepseekV41Cache`: slot 1 window KV; slot 2 compressed KV; slot 3 index K; slots 4/5 pending compressor KV/gates; slot 6 Engram history. Candidate and top-k selections are transient shared values regenerated for the current token from the durable source/index state.

## 8. Engram comparison

Both architectures keep Engram SSD-backed. DwarfStar hashes tokens in C, reads tables before the layer needing them, and commits `history` only after token success. oMLX uses tokenizer-normalized `NgramHash`, stores request-local lookback in cache slot 6, and can prefetch SSD/resident Engram tables with `mx.async_eval` overlap. oMLX can continue from M2 by seeding history/store identity; DwarfStar requires conversion into private `ds4_engram_history`.

## 9. MoE/expert comparison

DwarfStar has the most explicit expert residency machinery: resident/streaming caches, hotlists, selected-load/readahead, and Metal/CUDA selected expert paths. That is strong architecture evidence but is tied to the DwarfStar graph/checkpoint runtime.

oMLX uses grouped/sorted expert execution for larger chunks and short-token routed execution for decode/verification. The documented target baseline used no expert offload and stayed around 293 GB peak memory on M3 Ultra 512 GB, making resident official-checkpoint decode practical.

## 10. Speculative decode comparison

DwarfStar contains MTP/DSpark-like scratch, raw MTP cache, DSpark target hidden, verify suffix, prefix snapshots, and rollback frontiers. It is not selected for base decode because M2 admission is the hard gate.

oMLX DSpark/MTP is a coherent second-stage architecture: target `_forward` captures target hidden, DSpark stages propose draft blocks, target verification snapshots cache state, `mtp_partial_rollback` commits accepted prefixes and restores rejected suffix state. It requires preserved MTP weights and additional MTP stage cache. Milestone 4 should implement base target decode first, then enable DSpark after base continuation/correctness gates pass.

## 11. Long-context/memory comparison

Known official-checkpoint oMLX evidence on the target machine: 32K-200K prefill about 176-194 tok/s, decode about 29-37 tok/s, peak memory about 292.8-293.0 GB. This is implementation-scoped evidence, not correctness authority, but it proves practical official-checkpoint long-context decode on 512 GB.

DwarfStar evidence is not equivalent: available DwarfStar speed data is primarily DwarfStar checkpoint/quantization/hardware scoped, while ds41f DwarfStar-derived code has proven prefill topology/state handoff but not production official-checkpoint decode. DwarfStar's explicit streaming expert/cache architecture may be memory-efficient, but no no-replay M2 admission path exists today.

## 12. Concurrency comparison

DwarfStar is strongest as a single long-lived graph/session with explicit buffers; batching exists for prefill and speculative verify blocks but not as a general request-local scheduler comparable to oMLX.

oMLX already has request-local `DeepseekV41Cache`, cache `extract/merge/extend`, scheduler batching, prefix/paged cache opportunities, and request compaction. Production priority is single-session agent use, but oMLX does not foreclose later concurrency.

## 13. Correctness/qualification integration

For oMLX, ds41f gates attach at the adapter/session seam: validate first-token logits after M2 prefill, continuation offsets, reset/fork by cache copy/clear, compressed/index/candidate state by adapter inventory, Engram history by slot 6 and store identity, final logits by existing official validators, and transaction failure by not publishing mutated cache until a decode step succeeds.

For DwarfStar, equivalent gates would require a native graph import/export seam before tests could observe state without prompt replay. That is a large structural implementation before even reaching first-token decode.

## 14. State-admission probe results

Added comparison tool: `tools/probe_m3_state_admission.py`.

Artifact: `artifacts/m3/state-admission-probe.json`.

Result summary:

- oMLX: `ADMITTED_SYNTHETIC_NO_PROMPT_REPLAY`; constructed 40 layer cache objects/shape-equivalent fallback from a synthetic `PrefillContinuationState`, set all offsets to frontier 7, mapped source compressed/index/pending slots, and seeded Engram history shape. Local ds41f venv lacks `mlx_lm`, so the probe used a shape-only fallback rather than executing MLX kernels.
- DwarfStar: `NO_PUBLIC_STATE_ADMISSION_ABI`; source contains the expected private graph state terms but no public import/admission ABI for neutral continuation arrays.

This probe is not production decode implementation.

## 15. Composition analysis

No composition is selected. A coherent boundary would have to let one runtime own token execution, graph lifetime, cache mutation, Engram history, compressed/index state, rollback, and sampling commit. Splitting DwarfStar attention/expert buffers with oMLX request caches would require per-token cache conversion, duplicate session authorities, conflicting MLX vs C/Metal synchronization, and difficult rollback semantics. Using DwarfStar prefill plus oMLX decode is not a hybrid decode runtime; it is the already-required prefill-to-decode adapter boundary.

## 16. Selected architecture

**Selected base practical decode architecture: oMLX DeepSeek-V4.1 target decode architecture, adapted to consume `PrefillContinuationState` directly without prompt recomputation.**

The selected runtime is oMLX-style request-local `DeepseekV41Cache` ownership plus `LanguageModel._forward` target execution. DSpark/MTP speculative acceleration is selected as an optional/staged second step after base decode admission and correctness qualification.

## 17. Rejected alternatives

- **DwarfStar decode as Milestone 4 base:** rejected because it lacks a state-admission ABI from M2 neutral arrays, is tied to private C/Metal graph tensors and DwarfStar checkpoint/runtime assumptions, and would require a major import/export/runtime build before proving first-token continuation. Its explicit graph/expert design remains valuable evidence for future optimization.
- **Current native reference decode:** not a candidate; it remains correctness/reference evidence.
- **Composition:** rejected because no clean per-token state/lifetime boundary avoids duplicate execution, conversion, synchronization, and rollback conflicts.

## 18. Exact Milestone 4 implementation boundary

Milestone 4 starts with implementation, not another topology decision:

1. Create a production decode session type with oMLX-style cache ownership, e.g. `OmlxDecodeSession` under `ds41f_mlx/runtime/`.
2. Implement an adapter from `PrefillContinuationState` to 40 `DeepseekV41Cache` instances: slot 0 offsets, slot 1 window KV, slots 2/3 source compressed/index K, slots 4/5 pending compressor state, slot 6 Engram history, plus adapter validation for candidates/top-k/ownership/source order.
3. The model/cache owner is the oMLX `LanguageModel` plus request-local cache list; M2 prefill remains DwarfStar-derived and unchanged.
4. First-token execution entry point is `LanguageModel._forward`/`__call__` with one token and admitted cache, not prompt replay.
5. Runtime-facing seams: cache admission, reset, fork, continuation commit, failure rollback, logits return, Engram store identity, qualification hooks.
6. Base target decode path comes first. DSpark/MTP remains staged until base decode passes first-token, continuation, reset/fork, Engram, compressed/index, and logits gates.
7. Reuse: `PrefillContinuationState`, oMLX model/cache architecture, existing ds41f validators, artifact/probe tooling. Reference-only: current native decode loop and any prompt-recompute bridge.

Milestone 3 is complete when this document and plan updates are committed.

## 19. M4 numerical-trajectory re-evaluation note

M4 Block1 same-input evidence (`artifacts/m4/block1-same-input/result.json`) does not rewrite the historical Milestone 3 rejection of DwarfStar decode. That rejection was based on architecture/state-admission constraints: real DwarfStar decode state is private to `ds41_gpu_graph` C/Metal tensors, and no public no-replay state-admission ABI exists for the M2 neutral handoff.

New M4 evidence introduces a separate concern: oMLX Block1 reproduces its own production same-input endpoint, but the connected token15 trajectory relative to the official-source-derived path is classified as `CONNECTED_TRAJECTORY_COLLAPSE_CANDIDATE`. Therefore DwarfStar decode may be re-evaluated as a diagnostic/architecture comparison before further Layer2-downstream oMLX qualification. Such a comparison must initially answer whether DwarfStar stays materially closer from the same semantic prefix/state; it is not an automatic replacement production runtime and does not by itself solve the no-replay admission ABI constraint.

The first re-evaluation audit verified the pinned DwarfStar revision but blocked direct numerical comparison: the pinned DwarfStar V4.1 path is GGUF/projection-quantization based, while the official Flash Q path uses safetensors FP8 weights/scales. This preserves the Milestone 3 decision rationale and narrows any future DwarfStar diagnostic to a precision-preserving official-FP8 Q-path seam, not a production selector change.
