# P8 E2E dataflow and representation lifecycle audit

Status: **P8 optimization search complete**. P0-P7 are frozen and qualified. This document records the E2E representation audit and the implemented A/B candidate result. It deliberately does not add production `mx.eval`, synchronization, component barriers, kernel probes, or production semantic changes.

## 1. Frozen P8 attribution conclusion

The ~4.42 s P6 `persistent_source_eval` is primarily a **materialization point for upstream lazy computation** that produced persistent source/cache state. It is **not** evidence that P6 detach bookkeeping or row-detach copies themselves cost ~4.42 s.

Further `mx.eval` insertion is rejected as the primary P8 attribution method because it changes laziness, fusion and overlap. The five-cut upstream materialization staircase is preserved as raw evidence, but is classified as:

- **coverage validation: useful** — it prepaid the final persistent graph and reduced the final residual eval to near-zero.
- **performance attribution: too perturbing** — its cut sum inflated materialization by ~1.85x, so the split is not a reliable cost attribution.

P8 therefore changes question from “which component is slower: Attention, MoE, or HC?” to: **can the same correct model computation be expressed with fewer E2E representation boundaries?** Kernel/component optimization remains deferred until structural opportunities are exhausted or proven non-viable.

## 2. Authority inspected

Production ds41f path inspected:

- `ds41f_mlx/prefill_fp8_mlx/arena.py`
- `ds41f_mlx/prefill_fp8_mlx/block_runner.py`
- `ds41f_mlx/prefill_fp8_mlx/publications.py`
- `ds41f_mlx/prefill_fp8_mlx/p6_append.py`
- `ds41f_mlx/prefill_fp8_mlx/omlx_suffix_math.py`
- `ds41f_mlx/prefill_fp8_mlx/p8_optimizer.py`

Pinned DwarfStar authority inspected:

- `antirez/ds4` commit `0aaea5a238fb41a35106a551e73c8409dfb751ac`
- `$HOME/ds4/ds4.c`, especially `ds41_graph_prefill_sweep`, `ds41_carry_copy`, and carry allocation/capacity logic.

DwarfStar requires command order, offsets, row counts, publication points, invalid/commit transitions, and HC swap semantics. It does not prove that ds41f must physically represent layer-to-layer HC as one dense MLX `[1,N,4,5120]` parent at every internal layer boundary. DwarfStar has a compact carry abstraction plus copy-in/copy-out around command tiles; that is a logical carry/span contract, not necessarily a Python/MLX dense-array mandate.

## 3. Current E2E tensor/dataflow map: production P7-qualified 16384 source path

| object | producer | representation / shape / dtype | logical owner | physical storage | consumers | lifetime | persisted | copied | sliced | concat | packed/quantized | unpacked | written into arena | immediately re-read | boundary class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `input_ids` | `_make_segment_tensors` | MLX int64 `[1,count]` | RequestArena | MLX array | embedding, Engram hasher/history | segment | no | no | token segment selected before arena | no | no | no | no | no | SEMANTIC_REQUIRED token input |
| Engram hashes | `_make_segment_tensors` / oMLX hasher | hash tensor indexed `[batch,row,engram_layer]` | RequestArena `engram.hashes` | MLX/host-derived tensor | `_slice_engram_hashes` for layers 1,14 | segment until P6 detach/retire | history yes; hashes transient | no hot copy | yes per Engram layer command and microtile | no | no | no | no | yes by Engram calls | BACKEND_REQUIRED for SSD/hash contract |
| Engram history | oMLX `NgramHash` | cache slot6-compatible history | DeepseekV41Cache layer0 / arena | MLX tensor | P5 handoff/decode | request | yes | no | no | no | no | no | cache slot | no | SEMANTIC_REQUIRED |
| `carry.current` | embedding for layer0, then `swap()` from prior `next` | dense MLX `[1,count,4,5120]`, BF16-ish HC | RequestArena carry | MLX parent | `_slice_rows` -> `h_chunk`, decoder prepare, P6 cone copy | segment, narrowed after P6 | no full parent after detach | P6 cone copy | yes per command | no | Block internals may pack caches | no | no | yes next layer tile | IMPLEMENTATION_ONLY candidate |
| `carry.next` | zero/init then `_write_rows(h_out)` | dense MLX `[1,count,4,5120]` | RequestArena carry | MLX slice-update descriptor chain | swap to current, P6 next-cone copy | segment | bounded cone after P6 | P6 cone copy | P6 next cone | no | no | no | yes | yes after swap | IMPLEMENTATION_ONLY candidate |
| `carry.pre` | initial pre mask then `_write_rows(pre_out)` | dense MLX `[1,count,4]`, float/pre state | RequestArena carry | MLX slice-update descriptor chain | `_slice_rows` -> `pre_chunk`, decoder prepare, layer19 final, P6 cone copy | segment/cone | bounded cone after P6 | P6 cone copy | yes per command | no | no | no | yes | yes next layer | IMPLEMENTATION_ONLY candidate |
| `h_chunk` | `_slice_rows(carry.current)` | view `[1,rows,4,5120]` | runner local | MLX view/graph node | Engram, Block | command | no | no | yes | no | no | no | no | consumed by Block | IMPLEMENTATION_ONLY candidate |
| `pre_chunk` | `_slice_rows(carry.pre)` | view `[1,rows,4]` | runner local | MLX view/graph node | Engram scheduling, Block | command | no | no | yes | no | no | no | no | consumed by Block | IMPLEMENTATION_ONLY candidate |
| Engram 2048 outputs | `SchedulingCoordinator.apply_engram_micro_pipeline` / `layer.engram` | microtile `h` chunks `[1,<=2048,4,5120]` | runner local | MLX tensors backed by SSD donor rows + compute graph | reassembly | command | no | no | micro hashes sliced | yes into command tile | no | SSD rows dequantized by donor | no | yes by concat consumer | BACKEND_REQUIRED for SSD donor, implementation-only for reassembly |
| Engram 8192 reassembly | `_concat_sequence` in scheduler | `mx.concatenate(..., axis=1)` to `[1,8192,4,5120]` | runner local | lazy concat graph | Block | command | no | no | no | yes | no | no | no | yes Block | IMPLEMENTATION_ONLY candidate |
| Block `h_out` | pinned oMLX `Block.__call__` or suffix math | `[1,rows,4,5120]` | runner local until write | MLX graph | `_write_rows(carry.next)` | command -> next layer via arena | no | no | no | no | cache ops pack internal KV/index | no | yes | yes after swap/slice | IMPLEMENTATION_ONLY candidate |
| Block `pre_out` | pinned oMLX `Block.__call__` | `[1,rows,4]` | runner local until write | MLX graph | `_write_rows(carry.pre)` | command -> next layer | no | no | no | no | no | no | yes | yes after next layer slice | IMPLEMENTATION_ONLY candidate |
| window KV | Attention/cache update | packed activation, shape family `[1,window,head_dim]` | DeepseekV41Cache slot1 | cache object | P6 persistent eval, decode | request | yes | no | maybe internal | maybe cache append | yes `pack_activation` | no | cache slot | yes | SEMANTIC_REQUIRED |
| compressed KV | source layers 2,8,14,20 | packed FP4/E4M3-ish source KV | cache slot2 and publication shared `kv` | same tensor refs | consumers with compress ratio, P6 eval, decode | request | yes | no | maybe append/concat | source append | yes | no | cache/publication | yes | SEMANTIC_REQUIRED |
| index K | source/index layers | packed index key | cache slot3 and publication shared `index_k` | same tensor refs | index scores/topk, P6 eval, decode | request | yes | no | maybe append/concat | source append | yes | no | cache/publication | yes | SEMANTIC_REQUIRED |
| `idx` | indexer/topk | row-span tensor per command span | PublicationManager span | MLX tensor refs | consumer span lookup | until superseded/commit | no as full request state | no | maybe in `_row_value_for_span` | maybe if spanning multiple gens | no | no | no | yes | LIFETIME_REQUIRED visibility; concat is implementation-only if spanning |
| candidates | layer20 candidate source | row-span tensor | PublicationManager span | MLX tensor refs | later layers | transaction | no | no | maybe | maybe | no | no | no | yes | LIFETIME_REQUIRED visibility |
| compressor pending | source Attention | cache slots4/5 | DeepseekV41Cache | MLX tensors | P6 eval, next source grouping, decode | request | yes when present | no | no | maybe internal | maybe packed depending slot | no | cache slot | yes | SEMANTIC_REQUIRED |
| indexer pending | source/indexer | cache slot5 | DeepseekV41Cache | MLX tensor | next source/index, P6 eval | request | yes when present | no | no | maybe internal | maybe packed | no | cache slot | yes | SEMANTIC_REQUIRED |
| PublicationManager shared values | `capture_layer_outputs` | `SourceGeneration(value=<same tensor ref>)` | PublicationManager + arena metadata | Python refs to MLX tensors | `shared_for_span` | transaction/request | committed metadata yes | no by manager | only row-span partial consumption | only multi-generation row-span lookup | no | no | no | yes | OWNERSHIP/LIFETIME_REQUIRED metadata; physical copy already optimized mostly |
| layer19 encoder-final h/pre | `arena.apply_command(SWAP_HC_AFTER_LAYER)` before swap | full dense `carry.next.value` and `carry.pre.value` | arena | MLX dense parent | layer20 full-source publication | until P6 detach/retire | no | no | no | no | no | no | no | yes by layer20 source bridge | LIFETIME_REQUIRED if layer20 requires full source; assembly location candidate |
| layer20 full-source publication | `suffix_math.publish_full_source` | full-source h/pre inputs, outputs compressed/index publications | cache/publication | MLX packed cache tensors | decoder suffix and decode | request | yes outputs | maybe internal source append | no at ds41f seam | cache concat append | yes pack_activation | no | cache/publication | yes | SEMANTIC_REQUIRED output; full h/pre representation UNKNOWN |
| P6 retained decoder cone | `_owned_row_copy` in P6 detach | compact h <=2541, next <=2414, pre <=2541 | RequestArena/P6 owner | new compact MLX arrays via `mx.array()` branch | deferred decoder/P5 proof | until seal/handoff | yes bounded | yes | yes from full parent | no | no | no | arena rebound to compact | yes | OWNERSHIP_REQUIRED |
| DeepseekV41Cache | language_model.make_cache / runner | 40 cache objects with slots0..6 | request/P6 owner then P5 | Python objects holding MLX arrays | handoff, decode | request/decode | yes | no manager-level copy | internal ops | internal append | packed slots | no | cache slot assignment | yes | SEMANTIC_REQUIRED |

## 4. Actual current encoder tile lifecycle

Confirmed from production code, not simplified:

```text
arena.carry.current.value
  -> _slice_rows(... offset, rows) -> h_chunk

arena.carry.pre.value
  -> _slice_rows(... offset, rows) -> pre_chunk

if layer has Engram:
  arena.engram.hashes.value -> _slice_engram_hashes(...)
  P7 Engram microtiles -> mx.concatenate(axis=1) -> h_chunk

layer(h_chunk, pre_chunk, cache, shared, absolute_start, image_mask)
  -> h_out, pre_out

h_out  -> _write_rows(arena.carry.next.value, offset, rows, h_out)
pre_out -> _write_rows(arena.carry.pre.value,  offset, rows, pre_out)

SWAP_HC_AFTER_LAYER:
  if layer == 19: save encoder_final_h = carry.next.value; encoder_final_pre = carry.pre.value
  CarryState.swap(): current and next TensorSlot references swap

next layer:
  _slice_rows(arena.carry.current.value, same logical span) -> h_chunk
```

The physical sequence is a loop-internal write/read cycle: block output is written into a full-range parent, slot references swap, and the same tile is sliced back out for the next layer. `carry.pre` does not ping-pong but is still full-range slice-updated then sliced on the next layer.

## 5. Static representation-boundary frequency counts (no evals)

Method: command geometry from `P6AppendPlanner` and production code paths. Counts are operation-topology counts, not execution timing. “Carry slices” includes `h/pre` slices for every `ENCODE_ROWS` plus local-window decoder prepare slices; it excludes P6 owned cone slices which are listed separately. “Carry writes” is two `_write_rows` calls per `ENCODE_ROWS` (`h_out`, `pre_out`).

| request | P6 segments | ENCODE_ROWS source | ENCODE_ROWS suffix/full tail | carry slices | carry writes | carry swaps | Engram microtile slices | Engram concats | P6 detach commands | P6 owned cone copies | publication captures | publication shared-span lookups | cache slot writes/frontier advances | compressed/index publication ops | temp→persistent transitions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | one FINAL_ENCODER_DECODER(8192) | 20 | 20 suffix | 118 | 80 | 40 | 8 | 2 | 1 | 3 | 42 | 61 | >=40 frontier + cache mutations | 2 layer20 full-source captures + source captures | 1 P6 persistent materialization |
| 16384 | one FINAL_ENCODER_DECODER(16384) | 40 (two 8192 tiles/layer) | 20 suffix | 158 | 120 | 40 | 16 | 4 | 1 | 3 | 42 | 81 | >=60 frontier + cache mutations | 2 layer20 full-source captures + source captures | 1 P6 persistent materialization |
| A-24577 | ENCODER_SOURCE_ONLY(16384), FINAL_ENCODER_DECODER(8192), ORDINARY_COMPLETE_RANGE(1) | 60 | 20 suffix + 40 ordinary 1-row full-layer | 278 | 240 | 100 | 26 | 6 | 2 | 3 in final segment | 104 | 122 | >=120 frontier + cache mutations | source-only + final source captures | 2 source materializations plus final seal |

Other counted hot-path topology:

- `pack_activation` / quantization: window KV on cache fill/update, source compressed KV, index K, empty slot construction, and suffix source math. No duplicate identical-input repack was proven at ds41f boundaries; packing is driven by distinct cache/source tensors or empty-slot initialization.
- Publication capture: `capture_layer_outputs` stores tensor references in `SourceGeneration`; it does not copy tensor contents.
- Publication row-span lookup: exact span returns the same tensor; partial/multi-generation span may slice and concatenate.

## 6. Semantic vs implementation boundary classification

| boundary / operation | current purpose | semantic necessity | frequency driver | representation size | copy/view/transform | can eliminate? | candidate architecture | risk |
|---|---|---|---|---|---|---|---|---|
| `h_out -> carry.next` write | transport next-layer tile through arena | IMPLEMENTATION_ONLY candidate | every ENCODE_ROWS | tile into full range | MLX slice-update descriptor | yes if tile object preserves logical carry | Tile-native carry | graph ancestry/slot ownership |
| `carry.next -> swap -> carry.current` | DwarfStar HC layer transition | SEMANTIC_REQUIRED logical; physical dense swap implementation-only | every layer | slot refs | metadata swap | keep semantic, alter physical | TileCarryState vector swap | layer19 capture correctness |
| `carry.current -> h_chunk slice` | tile input to Block | IMPLEMENTATION_ONLY if tile already available | every ENCODE_ROWS | tile view | view graph node | yes | TileSpan direct input | absolute offsets must remain |
| `pre_out -> carry.pre` write | transport pre state | IMPLEMENTATION_ONLY candidate | every ENCODE_ROWS | tile into full range | slice-update descriptor | yes | independent pre tiles | pre lifetime/suffix cone |
| `carry.pre -> pre_chunk slice` | Block pre input | IMPLEMENTATION_ONLY candidate | every ENCODE_ROWS + prepare | tile view | view | yes | TileSpan direct pre | HC semantics |
| Engram microtile -> concat -> Block | donor 2048 limit adaptation | BACKEND_REQUIRED microtiles; concat implementation-only | Engram command >2048 | 2048 parts to 8192 | lazy concat | maybe | Engram writes tile outputs or Block accepts composite tile | donor overlap risk |
| source h/pre -> layer20 publish_full_source | source publication | SEMANTIC_REQUIRED output; full contiguous input UNKNOWN | layer20 source boundary | full count | full h/pre consumed by donor API | maybe concat once only | FullSourceAssemblyBoundary | hardest correctness surface |
| PublicationManager capture | frontier visibility | LIFETIME_REQUIRED/OWNERSHIP_REQUIRED | source layers/frontiers | refs | metadata only | no need | already zero-copy | visibility rules |
| Publication shared span exact lookup | consumer access | LIFETIME_REQUIRED | consumers | ref | metadata only | no need | already zero-copy | none |
| Publication shared span partial concat | row-span coverage | IMPLEMENTATION_ONLY when avoidable | only split spans | span fragments | slice/concat | maybe | TileSpan publication bridge | row coverage proofs |
| P6 persistent-source eval | detach/cache validity | LIFETIME_REQUIRED | P6 source completion | persistent cache tensors | mandatory materialization | no | keep | correctness |
| P6 cone extraction from full carry | retain bounded decoder cone | OWNERSHIP_REQUIRED copy; full-parent source maybe implementation-only | final P6 segment | <=2541 rows | slice + copy | partially | FinalConeExtractor from tiles | ownership proof |
| SSD Engram host/storage | model-static Engram | BACKEND_REQUIRED | Engram layers | selected rows | SSD read/dequant | no | keep | foreground fallback |
| cache slot assignment | persistent state | SEMANTIC_REQUIRED | every layer/source | packed tensors | object mutation | no | keep | P5/P6 cache authority |

## 7. Dense full-range carry consumers

| consumer | currently requires contiguous `[1,N,...]`? | classification | notes |
|---|---|---|---|
| Engram hash consumers | no for h/pre; hashes sliced by span | TILE_NATIVE | require span hash selection only |
| oMLX Block calls | no, accepts `h`, `pre`, `cache`, `shared`, `start`, `image_mask` for supplied rows | TILE_NATIVE | absolute_start preserves positions |
| source-layer compressor/indexer inside Block | no for command tile; uses current h/pre rows and shared/cache | TILE_NATIVE for layers <20 | packed cache outputs persistent |
| Publication logic | no for cumulative exact refs; row-span partial may concat | TILE_NATIVE / CAN_CONCAT_ONCE_AT_BOUNDARY | metadata visibility independent of dense carry |
| layer19 `encoder_final_h/pre` | current code stores full dense parent | CAN_CONCAT_ONCE_AT_BOUNDARY | only needed for layer20 full-source bridge |
| layer20 `publish_full_source` | donor API currently takes `h_full`, `pre_full` | REQUIRES_CONTIGUOUS_FULL_RANGE currently / UNKNOWN semantically | likely narrow assembly point if tile-native |
| P6 final decoder cone | no semantic need for all rows | TILE_NATIVE | can copy rows directly from final tile(s) if available |
| decoder suffix prepare | no, needs local 127-row windows and bounded q rows | TILE_NATIVE | direct last-tile extraction possible |
| DeepseekV41Cache | persistent packed cache tensors yes | REQUIRES_CONTIGUOUS per cache slot semantics, not carry | keep |

## 8. Layer19 -> layer20 audit

Current P6 source execution saves `arena.encoder_final_h = carry.next.value` and `arena.encoder_final_pre = carry.pre.value` at layer19 swap, then `suffix_math.publish_full_source(layer20, h_full, pre_full, ...)` publishes layer20 compressed KV and index K for the whole source. The donor API is full-source-shaped today.

Conclusion: zero full-range assembly is not established. The viable structural design is: keep tile-native h/pre through layers 0..19, then concatenate exactly once at the layer19->20 full-source boundary if `publish_full_source` cannot be taught to consume two 8192 tiles behind the same math. This moves assembly to the narrowest currently required boundary instead of reconstructing dense carry at every layer.

## 9. Source-layer representation transitions

Source layers 2, 8, 14, 20 produce/refresh raw attention/source state, compressed KV, index K, `idx`, and candidates. The current ds41f seam generally lets oMLX produce packed cache/publication representations directly. `PublicationManager` stores references, not content copies. `omlx_suffix_math.publish_full_source` fallback uses `pack_activation(rope(...))` for compressed KV and index K and may concatenate with previous cache values. These are semantically distinct source-cache transforms, not removable copies by inspection.

No sequence was found where ds41f unpacks a packed source tensor only to repack the same values later. Fusion opportunities, if any, are inside donor source math APIs and are deferred because donor math must not be modified in this task.

## 10. Packing/quantization boundaries

Hot-path inventory:

- window KV slot1 packing by Attention/cache path;
- compressed KV slot2 packing for source layers and layer20 full-source publication;
- index K slot3 packing for index source layers;
- `quantize_activation` for query/index scoring paths;
- empty packed slot construction in `_empty_cache_slot`/`_fill_empty_slots`.

Finding: no proven duplicate packing of identical logical input at ds41f orchestration boundaries. Packing outputs are reused through cache slots/publication references. Empty-slot packing is repeated by slot/layer initialization but cheap relative to full model tensors and semantically separate per cache object. Candidate `PACKING_REUSE` is therefore deferred.

## 11. PublicationManager zero-copy/copy audit

`capture_layer_outputs` creates `SourceGeneration(value=shared_after.get(key), ...)`; this is a Python reference to the same tensor object. `publish_frontier` moves metadata into visible dictionaries and `arena.publications.shared`. Exact `shared_for_span` returns the same cumulative or row-span tensor. Only row-span coverage across multiple generations performs slicing/concatenation in `_row_value_for_span`.

Conclusion: publication is already zero-copy for the common cumulative/exact-span path. It should not be the selected structural package. Preserve logical visibility and transaction authority.

## 12. Loop-invariant work

| item | classification | finding |
|---|---|---|
| publication topology | already precomputed | `PublicationTopology` built once; lookups are metadata only |
| `_required_publication_keys(layer)` | cheap metadata only | dictionary read; not retained as target |
| source-layer/Engram membership | cheap metadata only | config/layer checks; not graph construction driver |
| cache geometry/empty-slot shapes | candidate only for empty-slot initialization | not repeated on identical hot tensor values |
| compress ratios | cheap metadata only | donor source math uses config |
| DwarfStar command geometry | already precomputed | `SweepCommand` sequence authoritative |
| P7 Engram tile size/read-ahead | already policy/config | do not change |

No loop-invariant Python lookup was retained as a primary target because none appears capable of eliminating repeated full-tensor graph construction by itself.

## 13. Repeated views/slices and full-range state

Repeated views:

- `carry.current` h tile slice each `ENCODE_ROWS`;
- `carry.pre` tile slice each `ENCODE_ROWS` and decoder prepare;
- Engram hash slice per Engram command/microtile;
- Publication row-span slices only for partial coverage;
- P6/decoder local-window slices.

The concern is not Python view-object cost. The structural concern is parent graph retention: dense parent -> slice -> Block -> slice_update dense parent chains can keep full-range ancestry alive until P6 materialization. Tile-native carry would make the tile output the next layer input directly, reducing slice-update ancestry and dense-parent retention without inserting materialization.

Current full-range arena allocations to challenge:

- `carry.current`, `carry.next`, `carry.pre`: not all consumers need all rows every layer.
- `encoder_final_h/pre`: full range likely needed only at layer19->20 boundary.
- Engram hashes: span-access required; full hash table can remain because it is small and tied to token history.
- Publication shared values: persistent cache tensors are semantic; row-span temporary storage can remain metadata.

## 14. Cost model without barriers

Classes used: O(1) metadata reference, view creation, lazy graph node, full-tile representation construction, full-range representation construction, pack/quantize transformation, SSD host boundary, mandatory persistent materialization.

High-leverage current patterns:

- `frequency 80/120/240 × tile/full-parent slice-update × HC tile size` for carry writes (8192/16384/A).
- `frequency 118/158/278 × view/lazy slice × HC/pre size` for carry slices.
- `frequency 2/4/6 × Engram full-tile concat` for Engram reassembly.
- `frequency 1/1/2 × mandatory persistent materialization` for P6 source completion.
- `frequency 3 final × owned cone copy` for final segment; small measured relative to persistent eval.

This points to a structural package around carry/tile representation, not more attribution barriers.

## 15. Current vs idealized dataflow

Current:

```text
command tile
 -> slice h/pre from dense carry
 -> optional Engram 2048 microtiles
 -> concatenate/reassemble h tile
 -> Block
 -> write h/pre into dense arena parents
 -> swap h current/next slots
 -> slice same logical tile again next layer
 ...
 -> save full layer19 h/pre parent
 -> layer20 full-source bridge
 -> P6 persistent materialization
 -> slice/copy bounded final cone
```

Candidate:

```text
TileSpan(h, pre, absolute_start, rows, logical_offset)
 -> optional Engram transform returns TileSpan h
 -> Block consumes tile tensors unchanged
 -> h_out/pre_out become next-layer TileSpan directly
 -> tile-vector logical swap between current/next
 ...
 -> assemble once only if layer20 full-source bridge requires it
 -> P6 materializes persistent cache as today
 -> FinalConeExtractor copies directly from relevant retained tiles
```

Eliminated boundaries: repeated dense carry writes, repeated dense-parent slices, most full-range parent retention, and full-parent-to-cone extraction. Not eliminated: DwarfStar commands, logical swaps, publication frontiers, cache slots, P6 persistent materialization, SSD Engram boundary.

## 16. Selected architecture-level candidate

Historical candidate decision: **TILE_NATIVE_CARRY**. Final P8 result after implementation/A-B is **TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN**; see section 24. The candidate code is retained experimental/default OFF and is not production-selected.

Reasons:

- removes repeated E2E work rather than one micro-operation;
- bounded correctness surface: adapts ds41f orchestration/storage around unchanged oMLX Block math;
- preserves DwarfStar command offsets/order and PublicationManager/DeepseekV41Cache semantics;
- can be A/B tested behind a single switch;
- publication zero-copy is already mostly optimized; packing reuse and source fusion lack proven duplicate boundary.

## 17. Candidate package design

Conceptual components:

- `TileCarryState`: physical representation replacing dense `current/next/pre` parents with vectors/maps of `TileSpan` by command offset for current layer.
- `TileSpan`: tensor reference plus absolute/logical offset, rows, dtype/shape metadata, logical owner (`current`, `next`, `pre`), and optional parent proof.
- `TileExecutionBridge`: supplies `h_chunk`/`pre_chunk` to unchanged oMLX `Block`, records `h_out`/`pre_out` as next/pre tiles instead of `_write_rows` into dense parents.
- `PublicationViewBridge`: keeps PublicationManager API stable; accepts tile/exact span references and only concatenates row-span fragments when required.
- `FullSourceAssemblyBoundary`: at layer19->20, either calls donor with tile sequence if supported behind ds41f adapter, or concatenates exactly once to full h/pre.
- `FinalConeExtractor`: extracts P6 retained h/next/pre cone from tile spans directly, copying only owned compact rows required by P6.

## 18. Before/after structural operation counts for TILE_NATIVE_CARRY

For 16384 source path:

| operation | current | candidate |
|---|---:|---:|
| carry writes | 120 `_write_rows` | 0 dense writes; 120 tile bindings/metadata |
| carry slices | 158 | 0 for same-tile layer transport; suffix/window slices only if not already held as TileSpan |
| carry swaps | 40 TensorSlot swaps | 40 logical tile-vector swaps |
| Engram reassembly | 4 concats | 4 initially, or 0 if Engram tile bridge later emits command tile without concat |
| full-range layer assembly | implicit dense carry every layer | 0 through layers0..19; 1 possible layer19->20 assembly |
| P6 final-cone copies | 3 owned compact copies from full parents | 3 owned compact copies from tile spans, avoiding full-parent dependency |
| pack/quantize | unchanged | unchanged |
| P6 persistent materialization | 1 | 1 unchanged |

8192 current/candidate: 80 carry writes -> 0; 118 carry slices -> near 0 for layer transport; 2 Engram concats retained initially; 1 possible source assembly. A-24577 current/candidate: 240 carry writes -> 0; 278 carry slices -> near 0 for layer transport; 6 Engram concats retained initially; source assemblies only at true layer20 full-source boundaries.

## 19. Memory/lifetime implications

Current 16384 logical carry parents:

- `carry.current`: `[1,16384,4,5120]`
- `carry.next`: `[1,16384,4,5120]`
- `carry.pre`: `[1,16384,4]`

Tile-native for two 8192 encoder tiles would keep, per layer boundary, tile objects for current h tile0/tile1, next h tile0/tile1 under construction, and pre tile0/tile1. Old layer tiles become unreachable after the logical swap and after all consumers/publication references are released. It avoids constructing new dense slice-update parent ancestry per layer. Logical live bytes may be similar while both current and next tile vectors are live, but full-range dense parents and descriptor chains are avoided. Do not translate this into physical MLX memory predictions until implemented and measured.

Graph-depth effect: current pattern can build `dense parent -> slice_update -> descriptor replacement` repeatedly across layers, then slice from that parent. Candidate builds `tile output -> next layer input` directly, reducing slice-update ancestry, dense-parent retention, and concat ancestry except at explicit source assembly boundaries.

## 20. Correctness equivalence and frozen interfaces

Future implementation must preserve:

- `SweepPlan` and `SweepCommand` sequence, offsets, row counts, phases;
- `SWAP_HC_AFTER_LAYER` logical meaning;
- publication visibility contract and generation metadata;
- `DeepseekV41Cache` slot semantics/shapes/dtypes;
- SSD Engram donor contract and zero foreground fallback;
- P6 checkpoint validity, `P6_SOURCE_COMPLETE_AND_DETACH_CONE`, persistent-cache materialization, C/E/D/T transitions;
- P6 retained-cone geometry;
- P5 same-cache zero-replay handoff;
- same final all40 frontiers, compressed KV, index K, pending state, Engram history, publication generations.

Exact comparison remains required where current backend-local policy already requires exact packed state. Do not require arbitrary hidden-state identity beyond current policy.

## 21. Implementation plan for TILE_NATIVE_CARRY (not implemented here)

1. Add disabled A/B switch, e.g. `DS41F_P8_TILE_NATIVE_CARRY=1`, default off.
2. Introduce `TileSpan` and `TileCarryState` behind `RequestArena` without changing `SweepCommand` or public semantic interfaces.
3. Initial support only for P6/P7 qualified source path; fail closed to dense path for unsupported geometry.
4. On arena initialization, split or represent embedding h/pre as tile spans matching command offsets. If embedding naturally produces full tensor, allow one initial split/view; later layers must remain tile-native.
5. Modify the block runner through a bridge: for an `ENCODE_ROWS` command, look up the exact h/pre `TileSpan` instead of slicing dense parents; pass tensors unchanged to oMLX Block with the same `absolute_start`, cache and shared dict.
6. Store `h_out` as the next-layer tile for the same logical offset. Store `pre_out` independently as the pre tile for that offset. Preserve logical owners even if tensor refs are shared.
7. Implement logical swap of tile vectors at `SWAP_HC_AFTER_LAYER`; no dense array reconstruction.
8. Engram integration: keep P7 2048 donor microtiles and current overlap. Initially reassemble to one tile tensor per command exactly as today; later optional improvement can avoid concat only if Block/tile bridge supports composite tiles.
9. Publication integration: keep `PublicationManager` unchanged for cumulative refs. Row-span exact values can be tile refs; partial spans use existing slice/concat until a `PublicationViewBridge` proves exact coverage.
10. Source layers 2/8/14: unchanged Block/cache/publication math; only h/pre transport changes.
11. Layer19->20: implement `FullSourceAssemblyBoundary`. If donor `publish_full_source` still requires full tensors, concatenate tile0/tile1 once for h and pre, set `encoder_final_h/pre` to those assembled tensors, then call existing suffix math unchanged.
12. P6 detach: keep `_p6_eval_persistent_source_state` unchanged. Replace final cone source from dense parents with `FinalConeExtractor` over current/next/pre tiles; copy compact rows and rebind arena as today.
13. P5 compatibility: exported live `DeepseekV41Cache` and compact cone metadata must match current P6 handoff expectations.
14. Failure/rollback: any missing exact tile, publication mismatch, foreground Engram fallback, cache-frontier mismatch, or P6 ownership failure disables candidate and falls back before promotion; no mixed partial promotion.
15. Tests: add structural fake-tensor tests for tile lookup/swap/pre flow/layer19 assembly/final cone extraction; then run existing full suite and P7/P6/P5 qualification cases.
16. Benchmark only after implementation: 8192, 16384, A-24577 cold/warm E2E prefill tok/s, active/cache/peak memory, Engram bg/fg reads, P5 behavior. No internal eval barriers.

## 22. Rejected/deferred structural packages

- `PUBLICATION_ZERO_COPY_RESTRUCTURE`: rejected as primary because PublicationManager already stores tensor references and exact-span lookups are zero-copy. Keep only minor row-span concat cleanup as future secondary work.
- `SOURCE_REPRESENTATION_FUSION`: deferred. Source compressed/index outputs are semantically packed cache representations; no ds41f-level unpack/repack cycle was proven.
- `PACKING_REUSE`: deferred. No identical-input duplicate pack was found at the orchestration seam.
- `Attention/MoE/HC/kernel work`: explicitly deferred until tile-native carry is implemented and qualified, or proven not viable/no operation elimination.

## 23. Promotion rule and closure

Promotion required P0-P7 correctness preservation, structural operation elimination, repeatable E2E improvement outside noise, no meaningful memory regression, no hidden cold first-request problem, P7 foreground Engram fallback zero, and no new CPU fallback. The implemented tile-native candidate failed the repeatable E2E gain/memory selection requirement, so it was rejected and P8 optimization search is complete.

## 24. TILE_NATIVE_CARRY implementation and A/B result (2026-10-01)

Implementation status: **implemented behind disabled A/B switch** `DS41F_P8_TILE_NATIVE_CARRY=1`. Default remains OFF and the production selector is unchanged.

Implemented pieces:

- `TileSpan`: immutable metadata plus tensor reference (`value`, `logical_offset`, `rows`, `role`, optional absolute-start metadata). Creating a span does not copy tensor content.
- `TileCarryState`: request-local current/next/pre tile maps built from actual `ENCODE_ROWS` command spans. It follows planner offsets/row counts and does not define an independent tiling policy.
- Tile-native source transport in `OfficialFP8MLXBlockRunner`: source/encoder `ENCODE_ROWS` obtains h/pre directly from tile maps, calls unchanged pinned oMLX Block, and binds `h_out`/`pre_out` as next-layer tiles instead of calling `_write_rows`.
- Logical `SWAP_HC_AFTER_LAYER` remains in the command stream and command history; the tile bridge performs the physical tile-vector swap and preserves ping-pong spare semantics.
- Layer19 full-source bridge: ordered source tiles are assembled exactly once for current `publish_full_source(h_full, pre_full, ...)`. Single-tile source can reuse the tile directly.
- P6 final cone extraction: after unchanged `_p6_eval_persistent_source_state()`, the retained h/next/pre cones are extracted from tile spans and then copied through the same owned compact-copy proof path.
- Candidate admission is request/segment-start only. If not admitted before execution, dense remains available. Once tile-native execution begins, invariant failure aborts the request rather than falling back mid-request.

Structural tests: `PYTHONPATH=tests /Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 -m unittest discover -s tests -v` passed **97 tests OK**. Added tests cover default-off switch, admission, exact lookup, two-tile transport, logical swap, missing tile fail-closed, gap/overlap rejection, ordered full-source assembly, single-tile no-op assembly, same-tile/cross-tile cone extraction, and retirement of tile parents.

### 16384 same-revision A/B

Artifacts:

- dense control: `artifacts/p8-tile-native/dense-16384-v2.json`
- tile candidate: `artifacts/p8-tile-native/tile-16384-v2.json`
- P5 dense smoke: `artifacts/p8-tile-native/dense-p5-16384.json`
- P5 tile smoke: `artifacts/p8-tile-native/tile-p5-16384.json`

Both A/B runs used fresh process/model, deterministic 16384 tokens, P7 enabled, P8 verification, cold + 3 warm. No new eval barriers, no `mx.compile`, no custom Metal, no donor oMLX changes.

| path | cold s | warm1 | warm2 | warm3 | warm median | warm range | warm tok/s | bg Engram reads/run | fg fallback | cold write_rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 16.365340 | 16.338032 | 16.349117 | 16.207646 | 16.338032 | 0.141471 | 1002.81 | 32 | 0 | 120 |
| tile-native | 16.353992 | 16.245373 | 16.190254 | 16.248494 | 16.245373 | 0.058240 | 1008.53 | 32 | 0 | 40 |

Structural operation result for cold 16384:

- dense source+suffix `_write_rows`: 120 total.
- tile-native `_write_rows`: 40 total, all decoder suffix; source/encoder dense carry writes were eliminated.
- tile telemetry at P6 detach: `dense_carry_writes=0`, `dense_layer_transport_slices=0`, `logical_swaps=20`, `tile_input_bindings=80`, `tile_output_bindings=80`, `full_source_assemblies=1`, `final_cone_tile_slices=3`, `final_cone_minimal_concats=0`, `initial_tile_views=4`.

State/P7/P5 evidence:

- all 40 cache frontiers reached 16384 in both A/B runs;
- P7 foreground fallback stayed 0 and expected background microtile reads were preserved;
- cache slot shape/dtype inventory in P5 matrix remained valid for source layers and tail layers;
- P6 C/E/D/T completed at 16384;
- source coverage remained `{2:16384, 8:16384, 14:16384, 20:16384}`;
- P5 tile smoke PASS: prompt replay `0`, full cache repack `0`, export `false`, generated tokens `[1, 0]`, decode frontiers advanced.

Memory proxy from final P6 materialization boundary after-state:

| path | active bytes cold | cache bytes cold | peak bytes cold | active bytes warm typical | cache bytes warm typical | peak bytes warm typical |
|---|---:|---:|---:|---:|---:|---:|
| dense | 314,864,784,632 | 17,004,751,578 | 319,629,252,136 | ~315,152,036,4xx | ~17.38-17.42B | 319,916,503,956 |
| tile-native | 314,864,915,708 | 18,558,717,170 | 318,765,159,990 | ~315,152,167,5xx | ~18.93-18.98B | 319,052,411,806 |

Interpretation: tile-native eliminated the intended source carry writes/slices, but warm median improvement was only ~0.092659 s (~0.57%), smaller than the same-revision dense warm run-to-run range (~0.141471 s). Candidate cache-memory proxy was also ~1.5 GB higher at the sampled final boundary, while peak proxy was lower. This is not a repeatable practical E2E gain.

### P8 completion decision

Decision: **TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN**.

Because the coherent structural candidate removed the intended operations but did not produce repeatable E2E value outside observed noise, P8 optimization search is **COMPLETE**. The qualified production candidate for Milestone 6 is the existing dense P0-P7 path. Tile-native carry remains retained experimental code, default OFF, and not production-selected. Per P8 stop rule, do not continue into Engram concat tweaks, Attention/MoE/HC profiling, `mx.compile`, or custom kernels under this task. Attention/MoE/HC further optimization is not pursued / not required for M6 unless a future explicit product/runtime requirement reopens component optimization.
