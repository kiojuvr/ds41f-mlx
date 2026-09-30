# DwarfStar prefill architecture-gap matrix

Pinned DwarfStar authority audited: `/Users/kioju/ds4` at `0aaea5a238fb41a35106a551e73c8409dfb751ac` (expected `antirez/ds4`).  ds41f audited at `59e5dd2733bbafca689a8422ed4154fa00b2ca0a` before the serving guard change.

Companion bounded diagnostics artifact: `artifacts/current-task/bounded_diagnostics.json`.

| Mechanism | ds41f classification | Gap summary |
|---|---:|---|
| Sweep structure | PRESENT_BUT_NOT_ON_HOT_PATH | Planner exists; serving enters Python vertical slice and per-layer reference helper. |
| Encoder full-row execution | PRESENT_BUT_ONLY_DOCUMENTED | Native headers/plans describe it; production model math is not native full-row. |
| Decoder suffix execution | PRESENT_BUT_ONLY_DOCUMENTED | Deferred suffix geometry exists; not used by serving. |
| Deferred decoder work | PRESENT_BUT_ONLY_DOCUMENTED | Transactions/policy exist in native headers; Python executor runs full 40 layers. |
| Chunk geometry | PRESENT_BUT_NOT_ON_HOT_PATH | Planner records chunks; executor only accepts offset=0/full rows for current slice. |
| Carry-buffer lifetime | PRESENT_BUT_NOT_ON_HOT_PATH | Python ping-pong exists but copies/digests intermediates and uses reference math. |
| Intermediate tensor retention | MISSING | Layer records, digests, and copied arrays are retained for validation. |
| Metal command ownership | PRESENT_BUT_NOT_ON_HOT_PATH | Native scaffolding owns planner/buffer ops only; model hot path is Python/NumPy. |
| Graph construction/reuse | MISSING | Serving rebuilds native planner/reference math per request; no reused production prefill graph. |
| Synchronization/materialization boundaries | MISSING | Reference path materializes CPU arrays and computes digests per layer. |
| Weight preparation lifetime | MISSING | Validation helpers mmap/read tensors inside helper calls; no production weight lifetime. |
| Expert scheduling/cache seeding | PRESENT_BUT_ONLY_DOCUMENTED | Plan has seed steps; no active resident expert cache in vertical slice. |
| Engram prefetch/read-ahead | PRESENT_BUT_NOT_ON_HOT_PATH | Hash/reference Engram helpers run; no production read-ahead ownership. |
| Checkpoint-validity transitions | PRESENT_AND_ACTIVE | Transaction/commit state exists, but around reference execution. |
| Source publication timing | PRESENT_AND_ACTIVE | Shared publication/ownership lifecycle is active in Python executor. |
| Output-head timing | MISSING | Full final logits are always computed for `layers>=40`; serving prefix prefill should skip them. |
| Work DwarfStar deliberately avoids | MISSING | Serving path did validation block math, per-layer artifacts, copies, digests, and final head. |
| Official FP8 checkpoint applicability | INAPPLICABLE_TO_OFFICIAL_FP8_CHECKPOINT | DwarfStar GGUF kernels are not directly portable as official FP8 semantics. |

## Hot path before guard

`recipe token IDs -> DeepSeekRecipeRuntimeBackend.infer() -> tools.run_m4_omlx_base_decode_qualification.build_prefill_state() -> DwarfStarPrefillVerticalSliceExecutor.run() -> OfficialModelMath.execute_block() -> official-source-derived validation helper block()`.

Component classes: Python orchestration plus native planner scaffold, NumPy/reference validation math, then oMLX decode admission after prefill. This is **not** a production DwarfStar/Metal prefill path.

Classification if encountered: `PRODUCTION_PREFILL_REGRESSED_TO_REFERENCE_VERTICAL_SLICE`.

## Dominant structural cause

The current short-prompt pathology is caused by production serving entering a full official-source-derived NumPy/reference validation prefill. DwarfStar-derived topology is present but performance architecture is missing from the model-math hot path.

## First change

`ds41f_mlx/serving/deepseek_recipe_backend.py` now refuses the reference vertical-slice serving prefill by default. It can be enabled only for bounded diagnostics with `DS41F_ALLOW_REFERENCE_VERTICAL_SLICE_SERVING=1`.
