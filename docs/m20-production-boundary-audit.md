# M20 production-reachable dependency audit

Status: source audit completed before runtime adaptation. Subsequent bounded
compatibility/lifecycle, persistence/HTTP/tool/EOS, repeated-session and targeted
200K endpoint gates passed; see [M20](m20-omlx-release-migration.md). The seam
classifications below describe the migration requirements discovered by the audit.
Comparison: dev2 `b390b31e0c6831225fed0f24d278eb1db7fcb68b` plus its recorded
non-production patches versus clean release `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`.
mlx-lm changes from `ab1806e8` to `94cdcae13` are part of this dependency migration.

## Reachability

`OmlxRuntime.load_model` directly imports `deepseek_v41.loading.load`, not the
HTTP engine or generic oMLX model loader. ds41f's DENSE_P0_P7 executor uses
loaded V4.1 layer operations and publishes the seven cache slots. P5 passes the
live cache plus prefix history into `OMLXGenerationSession`; it imports
`mlx_lm.generate` and then `omlx.scheduler` for module-level monkey patches,
not a Scheduler instance. M8 extracts at idle boundaries and P6/P7 append in
place. M9 persists explicit `cache.cache` slots. Recipe owns rendering, tool
parsing and EOS suppression; oMLX does not own the serving protocol.

This audit does not qualify optional paths or review all of oMLX. Supporting
source diffs are in `artifacts/m20/reachable-{omlx,mlx-lm,mlx-cache}.diff`.

| Seam | Classification | Finding / required confirmation |
| --- | --- | --- |
| Direct model load | COMPATIBLE_CHANGE | `ced_prefill=False` added, preserve_mtp explicitly False in production. Official-source Engram discovery now accepts raw config and supports packed affine tables. No expert-offload mode requested. Load real checkpoint before acceptance. |
| V4.1 patch selection | COMPATIBLE_CHANGE | apply_patch forcibly installs vendor alias rather than setdefault. Direct loading already selects vendor model. No generic upstream V4.1 fallback permitted. |
| Seven-slot live cache | COMPATIBLE_CHANGE | Slot layout/meta v3 unchanged. extract trims padding by row offset/ratio; singleton extraction should keep exact logical shapes. New from_state override is important for new ArraysCache state representation. |
| ArraysCache state | COMPATIBLE_CHANGE | mlx-lm `.state` is now `(cache,left_padding,lengths)`, not seven-array list. ds41f production accesses explicit slots/`.cache`, including M9 persistence, rather than inheriting this serialization API. V4.1 from_state reconstructs its explicit custom state. Test save/restore. |
| BatchGenerator admission | COMPATIBLE_CHANGE | Existing insert(caches,all_tokens,samplers) retained. New normalization/length validation occurs before pending state insertion. ds41f supplies single concrete sampler/cache/history. |
| GenerationBatch terminal bootstrap | COMPATIBLE_CHANGE | Constructor still forwards one held-out token via `_step`; subsequent next consumes sampled token. Existing private `_generation_batch.prompt_cache` observation retained. StopSequences replaces SequenceStateMachine. |
| P5 handoff | UNCHANGED | ds41f handoff remains same-cache, held-out terminal, zero prefix replay/full-cache repack, one executable authority. Release behavior requires real-model proof, not source inference. |
| Append/advance | COMPATIBLE_CHANGE | Optional CED parameters default absent. Normal index origins and compressor/cache lifecycle retained. Attention old-window length now bounded by actual shape; ds41f suffix adapter already does this. |
| Rollback | UNCHANGED | No speculative rollback in MTP-OFF production. P6 failure invalidates append state rather than reconstructing/replaying. Do not introduce speculative trim/rollback semantics. |
| Cancel/extract/close | DS41F_ADAPTATION_REQUIRED | Both mlx-lm versions' extract_cache observes but does not remove a UID; close only restores wired limit. ds41f stop currently extracts without remove, leaving old GenerationBatch alive until GC after continuation. Explicit remove after extraction and deterministic close/failure draining are required for the ownership/leak gate. This is inherited behavior, not a release-only regression. |
| History ownership | COMPATIBLE_CHANGE | GenerationBatch.tokens remains consumed-cache history, not lookahead token. Response now carries all_tokens on natural finish; use backend-owned history and verify frontier. Prefix metadata must not become another executable cache. |
| Engram/index decode lifecycle | COMPATIBLE_CHANGE | Engram math/hasher and index key slot remain same; loading supports additional table metadata, storage dtype map extracted to constant. History slot 6 continues on layer 0. Repeated-turn and restore tests required. |
| Request vs persistent state | DS41F_ADAPTATION_REQUIRED | Per-request generators must be drained when idle/closed. Scheduler module has bounded sampler/processor registry, but registration happens in Scheduler methods not used by direct ds41f insert. No Scheduler instance or shared persistent GenerationBatch should be introduced. |
| Memory guard/admission | COMPATIBLE_CHANGE | Scheduler instance admission guards are not used by ds41f. New mlx-lm maybe_set_recommended_wired_limit replaces older model-byte threshold; synchronize/restore on close retained. Native memory ceilings/RSS need measurement. |
| Native loading | COMPATIBLE_CHANGE | MLX remains 0.32.2/nanobind 2.15.0. V4.1 uses glm_moe_dsa symbols plus dynamic Metal kernels. Rebuild for CPython 3.13, test actual array operations; import success alone is not execution proof. Changed GLM header concerns DSpark/Qwen high-index tie ordering, not the OFF V4.1 path. |
| Sampler/EOS | COMPATIBLE_CHANGE | Concrete sampler normalization, batched shared-sampler fast path. Single-row argmax policy retained. StopSequences matches stop_tokens and overrides length at EOS; recipe remains termination/protocol authority. Natural finish returns cache and consumed history. |
| Private APIs | COMPATIBLE_CHANGE | Direct BatchGenerator insert/next/extract/remove, private `_generation_batch.prompt_cache`, V4.1 `_config`, `_forward`, layer math, cache slots all remain reachable. Do not import removed SequenceStateMachine or old stats internals. Cheap real-array probe before full-model gates. |
| oMLX native prefill | NO_LONGER_APPROPRIATE_DEPENDENCY | Not the ds41f prefill authority. New CED and >=256-token layer eval/cache clearing in LanguageModel._forward must not replace DENSE_P0_P7. One-token decode is below backpressure threshold. |

## Historical patches

Upstream release encoding now sanitizes literal image placeholders to `[image]`.
It is not the same historical dev2 passthrough patch; do not describe byte-level
identity or copy it. processing.py is unchanged between upstream revisions and
the dev2 processing fix remains local. The historical test remains only in dev2.
All three retain their prior non-production classification: ds41f text prompts
come from DeepSeek-recipe, not oMLX process_image_messages/processor encoding.
No release source patch is justified by these differences.

## Adaptation boundary

Only deterministic generator ownership/cleanup and backend-owned final history
are identified for ds41f adaptation at this point. Do not make release emulate
removed internal stats/state-machine structures. Preserve ds41f prefill and all
OFF selectors. Defaults/provenance pins stay dev2 until promotion.
