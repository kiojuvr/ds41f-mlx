# Implementation plan

This roadmap converts the runtime strategy into a finite sequence of milestones. It is not an optimization backlog. Runtime behavior, selectors, API integration, and multimodal work must not change until the relevant architecture milestone authorizes that work.

## Milestone 1 — Architecture restoration audit

Status: **complete**. See `docs/milestone-1-architecture-audit.md` for the concrete inventory, source evidence, component classifications, missing seams, and Milestone 2 starting frontier.

Deliverables:

- current component classification across `native/*`, DwarfStar-derived files, oMLX bindings, and validators;
- DwarfStar prefill implementation-state inventory, including pinned revision, sweep topology, planner state, native C ownership, Metal submission scaffolds, official data-plane primitives, measured artifacts, and missing full-model connection;
- oMLX runtime inventory, including checkpoint loading, DeepSeek-V4.1 decode topology, MTP/DSpark behavior, state/cache layout, memory behavior, and recorded baselines;
- legacy-native/reference inventory, including what is reusable as production component versus reference/qualification evidence;
- explicit reuse/replace decisions and missing production seams.

Rules:

- no model optimization;
- no selector changes;
- no long benchmark ladders;
- qualification results remain scoped to the implementation that produced them.

## Milestone 2 — Restore DwarfStar-derived prefill production path

Status: **complete**. See `docs/milestone-2-prefill-restoration-status.md` for the bounded official-checkpoint DwarfStar-derived production-prefill path through all 40 transformer layers, final logits, transaction commit, and executable neutral prefill continuation-state handoff.

Continue from the furthest real implementation point already present. Do not restart from a blank implementation.

Deliverables:

- DwarfStar-derived V4.1 sweep/lifetime topology connected to real official model execution;
- official checkpoint data path preserved, without adopting incompatible DwarfStar checkpoint or quantization assumptions;
- native carry/state/publication ownership reconciled with official semantics contracts;
- real full-model prefill path with DwarfStar-derived topology;
- correct state handoff into the selected/temporary decode boundary;
- promotion gates using current `ds41f` correctness contracts.

Goal:

```text
real full-model prefill
official checkpoint
DwarfStar-derived topology
correct state handoff into decode
```

## Milestone 3 — Decode architecture selection

Status: **complete**. See `docs/milestone-3-decode-architecture-decision.md`.

Decision: select oMLX `0.7.0.dev2` DeepSeek-V4.1 target decode architecture as the base production decode architecture, adapted to consume `PrefillContinuationState` without prompt recomputation. The selected topology is request-local `DeepseekV41Cache` ownership plus `LanguageModel._forward` target execution. DSpark/MTP speculative acceleration is staged after the base target decode adapter and correctness gates pass.

Rejected:

- DwarfStar decode as Milestone 4 base, because the inspected real decode state is private to `ds41_gpu_graph` C/Metal tensors and no public no-replay state-admission ABI exists for the M2 neutral handoff.
- Composition, because no clean state/lifetime/interface boundary avoids duplicate execution, cache conversion, conflicting graph ownership, and rollback ambiguity.
- Current native reference decode, which remains correctness/reference evidence only.

## Milestone 4 — Implement selected decode architecture

Status: **in progress / incomplete**. See `docs/milestone-4-base-decode-status.md`.

Implemented so far: real oMLX `DeepseekV41Cache` admission from live `PrefillContinuationState`, `OMLXDecodeSession`, base target execution, continuation, Engram state update, reset, fork, injected-failure rollback, removal of diagnostic cache scalar reads from the production token loop, native oMLX custom-kernel build/provenance, R/O/D full-logits triangulation, layer0-slot1 pack/unpack analysis, and P0-P5 execution-path controls. Current correctness frontier has advanced compositionally: Block0 COMPLETE; Engram@1 COMPLETE; Block1 COMPLETE; Layer2 COMPLETE; Layers3-7 COMPLETE; Layer8 COMPLETE; Layers9-13 COMPLETE; Engram@14 COMPLETE; Layer14 COMPLETE; Layers15-19 COMPLETE; Layer20 source/candidate generation COMPLETE; Layers21-23 COMPLETE; Layer24 candidate consumer/index refresh COMPLETE; Layer25 refreshed-topk consumption COMPLETE; Layers26-39 COMPLETE; final HC collapse COMPLETE; final RMSNorm COMPLETE; ParallelHead COMPLETE; bounded prefix `[0,3]` compositional correctness COMPLETE; committed continuation-state correctness COMPLETE; no-replay corrected-state admission COMPLETE; full admission semantic round-trip COMPLETE after preserving model-semantic compressed-KV FP4/E4M3 physical payloads. M2 correctness requalification is complete and no longer depends on oMLX decode correctness. The former parent prefill artifact failure was a validator-authority issue: stale historical connected bit-identity gates survived after M4 changed correctness authority to compositional reviewed boundaries. Current M4 frontier: **global numerical/behavioral stability policy for official-compatible decode**, not Layer2 continuation. Boundary13e remains the qualified source-derived authority, and real-loaded capture proved the earlier Layer2 sparse-call mismatch is downstream. Block0, Engram@1, and Block1 same-input correctness are COMPLETE on the exact inputs consumed by production. Block1 `AA` reproduces captured actual oMLX `x_out`/`ffn_pre`, so the connected Block1 mismatch is not a Block1 implementation defect. The causal seed audit shows the connected Block1 divergence is `OFFICIAL_PATH_AMPLIFICATION_OF_QUALIFIED_NUMERICAL_SEED`: one contract-valid 1-ULP Block0 Attention-output seed, injected into the expected official path, reproduces the actual downstream trajectory within existing contracts, including exact Block1 `q_rotary` and max-1-ULP Block1 `x_out`. Thus the 30k-ULP hidden-state divergence is real but is not currently evidence of oMLX-specific Q-path instability. The first DwarfStar re-evaluation audit remains valid and blocked direct comparison as `DWARFSTAR_DIRECT_TRAJECTORY_COMPARISON_BLOCKED`: pinned DwarfStar decode consumes GGUF dense BF16/Q8_0/Q4_K/Q4_0 projections, while official Block1 q projections are safetensors F8_E4M3 weights with F8_E8M0 scales. Do not compare against DwarfStar Q2/Q4 GGUF as an architecture-fidelity result, and do not build a DwarfStar FP8 Q path merely to chase closeness to one hidden-state reference trajectory. A lightweight isolated replay of real pinned oMLX Block0 Attention reproduces the captured endpoint exactly, but subsequent evidence corrects the causal attribution: the local `wq_b` 1-ULP difference is erased by `sparse_output`, and the local `wo_a` 1-ULP difference is erased by `wo_b` activation quantization. The surviving seed is at `wo_b`: the activation-quantized input is exact (digest `60c026...`), while `wo_b` `mx.quantized_matmul` returns final Attention `[3758] = 0x3f6d` instead of official `0x3f6c`. The `wq_b` M=16 result is retained only as the first tested full-row-exact M in that coarse sweep, not as a dispatch-threshold claim. The new `wo_b` topology sweep over M=1..12,16,32 finds duplicate rows equal but no existing MLX quantized topology exactifies the row; dequantized MLX dense also returns `0x3f6d`. Classification: `BLOCK0_WO_B_MLX_QUANTIZED_REDUCTION_FAMILY_DIVERGENCE`. The next single frontier is a canonical length-1 `wo_b` FP8 reduction kernel or an isolated CUDA oracle. This does not change the production selector.

Starting boundary from Milestone 3:

- implement a production oMLX-derived decode session type, e.g. under `ds41f_mlx/runtime/`;
- implement a no-prompt-replay adapter from `PrefillContinuationState` to 40 `DeepseekV41Cache` objects: slot 0 offsets, slot 1 window KV, slots 2/3 compressed/index state, slots 4/5 pending compressor state, slot 6 Engram history, plus validation for candidates/top-k/ownership/source order;
- model/cache owner is oMLX `LanguageModel` plus request-local cache list;
- first-token entry point is one-token `LanguageModel._forward`/`__call__` against admitted cache;
- preserve reset, fork, continuation commit, failure rollback, logits, Engram store identity, and qualification hooks;
- implement base target decode first; enable DSpark/MTP only after base decode gates pass;
- keep current native decode reference-only and leave DwarfStar-derived prefill unchanged.

Deliverables:

- selected oMLX-derived decode topology connected to the same production session/state boundary as prefill;
- official-semantics qualification gates preserved;
- practical single-stream decode restored before optional speculative acceleration, unless evidence from the genuine standard oMLX MTP-OFF path proves practical speed structurally starts at DSpark/MTP;
- MTP/speculative decoding evaluated only as a staged part of the selected architecture;
- reset, fork, continuation, and failure atomicity preserved.

Success is not defined as merely exceeding the current `0.31 tok/s` reference baseline. The target is practical local operation comparable to known oMLX-class behavior.

## Milestone 5 — Unified production runtime qualification

Verify the unified runtime for:

- prefill to first decode;
- continuation;
- reset;
- fork;
- long-lived session state;
- Engram;
- long context;
- memory behavior.

The historical native/reference implementation may be used as bounded evidence or oracle where appropriate, but it must not dictate execution topology. Qualification must be recorded as implementation-scoped.

## Milestone 6 — Performance qualification

Measure end-to-end performance with exact runtime/checkpoint/build/hardware provenance:

- short context;
- 32K;
- 64K;
- 128K;
- 200K or another documented practical upper target;
- prefill;
- decode;
- TTFT;
- memory.

Compare with the recorded oMLX baseline. Do not define success as merely faster than the current native reference decode rate.

## Milestone 7 — Serving integration

Use official DeepSeek `deepseek-recipe` as the protocol/prompt/response layer.

Target shape:

```text
HTTP transport
  ↓
deepseek-recipe
  ↓
ds41f backend interface
  ↓
production runtime
```

The project should not independently reinvent Chat Completions conversion, Responses conversion, DeepSeek V4.1 prompt encoding, tool-call parsing, thinking parsing, or stream response formatting.

## Later goals

- multimodal support;
- broader serving qualification;
- additional speculative execution;
- release hardening.

## Component disposition matrix

| Component | Current role | Intended future role | Retain/reuse/replace | Reason | Next milestone |
| --- | --- | --- | --- | --- | --- |
| `native/*` | Executable native model core and current correctness/reference runtime | Reference/qualification plus selectively reusable production components | Retain; reuse selectively | Valuable state, checkpoint, Engram, MoE/HC, generation, and tests; not automatically production topology | 1, 5 |
| `ds41f_mlx/dwarfstar_*` | DwarfStar-derived planners and prefill adapter evidence | Production prefill architecture line | Retain and continue | Captures pinned V4.1 sweep and interrupted implementation intent | 1, 2 |
| `ds41f_mlx/native/*` | C/Metal prefill ownership/submission/primitive scaffold | Candidate production prefill substrate | Retain and extend during prefill milestone | Owns carry buffers, command stream, bounded Metal submission, official primitive checks | 1, 2 |
| `ds41f_mlx/native_prefill.py` | Python bridge to native prefill ABI | Tooling bridge for restored prefill path | Retain | Needed to inspect/build existing native prefill seam | 1, 2 |
| `ds41f_mlx/m2_layer_major.py` | oMLX-compatible layer-major diagnostic | Tool/prototype evidence | Retain as diagnostic | Proved loop inversion alone was flat; helps avoid repeating rejected path | 1, 2 |
| `ds41f_mlx/m2_state_publication.py` | Explicit publication-frontier diagnostic | Tool/reference for state seam | Retain/reuse concepts | Encodes useful producer/consumer frontiers | 1, 2, 5 |
| `ds41f_mlx/runtime/omlx_*` | Thin oMLX bridge and worker | Decode candidate inventory/comparison harness | Retain | oMLX has strong real baseline and may supply selected decode architecture | 1, 3 |
| correctness artifacts / validators | Official-source-derived gates | Qualification gates for every implementation | Retain and extend | Correctness authority below official semantics; implementation-scoped | all |

## Planning guardrails

- Architecture changes require documentation first in `docs/runtime-strategy.md` and this file.
- Correctness fixes must preserve architecture intent or explicitly document structural consequences.
- Performance evidence must be end-to-end and provenance-recorded.
- Reference code may stay slow if it remains useful correctness evidence.
- No API or multimodal work should begin before runtime architecture restoration reaches the serving milestone.
