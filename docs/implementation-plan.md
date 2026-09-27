# Implementation plan

This roadmap converts the runtime strategy into a finite sequence of milestones. It is not an optimization backlog. Runtime behavior, selectors, API integration, and multimodal work must not change until the relevant architecture milestone authorizes that work.

## Milestone 1 — Architecture restoration audit

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

Perform a bounded architecture comparison before implementing decode.

Candidates:

- DwarfStar DeepSeek-V4.1 decode architecture;
- oMLX `0.7.0.dev2` DeepSeek-V4.1-Flash decode architecture;
- a composition only if state/lifetime/interface boundaries are explicit.

Compare at minimum:

- token execution topology;
- layer scheduling;
- graph ownership/lifetime;
- MLX/Metal evaluation and synchronization boundaries;
- attention decode path;
- KV/state ownership and update strategy;
- MoE routing and routed expert scheduling;
- resident/streamed expert strategy;
- MTP/speculative decoding architecture;
- state publication and transaction semantics;
- long-context behavior;
- memory residency;
- batch/concurrency implications;
- ability to preserve official DeepSeek semantics;
- difficulty of integrating existing `ds41f` correctness contracts.

Deliverable: an architecture decision document or an update to this plan. Do not begin decode implementation before this decision unless repository evidence already makes the decision conclusive.

## Milestone 4 — Implement selected decode architecture

Deliverables:

- selected decode topology connected to the same production session/state boundary as prefill;
- official-semantics qualification gates preserved;
- practical single-stream decode restored before optional speculative acceleration;
- MTP/speculative decoding evaluated only as part of the selected architecture where applicable;
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
